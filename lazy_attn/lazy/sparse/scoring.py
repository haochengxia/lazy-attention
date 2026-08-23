"""The Quest bound for every cached block, in one kernel.

The torch formulation of this is a gather, an `abs`, two batched GEMVs and two
reductions, and it is slow for a reason that has nothing to do with arithmetic.
Scoring 5404 blocks needs 44 MFLOP and reads 11 MB of descriptors -- about
20 microseconds of this card. It was taking 570.

Two causes, both structural:

* **The query broadcast is materialised.** A block's score needs its own
  document's query, so the torch version gathers `q` out to every block,
  building a `[blocks, kv_head, group, dim]` tensor -- 44 MB at M=600, written
  and then read twice more by `abs` and the GEMVs. The query itself is 1.2 MB.
  There are only ever `num_documents` distinct values in that tensor.
* **43 232 batched GEMVs of shape 4x64x1.** cuBLAS has nothing useful to do
  with a batch of matrix-vector products that small.

Here each program owns one (request, block), reads that block's box, reads its
document's query straight out of the small per-document table, and writes one
scalar. The broadcast happens in registers and never reaches memory; blocks of
the same document hit the same query lines in L2.

This covers `scorer=quest`, which is the default and the only one that runs in
a serving path. `oracle`, `centroid` and `random` stay on the torch path in
`router.py`: they are evaluation arms, `oracle` deliberately reads the whole KV
cache, and none of them is on a hot path worth a second kernel.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from lazy.sparse.descriptors import MAX, MIN


@triton.jit
def _quest_score_kernel(
    scores_ptr,
    q_ptr,
    desc_ptr,
    valid_ptr,
    phys_ptr,
    doc_ptr,
    num_blocks,
    num_phys,
    num_docs,
    q_stride_r,
    q_stride_m,
    q_stride_h,
    d_stride_p,
    d_stride_h,
    d_stride_s,
    NUM_KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    MIN_SLOT: tl.constexpr,
    MAX_SLOT: tl.constexpr,
    AGG_SUM: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_blocks
    col = pid % num_blocks
    idx = row * num_blocks + col

    # Clamped rather than masked: a row outside this request's own extent still
    # holds a real physical id belonging to some other request, and its score is
    # discarded by `_select`'s candidate mask. Clamping only guards against a
    # sentinel, and costs two instructions against a branch.
    phys = tl.load(phys_ptr + idx)
    phys = tl.minimum(tl.maximum(phys, 0), num_phys - 1)
    described = tl.load(valid_ptr + phys) != 0

    doc = tl.load(doc_ptr + idx)
    doc = tl.minimum(tl.maximum(doc, 0), num_docs - 1)

    dim = tl.arange(0, BLOCK_D)
    dim_mask = dim < HEAD_SIZE
    grp = tl.arange(0, BLOCK_G)
    grp_mask = grp < GROUP

    best = float("-inf")
    for head in range(NUM_KV_HEADS):
        base = desc_ptr + phys * d_stride_p + head * d_stride_h
        lo = tl.load(base + MIN_SLOT * d_stride_s + dim,
                     mask=dim_mask, other=0.0).to(tl.float32)
        hi = tl.load(base + MAX_SLOT * d_stride_s + dim,
                     mask=dim_mask, other=0.0).to(tl.float32)
        # The GQA group's queries for this kv head, [BLOCK_G, BLOCK_D].
        query = tl.load(q_ptr + row * q_stride_r + doc * q_stride_m +
                        (head * GROUP + grp[:, None]) * q_stride_h +
                        dim[None, :],
                        mask=grp_mask[:, None] & dim_mask[None, :],
                        other=0.0).to(tl.float32)
        # sum_d max(q_d*lo_d, q_d*hi_d) -- the bound in its original form. The
        # centre/half-width rewrite exists in torch only to turn an elementwise
        # max over a materialised product into two contractions; here nothing
        # is materialised, so the direct form is both cheaper and exact.
        bound = tl.sum(tl.maximum(query * lo[None, :], query * hi[None, :]),
                       axis=1)
        if AGG_SUM:
            aggregated = tl.sum(tl.where(grp_mask, bound, 0.0), axis=0)
        else:
            aggregated = tl.max(tl.where(grp_mask, bound, float("-inf")),
                                axis=0)
        best = tl.maximum(best, aggregated)

    # Undescribed blocks score +inf, i.e. are always read: a lifecycle bug
    # degrades to a dense walk rather than to scoring one document's keys
    # against another's statistics.
    tl.store(scores_ptr + idx, tl.where(described, best, float("inf")))


# The kernel reads the descriptor through pointer arithmetic and converts to
# fp32; anything Triton can load and cast works. fp8 is excluded because its
# two variants need an explicit reinterpret the torch path already handles.
FUSED_DESC_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def can_fuse(scorer: str, desc: Optional[torch.Tensor]) -> bool:
    return (scorer == "quest" and desc is not None
            and desc.dtype in FUSED_DESC_DTYPES)


def score_blocks(q_by_doc: torch.Tensor, desc: torch.Tensor,
                 valid: torch.Tensor, phys: torch.Tensor,
                 doc_id: torch.Tensor, num_kv_heads: int,
                 gqa_agg: str) -> torch.Tensor:
    """`[R, MB]` fp32 scores. See the module docstring for the layout."""
    num_reqs, num_blocks = phys.shape
    num_q_heads, head_size = q_by_doc.shape[2], q_by_doc.shape[3]
    scores = torch.empty((num_reqs, num_blocks),
                         dtype=torch.float32,
                         device=phys.device)
    _quest_score_kernel[(num_reqs * num_blocks, )](
        scores,
        q_by_doc,
        desc,
        # bool and int8 share a storage layout, and Triton has no bool load.
        valid.view(torch.int8),
        phys,
        doc_id,
        num_blocks,
        desc.shape[0],
        q_by_doc.shape[1],
        q_by_doc.stride(0),
        q_by_doc.stride(1),
        q_by_doc.stride(2),
        desc.stride(0),
        desc.stride(1),
        desc.stride(2),
        NUM_KV_HEADS=num_kv_heads,
        GROUP=num_q_heads // num_kv_heads,
        BLOCK_G=triton.next_power_of_2(num_q_heads // num_kv_heads),
        BLOCK_D=triton.next_power_of_2(head_size),
        HEAD_SIZE=head_size,
        MIN_SLOT=MIN,
        MAX_SLOT=MAX,
        AGG_SUM=(gqa_agg == "sum"),
    )
    return scores
