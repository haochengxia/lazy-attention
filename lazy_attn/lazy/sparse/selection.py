"""Selection, compaction and the walk-table build, as one kernel.

`Router._score_blocks` is already a single Triton launch. Everything after it
is not: ranking the candidates, applying the budget and the sink stripe,
compacting the kept rows to the left and rebuilding `seq_lens` is about 470
elementwise PyTorch operations over `[R, W]` tensors, and each line of it is
its own kernel. Measured at 600 documents, one `route` call issues **477 host
operations to submit 221 CUDA kernels -- 2.78 ms of host time for 0.79 ms of
GPU work**. The arithmetic is trivial; the dispatch is the cost.

It collapses because the block table has a hard width bound. It is sized for
`max_model_len`, so at this model's 131k context it is `131072 / 16 = 8192`
columns, and a request's score vector is therefore at most **32 KB of fp32** --
small enough to sit in one program's registers. So one program per request can
hold the whole row and do the entire tail locally, with no round trip through
global memory between steps and no second launch.

**Selection without a sort.** The torch path ranks with `argsort` and scatters
a per-request cutoff, because `topk` would need `k` on the host. Here the same
answer comes from a binary search on the score's bit pattern: 32 iterations,
each one a masked count of "how many candidates score at least this", converging
on the exact `budget`-th largest key. Floats are mapped to an order-preserving
unsigned key first, the standard sign-magnitude flip, so integer comparison
reproduces float comparison including the `-inf` that masks non-candidates.
Ties at the threshold are broken by lower index, which the torch path leaves to
whatever the radix sort did -- so the two agree exactly whenever scores are
distinct, and both are valid selections when they are not.

**What it does not do.** `doc` and `prefix` granularity stay on the torch path:
`prefix` needs a per-document scatter-reduce after selection and `doc` ranks
documents rather than pages, and neither is the default. `can_fuse_selection`
is what decides, and the fallback is the original code unchanged -- this file
adds a path, it does not replace one. `tests/sparse/test_selection.py` holds the
two together.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# `_FORCED` in router.py: the sink stripe's score bonus. Large enough to sort
# above any real Quest bound, finite so it survives the float-to-key mapping.
_FORCED = 1e30
_FORCED_TL = tl.constexpr(_FORCED)
# One program holds a whole row, so the row has to fit. This is the block
# table's own bound (`max_model_len / block_size`), not a new restriction --
# but a model with a longer context than 128k at block 16 would exceed it, and
# `can_fuse_selection` sends that case back to the torch path rather than
# silently truncating the corpus.
MAX_FUSED_WIDTH = 8192


@triton.jit
def _order_key(score):
    """Map float32 to an unsigned key whose integer order is the float order.

    Sign-magnitude: non-negative floats keep their bits and gain the top bit;
    negative floats are complemented, which reverses them. Done in int64 so the
    zero-extension is explicit and there is no unsigned arithmetic to get wrong.
    """
    bits = score.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)


@triton.jit
def _rint(x):
    """Round half to even, matching `torch.round`.

    `floor(x + 0.5)` is not the same function: it sends 2.5 to 3 where torch
    sends it to 2, and a budget of `round(n * 0.25)` lands on a half often
    enough for that to change which rows are read.
    """
    down = tl.floor(x)
    half = (x - down) == 0.5
    even = (down * 0.5) == tl.floor(down * 0.5)
    return tl.where(half, tl.where(even, down, down + 1.0), tl.floor(x + 0.5))


@triton.jit
def kernel_select_and_compact(
        walk_ptr,
        out_lens_ptr,
        stat_kept_ptr,
        stat_dense_ptr,
        stat_guard_ptr,
        packed_ptr,
        scores_ptr,
        doc_id_ptr,
        seq_lens_ptr,
        num_doc_blocks_ptr,
        routable_ptr,
        width,
        packed_stride,
        scores_stride,
        doc_id_stride,
        walk_stride,
        block_size,
        budget_frac,
        BUDGET_ABS: tl.constexpr,
        KEEP_DOC0: tl.constexpr,
        SINK_STRIPE: tl.constexpr,
        BLOCK_W: tl.constexpr):
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_W)
    live = offs < width

    packed = tl.load(packed_ptr + req * packed_stride + offs, mask=live,
                     other=0)
    doc_id = tl.load(doc_id_ptr + req * doc_id_stride + offs, mask=live,
                     other=-1).to(tl.int32)
    score = tl.load(scores_ptr + req * scores_stride + offs, mask=live,
                    other=float("-inf"))
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    num_doc_blocks = tl.load(num_doc_blocks_ptr + req).to(tl.int32)
    routable = tl.load(routable_ptr + req) != 0

    # -- geometry, the same derivation `Router._geometry` does ---------------
    num_blocks = (seq_len + block_size - 1) // block_size
    block_valid = live & (offs < num_blocks)
    doc_mask = block_valid & (offs < num_doc_blocks)
    tail = block_valid & (doc_mask == 0)
    if KEEP_DOC0:
        preamble = doc_mask & (doc_id == 0)
        candidate = doc_mask & (doc_id > 0)
    else:
        preamble = doc_mask & False
        candidate = doc_mask
    free_rows = preamble | tail

    num_candidates = tl.sum(candidate.to(tl.int32), axis=0)
    if BUDGET_ABS >= 0:
        budget = tl.minimum(BUDGET_ABS // block_size, num_candidates)
    else:
        scaled = _rint(num_candidates.to(tl.float32) * budget_frac).to(tl.int32)
        budget = tl.minimum(tl.maximum(scaled, 1), num_candidates)

    # -- sink stripe, with §9b's budget guard --------------------------------
    if SINK_STRIPE:
        previous = tl.load(doc_id_ptr + req * doc_id_stride + offs - 1,
                           mask=live & (offs > 0), other=-2).to(tl.int32)
        stripe = candidate & ((offs == 0) | (doc_id != previous))
        affordable = (tl.sum(stripe.to(tl.int32), axis=0) * 2) <= budget
        tl.atomic_add(stat_guard_ptr,
                      tl.where(affordable, 0, 1).to(tl.int64))
        score = score + tl.where(stripe & affordable, _FORCED_TL, 0.0)

    # -- top-`budget` by score, by bisection on the key ----------------------
    priority = tl.where(candidate, score, float("-inf"))
    key = _order_key(priority)

    # Invariant: at least `budget` candidates have key >= lo, fewer than
    # `budget` have key >= hi. Thirty-two halvings of a 32-bit space close it
    # exactly, so `lo` ends on the budget-th largest key itself.
    lo = tl.zeros((), dtype=tl.int64)
    hi = tl.full((), 1 << 32, dtype=tl.int64)
    for _ in range(32):
        mid = (lo + hi) // 2
        count = tl.sum((candidate & (key >= mid)).to(tl.int32), axis=0)
        take = count >= budget
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)

    above = candidate & (key > lo)
    num_above = tl.sum(above.to(tl.int32), axis=0)
    # Everything tied at the threshold, taken in index order until the budget
    # is met. The torch path leaves this to the sort's own tie-break; with
    # distinct scores the two coincide, and with ties both are valid.
    equal = candidate & (key == lo)
    equal_rank = tl.cumsum(equal.to(tl.int32), axis=0) - 1
    chosen = above | (equal & (equal_rank < (budget - num_above)))
    chosen = tl.where(budget > 0, chosen, candidate & False)

    keep = free_rows | chosen

    # -- compaction: kept rows to the left, in their original order ----------
    kept_rank = tl.cumsum(keep.to(tl.int32), axis=0)
    kept_total = tl.sum(keep.to(tl.int32), axis=0)
    dropped_rank = tl.cumsum((keep == 0).to(tl.int32), axis=0) + kept_total
    destination = tl.where(keep, kept_rank, dropped_rank) - 1

    # A non-routable row keeps its dense table: the identity permutation is
    # exactly `torch.where(routable, sub_walk, packed)` without a second pass.
    destination = tl.where(routable, destination, offs)
    tl.store(walk_ptr + req * walk_stride + destination, packed, mask=live)

    kept_doc_rows = tl.sum((keep & doc_mask).to(tl.int32), axis=0)
    rows_dense = tl.sum(block_valid.to(tl.int32), axis=0)
    tail_len = seq_len - num_doc_blocks * block_size
    sub_len = kept_doc_rows * block_size + tail_len
    tl.store(out_lens_ptr + req, tl.where(routable, sub_len, seq_len))

    rows_kept = tl.where(routable, tl.sum(keep.to(tl.int32), axis=0),
                         rows_dense)
    tl.atomic_add(stat_kept_ptr, rows_kept.to(tl.int64))
    tl.atomic_add(stat_dense_ptr, rows_dense.to(tl.int64))


def can_fuse_selection(config, width: int) -> bool:
    """Whether this step's selection is one the kernel reproduces exactly.

    Conservative on purpose: anything outside the fused path's contract falls
    back to the torch implementation, which is unchanged and remains the
    reference the tests compare against.
    """
    return (config.granularity == "page"
            and config.budget_docs == 0
            and config.sink_stripe in ("off", "selected", "all")
            and 0 < width <= MAX_FUSED_WIDTH)


def select_and_compact(packed: torch.Tensor, scores: torch.Tensor,
                       doc_id: torch.Tensor, seq_lens: torch.Tensor,
                       num_doc_blocks: torch.Tensor, routable: torch.Tensor,
                       block_size: int, config,
                       stat_kept: torch.Tensor, stat_dense: torch.Tensor,
                       stat_guard: torch.Tensor):
    """One launch: geometry, selection, compaction, lengths and counters.

    Returns `(walk, out_lens, rows_kept, rows_dense)` where the last two are
    `None` -- they are accumulated into the device counters directly, and
    materialising per-request copies would put two more reductions back on the
    host path this exists to empty.
    """
    num_reqs, width = packed.shape
    walk = torch.empty_like(packed)
    out_lens = torch.empty((num_reqs, ), dtype=torch.int32,
                           device=packed.device)

    absolute = (int(config.budget_tokens) if config.budget_tokens > 1.0
                else -1)
    kernel_select_and_compact[(num_reqs, )](
        walk_ptr=walk,
        out_lens_ptr=out_lens,
        stat_kept_ptr=stat_kept,
        stat_dense_ptr=stat_dense,
        stat_guard_ptr=stat_guard,
        packed_ptr=packed,
        scores_ptr=scores,
        doc_id_ptr=doc_id,
        seq_lens_ptr=seq_lens,
        num_doc_blocks_ptr=num_doc_blocks,
        routable_ptr=routable,
        width=width,
        packed_stride=packed.stride(0),
        scores_stride=scores.stride(0),
        doc_id_stride=doc_id.stride(0),
        walk_stride=walk.stride(0),
        block_size=block_size,
        budget_frac=float(config.budget_tokens),
        BUDGET_ABS=absolute,
        KEEP_DOC0=bool(config.keep_doc0),
        SINK_STRIPE=config.sink_stripe != "off",
        BLOCK_W=triton.next_power_of_2(width),
    )
    return walk, out_lens
