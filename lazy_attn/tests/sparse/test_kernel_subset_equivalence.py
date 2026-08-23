"""T2. The compacted walk table means what we claim it means, to the kernel.

Every other test in this directory checks the walk table as data. This one
checks it as *instructions*: it runs the real Triton decode kernel over a
compacted table and compares against attention computed in torch over exactly
the keys that table selects, with each document's query de-rotation and each
document's padding mask applied by hand.

That closes the loop the whole design rests on -- PROJECT.md's invariant 6,
"dropping a document while keeping every retained row's original Δ reproduces
the dense layout's positions with holes, so sparse output = exact subset
attention". Four separate things have to be right at once for this to pass, and
each of them is a plausible way to be silently wrong:

* the walk table addresses the intended physical blocks,
* `seq_lens` stops the walk in the right place and masks the final tail row,
* `q_mask` still excludes each document's padding after its row has moved,
* the per-document rotation offset survives compaction, including the tail's
  `q_offset == 0` rows that inherit the rotation of the row before them.
"""
import pytest
import torch
import triton

from conftest import rope_table
from lazy.attention.ops.models.llama_v1 import kernel_paged_attention_2d_llama
from lazy.sparse.descriptors import DescriptorStore
from lazy.sparse.router import Router, RouterConfig, derotate_query
from lazy.utils.rotation import MAX_PACKED_Q_MASK, MAX_PACKED_Q_OFFSET, \
    PACKED_Q_OFFSET_SHIFT

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

HEAD_SIZE = 64
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
X = 8
DTYPE = torch.float32


def _run_kernel(query, key_cache, value_cache, table, seq_lens, cos_sin):
    output = torch.empty_like(query)
    kernel_paged_attention_2d_llama[(1, NUM_KV_HEADS)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=table,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=HEAD_SIZE**-0.5,
        k_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        v_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        num_query_heads=NUM_Q_HEADS,
        num_queries_per_kv=NUM_Q_HEADS // NUM_KV_HEADS,
        num_queries_per_kv_padded=max(
            triton.next_power_of_2(NUM_Q_HEADS // NUM_KV_HEADS), 16),
        block_table_stride=table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        IN_PRECISION="ieee",
        BLOCK_SIZE=16,
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=triton.next_power_of_2(HEAD_SIZE),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=torch.tensor([0, 1],
                                         dtype=torch.int32,
                                         device="cuda"),
        rotary_dim=HEAD_SIZE,
        rotary_dim_pow2=triton.next_power_of_2(HEAD_SIZE),
        is_neox_style=True,
        is_lazy_ptr=torch.tensor([True], device="cuda"),
        q_offset_ptr=None,
        q_mask_ptr=None,
        cos_sin_cache_ptr=cos_sin,
        IGNORE_Q_MASK=False,
        COMPUTE_COS_SIN=False,
    )
    return output


def _reference(query, key_cache, value_cache, table, seq_len, cos_sin,
               block_size=16):
    """Attention over exactly the keys the walk table selects, in torch.

    Deliberately written as the kernel's contract rather than as a rewrite of
    its body: walk the rows `cdiv(seq_len, BLOCK_SIZE)` says to walk, mask each
    by `q_mask` and by the running `seq_offset < seq_len` boundary, de-rotate
    the query per row's offset -- carrying the previous rotation forward
    through `q_offset == 0` rows as the kernel does -- and softmax over
    whatever survives.
    """
    trips = (seq_len + block_size - 1) // block_size
    scores, values = [], []
    rotated = query.float()

    for j in range(trips):
        entry = int(table[0, j])
        block = entry >> 32
        offset = (entry >> PACKED_Q_OFFSET_SHIFT) & MAX_PACKED_Q_OFFSET
        mask = entry & MAX_PACKED_Q_MASK

        if offset == 1:
            rotated = query.float()
        elif offset != 0:
            rotated = derotate_query(
                query, torch.tensor([[offset]], device=query.device),
                cos_sin, HEAD_SIZE)[:, 0]
        # offset == 0 keeps whatever the previous row installed.

        keys = key_cache[block].permute(0, 2, 1, 3).reshape(
            NUM_KV_HEADS, block_size, HEAD_SIZE).float()
        vals = value_cache[block].permute(0, 2, 1).float()  # [H, P, D]

        live = torch.arange(block_size, device=query.device)
        live = (live < (block_size - mask)) & ((j * block_size + live) < seq_len)
        if not bool(live.any()):
            continue

        # [H_q, P] : each query head against its own KV head's keys.
        per_head = torch.einsum(
            "qd,qpd->qp", rotated[0],
            keys.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=0))
        scores.append(
            (per_head * HEAD_SIZE**-0.5).masked_fill(~live[None, :],
                                                     float("-inf")))
        values.append(vals.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS,
                                             dim=0))

    weights = torch.softmax(torch.cat(scores, dim=1), dim=-1)
    return torch.einsum("qp,qpd->qd", weights, torch.cat(values,
                                                         dim=1))[None, :, :]


@pytest.mark.parametrize("budget,stripe", [(1.0, "off"), (0.5, "off"),
                                           (0.25, "selected"), (0.1, "off")])
def test_kernel_over_walk_table_equals_subset_attention(layout_factory, budget,
                                                        stripe):
    torch.manual_seed(0)
    # Ragged documents, so every one of them carries padding that `q_mask` has
    # to keep excluding once its row has moved.
    layout = layout_factory([60, 48, 77, 93, 29], tail_tokens=20)
    num_blocks = int(layout.block_ids.max()) + 2

    key_cache = torch.randn(num_blocks,
                            NUM_KV_HEADS,
                            HEAD_SIZE // X,
                            layout.block_size,
                            X,
                            dtype=DTYPE,
                            device="cuda")
    value_cache = torch.randn(num_blocks,
                              NUM_KV_HEADS,
                              HEAD_SIZE,
                              layout.block_size,
                              dtype=DTYPE,
                              device="cuda")
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, dtype=DTYPE, device="cuda")
    cos_sin = rope_table(HEAD_SIZE, 4096).to(DTYPE)

    store = DescriptorStore("bf16")
    store.fill("layer", key_cache, layout.block_ids,
               layout.block_size - layout.q_mask.to(torch.int32))

    walk = Router(RouterConfig(budget_tokens=budget, sink_stripe=stripe),
                  store).route(layer_name="layer",
                               query=query,
                               cos_sin_cache=cos_sin,
                               rotary_dim=HEAD_SIZE,
                               num_kv_heads=NUM_KV_HEADS,
                               key_cache=key_cache,
                               **layout.route_kwargs())

    got = _run_kernel(query, key_cache, value_cache, walk.block_table,
                      walk.seq_lens, cos_sin)
    want = _reference(query, key_cache, value_cache, walk.block_table,
                      int(walk.seq_lens[0]), cos_sin, layout.block_size)

    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


def test_dense_and_sparse_agree_on_the_pages_they_share(layout_factory):
    """The subset claim, stated as a difference rather than a reconstruction.

    Running the kernel at budget=1.0 and at a tight budget must differ *only*
    because keys were dropped: recomputing the tight-budget reference from the
    dense table restricted to the selected rows has to reproduce the sparse
    kernel output. If compaction perturbed a rotation or a mask, these diverge.
    """
    torch.manual_seed(1)
    layout = layout_factory([60, 48, 77], tail_tokens=20)
    num_blocks = int(layout.block_ids.max()) + 2

    key_cache = torch.randn(num_blocks, NUM_KV_HEADS, HEAD_SIZE // X,
                            layout.block_size, X, dtype=DTYPE, device="cuda")
    value_cache = torch.randn(num_blocks, NUM_KV_HEADS, HEAD_SIZE,
                              layout.block_size, dtype=DTYPE, device="cuda")
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, dtype=DTYPE, device="cuda")
    cos_sin = rope_table(HEAD_SIZE, 4096).to(DTYPE)

    store = DescriptorStore("bf16")
    store.fill("layer", key_cache, layout.block_ids,
               layout.block_size - layout.q_mask.to(torch.int32))
    router = Router(RouterConfig(budget_tokens=0.34, sink_stripe="off"), store)
    walk = router.route(layer_name="layer",
                        query=query,
                        cos_sin_cache=cos_sin,
                        rotary_dim=HEAD_SIZE,
                        num_kv_heads=NUM_KV_HEADS,
                        key_cache=key_cache,
                        **layout.route_kwargs())

    dense = _run_kernel(query, key_cache, value_cache, layout.packed,
                        layout.seq_lens, cos_sin)
    sparse = _run_kernel(query, key_cache, value_cache, walk.block_table,
                         walk.seq_lens, cos_sin)
    assert not torch.allclose(dense, sparse), (
        "the tight budget dropped nothing -- this test would prove nothing")

    torch.testing.assert_close(sparse,
                               _reference(query, key_cache, value_cache,
                                          walk.block_table,
                                          int(walk.seq_lens[0]), cos_sin,
                                          layout.block_size),
                               rtol=2e-3,
                               atol=2e-3)
