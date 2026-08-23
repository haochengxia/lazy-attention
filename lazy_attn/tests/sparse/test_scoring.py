"""The fused scorer and the torch scorer must agree.

`router.py` keeps both: one kernel for `scorer=quest`, and the torch
formulation for the evaluation scorers and for descriptor dtypes the kernel
does not load. Two implementations of one bound is exactly the arrangement that
silently drifts, so these hold them to each other on the properties that matter
-- the ranking they induce, the `+inf` an undescribed block gets, and the GQA
aggregation -- rather than only on a single happy-path array.
"""
import pytest
import torch

from conftest import Layout, rope_table
from lazy.sparse.descriptors import DescriptorStore
from lazy.sparse.router import Router, RouterConfig, derotate_query
from lazy.sparse.scoring import can_fuse, score_blocks

NUM_KV_HEADS = 4
NUM_Q_HEADS = 16
HEAD_SIZE = 32


def _fixture(doc_lens=(40, 70, 30, 90), tail=48, seed=0):
    torch.manual_seed(seed)
    layout = Layout(list(doc_lens), tail, block_size=16)
    x = 8
    key_cache = torch.randn(layout.num_blocks + 4,
                            NUM_KV_HEADS,
                            HEAD_SIZE // x,
                            16,
                            x,
                            device="cuda",
                            dtype=torch.bfloat16)
    store = DescriptorStore(dtype="bf16")
    valid_lens = torch.full((layout.block_ids.numel(), ),
                            16,
                            dtype=torch.int32,
                            device="cuda")
    store.fill("layer", key_cache, layout.block_ids, valid_lens)

    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")
    cos_sin = rope_table(HEAD_SIZE, 8192)
    q_by_doc = derotate_query(query, layout.doc_offsets, cos_sin, HEAD_SIZE)
    phys = (layout.packed >> 32).to(torch.int64)
    return layout, store, q_by_doc, phys


def _torch_scores(store, q_by_doc, phys, doc_id, gqa_agg="max"):
    """The torch path, reached by asking for a scorer the kernel declines."""
    router = Router(RouterConfig(scorer="quest", gqa_agg=gqa_agg), store)
    desc, _ = store.raw("layer")
    assert can_fuse("quest", desc), "fixture should be fusable"
    # Force the fallback by hiding the descriptor tensor from `can_fuse`.
    original = store.raw
    store.raw = lambda name: (None, None)
    try:
        return router._score_blocks("layer", q_by_doc, phys, doc_id,
                                    NUM_KV_HEADS, None)
    finally:
        store.raw = original


@pytest.mark.parametrize("gqa_agg", ["max", "sum"])
def test_fused_scores_match_torch(gqa_agg):
    layout, store, q_by_doc, phys = _fixture()
    desc, valid = store.raw("layer")

    fused = score_blocks(q_by_doc, desc, valid, phys, layout.doc_id,
                         NUM_KV_HEADS, gqa_agg)
    reference = _torch_scores(store, q_by_doc, phys, layout.doc_id, gqa_agg)

    # Document rows are the ones either path is asked about; the torch version
    # leaves tail rows at whatever its tiling wrote, and `_select` masks them.
    docs = layout.doc_id[0] >= 0
    torch.testing.assert_close(fused[0][docs],
                               reference[0][docs],
                               rtol=2e-2,
                               atol=2e-2)


def test_fused_scores_induce_the_same_ranking():
    """Tighter than closeness: selection only ever uses the order."""
    layout, store, q_by_doc, phys = _fixture(seed=3)
    desc, valid = store.raw("layer")

    fused = score_blocks(q_by_doc, desc, valid, phys, layout.doc_id,
                         NUM_KV_HEADS, "max")
    reference = _torch_scores(store, q_by_doc, phys, layout.doc_id)

    docs = (layout.doc_id[0] >= 0).nonzero(as_tuple=True)[0]
    assert torch.equal(docs[fused[0][docs].argsort()],
                       docs[reference[0][docs].argsort()])


def test_undescribed_blocks_score_positive_infinity():
    """The lifecycle guarantee: never described means always read."""
    layout, store, q_by_doc, phys = _fixture()
    desc, valid = store.raw("layer")
    victim = int(phys[0, 1])
    valid[victim] = False

    fused = score_blocks(q_by_doc, desc, valid, phys, layout.doc_id,
                         NUM_KV_HEADS, "max")
    assert torch.isinf(fused[0, 1]) and fused[0, 1] > 0
    assert torch.isfinite(fused[0, 0])


def test_the_bound_is_an_upper_bound_on_true_attention():
    """What the whole scheme rests on: a low bound means a low true q.k.

    Computed against the stored keys directly, so this fails if the kernel's
    box arithmetic drifts from what the descriptors mean.
    """
    layout, store, q_by_doc, phys = _fixture(seed=7)
    desc, valid = store.raw("layer")
    fused = score_blocks(q_by_doc, desc, valid, phys, layout.doc_id,
                         NUM_KV_HEADS, "max")

    for col in range(layout.num_doc_blocks):
        doc = int(layout.doc_id[0, col])
        block = int(phys[0, col])
        stored = store._desc["layer"][block].float()  # [H, 2, D]
        lo, hi = stored[:, 0, :], stored[:, 1, :]
        q = q_by_doc[0, doc].reshape(NUM_KV_HEADS, -1, q_by_doc.shape[3])
        true_bound = torch.maximum(q * lo[:, None, :],
                                   q * hi[:, None, :]).sum(-1)
        expected = true_bound.amax(-1).amax(-1)
        assert fused[0, col] >= expected - 1e-2
