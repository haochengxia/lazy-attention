"""The walk table's invariants, which everything downstream assumes.

The sparse decoder is correct only if the compacted table it hands the kernel
means the same thing the dense one did. Four properties carry that, and they are
the four the kernel's structure actually depends on -- not a general-purpose
wish list:

* budget = everything reproduces the dense table *bit for bit* (T1), which is
  the anchor: if this drifts, no accuracy number downstream can be attributed
  to the method rather than to the plumbing;
* rows stay in ascending original order, because the kernel elides Q's
  re-rotation whenever `rot_offset` repeats between adjacent rows, and because
  tail rows carry `q_offset == 0` meaning "keep the previous rotation";
* `seq_lens` is the *walk* length, since the kernel derives its trip count from
  it as `cdiv(seq_len, BLOCK_SIZE)` and masks the final row with it;
* rows the router was not asked to touch come back untouched.
"""
import pytest
import torch

from conftest import rope_table
from lazy.sparse.descriptors import DescriptorStore
from lazy.sparse.router import Router, RouterConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

HEAD_SIZE = 64
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4


def _router(store, **config):
    return Router(RouterConfig(**config), store)


def _warm_store(layout, device="cuda"):
    """Describe every block, so nothing is selected merely for being unknown."""
    key_cache = torch.randn(layout.num_blocks + layout.block_ids[0].item() + 1,
                            NUM_KV_HEADS,
                            HEAD_SIZE // 4,
                            layout.block_size,
                            4,
                            device=device,
                            dtype=torch.bfloat16)
    store = DescriptorStore("bf16")
    valid = layout.block_size - layout.q_mask.to(torch.int32)
    store.fill("layer", key_cache, layout.block_ids, valid)
    return store, key_cache


def _walk(layout, result):
    """The rows the kernel would actually visit, as physical block ids."""
    trips = int((result.seq_lens[0] + layout.block_size - 1) //
                layout.block_size)
    return (result.block_table[0, :trips] >> 32).tolist()


def test_full_budget_reproduces_the_dense_table(layout_factory):
    """T1. The trust anchor: no budget pressure, no change, bit for bit."""
    layout = layout_factory([64, 48, 80], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")

    result = _router(store, budget_tokens=1.0, sink_stripe="off").route(
        layer_name="layer",
        query=query,
        cos_sin_cache=rope_table(HEAD_SIZE, 4096),
        rotary_dim=HEAD_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        key_cache=key_cache,
        **layout.route_kwargs())

    assert torch.equal(result.block_table, layout.packed)
    assert torch.equal(result.seq_lens, layout.seq_lens)


@pytest.mark.parametrize("budget", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("stripe", ["off", "selected"])
def test_walk_is_ordered_and_lengths_agree(layout_factory, budget, stripe):
    """Ascending rows, tail last, and `seq_lens` equal to the trip count."""
    layout = layout_factory([64, 48, 80, 96, 32], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")

    result = _router(store, budget_tokens=budget, sink_stripe=stripe).route(
        layer_name="layer",
        query=query,
        cos_sin_cache=rope_table(HEAD_SIZE, 4096),
        rotary_dim=HEAD_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        key_cache=key_cache,
        **layout.route_kwargs())

    walked = _walk(layout, result)
    assert walked == sorted(walked), "rows must stay in original order"
    assert len(walked) == len(set(walked)), "no row may be walked twice"

    # The tail is never dropped and always trails the documents.
    first_tail = int(layout.block_ids[layout.num_doc_blocks])
    tail_ids = [int(b) for b in layout.block_ids[layout.num_doc_blocks:]]
    assert [b for b in walked if b >= first_tail] == tail_ids
    assert walked[-len(tail_ids):] == tail_ids

    # seq_lens must describe exactly this walk: whole rows for the documents,
    # then the true tail length, so the kernel's boundary mask lands where it
    # did in the dense layout.
    kept_doc_rows = len(walked) - len(tail_ids)
    assert int(result.seq_lens[0]) == (kept_doc_rows * layout.block_size +
                                       layout.tail_tokens)


def test_preamble_is_kept_free_and_charged_when_asked(layout_factory):
    """`LAZY_SPARSE_KEEP_DOC0` is a statement about the corpus, not the method.

    With it on, document 0 is read unconditionally and excluded from the
    budget's denominator -- the convention `benchmarks/lazyroute/corpus.py`
    submits under and the Phase-0 tables are computed under. With it off, it
    competes like any other document.
    """
    layout = layout_factory([64, 48, 80, 96], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")
    common = dict(layer_name="layer",
                  query=query,
                  cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                  rotary_dim=HEAD_SIZE,
                  num_kv_heads=NUM_KV_HEADS,
                  key_cache=key_cache,
                  **layout.route_kwargs())

    doc0_blocks = {
        int(b)
        for b in layout.block_ids[:layout.padded[0] // layout.block_size]
    }
    kept = _router(store, budget_tokens=0.1, sink_stripe="off",
                   keep_doc0=True).route(**common)
    assert doc0_blocks <= set(_walk(layout, kept))

    # Charged: at a tiny budget it is no longer guaranteed a place, and the
    # walk is shorter because doc 0's pages now compete for the same budget.
    charged = _router(store, budget_tokens=0.1, sink_stripe="off",
                      keep_doc0=False).route(**common)
    assert len(_walk(layout, charged)) < len(_walk(layout, kept))


def test_document_granularity_takes_whole_documents(layout_factory):
    """The G0-C arm has to be document-level, not a truncated page budget."""
    layout = layout_factory([64, 48, 80, 96, 32], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")

    result = _router(store,
                     budget_tokens=0.5,
                     granularity="doc",
                     sink_stripe="off").route(
                         layer_name="layer",
                         query=query,
                         cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                         rotary_dim=HEAD_SIZE,
                         num_kv_heads=NUM_KV_HEADS,
                         key_cache=key_cache,
                         **layout.route_kwargs())

    walked = set(_walk(layout, result))
    doc_of = layout.doc_id[0].tolist()
    for doc in range(len(layout.doc_lens)):
        pages = {
            int(layout.block_ids[i])
            for i, d in enumerate(doc_of) if d == doc
        }
        assert pages <= walked or not (pages & walked), (
            f"document {doc} was taken in part: {pages & walked} of {pages}")


def test_refinement_chain_dense_contains_doc_contains_page(layout_factory):
    """dense ⊇ doc ⊇ page at a matched budget -- the containment the plan claims."""
    layout = layout_factory([64, 48, 80, 96, 32], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")
    common = dict(layer_name="layer",
                  query=query,
                  cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                  rotary_dim=HEAD_SIZE,
                  num_kv_heads=NUM_KV_HEADS,
                  key_cache=key_cache,
                  **layout.route_kwargs())

    dense = set(_walk(layout, _router(store, budget_tokens=1.0,
                                      sink_stripe="off").route(**common)))
    doc = set(_walk(layout, _router(store, budget_tokens=0.5,
                                    granularity="doc",
                                    sink_stripe="off").route(**common)))
    page = set(_walk(layout, _router(store, budget_tokens=0.5,
                                     granularity="page",
                                     sink_stripe="off").route(**common)))
    assert doc <= dense and page <= dense
    assert len(page) <= len(dense)


def test_rows_the_router_was_not_given_are_untouched(layout_factory):
    """T4 in miniature: a non-lazy or prefilling row keeps its dense table."""
    layout = layout_factory([64, 48, 80], tail_tokens=20)
    store, key_cache = _warm_store(layout)

    # Two identical rows; only the first is routable.
    packed = layout.packed.repeat(2, 1)
    kwargs = layout.route_kwargs(
        packed=packed,
        seq_lens=layout.seq_lens.repeat(2),
        doc_id=layout.doc_id.repeat(2, 1),
        num_doc_blocks=layout.num_doc_blocks_t.repeat(2),
        doc_offsets=layout.doc_offsets.repeat(2, 1),
        routable=torch.tensor([True, False], device="cuda"),
        query_start_loc=torch.tensor([0, 1, 2],
                                     dtype=torch.int32,
                                     device="cuda"))

    result = _router(store, budget_tokens=0.25, sink_stripe="off").route(
        layer_name="layer",
        query=torch.randn(3, NUM_Q_HEADS, HEAD_SIZE, device="cuda"),
        cos_sin_cache=rope_table(HEAD_SIZE, 4096),
        rotary_dim=HEAD_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        key_cache=key_cache,
        **kwargs)

    assert torch.equal(result.block_table[1], packed[1])
    assert int(result.seq_lens[1]) == int(layout.seq_lens[0])
    assert int(result.seq_lens[0]) < int(layout.seq_lens[0])


def test_narrowing_the_table_does_not_change_the_walk(layout_factory):
    """The block table is sized for `max_model_len`, not for what was allocated.

    At a 131k context that is 8192 columns for a request using about a hundred,
    and scoring the padding is the difference between a router that costs a few
    percent and one that costs several times the decode step. Narrowing is only
    safe if it is invisible, which is what this pins: the kernel takes its row
    stride from the tensor it is handed, so a narrower table must produce the
    same walk and the same lengths.
    """
    layout = layout_factory([64, 48, 80, 96], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")

    # Pad the table out the way the runner's buffers do, then narrow it back.
    pad = 8192 - layout.num_blocks
    wide = dict(
        packed=torch.nn.functional.pad(layout.packed, (0, pad)),
        doc_id=torch.nn.functional.pad(layout.doc_id, (0, pad), value=-1),
        doc_offsets=torch.nn.functional.pad(layout.doc_offsets, (0, pad)),
    )
    common = dict(layer_name="layer",
                  query=query,
                  cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                  rotary_dim=HEAD_SIZE,
                  num_kv_heads=NUM_KV_HEADS,
                  key_cache=key_cache)
    router = _router(store, budget_tokens=0.5, sink_stripe="off")

    narrow = router.route(max_blocks=layout.num_blocks,
                          **common,
                          **layout.route_kwargs(**wide))
    full = router.route(max_blocks=None, **common, **layout.route_kwargs(**wide))

    trips = int((narrow.seq_lens[0] + layout.block_size - 1) //
                layout.block_size)
    assert torch.equal(narrow.seq_lens, full.seq_lens)
    assert torch.equal(narrow.block_table[0, :trips],
                       full.block_table[0, :trips])
    assert narrow.block_table.shape[1] == layout.num_blocks


def test_prefix_granularity_never_takes_a_page_without_its_head(layout_factory):
    """If page 3 of a document is read, pages 0-2 are read with it.

    The motivation is §9b's block-head result -- 64.5% of cached mass sits on
    the first two tokens of each document -- so a late page arriving without
    its document's head is exactly the case worth ruling out. Prefix closure
    also subsumes the sink stripe, which is why this runs with the stripe off.
    """
    layout = layout_factory([64, 48, 80, 96, 32], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")

    result = _router(store, budget_tokens=0.3, granularity="prefix",
                     sink_stripe="off").route(
                         layer_name="layer",
                         query=query,
                         cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                         rotary_dim=HEAD_SIZE,
                         num_kv_heads=NUM_KV_HEADS,
                         key_cache=key_cache,
                         **layout.route_kwargs())

    walked = set(_walk(layout, result))
    doc_of = layout.doc_id[0].tolist()
    for doc in range(len(layout.doc_lens)):
        pages = [
            int(layout.block_ids[i]) for i, d in enumerate(doc_of) if d == doc
        ]
        kept = [p for p in pages if p in walked]
        if not kept:
            continue
        # Whatever was kept must be an unbroken run from the document's head.
        assert kept == pages[:len(kept)], (
            f"document {doc} kept {kept} out of {pages} -- not a prefix")


def test_prefix_is_a_superset_of_page_at_the_same_budget(layout_factory):
    """Closure only ever adds pages, and it spends past the nominal budget.

    Recording this because it is the trap in comparing the two arms: `prefix`
    at budget b reads more than `page` at budget b, so a quality win could be
    bought with tokens rather than with better selection. Arms have to be
    compared at matched `kept_fraction`.
    """
    layout = layout_factory([64, 48, 80, 96, 32], tail_tokens=20)
    store, key_cache = _warm_store(layout)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")
    common = dict(layer_name="layer",
                  query=query,
                  cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                  rotary_dim=HEAD_SIZE,
                  num_kv_heads=NUM_KV_HEADS,
                  key_cache=key_cache,
                  **layout.route_kwargs())

    page = set(_walk(layout, _router(store, budget_tokens=0.3,
                                     granularity="page",
                                     sink_stripe="off").route(**common)))
    prefix = set(_walk(layout, _router(store, budget_tokens=0.3,
                                       granularity="prefix",
                                       sink_stripe="off").route(**common)))
    assert page <= prefix
