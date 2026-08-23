"""The fused selector and the torch selector must choose the same rows.

`selection.py` reproduces, in one kernel, what `Router._geometry`,
`Router._select`, `Router.compact` and the `torch.where(routable, ...)` merge do
in about 470 PyTorch operations. That is the arrangement most likely to drift
silently: the fused path is the default, so a divergence would show up as a
quietly different set of pages read, not as a crash.

So these compare against the torch path directly, on the objects the decode
kernel actually consumes -- the walk table and its `seq_lens` -- across the
layouts that exercise the parts easiest to get wrong:

* the **budget**, including `round`-half-to-even, which `floor(x + 0.5)` gets
  wrong at exactly the sizes a 0.25 budget produces;
* the **sink stripe and its guard**, which withdraws when the stripe costs more
  than half the budget and so changes the selection discontinuously;
* **`keep_doc0`**, which moves a whole document between "free" and "charged";
* **non-routable rows**, which must pass their dense table through untouched;
* **ties**, where the two implementations are allowed to differ, and the test
  says what must still hold rather than pretending they cannot occur.

Scores are drawn distinct where the selection has to match exactly. That is not
papering over ties: with ties the torch path's answer depends on the radix
sort's internal order, so there is no "correct" set to compare against, and the
tie test below asserts the invariants that survive instead.
"""
import pytest
import torch

from conftest import Layout
from lazy.sparse.router import Router, RouterConfig
from lazy.sparse.selection import (MAX_FUSED_WIDTH, can_fuse_selection,
                                   select_and_compact)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the selector is a CUDA kernel")


def _counters():
    zero = lambda: torch.zeros((), dtype=torch.long, device="cuda")
    return zero(), zero(), zero()


def _torch_reference(config, layout, scores, routable):
    """Geometry + select + compact + merge, exactly as `route` runs them."""
    router = Router(config, None)
    router._ensure_counters(scores.device)
    geometry = router._geometry(layout.packed, layout.seq_lens, layout.doc_id,
                                layout.doc_offsets, layout.num_doc_blocks_t,
                                layout.block_size, None)
    keep = router._select(scores, geometry)
    walk, lens = router.compact(geometry.packed, keep, geometry.doc_mask,
                                geometry.tail_len, layout.block_size)
    walk = torch.where(routable[:, None], walk, geometry.packed)
    lens = torch.where(routable, lens, layout.seq_lens.int())
    kept = torch.where(routable, keep.sum(1).int(),
                       geometry.block_valid.sum(1).int())
    return walk, lens, kept, geometry.block_valid.sum(1).int()


def _fused(config, layout, scores, routable):
    kept, dense, guard = _counters()
    walk, lens = select_and_compact(layout.packed, scores, layout.doc_id,
                                    layout.seq_lens, layout.num_doc_blocks_t,
                                    routable, layout.block_size, config,
                                    kept, dense, guard)
    return walk, lens, kept, dense


def _distinct_scores(layout, seed=0):
    """Distinct values, so the selection has a unique correct answer."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randperm(layout.num_blocks, generator=generator,
                            device="cuda").float()
    return scores[None, :] * 0.5 - 3.0


@pytest.mark.parametrize("doc_lens,tail", [
    ([40, 70, 30, 90], 48),           # ragged padding, four documents
    ([16] * 20, 32),                  # many equal documents
    ([600], 64),                      # one long document, many pages
    ([13, 200, 47, 13, 160], 16),     # very uneven document lengths
    ([16, 16], 0),                    # no tail at all
])
@pytest.mark.parametrize("budget", [0.25, 0.5, 0.1])
def test_fused_selection_matches_torch(doc_lens, tail, budget):
    layout = Layout(list(doc_lens), tail)
    config = RouterConfig(budget_tokens=budget)
    scores = _distinct_scores(layout)
    assert can_fuse_selection(config, layout.num_blocks)

    want, want_lens, want_kept, want_dense = _torch_reference(
        config, layout, scores, layout.routable)
    got, got_lens, got_kept, got_dense = _fused(config, layout, scores,
                                                layout.routable)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_lens, want_lens)
    assert int(got_kept) == int(want_kept.sum())
    assert int(got_dense) == int(want_dense.sum())


@pytest.mark.parametrize("sink", ["off", "selected"])
@pytest.mark.parametrize("keep_doc0", [True, False])
def test_stripe_and_doc0_switches_agree(sink, keep_doc0):
    layout = Layout([40, 70, 30, 90, 25], 48)
    config = RouterConfig(budget_tokens=0.25, sink_stripe=sink,
                          keep_doc0=keep_doc0)
    scores = _distinct_scores(layout, seed=3)
    want, want_lens, _, _ = _torch_reference(config, layout, scores,
                                             layout.routable)
    got, got_lens, _, _ = _fused(config, layout, scores, layout.routable)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_lens, want_lens)


def test_the_stripe_guard_trips_on_a_tight_budget():
    """Many documents at a small budget: the stripe costs more than it saves.

    This is the discontinuity §9b's guard introduces, and it is the case where
    a fused reimplementation would most plausibly disagree -- so assert both
    that it fires and that both paths agree once it has.
    """
    layout = Layout([16] * 40, 32)
    config = RouterConfig(budget_tokens=0.1, sink_stripe="selected")
    scores = _distinct_scores(layout, seed=5)
    kept, dense, guard = _counters()
    select_and_compact(layout.packed, scores, layout.doc_id, layout.seq_lens,
                       layout.num_doc_blocks_t, layout.routable,
                       layout.block_size, config, kept, dense, guard)
    assert int(guard) == 1, "40 documents at budget 0.1 must trip the guard"
    want, _, _, _ = _torch_reference(config, layout, scores, layout.routable)
    got, _, _, _ = _fused(config, layout, scores, layout.routable)
    torch.testing.assert_close(got, want)


def test_a_non_routable_row_passes_its_table_through():
    layout = Layout([40, 70, 30], 48)
    config = RouterConfig(budget_tokens=0.25)
    scores = _distinct_scores(layout, seed=7)
    routable = torch.tensor([False], device="cuda")
    got, got_lens, _, _ = _fused(config, layout, scores, routable)
    torch.testing.assert_close(got, layout.packed)
    torch.testing.assert_close(got_lens, layout.seq_lens.int())


def test_an_absolute_token_budget_agrees():
    layout = Layout([40, 70, 30, 90], 48)
    config = RouterConfig(budget_tokens=256.0)
    scores = _distinct_scores(layout, seed=11)
    want, want_lens, _, _ = _torch_reference(config, layout, scores,
                                             layout.routable)
    got, got_lens, _, _ = _fused(config, layout, scores, layout.routable)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_lens, want_lens)


def test_tied_scores_still_produce_a_legal_selection():
    """With ties there is no unique answer, so assert what must hold anyway.

    The torch path breaks ties by whatever its radix sort did; the kernel takes
    the lowest index. Both are valid, so the contract is the *shape* of the
    result: the budget is respected, free rows survive, and the kept rows stay
    in ascending order -- which is the invariant the decode kernel's rotation
    elision depends on.
    """
    layout = Layout([16] * 12, 32)
    config = RouterConfig(budget_tokens=0.25, sink_stripe="off")
    scores = torch.zeros((1, layout.num_blocks), device="cuda")
    walk, lens = select_and_compact(layout.packed, scores, layout.doc_id,
                                    layout.seq_lens, layout.num_doc_blocks_t,
                                    layout.routable, layout.block_size, config,
                                    *_counters())
    _, want_lens, _, _ = _torch_reference(config, layout, scores,
                                          layout.routable)
    assert int(lens[0]) == int(want_lens[0]), "budget must be spent identically"

    rows = int(lens[0]) // layout.block_size
    doc_rows = walk[0, :rows]
    offsets = (doc_rows >> 32).tolist()
    assert offsets == sorted(offsets), "kept rows must stay ascending"


def test_a_table_wider_than_the_kernel_falls_back():
    config = RouterConfig()
    assert can_fuse_selection(config, MAX_FUSED_WIDTH)
    assert not can_fuse_selection(config, MAX_FUSED_WIDTH + 1)
    assert not can_fuse_selection(RouterConfig(granularity="prefix"), 128)
    assert not can_fuse_selection(RouterConfig(granularity="doc"), 128)
