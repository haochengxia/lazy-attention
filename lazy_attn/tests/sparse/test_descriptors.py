"""What the descriptor promises, and what happens when it cannot deliver.

The box is only a bound on the keys a block actually holds if it was built from
exactly those keys -- so padding has to be out, a refill has to overwrite, and a
block nobody described must not be scored against whatever was in memory.

The last of those is the interesting one. There is no invalidation hook: the
block pool lives in the EngineCore process and these tensors live on the
worker's GPU, so no call crosses. `descriptors.py` argues the hook is
unnecessary (a descriptor is a pure function of its block's contents, and every
write of a document block runs the fill path), and the `valid` flag is the
backstop for anything that argument misses. The tests below pin the backstop:
an undescribed block is *read*, never scored against stale statistics, so a
lifecycle bug degrades to dense rather than to a wrong answer.
"""
import pytest
import torch

from conftest import rope_table
from lazy.sparse.descriptors import (MAX, MEAN, MIN, DescriptorStore,
                                     block_valid_lens, newly_complete_blocks)
from lazy.sparse.router import Router, RouterConfig, derotate_query

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

HEAD_SIZE = 32
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
X = 4
BLOCK = 16


def _cache(num_blocks=12):
    return torch.randn(num_blocks,
                       NUM_KV_HEADS,
                       HEAD_SIZE // X,
                       BLOCK,
                       X,
                       device="cuda",
                       dtype=torch.bfloat16)


def _rows(cache, block):
    """`[kv_head, block_size, head_size]` -- the paged layout, reassembled."""
    return cache[block].permute(0, 2, 1, 3).reshape(NUM_KV_HEADS, BLOCK,
                                                    HEAD_SIZE).float()


def test_padding_rows_stay_out_of_the_box():
    """A document's padding holds real `<pad>` keys; including them widens the
    box toward keys no query will ever attend to."""
    cache = _cache()
    store = DescriptorStore("bf16")
    blocks = torch.tensor([2, 5], device="cuda")
    lens = torch.tensor([BLOCK, 11], device="cuda", dtype=torch.int32)
    store.fill("l", cache, blocks, lens)

    box, valid = store.boxes("l", blocks)
    assert bool(valid.all())
    for i, (block, live) in enumerate(zip([2, 5], [BLOCK, 11])):
        rows = _rows(cache, block)[:, :live, :]
        assert torch.equal(box[i, :, MIN], rows.amin(1))
        assert torch.equal(box[i, :, MAX], rows.amax(1))

    # And the excluded rows genuinely mattered: the full-block box is wider.
    store.fill("full", cache, blocks[1:], torch.tensor([BLOCK],
                                                       device="cuda",
                                                       dtype=torch.int32))
    wide, _ = store.boxes("full", blocks[1:])
    assert (wide[0, :, MAX] >= box[1, :, MAX]).all()
    assert (wide[0, :, MIN] <= box[1, :, MIN]).all()
    assert not torch.equal(wide[0], box[1])


def test_a_refill_overwrites():
    """The mechanism the no-invalidation argument depends on: a block written
    again is described again, so it can never carry another document's box."""
    store = DescriptorStore("bf16")
    blocks = torch.tensor([3], device="cuda")
    lens = torch.tensor([BLOCK], device="cuda", dtype=torch.int32)

    first = _cache()
    store.fill("l", first, blocks, lens)
    before, _ = store.boxes("l", blocks)

    second = _cache()
    store.fill("l", second, blocks, lens)
    after, _ = store.boxes("l", blocks)

    assert not torch.equal(before, after)
    assert torch.equal(after[0, :, MIN], _rows(second, 3).amin(1))
    store.verify = True
    store.check("l", second, blocks, lens)  # raises if stale


def test_an_undescribed_block_is_read_not_guessed(layout_factory):
    """A block with no descriptor scores `+inf`, so it survives any budget.

    This is the direction a lifecycle bug has to fail in: reading a block we
    did not need is a wasted page, reading one against another document's
    statistics is a wrong answer.
    """
    layout = layout_factory([64, 48, 80, 96], tail_tokens=20)
    cache = _cache(int(layout.block_ids.max()) + 2)
    store = DescriptorStore("bf16")

    # Describe everything except one page in the middle of the corpus.
    described = torch.cat([layout.block_ids[:5], layout.block_ids[6:]])
    store.fill("l", cache, described,
               torch.full_like(described, BLOCK, dtype=torch.int32))
    orphan = int(layout.block_ids[5])

    _, valid = store.boxes("l", layout.block_ids)
    assert not bool(valid[5]) and bool(valid[4])

    walk = Router(RouterConfig(budget_tokens=0.1, sink_stripe="off"),
                  store).route(layer_name="l",
                               query=torch.randn(1,
                                                 NUM_Q_HEADS,
                                                 HEAD_SIZE,
                                                 device="cuda"),
                               cos_sin_cache=rope_table(HEAD_SIZE, 4096),
                               rotary_dim=HEAD_SIZE,
                               num_kv_heads=NUM_KV_HEADS,
                               key_cache=cache,
                               **layout.route_kwargs())

    trips = int((walk.seq_lens[0] + BLOCK - 1) // BLOCK)
    assert orphan in (walk.block_table[0, :trips] >> 32).tolist()


def test_the_mean_vector_is_only_allocated_when_asked():
    """(min+max)/2 is the box centre, not the mean, so the centroid scorer gets
    its own slot rather than a plausible-looking substitute."""
    cache = _cache()
    blocks = torch.tensor([1], device="cuda")
    lens = torch.tensor([BLOCK], device="cuda", dtype=torch.int32)

    lean = DescriptorStore("bf16")
    lean.fill("l", cache, blocks, lens)
    assert lean.boxes("l", blocks)[0].shape[2] == 2

    full = DescriptorStore("bf16", with_mean=True)
    full.fill("l", cache, blocks, lens)
    box = full.boxes("l", blocks)[0]
    assert box.shape[2] == 3
    torch.testing.assert_close(box[0, :, MEAN],
                               _rows(cache, 1).mean(1).to(torch.bfloat16).float(),
                               rtol=1e-2,
                               atol=1e-2)
    assert not torch.allclose(box[0, :, MEAN],
                              (box[0, :, MIN] + box[0, :, MAX]) / 2)


def test_derotation_reproduces_the_kernels_own_arithmetic():
    """The identity the descriptor argument rests on.

    Scoring a de-rotated query against locally-framed stored keys is only
    Quest's math *inside the document's frame* if the de-rotation is the one the
    kernel performs. The reference below is transcribed elementwise from
    `llama_v1.py`, deliberately without applying the algebra the router uses to
    collapse it, so the two disagree if that algebra is wrong.
    """
    torch.manual_seed(0)
    cos_sin = rope_table(HEAD_SIZE, 4096)
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device="cuda")

    def kernel_form(q, offset):
        embed = HEAD_SIZE // 2
        rev = (torch.arange(HEAD_SIZE, device=q.device) +
               embed) % HEAD_SIZE
        q_rev = q[..., rev]
        first = torch.arange(HEAD_SIZE, device=q.device) < embed
        q1 = torch.where(first, q, q_rev)
        q2 = torch.where(first, q_rev, q)
        cols = torch.arange(HEAD_SIZE, device=q.device) % embed
        base = (offset - 1) * HEAD_SIZE
        cos = cos_sin.flatten()[base + cols]
        sin = cos_sin.flatten()[base + embed + cols]
        return torch.where(first, q1 * cos + q2 * sin, -q1 * sin + q2 * cos)

    for offset in (1, 2, 17, 65, 513):
        mine = derotate_query(
            query, torch.tensor([[offset]], device="cuda", dtype=torch.int32),
            cos_sin, HEAD_SIZE)[0, 0]
        torch.testing.assert_close(mine,
                                   kernel_form(query[0].float(), offset),
                                   rtol=1e-5,
                                   atol=1e-5)


@pytest.mark.parametrize("computed,scheduled,expected", [
    (0, 100, (0, 6)),
    (100, 60, (6, 10)),
    (0, 16, (0, 1)),
    (160, 0, (10, 10)),
])
def test_chunked_prefill_completes_each_block_exactly_once(
        computed, scheduled, expected):
    """A block split across two chunks is described by the chunk that finishes
    it, and by that one only."""
    assert newly_complete_blocks(computed, scheduled, 16) == expected


def test_valid_lens_charge_padding_to_the_last_block_only():
    lens = block_valid_lens(torch.arange(5), num_prompt_tokens=80,
                            true_len=76, block_size=16)
    assert lens.tolist() == [16, 16, 16, 16, 12]
