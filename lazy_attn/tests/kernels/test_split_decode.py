"""The split decode kernel must answer exactly what the serial one answers.

`LAZY_SPLIT_KV` cuts the block walk into ranges and merges the partial
softmaxes afterwards. That is a numerically different route to the same value,
and three things in it are easy to get subtly wrong in ways that a single
smoke test would not catch:

* the **softmax merge** itself, which has to rescale each split to a common max
  before summing, and which silently produces a plausible-looking wrong answer
  if the normaliser is applied in the wrong pass;
* the **rotation elision**, since each split restarts its `prev_rot_offset` and
  so must re-rotate at its own first row rather than inherit the previous
  split's Q;
* **`q_mask` and the sequence boundary**, which are indexed by the absolute row
  and would silently mask the wrong positions if a split passed a local index.

So these compare against the shipped kernel across document layouts that
exercise ragged padding, many rotation changes, and splits that fall off the
end of the walk.
"""
import pytest
import torch

from lazy.attention.ops.models.llama_split import (MAX_SPLITS,
                                                   MIN_BLOCKS_PER_SPLIT,
                                                   choose_splits)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="decode kernels are CUDA-only")


def _run(monkeypatch, split: bool, **kwargs):
    import lazy.utils.variants as variants
    monkeypatch.setattr(variants, "lazy_split_kv_enabled", lambda: split)
    import lazy.attention.ops.chunked_prefill_paged_decode as dispatch
    monkeypatch.setattr(dispatch, "lazy_split_kv_enabled", lambda: split)
    output = torch.zeros_like(kwargs["query"])
    dispatch.chunked_prefill_paged_decode(output=output, **kwargs)
    return output


def _case(doc_lens, tail_tokens, num_kv_heads=4, num_q_heads=16,
          head_size=32, block_size=16, seed=0, lazy=True):
    """One decode step, in the layout the dispatcher expects."""
    import numpy as np
    from lazy.core.sched.scheduler import metadata_for_lazy_attention
    from lazy.utils.rotation import PACKED_Q_OFFSET_SHIFT

    torch.manual_seed(seed)
    padded = [((n + block_size - 1) // block_size) * block_size
              for n in doc_lens]

    class _Request:
        document_lens = doc_lens
        document_lens_padded = padded

    q_offset, q_mask = metadata_for_lazy_attention(_Request(), block_size)
    num_doc_blocks = sum(padded) // block_size
    tail_blocks = (tail_tokens + block_size - 1) // block_size
    # The scheduler emits one entry per document block *plus* one for the query
    # block, so the table has to hold that entry even when the tail rounds to
    # zero blocks. The kernel still walks only `cdiv(seq_len, block_size)`.
    num_blocks = max(num_doc_blocks + tail_blocks, len(q_offset))

    offsets = np.zeros(num_blocks, dtype=np.int64)
    masks = np.zeros(num_blocks, dtype=np.int64)
    offsets[:len(q_offset)] = q_offset
    masks[:len(q_mask)] = q_mask
    block_ids = np.arange(1, 1 + num_blocks, dtype=np.int64)
    packed = ((torch.from_numpy(block_ids) << 32)
              | (torch.from_numpy(offsets) << PACKED_Q_OFFSET_SHIFT)
              | torch.from_numpy(masks)).cuda()[None, :]

    x = 8
    total_blocks = num_blocks + 2
    key_cache = torch.randn(total_blocks, num_kv_heads, head_size // x,
                            block_size, x, device="cuda",
                            dtype=torch.bfloat16)
    value_cache = torch.randn(total_blocks, num_kv_heads, head_size,
                              block_size, device="cuda",
                              dtype=torch.bfloat16)
    query = torch.randn(2, num_q_heads, head_size, device="cuda",
                        dtype=torch.bfloat16)
    seq_lens = torch.tensor([num_doc_blocks * block_size + tail_tokens],
                            dtype=torch.int32, device="cuda")

    inv = 1.0 / (10000**(torch.arange(0, head_size, 2, device="cuda").float()
                         / head_size))
    angles = torch.arange(70000, device="cuda").float()[:, None] * inv[None, :]
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1)

    return dict(
        query=query,
        key=torch.zeros(1, num_kv_heads, head_size, device="cuda",
                        dtype=torch.bfloat16),
        value=torch.zeros(1, num_kv_heads, head_size, device="cuda",
                          dtype=torch.bfloat16),
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=packed,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.item()),
        max_query_len=1,
        k_scale=torch.tensor(1.0, device="cuda"),
        v_scale=torch.tensor(1.0, device="cuda"),
        rotary_dim=head_size,
        cos_sin_cache=cos_sin,
        is_lazy=torch.tensor([lazy], device="cuda"),
        packed_block_table=packed,
    )


@pytest.mark.parametrize("doc_lens,tail", [
    ([40, 70, 30, 90], 48),          # ragged padding, four rotation changes
    ([16, 16, 16], 16),              # exact blocks, no padding at all
    ([200], 32),                     # one long document, one rotation
    ([13] * 12, 64),                 # many short documents, heavy padding
    ([16], 16),                      # fewer blocks than splits
])
def test_split_matches_serial_decode(monkeypatch, doc_lens, tail):
    case = _case(doc_lens, tail)
    serial = _run(monkeypatch, False, **case)
    split = _run(monkeypatch, True, **case)
    torch.testing.assert_close(split, serial, rtol=2e-2, atol=2e-2)


def test_split_matches_serial_for_non_lazy_rows(monkeypatch):
    """The non-lazy branch takes Q unrotated; the split kernel must too."""
    case = _case([40, 70], 48, lazy=False)
    serial = _run(monkeypatch, False, **case)
    split = _run(monkeypatch, True, **case)
    torch.testing.assert_close(split, serial, rtol=2e-2, atol=2e-2)


def test_a_walk_shorter_than_the_split_count_still_agrees(monkeypatch):
    """Splits past the end of the walk must contribute nothing, not NaN."""
    case = _case([16, 16], 0)
    serial = _run(monkeypatch, False, **case)
    split = _run(monkeypatch, True, **case)
    assert torch.isfinite(split).all()
    torch.testing.assert_close(split, serial, rtol=2e-2, atol=2e-2)


def test_split_count_never_starves_a_split():
    """A split shorter than its minimum is not worth its launch or its merge."""
    assert choose_splits(1, 8, MIN_BLOCKS_PER_SPLIT, 70) == 1
    assert choose_splits(1, 8, 4096, 70) <= MAX_SPLITS
    # Enough sequences and heads already fill the card; do not split further.
    assert choose_splits(64, 8, 4096, 70) == 1
