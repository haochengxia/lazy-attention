"""The packed entry has to survive the large-M regime, field by field.

The other rotation tests check the *bound*; this one checks the *packing* at a
document count the 16-bit layout could not express at all (200 documents of 400
tokens needs a rotation offset of ~80k). It walks the real chain -- scheduler
metadata, the runner's pack expression, the kernel's unpack expression -- and
asserts every entry comes back out unchanged, because the failure mode of a
mis-sized field is silence: a wrong physical block and a wrong rotation, never
an exception.

CPU-only: the pack is three shifts on an int64 tensor, so it needs no device.
"""
import numpy as np
import pytest
import torch

from conftest import make_lazy_request
from lazy.core.sched.scheduler import metadata_for_lazy_attention
from lazy.engine.processor import _validate_rotation_offsets
from lazy.utils.rotation import (MAX_PACKED_Q_MASK, MAX_PACKED_Q_OFFSET,
                                 PACKED_Q_OFFSET_SHIFT, max_rotation_offset)

BLOCK_SIZE = 16


def pack(block_ids, q_offset, q_mask) -> torch.Tensor:
    """Exactly what `_rebuild_packed_block_table` does, on one row."""
    packed = block_ids.clone()
    packed.bitwise_left_shift_(32)
    packed.bitwise_or_(q_offset.to(torch.int64) << PACKED_Q_OFFSET_SHIFT)
    packed.bitwise_or_(q_mask.to(torch.int64))
    return packed


def unpack(packed: torch.Tensor):
    """Exactly what `kernel_paged_attention_2d_llama` does, per entry."""
    return (packed >> 32,
            (packed >> PACKED_Q_OFFSET_SHIFT) & MAX_PACKED_Q_OFFSET,
            packed & MAX_PACKED_Q_MASK)


@pytest.mark.unit
@pytest.mark.parametrize("num_docs,doc_len", [
    (200, 400),  # the corpus-as-cache case: offset ~80k, past the old field
    (200, 397),  # ... ragged, so every document also carries padding
    (1100, 61),  # many small documents, each carrying padding
])
def test_large_document_sets_pack_losslessly(num_docs, doc_len):
    document_lens = [doc_len] * num_docs
    padded = [((doc_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
              ] * num_docs
    _validate_rotation_offsets(document_lens, padded)  # admitted
    assert max_rotation_offset(document_lens, padded) > 0xFFFF  # ... and only now

    request = make_lazy_request(
        documents_token_ids_padded=[[0] * length for length in padded],
        document_lens=document_lens,
        document_lens_padded=padded,
    )
    q_offset, q_mask = metadata_for_lazy_attention(request, BLOCK_SIZE)
    num_blocks = len(q_offset)

    # Block ids the allocator could plausibly hand out, including a large one:
    # the offset field must not reach up into them.
    block_ids = torch.arange(num_blocks, dtype=torch.int64)
    block_ids[-1] = (1 << 31) - 1

    packed = pack(block_ids, torch.tensor(q_offset, dtype=torch.int32),
                  torch.tensor(q_mask, dtype=torch.int32))
    got_blocks, got_offset, got_mask = unpack(packed)

    assert torch.equal(got_blocks, block_ids)
    assert got_offset.tolist() == list(q_offset)
    assert got_mask.tolist() == list(q_mask)
    # Each document's last block carries its padding, and nothing else does.
    assert max(q_mask) <= MAX_PACKED_Q_MASK
    assert np.count_nonzero(q_mask) == (num_docs if doc_len % BLOCK_SIZE else 0)
