"""A rotation offset must fit the packed block table's q_offset field.

`[physical_block_idx:32 | q_offset:24 | q_mask:8]` gives q_offset 24 bits, so an
offset past 0xFFFFFF shifts into the block index: the kernel reads a different
(possibly out-of-range) block and de-rotates by a wrapped position. Nothing
raises and nothing looks wrong -- the request is simply answered incorrectly,
which is why this is checked rather than clamped.

The mask field is the narrow one, and fails the same way one field down: a
padding past 0xFF carries into the rotation offset.

q_offset is the +1-biased running sum of padded document lengths, so the limit
is on a request's *document region*, not on its context length.

The buffer fill is exercised on a stand-in `self`: it only touches the runner's
own arrays, so this stays CPU-only.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lazy.utils.rotation import (MAX_PACKED_Q_MASK, MAX_PACKED_Q_OFFSET,
                                 PACKED_Q_OFFSET_SHIFT)
from lazy.worker.gpu_model_runner import LazyGPUModelRunner

NUM_BLOCKS = 8


def make_runner_stub(q_offset, q_mask=None) -> SimpleNamespace:
    num_reqs = 1
    offset_cpu = torch.zeros((num_reqs, NUM_BLOCKS), dtype=torch.int32)
    mask_cpu = torch.zeros((num_reqs, NUM_BLOCKS), dtype=torch.int32)
    variant_cpu = torch.zeros(num_reqs, dtype=torch.int32)
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req-0"]),
        requests={
            "req-0":
            SimpleNamespace(is_lazy=True, lazy_variant=1, q_offset=q_offset,
                            q_mask=q_mask
                            if q_mask is not None else [0] * len(q_offset)),
        },
        is_lazy_req_cpu=torch.zeros(num_reqs, dtype=torch.bool),
        lazy_variant_cpu=variant_cpu,
        lazy_variant_np=variant_cpu.numpy(),
        lazy_offset_cpu=offset_cpu,
        lazy_offset_np=offset_cpu.numpy(),
        lazy_mask_cpu=mask_cpu,
        lazy_mask_np=mask_cpu.numpy(),
        is_lazy_req=torch.zeros(num_reqs, dtype=torch.bool),
        lazy_variant=variant_cpu.clone(),
        lazy_offset=offset_cpu.clone(),
        lazy_mask=mask_cpu.clone(),
    )


@pytest.mark.unit
def test_offset_at_the_limit_is_accepted():
    runner = make_runner_stub([1, MAX_PACKED_Q_OFFSET])

    LazyGPUModelRunner._refresh_lazy_metadata_buffers(runner)

    assert int(runner.lazy_offset_np[0, 1]) == MAX_PACKED_Q_OFFSET


@pytest.mark.unit
def test_offset_past_the_limit_is_refused():
    runner = make_runner_stub([1, MAX_PACKED_Q_OFFSET + 1])

    with pytest.raises(ValueError, match="past the"):
        LazyGPUModelRunner._refresh_lazy_metadata_buffers(runner)


@pytest.mark.unit
def test_mask_at_the_limit_is_accepted():
    runner = make_runner_stub([1, 1], q_mask=[0, MAX_PACKED_Q_MASK])

    LazyGPUModelRunner._refresh_lazy_metadata_buffers(runner)

    assert int(runner.lazy_mask_np[0, 1]) == MAX_PACKED_Q_MASK


@pytest.mark.unit
def test_mask_past_the_limit_is_refused():
    runner = make_runner_stub([1, 1], q_mask=[0, MAX_PACKED_Q_MASK + 1])

    with pytest.raises(ValueError, match="past the"):
        LazyGPUModelRunner._refresh_lazy_metadata_buffers(runner)


@pytest.mark.unit
def test_what_the_refusal_prevents():
    """Without the check, the overflow lands in the physical block index."""
    offset = MAX_PACKED_Q_OFFSET + 1
    block_id = 6  # even, so the carried bit actually changes it
    packed = (np.int64(block_id) << 32) | (np.int64(offset)
                                           << PACKED_Q_OFFSET_SHIFT)

    # What the kernel unpacks back out.
    assert (packed >> 32) & 0xFFFFFFFF == block_id + 1  # not block 7
    # and the rotation is lost entirely
    assert (packed >> PACKED_Q_OFFSET_SHIFT) & MAX_PACKED_Q_OFFSET == 0


@pytest.mark.unit
def test_the_three_fields_round_trip():
    """Pack the way the runner does, unpack the way the kernel does."""
    block_id, offset, mask = 0x7FFFFFFF, MAX_PACKED_Q_OFFSET, MAX_PACKED_Q_MASK

    packed = ((np.int64(block_id) << 32)
              | (np.int64(offset) << PACKED_Q_OFFSET_SHIFT)
              | np.int64(mask))

    assert packed > 0  # no sign bit: the entry stays a positive int64
    assert (packed >> 32) == block_id
    assert (packed >> PACKED_Q_OFFSET_SHIFT) & MAX_PACKED_Q_OFFSET == offset
    assert packed & MAX_PACKED_Q_MASK == mask
