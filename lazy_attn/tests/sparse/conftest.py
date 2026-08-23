"""A synthetic lazy request, in exactly the layout the runner would hand over.

Every sparse test needs the same object: a packed block table over M documents
followed by a dense tail, plus the routing geometry the runner derives from the
scheduler's `q_offset`. Building it here keeps each test to the one property it
is about, and keeps the layout honest -- the offsets below are produced by the
*real* `metadata_for_lazy_attention`, not hand-written, so a change to the
scheduler's convention breaks these tests rather than silently diverging from
them.
"""
import numpy as np
import pytest
import torch

from lazy.core.sched.scheduler import metadata_for_lazy_attention
from lazy.utils.rotation import PACKED_Q_OFFSET_SHIFT

BLOCK_SIZE = 16


class Layout:
    """The tensors `Router.route` expects, for one single-request batch."""

    def __init__(self, doc_lens, tail_tokens, block_size=BLOCK_SIZE,
                 device="cuda", first_block=1):
        self.block_size = block_size
        self.doc_lens = list(doc_lens)
        self.padded = [((n + block_size - 1) // block_size) * block_size
                       for n in doc_lens]

        request = _FakeRequest(self.doc_lens, self.padded)
        q_offset, q_mask = metadata_for_lazy_attention(request, block_size)

        self.num_doc_blocks = sum(self.padded) // block_size
        tail_blocks = (tail_tokens + block_size - 1) // block_size
        num_blocks = self.num_doc_blocks + tail_blocks

        # The scheduler emits one entry per document block plus one for the
        # query block; later tail blocks inherit offset 0, meaning "keep the
        # previous rotation", which is why they must stay after that entry.
        offsets = np.zeros(num_blocks, dtype=np.int64)
        masks = np.zeros(num_blocks, dtype=np.int64)
        offsets[:len(q_offset)] = q_offset
        masks[:len(q_mask)] = q_mask

        block_ids = np.arange(first_block, first_block + num_blocks,
                              dtype=np.int64)
        packed = ((torch.from_numpy(block_ids) << 32)
                  | (torch.from_numpy(offsets) << PACKED_Q_OFFSET_SHIFT)
                  | torch.from_numpy(masks))

        doc_id = np.full(num_blocks, -1, dtype=np.int32)
        cursor = 0
        for idx, padded in enumerate(self.padded):
            span = padded // block_size
            doc_id[cursor:cursor + span] = idx
            cursor += span
        doc_offsets = np.zeros(num_blocks, dtype=np.int32)
        starts = np.flatnonzero(
            np.r_[True, np.diff(offsets[:self.num_doc_blocks]) != 0])
        doc_offsets[:len(starts)] = offsets[:self.num_doc_blocks][starts]

        self.block_ids = torch.as_tensor(block_ids, device=device)
        self.packed = packed.to(device)[None, :]
        self.q_mask = torch.from_numpy(masks).to(device)
        self.doc_id = torch.as_tensor(doc_id, device=device)[None, :]
        self.doc_offsets = torch.as_tensor(doc_offsets, device=device)[None, :]
        self.num_doc_blocks_t = torch.tensor([self.num_doc_blocks],
                                             dtype=torch.int32,
                                             device=device)
        self.seq_lens = torch.tensor(
            [self.num_doc_blocks * block_size + tail_tokens],
            dtype=torch.int32,
            device=device)
        self.routable = torch.tensor([True], device=device)
        self.query_start_loc = torch.tensor([0, 1],
                                            dtype=torch.int32,
                                            device=device)
        self.num_blocks = num_blocks
        self.tail_tokens = tail_tokens

    def route_kwargs(self, **overrides):
        kwargs = dict(packed=self.packed,
                      seq_lens=self.seq_lens,
                      doc_id=self.doc_id,
                      num_doc_blocks=self.num_doc_blocks_t,
                      doc_offsets=self.doc_offsets,
                      routable=self.routable,
                      query_start_loc=self.query_start_loc,
                      block_size=self.block_size)
        kwargs.update(overrides)
        return kwargs


class _FakeRequest:
    """Just enough of `LazyRequest` for `metadata_for_lazy_attention`."""

    def __init__(self, document_lens, document_lens_padded):
        self.document_lens = document_lens
        self.document_lens_padded = document_lens_padded


@pytest.fixture
def layout_factory():
    return Layout


def rope_table(rotary_dim: int, max_pos: int, device="cuda") -> torch.Tensor:
    """A real Llama RoPE table: cos in the first half, sin in the second.

    Real rather than random because the kernel special-cases `q_offset == 1` to
    "leave Q alone", which is only the same thing as rotating at position 0
    when the table actually has cos=1, sin=0 there.
    """
    inv = 1.0 / (10000**(torch.arange(0, rotary_dim, 2, device=device).float()
                         / rotary_dim))
    angles = torch.arange(max_pos, device=device).float()[:, None] * inv[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)
