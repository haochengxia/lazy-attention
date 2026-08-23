"""What the packed block table can represent, in one place.

The decode kernels read `[physical_block_idx:32 | q_offset:24 | q_mask:8]`, so
q_offset has 24 bits. Whether a request fits has to be decided at admission --
by then the engine is committed -- while the packing itself happens in the model
runner, so the bound is defined here and both sides import it.

The split was 16/16 until the fields were rebalanced: q_mask only ever holds a
document's padding, which is at most `block_size - 1`, so 8 bits cover every
block size vLLM offers (<= 128) with room to spare, while q_offset accumulates
the true lengths of every document ahead of the last one and ran out at ~65
documents of 1k tokens -- the regime the corpus-as-cache experiments live in.
"""
from __future__ import annotations

from typing import Sequence

# Field widths of the packed int64 entry, low bits first:
#   [physical_block_idx:32 | q_offset:24 | q_mask:8]
PACKED_Q_MASK_BITS = 8
PACKED_Q_OFFSET_BITS = 24
PACKED_Q_OFFSET_SHIFT = PACKED_Q_MASK_BITS

# Past this, the shifted offset carries into the physical-block field: the
# kernel reads a different (possibly out-of-range) block and de-rotates by a
# wrapped position, with nothing raised anywhere.
MAX_PACKED_Q_OFFSET = (1 << PACKED_Q_OFFSET_BITS) - 1

# Same failure one field down: a mask past this carries into q_offset, so the
# document is de-rotated to the wrong position *and* keeps its padding. The
# value is a document's padding, so this is really a bound on block_size.
MAX_PACKED_Q_MASK = (1 << PACKED_Q_MASK_BITS) - 1
MAX_PACKED_BLOCK_SIZE = MAX_PACKED_Q_MASK + 1


def max_rotation_offset(document_lens: Sequence[int],
                        document_lens_padded: Sequence[int]) -> int:
    """The largest +1-biased q_offset `metadata_for_lazy_attention` will emit.

    Document `d` rotates by the total padding plus the *true* lengths of the
    documents before it, so the largest is the one on the last document:

        max = sum(padded) - true[-1] + 1

    Note what this is not. It is not the size of the document region: a single
    block-aligned document has no padding and nothing before it, so its offset
    is 1 no matter how long it is. What it bounds is everything ahead of the
    last document.
    """
    if not document_lens:
        return 0
    total_padding = sum(
        padded - true for true, padded in zip(document_lens,
                                              document_lens_padded))
    return total_padding + sum(document_lens[:-1]) + 1
