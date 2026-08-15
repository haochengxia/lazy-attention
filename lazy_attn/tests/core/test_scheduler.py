# Test the scheduling logic

from itertools import chain

import pytest

from lazy.core.sched.scheduler import (metadata_for_lazy_attention,
                                       metadata_for_mepic)
from lazy.request import LazyRequest

BLOCK_SIZE = 8

# Four documents, right-padded to a whole number of blocks. The padding is what
# the rotation metadata has to account for: real lengths [6, 15, 8, 2] padded
# to [8, 16, 8, 8].
DOCUMENTS_TOKEN_IDS_PADDED = [
    [1, 2, 3, 4, 5, 6, 128001, 128001],
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 128001],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 128001, 128001, 128001, 128001, 128001, 128001],
]
DOCUMENT_LENS = [6, 15, 8, 2]
DOCUMENT_LENS_PADDED = [8, 16, 8, 8]


@pytest.fixture
def mock_request():
    from vllm import SamplingParams

    return LazyRequest(
        request_id="test_request",
        prompt_token_ids=[1, 2, 3, 4],
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=SamplingParams(max_tokens=1),
        eos_token_id=128001,
        arrival_time=0.0,
        documents_token_ids_padded=DOCUMENTS_TOKEN_IDS_PADDED,
        document_lens=DOCUMENT_LENS,
        document_lens_padded=DOCUMENT_LENS_PADDED,
    )


@pytest.mark.unit
def test_metadata_for_lazy_attention(mock_request):
    q_offset, q_mask = metadata_for_lazy_attention(mock_request, BLOCK_SIZE)

    num_doc_blocks = sum(DOCUMENT_LENS_PADDED) // BLOCK_SIZE
    # One entry per document block, plus one for the query/decode block.
    assert len(q_offset) == len(q_mask) == num_doc_blocks + 1

    # Every block of a document carries the same rotation offset, biased by +1
    # so that 0 stays free as a sentinel. The offset of document i is the total
    # padding ahead of it plus the real length of every earlier document --
    # i.e. how far the document has to rotate to sit where the unpadded
    # concatenation would have put it.
    total_padding = sum(p - l
                        for p, l in zip(DOCUMENT_LENS_PADDED, DOCUMENT_LENS))
    expected_offset = []
    expected_mask = []
    rot = total_padding
    for doc_idx, padded_len in enumerate(DOCUMENT_LENS_PADDED):
        blocks = padded_len // BLOCK_SIZE
        expected_offset += [rot + 1] * blocks
        # Only the last block of a document is partially padded.
        expected_mask += [0] * (blocks - 1)
        expected_mask += [padded_len - DOCUMENT_LENS[doc_idx]]
        rot += DOCUMENT_LENS[doc_idx]
    # The query block keeps the original global RoPE orientation.
    expected_offset.append(1)
    expected_mask.append(0)

    assert q_offset == expected_offset
    assert q_mask == expected_mask
    # Pinned so a change in the encoding is visible rather than silently
    # tracked by the formula above.
    assert q_offset == [10, 16, 16, 31, 39, 1]
    assert q_mask == [2, 0, 1, 0, 6, 0]


@pytest.mark.unit
def test_metadata_for_mepic_zeroes_offsets(mock_request):
    # MEPIC rotates cached keys inside the attention kernel, so the scheduler
    # hands it no rotation offsets -- only the padding mask.
    _, lazy_mask = metadata_for_lazy_attention(mock_request, BLOCK_SIZE)
    q_offset, q_mask = metadata_for_mepic(mock_request, BLOCK_SIZE)

    assert q_offset == [0] * len(q_offset)
    assert q_mask == lazy_mask


@pytest.mark.unit
def test_documents_merge_in_front_of_the_prompt(mock_request):
    query_tokens = list(mock_request.prompt_token_ids)
    mock_request.merge_documents()

    assert mock_request.prompt_token_ids == (
        list(chain.from_iterable(DOCUMENTS_TOKEN_IDS_PADDED)) + query_tokens)
    assert mock_request.num_prompt_tokens == len(mock_request.prompt_token_ids)
    assert list(mock_request.all_token_ids) == mock_request.prompt_token_ids
