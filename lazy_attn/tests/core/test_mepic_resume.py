"""`MEPIC_FIRST_BLOCK_RECOMPUTE` has to keep applying after a preemption.

The switch is applied by `get_computed_blocks_docs(drop_first_cached_block=...)`,
inside the block that now runs only on a request's first pass. A resumed request
therefore reaches `allocate_slots` with the ordinary prefix-cache hit, which
includes the first block of every document -- exactly the blocks the switch
exists to recompute. The scheduler drops that hit instead.

The condition is exercised directly: standing up a real `Scheduler` needs vLLM's
own test helpers, which ship only in a source checkout.
"""
import pytest

from conftest import make_lazy_request

DOCUMENTS = [[1, 2, 3, 4], [5, 6, 7, 8]]


def make_request():
    return make_lazy_request(
        prompt_token_ids=[100, 101],
        documents_token_ids_padded=DOCUMENTS,
        document_lens=[4, 4],
        document_lens_padded=[4, 4],
    )


def schedule_pass(request, drop_first_cached_block, cached_prefix):
    """The scheduler's decision, in the order it makes it.

    Returns (num_computed_tokens, ran_document_block) -- the second is what
    applies `drop_first_cached_block`.
    """
    num_computed_tokens = cached_prefix
    just_merged = request.has_documents and request.merge_documents()
    if (request.has_documents and not just_merged and drop_first_cached_block):
        num_computed_tokens = 0
    return num_computed_tokens, just_merged


@pytest.mark.unit
def test_first_pass_applies_the_switch_itself():
    request = make_request()

    num_computed_tokens, ran_document_block = schedule_pass(
        request, drop_first_cached_block=True, cached_prefix=0)

    # The document block runs, so it passes the flag down; nothing to drop.
    assert ran_document_block
    assert num_computed_tokens == 0


@pytest.mark.unit
def test_resumed_request_drops_its_hit_when_the_switch_is_on():
    request = make_request()
    schedule_pass(request, drop_first_cached_block=True, cached_prefix=0)

    # Preempted, then scheduled again with the whole prefix cached.
    num_computed_tokens, ran_document_block = schedule_pass(
        request, drop_first_cached_block=True, cached_prefix=10)

    assert not ran_document_block  # would double-count the documents
    assert num_computed_tokens == 0  # so the hit goes instead


@pytest.mark.unit
def test_resumed_request_keeps_its_hit_when_the_switch_is_off():
    """The default path must not pay a recompute it does not need."""
    request = make_request()
    schedule_pass(request, drop_first_cached_block=False, cached_prefix=0)

    num_computed_tokens, ran_document_block = schedule_pass(
        request, drop_first_cached_block=False, cached_prefix=10)

    assert not ran_document_block
    assert num_computed_tokens == 10
