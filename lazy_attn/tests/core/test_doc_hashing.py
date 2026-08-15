"""Hashing rules the per-document KV cache depends on.

The lazy paths hash documents outside vLLM's `hash_request_tokens`, so the
properties upstream gets for free -- cache-salt isolation, and agreement
between the hash a parent computes for a document and the hash the spawned
document request writes -- have to be checked here.
"""
from __future__ import annotations

import pytest

import lazy.__vllm__  # noqa: F401  applies the patches

from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import hash_request_tokens

from lazy.core.kv_cache_utils import (hash_request_tokens_docs,
                                      hash_request_tokens_with_doc_hash)
from lazy.core.sched.scheduler import build_document_request
from lazy.request import LazyRequest

BLOCK_SIZE = 4
DOCUMENTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]]


def make_request(cache_salt=None, request_id="r0"):
    from vllm import SamplingParams

    return LazyRequest(
        request_id=request_id,
        prompt_token_ids=[100, 101, 102, 103],
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=SamplingParams(max_tokens=1),
        eos_token_id=None,
        arrival_time=0.0,
        cache_salt=cache_salt,
        documents_token_ids_padded=DOCUMENTS,
        document_lens=[len(d) for d in DOCUMENTS],
        document_lens_padded=[len(d) for d in DOCUMENTS],
        document_seq_hash="abc123",
    )


@pytest.mark.unit
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_salt_isolates_document_hashes(hash_fn):
    """Two requests with different salts must not share document blocks."""
    unsalted = hash_request_tokens_docs(hash_fn, BLOCK_SIZE, make_request())
    salted = hash_request_tokens_docs(hash_fn, BLOCK_SIZE,
                                      make_request(cache_salt="tenant-a"))
    other = hash_request_tokens_docs(hash_fn, BLOCK_SIZE,
                                     make_request(cache_salt="tenant-b"))

    for doc_idx in range(len(DOCUMENTS)):
        assert salted[doc_idx] != unsalted[doc_idx]
        assert salted[doc_idx] != other[doc_idx]

    # Same salt, same documents -> same hashes, or nothing is ever reused.
    again = hash_request_tokens_docs(hash_fn, BLOCK_SIZE,
                                     make_request(cache_salt="tenant-a",
                                                  request_id="r1"))
    assert again == salted


@pytest.mark.unit
@pytest.mark.parametrize("cache_salt", [None, "tenant-a"])
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_document_hashes_match_the_spawned_request(hash_fn, cache_salt):
    """The parent's per-document hashes must equal what the document request
    writes to the cache -- otherwise the parent never sees a hit and keeps
    respawning the document."""
    parent = make_request(cache_salt=cache_salt)
    by_parent = hash_request_tokens_docs(hash_fn, BLOCK_SIZE, parent)

    for doc_idx in range(len(DOCUMENTS)):
        # The real spawn path, so a field the scheduler stops forwarding shows
        # up here as a hash mismatch.
        doc_request = build_document_request(parent, doc_idx)
        by_document = hash_request_tokens(hash_fn, BLOCK_SIZE, doc_request)
        assert by_parent[doc_idx] == by_document


@pytest.mark.unit
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_salt_isolates_query_hashes(hash_fn):
    """The query blocks chained behind the document-sequence hash carry the
    salt too."""
    unsalted = hash_request_tokens_with_doc_hash(hash_fn, BLOCK_SIZE,
                                                 make_request())
    salted = hash_request_tokens_with_doc_hash(
        hash_fn, BLOCK_SIZE, make_request(cache_salt="tenant-a"))

    assert unsalted and salted
    assert salted != unsalted


@pytest.mark.unit
def test_pooling_request_has_no_structured_output():
    """Pooling requests carry no sampling_params; the scheduler still asks."""
    from vllm.pooling_params import PoolingParams

    request = LazyRequest(
        request_id="pool",
        prompt_token_ids=[1, 2, 3],
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=None,
        pooling_params=PoolingParams(),
        eos_token_id=None,
        arrival_time=0.0,
    )
    assert request.use_structured_output is False
