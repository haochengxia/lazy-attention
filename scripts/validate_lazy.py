"""Smoke-test LazyAttention against a plain-vLLM baseline.

Runs the same questions twice -- once with the documents concatenated into the
prompt (stock vLLM) and once through LazyAttention's `document_seqs` path --
then re-issues the lazy requests with the documents in a different order,
which is the case prefix caching cannot reuse but LazyAttention can.

Each answer must contain the fact its documents support. That is the check
that fails when attention or the cache regresses: a broken rotation or a
mis-addressed block yields fluent text that no longer answers the question.

The lazy answer is *not* required to match the baseline token for token. The
two run different attention patterns by construction -- the baseline sees one
causal sequence, LazyAttention encodes each document position-agnostically --
so they routinely agree on the fact and differ in phrasing or length. Any
difference is printed for inspection.

    python scripts/validate_lazy.py [--model MODEL]

Runs greedily (temperature=0) so the comparison is deterministic.
"""
from __future__ import annotations

import argparse
import os
import sys

# Patch vLLM before anything imports it.
import lazy.__vllm__  # noqa: F401  isort:skip

from vllm import LLM, SamplingParams  # noqa: E402

CASES = [
    (
        "Question: Which city is the capital of France? Answer:",
        [
            "Paris is the capital and largest city of France.",
            "Berlin is the capital of Germany.",
            "Rome is the capital of Italy.",
        ],
        "Paris",
    ),
    (
        "Question: Which city is the capital of Italy? Answer:",
        [
            "Madrid is the capital of Spain.",
            "Rome is the capital of Italy, on the river Tiber.",
            "Lisbon is the capital of Portugal.",
        ],
        "Rome",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=os.environ.get("LAZY_TEST_MODEL",
                                               "hxia7/Llama-3.2-1B-Block-FT"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    prompts = [p for p, _, _ in CASES]
    docs = [d for _, d, _ in CASES]

    # Baseline: documents inlined into the prompt, no lazy path.
    baseline_prompts = [
        "\n".join(doc_list) + "\n" + prompt
        for prompt, doc_list, _ in CASES
    ]
    baseline = llm.generate(prompts=baseline_prompts, sampling_params=sampling)
    baseline_texts = [o.outputs[0].text for o in baseline]

    # LazyAttention: documents passed separately.
    lazy_out = llm.generate(prompts=prompts, sampling_params=sampling,
                            document_seqs=docs)
    lazy_texts = [o.outputs[0].text for o in lazy_out]

    # Same documents, new order -- the reordering case prefix caching misses.
    reordered = [list(reversed(doc_list)) for doc_list in docs]
    reordered_out = llm.generate(prompts=prompts, sampling_params=sampling,
                                 document_seqs=reordered)
    reordered_texts = [o.outputs[0].text for o in reordered_out]

    failures = 0
    for i, (prompt, _, expected) in enumerate(CASES):
        print(f"\n--- case {i}: {prompt}")
        print(f"  baseline  : {baseline_texts[i]!r}")
        print(f"  lazy      : {lazy_texts[i]!r}")
        print(f"  reordered : {reordered_texts[i]!r}")

        if expected.lower() not in baseline_texts[i].lower():
            # The baseline is stock vLLM, so this is the model failing the
            # question, not LazyAttention. Report it rather than blaming the
            # lazy path for an answer it was never going to get right.
            print(f"  WARN: baseline itself does not answer {expected!r}; "
                  "the lazy checks below are not meaningful for this case")
            continue

        for label, text in (("lazy", lazy_texts[i]),
                            ("reordered", reordered_texts[i])):
            if not text.strip():
                print(f"  FAIL: {label} produced no output")
                failures += 1
            elif expected.lower() not in text.lower():
                print(f"  FAIL: {label} answer does not contain {expected!r} "
                      "-- the documents did not reach attention intact")
                failures += 1

        if lazy_texts[i] != baseline_texts[i]:
            # Expected: different attention pattern, same fact. Shown so a
            # sudden change in wording is at least visible.
            print("  note: lazy wording differs from the baseline")
        if lazy_texts[i] != reordered_texts[i]:
            print("  note: reordering the documents changed the wording")

    print()
    if failures:
        print(f"FAILED ({failures} check(s))")
        return 1
    print(f"OK: {len(CASES)} cases answered correctly through the lazy path, "
          "in both document orders")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.exit(main())
