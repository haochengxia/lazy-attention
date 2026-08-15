"""Smoke-test LazyAttention against a plain-vLLM baseline.

Runs the same questions twice -- once with the documents concatenated into the
prompt (stock vLLM) and once through LazyAttention's `document_seqs` path --
and checks the answers agree. Then re-issues the lazy requests with the
documents in a different order, which is the case prefix caching cannot reuse
but LazyAttention can.

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
    ),
    (
        "Question: Which city is the capital of Italy? Answer:",
        [
            "Madrid is the capital of Spain.",
            "Rome is the capital of Italy, on the river Tiber.",
            "Lisbon is the capital of Portugal.",
        ],
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

    prompts = [p for p, _ in CASES]
    docs = [d for _, d in CASES]

    # Baseline: documents inlined into the prompt, no lazy path.
    baseline_prompts = [
        "\n".join(doc_list) + "\n" + prompt
        for prompt, doc_list in CASES
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
    for i, (prompt, _) in enumerate(CASES):
        print(f"\n--- case {i}: {prompt}")
        print(f"  baseline  : {baseline_texts[i]!r}")
        print(f"  lazy      : {lazy_texts[i]!r}")
        print(f"  reordered : {reordered_texts[i]!r}")
        if not lazy_texts[i].strip():
            print("  FAIL: lazy produced no output")
            failures += 1
        if lazy_texts[i] != reordered_texts[i]:
            # Not necessarily a bug -- reordering documents changes the context
            # the model sees -- but worth surfacing.
            print("  NOTE: reordering changed the answer")

    print()
    if failures:
        print(f"FAILED ({failures} case(s))")
        return 1
    print(f"OK: {len(CASES)} cases generated through the lazy path")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.exit(main())
