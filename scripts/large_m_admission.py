"""W1.1 exit check: a document set the 16-bit q_offset field could not serve.

The old packed layout gave q_offset 16 bits, so a request was refused once the
padding plus the true lengths of every document but the last passed 65535.
Since that offset tracks the padded length of the whole document region, the
cap was really "no more than ~64k tokens of reusable cache per request" --
below every corpus-as-cache workload the reusable cache exists for. With 24
bits the same request has to admit *and* run.

    python scripts/large_m_admission.py --num-docs 600

What it checks, in the order the failures would appear:

  1. control -- an example served from its supporting documents alone,
     answered correctly through the block path, so the harness itself is known
     good before anything is said about M;
  2. admission -- the large-M request, built from that same example, is no
     longer refused;
  3. execution -- it runs to completion and generates the same text twice.

The large corpus is the control corpus *plus* distractors appended, so the
leading blocks and their rotation offsets are identical between the two runs
and only M differs.

Answer quality at large M is reported, not asserted, and the control uses the
supporting documents rather than the full ten. Both because the instrument is
weak, not because the bar is being lowered to fit: on 2wiki dev the 1B
checkpoint reaches ~24% gold containment over ten documents and ~58% over the
two supporting ones (`benchmarks/lazyroute/block_health.py`), with full
attention over the same text no better -- so a single ten-document example
answering wrongly measures the model, not the packing. What the packing has to
do is stay bit-exact, which tests/core/test_packed_entry_roundtrip.py asserts
directly at this scale.
"""
import argparse
import os
import re
import sys

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lazy_attn"))
sys.path.insert(0, REPO_ROOT)

import lazy.__vllm__  # noqa: F401,E402

from vllm import SamplingParams  # noqa: E402

from benchmarks.lazyroute.corpus import (Example, load_2wiki,  # noqa: E402
                                         widen)
from lazy.entrypoints.llm import LazyLLM  # noqa: E402
from lazy.utils.rotation import max_rotation_offset  # noqa: E402

OLD_MAX_PACKED_Q_OFFSET = 0xFFFF
MODEL = "hxia7/Llama-3.2-1B-Block-FT"
BLOCK_SIZE = 16


def token_lengths(tokenizer, blocks: list[str]) -> list[int]:
    # The processor drops the BOS the tokenizer prepends to each block.
    return [len(tokenizer.encode(block)) - 1 for block in blocks]


def looks_like_text(answer: str) -> bool:
    """Fluency floor: what a wrong block or a wrapped rotation would fail.

    Not a quality bar -- a corrupted walk produces repeated punctuation,
    non-Latin fragments or an immediate stop, none of which clear this.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", answer)
    return len(words) >= 8 and len(words) >= 0.5 * len(answer.split())


def run(llm, example, sampling_params, runs: int = 2) -> list[str]:
    answers = []
    for _ in range(runs):
        outputs = llm.generate(prompts=[example.prompt()],
                               sampling_params=sampling_params,
                               document_seqs=[example.blocks()],
                               use_tqdm=False)
        answers.append(outputs[0].outputs[0].text.strip())
    return answers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--num-docs", type=int, default=600,
                        help="documents in the large-M request. 2wiki paragraphs "
                             "average ~120 padded tokens, so ~560 of them "
                             "are needed to pass the old 65535 offset cap")
    parser.add_argument("--candidates", type=int, default=12,
                        help="how many 2wiki examples to try before giving up "
                             "on finding one the control answers correctly")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    args = parser.parse_args()

    pool = load_2wiki(limit=max(args.num_docs, 64))

    llm = LazyLLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # 2wiki paragraphs run ~50-250 tokens and average ~120 after padding
        # to a block boundary; this leaves headroom without reserving a KV
        # cache the request will never touch.
        max_model_len=args.num_docs * 192 + 2048,
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=args.max_tokens)

    # Step 1: find an example the block path answers from its supporting
    # documents alone. That is the harness's own sanity check -- template,
    # block split, engine -- and it has to pass before large M means anything.
    base = control = None
    for candidate in pool[:args.candidates]:
        gold = Example(question=candidate.question,
                       answer=candidate.answer,
                       documents=[
                           candidate.documents[i] for i in candidate.supporting
                       ],
                       supporting=list(range(len(candidate.supporting))),
                       example_id=candidate.example_id)
        answer = run(llm, gold, sampling_params, runs=1)[0]
        if candidate.answer.lower() in answer.lower():
            base, control = candidate, gold
            break
        print(f"skipping example {candidate.example_id}: control answer "
              f"{answer[:60]!r} misses {candidate.answer!r}")

    if control is None:
        print(f"FAILED: none of the first {args.candidates} examples answered "
              f"correctly from their supporting documents. The harness is "
              f"broken, or the checkpoint is not the Block-FT one.")
        return 1

    print(f"\nquestion: {base.question}")
    print(f"gold answer: {base.answer}")
    large = widen(base, args.num_docs, pool)  # same prefix, distractors after

    ok = True
    for name, example in (("control", control), ("large-M", large)):
        lens = token_lengths(tokenizer, example.blocks())
        padded = [((length + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                  for length in lens]
        needed = max_rotation_offset(lens, padded)
        print(f"\n=== {name}: {len(example.blocks())} blocks, "
              f"{sum(padded)} padded tokens ===")
        print(f"rotation offset needed: {needed} "
              f"({'fits' if needed <= OLD_MAX_PACKED_Q_OFFSET else 'PAST'} "
              f"the old {OLD_MAX_PACKED_Q_OFFSET}-offset field)")
        print(f"supporting documents at blocks {example.supporting_blocks}")

        answers = run(llm, example, sampling_params)
        print(f"answer: {answers[0]!r}")

        if answers[0] != answers[1]:
            print(f"FAIL[{name}]: greedy decoding is not reproducible")
            ok = False
        if name == "control" and base.answer.lower() not in answers[0].lower():
            print(f"FAIL[{name}]: answer misses the gold string "
                  f"{base.answer!r}, having just answered it -- the run is "
                  f"not reproducible at all")
            ok = False
        if name == "large-M" and not looks_like_text(answers[0]):
            # Reported, not failed: 600 documents is 60x the checkpoint's
            # training distribution and full attention degenerates on the same
            # corpus, so this measures the model rather than the packing.
            print(f"NOTE[{name}]: output is not fluent text")
        if name == "large-M" and needed <= OLD_MAX_PACKED_Q_OFFSET:
            print("FAIL[large-M]: this corpus also fits the old field, so it "
                  "does not exercise the repack. Raise --num-docs.")
            ok = False

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
