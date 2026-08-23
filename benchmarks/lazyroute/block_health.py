"""Is the 1B Block-FT checkpoint usable as the cheap iteration instrument?

PROJECT.md leans on `hxia7/Llama-3.2-1B-Block-FT` for every Phase-0 sweep, with
the 8B checkpoint kept for headline numbers. That only works if the 1B model
answers 2wiki sensibly *through the block path* -- attention-mass statistics
collected from a model that is producing word salad measure the word salad.

So: accuracy on 2wiki dev at the standard ten-document setting, three ways.

    block   documents as separate cache blocks (document_seqs) -- the object
            under study; documents cannot see each other
    inline  the same text as one prompt (full attention) -- the ceiling, and
            the control that separates "block attention is lossy here" from
            "this model cannot do this task"
    gold    the two supporting documents only, as blocks -- separates a
            routing failure (too many distractors) from a block-attention one

Scoring is substring containment of the gold answer, the same rough measure
`benchmarks/grade_accuracy.py` starts from; it is a floor, not an EM number.
Alongside it, a degeneracy rate: the fraction of answers that collapse into
repetition ("is a film is a film"), which is a different failure from being
wrong and the one that says a context has stopped being usable at all.

    python benchmarks/lazyroute/block_health.py --examples 50
    python benchmarks/lazyroute/block_health.py --doc-counts 10,20,40,80

`--doc-counts` widens each example with distractors from other questions and
sweeps M, which locates the context length past which this checkpoint stops
producing text -- the ceiling every large-M experiment has to sit under.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "lazy_attn"))
sys.path.insert(0, REPO_ROOT)

import lazy.__vllm__  # noqa: F401,E402

from vllm import SamplingParams  # noqa: E402

from benchmarks.lazyroute.corpus import (Example, load_2wiki,  # noqa: E402
                                         widen)
from lazy.entrypoints.llm import LazyLLM  # noqa: E402

MODEL = "hxia7/Llama-3.2-1B-Block-FT"


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def contains_gold(answer: str, gold: str) -> bool:
    return normalise(gold) in normalise(answer)


def degenerate(answer: str) -> bool:
    """Repetition of the kind a broken context produces, not a wrong answer."""
    words = normalise(answer).split()
    if len(words) < 8:
        return True
    # A fluent sentence of this length repeats few words; "is a film is a film"
    # collapses to a handful of types.
    return len(set(words)) < 0.35 * len(words)


def gold_only(example: Example) -> Example:
    return Example(question=example.question,
                   answer=example.answer,
                   documents=[example.documents[i] for i in example.supporting],
                   supporting=list(range(len(example.supporting))),
                   example_id=example.example_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--show", type=int, default=3,
                        help="print this many answers per arm")
    parser.add_argument("--doc-counts", default="",
                        help="comma-separated M values to sweep instead of "
                             "running the three fixed arms")
    args = parser.parse_args()

    doc_counts = [int(m) for m in args.doc_counts.split(",") if m.strip()]
    examples = load_2wiki(limit=max(args.examples, max(doc_counts or [0])))
    llm = LazyLLM(model=args.model,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=(max(doc_counts) * 192 +
                                 2048) if doc_counts else 8192,
                  enable_prefix_caching=True,
                  enforce_eager=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def answer_for(example: Example, inline: bool) -> str:
        blocks = example.blocks()
        if inline:
            return llm.generate(prompts=["".join(blocks) + example.prompt()],
                                sampling_params=sampling,
                                use_tqdm=False)[0].outputs[0].text.strip()
        return llm.generate(prompts=[example.prompt()],
                            sampling_params=sampling,
                            document_seqs=[blocks],
                            use_tqdm=False)[0].outputs[0].text.strip()

    if doc_counts:
        print(f"2wiki dev, {args.examples} examples widened with distractors, "
              f"{args.model.split('/')[-1]}")
        print(f"{'M':>5s} {'arm':8s} {'contains gold':>14s} "
              f"{'degenerate':>12s}")
        for num_docs in doc_counts:
            for inline in (False, True):
                hits, bad = [], []
                for example in examples[:args.examples]:
                    wide = widen(example, num_docs, examples)
                    answer = answer_for(wide, inline)
                    hits.append(contains_gold(answer, example.answer))
                    bad.append(degenerate(answer))
                print(f"{num_docs:5d} {'inline' if inline else 'block':8s} "
                      f"{100.0 * sum(hits) / len(hits):13.1f}% "
                      f"{100.0 * sum(bad) / len(bad):11.1f}%")
        return 0

    arms = {
        "block": lambda ex: (ex, False),
        "inline": lambda ex: (ex, True),
        "gold": lambda ex: (gold_only(ex), False),
    }
    scores: dict[str, list[bool]] = {name: [] for name in arms}
    broken: dict[str, list[bool]] = {name: [] for name in arms}

    for name, build in arms.items():
        shown = 0
        for example in examples[:args.examples]:
            target, inline = build(example)
            answer = answer_for(target, inline)
            scores[name].append(contains_gold(answer, example.answer))
            broken[name].append(degenerate(answer))
            if shown < args.show:
                shown += 1
                print(f"[{name}] gold={example.answer!r}\n"
                      f"        answer={answer[:150]!r}")

    print(f"\n2wiki dev, first {args.examples} examples, "
          f"{args.model.split('/')[-1]}")
    print(f"{'arm':8s} {'contains gold':>14s} {'degenerate':>12s}")
    for name in arms:
        hit = 100.0 * sum(scores[name]) / len(scores[name])
        bad = 100.0 * sum(broken[name]) / len(broken[name])
        print(f"{name:8s} {hit:13.1f}% {bad:11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
