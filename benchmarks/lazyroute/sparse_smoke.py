"""Does the engine-side router actually route, and does the answer survive?

Two questions the unit tests cannot answer, because both are about the whole
engine rather than the pieces:

1. **Is anything being dropped?** A router that silently no-ops -- an empty
   `routable` mask, a geometry buffer that never got filled -- produces exactly
   the same answers as dense, so "the answers look right" is not evidence that
   sparse decode ran. The router's own walk-length accounting is. It reaches us
   through the log rather than a return value because the router lives in the
   EngineCore's worker process: an accessor called from here would read a
   router that never ran.
2. **Does the answer survive?** Reported as agreement with a dense run over the
   same corpus, not as an accuracy rate. §9b's standing rule: at 1B, EM
   adjudicates nothing (random selection "beat" dense by 3.3 points), while
   generation identity ranked the policies in exactly the KL order.

Corpus and template come from `corpus.py`, which is the single place that
builds blocks and prompts -- these checkpoints want the Tulu markers, and
handed Llama-3 headers the 1B model denies text that is plainly in front of it.
The preamble is block 0 there, so `LAZY_SPARSE_KEEP_DOC0` keeps its default.

    python benchmarks/lazyroute/sparse_smoke.py --examples 8 --docs 10
    LAZY_SPARSE=1 LAZY_SPARSE_BUDGET_TOKENS=0.25 LAZY_SPARSE_PROFILE=50 \
        python benchmarks/lazyroute/sparse_smoke.py --examples 8 --docs 10
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "lazy_attn"))
sys.path.insert(0, os.path.dirname(_HERE))

import lazy.__vllm__  # noqa: F401,E402  (installs the patches)

import vllm.transformers_utils.tokenizer as _vtok  # noqa: E402

_orig = _vtok.get_cached_tokenizer
_vtok.get_cached_tokenizer = lambda t: (
    setattr(t, "all_special_tokens_extended", t.all_special_tokens) or _orig(t)
) if not hasattr(t, "all_special_tokens_extended") else _orig(t)

from vllm import SamplingParams  # noqa: E402

from lazy.entrypoints.llm import LazyLLM  # noqa: E402
from lazyroute.corpus import load_2wiki, widen  # noqa: E402

MODEL = "hxia7/Llama-3.2-1B-block-FT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--docs", type=int, default=10,
                        help="corpus size per example, padded with distractors")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dump", default="",
                        help="write generations here, to diff against a "
                             "dense run with --compare")
    parser.add_argument("--compare", default="",
                        help="a --dump file from a dense run; report identity")
    args = parser.parse_args()

    pool = load_2wiki(limit=max(args.examples * 4, 40))
    examples = [
        widen(ex, args.docs, pool) if args.docs > len(ex.documents) else ex
        for ex in pool[:args.examples]
    ]

    llm = LazyLLM(model=args.model,
                  gpu_memory_utilization=0.85,
                  enable_prefix_caching=True,
                  trust_remote_code=True,
                  enforce_eager=True)
    outputs = llm.generate(
        prompts=[ex.prompt() for ex in examples],
        sampling_params=SamplingParams(temperature=0.0,
                                       max_tokens=args.max_tokens),
        document_seqs=[ex.blocks() for ex in examples])

    generations = [out.outputs[0].text.strip() for out in outputs]
    gold_hits = sum(
        ex.answer.lower() in gen.lower()
        for ex, gen in zip(examples, generations))
    print(f"EXAMPLES: {len(examples)} docs_per_example={args.docs}")
    print(f"GOLD_CONTAINMENT: {gold_hits}/{len(examples)}")

    if args.dump:
        with open(args.dump, "w") as handle:
            json.dump(generations, handle)
        print(f"DUMPED: {args.dump}")

    if args.compare:
        with open(args.compare) as handle:
            dense = json.load(handle)
        identical = sum(a == b for a, b in zip(dense, generations))
        print(f"IDENTICAL_TO_DENSE: {identical}/{len(generations)}")

    sparse = os.environ.get("LAZY_SPARSE", "").strip().lower() in ("1", "true",
                                                                  "yes", "on")
    if sparse and not os.environ.get("LAZY_SPARSE_PROFILE", "").strip():
        print("NOTE: set LAZY_SPARSE_PROFILE=50 to see the router's "
              "kept_fraction in the worker log -- without it this run cannot "
              "show that anything was dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
