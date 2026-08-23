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

# Subspan EM, the repo's own 2wiki metric: normalise away case, punctuation and
# articles on both sides, then ask whether the gold answer appears anywhere in
# the generation. Imported rather than reimplemented so this reports the same
# number `blockbench` does -- a private near-copy is how two arms of the same
# experiment quietly stop being comparable.
from blockbench import qa_em_score  # noqa: E402

MODEL = "hxia7/Llama-3.2-1B-block-FT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--docs", type=int, default=10,
                        help="corpus size per example, padded with distractors")
    # These checkpoints restate the question before answering ("What is the
    # best answer for the question: ..."), so a short budget measures
    # truncation rather than retrieval: at 32 tokens both dense and sparse
    # score ~1/8 purely because the answer had not been reached yet.
    parser.add_argument("--max-tokens", type=int, default=128)
    # Both arms must batch identically or the comparison is not paired. It is
    # also a memory bound: the router's scoring tile is sized against the batch
    # (see `score_tile_blocks`), and on a 16 GB card an unbounded batch of
    # 10-document requests leaves it nothing to work in.
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--dump", default="",
                        help="write generations here, to diff against a "
                             "dense run with --compare")
    parser.add_argument("--compare", default="",
                        help="a --dump file from a dense run; report identity")
    parser.add_argument("--verbose", action="store_true",
                        help="with --compare, print the examples whose "
                             "correctness flipped, both directions")
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
                  enforce_eager=True,
                  max_num_seqs=args.max_num_seqs)
    outputs = llm.generate(
        prompts=[ex.prompt() for ex in examples],
        sampling_params=SamplingParams(temperature=0.0,
                                       max_tokens=args.max_tokens),
        document_seqs=[ex.blocks() for ex in examples])

    generations = [out.outputs[0].text.strip() for out in outputs]
    hits = sum(
        qa_em_score(gen, [ex.answer])
        for ex, gen in zip(examples, generations))
    print(f"EXAMPLES: {len(examples)} docs_per_example={args.docs} "
          f"max_tokens={args.max_tokens}")
    print(f"SUBSPAN_EM: {hits:.0f}/{len(examples)} "
          f"({hits / len(examples):.3f})")
    truncated = sum(
        len(out.outputs[0].token_ids) >= args.max_tokens for out in outputs)
    print(f"HIT_TOKEN_LIMIT: {truncated}/{len(examples)}")

    if args.dump:
        with open(args.dump, "w") as handle:
            json.dump(generations, handle)
        print(f"DUMPED: {args.dump}")

    if args.compare:
        with open(args.compare) as handle:
            dense = json.load(handle)
        identical = sum(a == b for a, b in zip(dense, generations))
        print(f"IDENTICAL_TO_DENSE: {identical}/{len(generations)}")

        # Paired outcomes are what carry signal here: comparing two EM *rates*
        # off a ~0.27 baseline resolves nothing at any sample size we can
        # afford, but "which examples flipped, and in which direction" is a
        # McNemar table and is readable at n=40.
        won, lost = [], []
        for idx, (ex, before, after) in enumerate(
                zip(examples, dense, generations)):
            d_hit = qa_em_score(before, [ex.answer])
            s_hit = qa_em_score(after, [ex.answer])
            if d_hit and not s_hit:
                lost.append(idx)
            elif s_hit and not d_hit:
                won.append(idx)
        print(f"MCNEMAR: dense_only={len(lost)} sparse_only={len(won)} "
              f"(indices lost={lost} won={won})")

        if args.verbose:
            for idx in sorted(set(lost) | set(won)):
                ex = examples[idx]
                print(f"\n--- example {idx} | gold={ex.answer!r} | "
                      f"supporting_blocks={ex.supporting_blocks} ---")
                print(f"  dense : {dense[idx][:220]!r}")
                print(f"  sparse: {generations[idx][:220]!r}")

    sparse = os.environ.get("LAZY_SPARSE", "").strip().lower() in ("1", "true",
                                                                  "yes", "on")
    if sparse and not os.environ.get("LAZY_SPARSE_PROFILE", "").strip():
        print("NOTE: set LAZY_SPARSE_PROFILE=50 to see the router's "
              "kept_fraction in the worker log -- without it this run cannot "
              "show that anything was dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
