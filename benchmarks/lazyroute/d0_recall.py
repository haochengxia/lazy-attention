"""D0.1 / D0.2: is there anything to route to, and can a cheap scorer find it?

Both questions are asked of the same logged object (`probe.py`), so the scorer
ladder is measured against the oracle it is trying to approximate rather than
against a separately-built reference.

    D0.1  oracle recall@budget -- attend to the highest-mass pages (or
          documents) up to a token budget; what fraction of the true attention
          mass does that capture? A curve that rises steeply is the existence
          claim: each decode token needs few blocks.

    D0.2  the same curve with the selection made by a scorer that a router
          could actually afford -- the Quest min/max box over a page's stored
          keys, a centroid, or nothing at all (random) -- and with the box
          computed with and without each document's first tokens, which are
          outliers in a checkpoint with no trained sink.

Budgets are in *tokens*, not pages or documents, because that is what decode
bytes are charged in and it is the only unit under which page-level and
document-level selection can be compared (G0-C).

Two normalisations are reported and they answer different questions:

    all           over every cached token, preamble included. The preamble is
                  block 0 and holds the sequence's first token, which is the
                  attention sink -- so this number is dominated by a block the
                  router keeps unconditionally.
    documents     the preamble excluded from both the budget and the mass:
                  routing among the reference documents, which is the actual
                  decision.

    python benchmarks/lazyroute/d0_recall.py --examples 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from benchmarks.lazyroute.corpus import load_2wiki  # noqa: E402
from benchmarks.lazyroute.probe import (PAGE, BlockProbe,  # noqa: E402
                                        ProbeResult)

MODEL = "hxia7/Llama-3.2-1B-Block-FT"
BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75)
SCORERS = ("oracle", "quest", "quest_nosink", "centroid", "random")


@dataclass
class Descriptors:
    """Per-page summaries of the stored keys, in each document's local frame.

    Built once per document at cache time -- which is the whole point: they are
    a property of the block, not of the request that happens to be reading it.
    """
    kmin: torch.Tensor  # [layers, kv_heads, pages, head_dim]
    kmax: torch.Tensor
    centroid: torch.Tensor


def build_descriptors(result: ProbeResult, num_layers: int,
                      sink_exclude: int = 0) -> Descriptors:
    """Segmented min/max/mean over each page's keys.

    `sink_exclude` drops a document's first tokens from the statistics. Neither
    Block-FT checkpoint trained a sink token, so those keys are outliers; a box
    stretched to contain them is loose for every query (PROJECT.md §1.9).
    The tokens are still attendable -- only the *summary* ignores them, and a
    page left with nothing gets a box that scores -inf and is never selected on
    its own merits.
    """
    layout = result.layout
    kmin, kmax, centroid = [], [], []
    for doc_idx, document in enumerate(result.documents):
        for offset in range(0, document.length, PAGE):
            stop = min(offset + PAGE, document.length)
            start = max(offset, sink_exclude) if offset < sink_exclude else offset
            if start >= stop:  # a page entirely inside the excluded prefix
                start = offset
            # [layers, kv_heads, tokens, head_dim]
            keys = torch.stack([
                document.keys[layer][:, start:stop].float()
                for layer in range(num_layers)
            ])
            kmin.append(keys.amin(dim=2))
            kmax.append(keys.amax(dim=2))
            centroid.append(keys.mean(dim=2))
    stack = lambda values: torch.stack(values, dim=2)  # -> [L, H, pages, D]
    assert len(kmin) == layout.num_pages
    return Descriptors(kmin=stack(kmin), kmax=stack(kmax),
                       centroid=stack(centroid))


def rotated_queries(probe: BlockProbe, result: ProbeResult,
                    trace) -> torch.Tensor:
    """The decode query as each document's frame sees it.

    One rotary call for the whole trace: layers and heads are folded into the
    head axis and the per-document rotation angles into the sequence axis.
    Returns [layers, q_heads, num_docs, head_dim].
    """
    layout = result.layout
    num_docs = layout.num_docs
    offsets = trace.position - layout.frame_offset  # [num_docs]
    query = trace.query.to(probe.dtype)  # [layers, q_heads, head_dim]
    folded = query.reshape(-1, 1, probe.head_dim).expand(-1, num_docs, -1)
    rotated = probe._rotate(folded.contiguous(), offsets)
    return rotated.reshape(probe.num_layers, probe.num_q_heads, num_docs,
                           probe.head_dim).float()


def page_scores(probe: BlockProbe, result: ProbeResult, trace,
                descriptors: dict[str, Descriptors],
                generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Every scorer's view of every page, as [layers, kv_heads, pages].

    Scores are computed per query head and then aggregated over the GQA group
    by max, which is the router's default: a page one head in the group needs
    has to survive selection for the whole group, since they share a walk.
    """
    layout = result.layout
    per_page_query = rotated_queries(probe, result, trace)[:, :, layout.page_doc]
    scores: dict[str, torch.Tensor] = {}

    def group_max(per_q_head: torch.Tensor) -> torch.Tensor:
        shaped = per_q_head.reshape(probe.num_layers, probe.num_kv_heads,
                                    probe.group, layout.num_pages)
        return shaped.amax(dim=2)

    for name, desc in descriptors.items():
        # A KV head's descriptor is shared by its whole query group.
        expand = lambda tensor: tensor.repeat_interleave(probe.group, dim=1)
        if name.startswith("quest"):
            kmin, kmax = expand(desc.kmin), expand(desc.kmax)
            bound = torch.maximum(per_page_query * kmin, per_page_query * kmax)
            scores[name] = group_max(bound.sum(-1))
        else:
            centroid = expand(desc.centroid)
            scores[name] = group_max((per_page_query * centroid).sum(-1))
    scores["random"] = torch.rand(
        (probe.num_layers, probe.num_kv_heads, layout.num_pages),
        generator=generator, device=per_page_query.device)
    return scores


def true_page_mass(result: ProbeResult, trace, probe: BlockProbe) -> torch.Tensor:
    """Attention mass per page, summed over each GQA group: [L, kv_heads, pages]."""
    layout = result.layout
    per_page = torch.zeros(probe.num_layers, probe.num_q_heads,
                           layout.num_pages, device=trace.attention.device)
    per_page.index_add_(2, layout.page_of_token, trace.attention)
    return per_page.reshape(probe.num_layers, probe.num_kv_heads, probe.group,
                            layout.num_pages).sum(dim=2)


def recall_curve(scores: torch.Tensor, mass: torch.Tensor,
                 unit_tokens: torch.Tensor, budgets=BUDGETS,
                 forced: torch.Tensor = None) -> torch.Tensor:
    """Mass captured by taking units in score order until the budget runs out.

    `unit_tokens` is what each unit costs, so the same function serves page
    selection and document selection and the two are comparable at equal cost
    (G0-C). Returns [layers, kv_heads, len(budgets)].
    """
    if forced is not None:
        # A stripe policy: these units are taken before anything is scored,
        # and they are charged for. Sorting them to the front reproduces
        # "keep these, then spend what is left by score" exactly.
        scores = scores.masked_fill(forced, float("inf"))
    order = scores.argsort(dim=-1, descending=True)
    ordered_mass = mass.gather(-1, order)
    ordered_cost = unit_tokens.expand_as(order).gather(-1, order)
    cumulative_mass = ordered_mass.cumsum(-1)
    cumulative_cost = ordered_cost.cumsum(-1)
    total_mass = mass.sum(-1, keepdim=True).clamp_min(1e-9)
    total_cost = float(unit_tokens.sum())

    curves = []
    for budget in budgets:
        allowed = cumulative_cost <= max(budget * total_cost, 1.0)
        captured = torch.where(allowed, cumulative_mass,
                               torch.zeros_like(cumulative_mass)).amax(-1)
        curves.append(captured / total_mass.squeeze(-1))
    return torch.stack(curves, dim=-1)


def to_documents(scores: torch.Tensor, mass: torch.Tensor,
                 page_doc: torch.Tensor, num_docs: int):
    """Page scores/mass -> document scores/mass. Score is max over pages (R2)."""
    shape = scores.shape[:-1] + (num_docs, )
    doc_scores = torch.full(shape, float("-inf"), device=scores.device)
    doc_scores.index_reduce_(-1, page_doc, scores, "amax", include_self=True)
    doc_mass = torch.zeros(shape, device=mass.device)
    doc_mass.index_add_(-1, page_doc, mass)
    return doc_scores, doc_mass


def selection_overlap(mass_by_step: list[torch.Tensor],
                      fraction: float = 0.25) -> tuple[float, float]:
    """How much a step's chosen page set is still the previous step's (D0.3).

    Two numbers, because they answer different questions: consecutive overlap
    sets the cost of an event-driven refresh, and overlap against step 0 says
    whether the set a router picks when it first reads the question survives
    the whole answer -- which is the difference between per-token routing and
    running a retriever once.
    """
    if len(mass_by_step) < 2:
        return float("nan"), float("nan")
    num_pages = mass_by_step[0].shape[-1]
    k = max(1, round(fraction * num_pages))
    masks = []
    for mass in mass_by_step:
        mask = torch.zeros_like(mass, dtype=torch.bool)
        mask.scatter_(-1, mass.topk(k, dim=-1).indices, True)
        masks.append(mask)

    def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(((a & b).sum(-1).float() /
                      (a | b).sum(-1).clamp_min(1).float()).mean())

    consecutive = [jaccard(masks[i - 1], masks[i]) for i in range(1, len(masks))]
    from_first = [jaccard(masks[0], masks[i]) for i in range(1, len(masks))]
    return float(np.mean(consecutive)), float(np.mean(from_first))


def study(probe: BlockProbe, examples, args, request_local: bool) -> dict:
    generator = torch.Generator(device=probe.device).manual_seed(args.seed)
    collected: dict[tuple[str, str, str], list[torch.Tensor]] = {}
    diagnostics: list[dict] = []
    steps_seen: list[int] = []
    overlaps: list[tuple[float, float]] = []
    box_widths: list[float] = []
    checked = False

    for index, example in enumerate(examples):
        # The answer alone, not a lead-in phrase: "According to the provided
        # search documents," is generic text that needs no retrieval, and
        # padding the trace with it would dilute exactly the steps the study is
        # about. Step 0 is the last prompt token -- the step that has just read
        # the question and has not yet committed to anything.
        result = probe.run(example.blocks(), example.prompt(), example.answer,
                           max_steps=args.max_steps,
                           request_local=request_local)
        if not checked:
            check = probe.check_equivalence(result)
            print(f"  local-frame scoring vs model attention: "
                  f"max_abs_diff={check['max_abs_diff']:.4f} "
                  f"({'ok' if check['ok'] else 'FAILED'})")
            assert check["ok"], "scoring frame disagrees with the model"
            checked = True

        layout = result.layout
        descriptors = {
            "quest": build_descriptors(result, probe.num_layers, 0),
            "quest_nosink": build_descriptors(result, probe.num_layers,
                                              args.sink_exclude),
            "centroid": build_descriptors(result, probe.num_layers, 0),
        }
        box_widths.append(
            float((descriptors["quest"].kmax -
                   descriptors["quest"].kmin).mean()))
        page_len = layout.page_len.float()
        doc_len = layout.doc_len.float()
        keep = layout.page_doc != 0  # everything but the preamble block
        local_pos = (torch.arange(layout.num_tokens, device=probe.device) -
                     layout.doc_start[layout.doc_of_token])
        block_head = local_pos < args.sink_exclude
        # The stripe: page 0 of every document, which is where the block-head
        # tokens live. Query-independent, so whatever it captures is not
        # evidence for routing.
        first_page = torch.zeros(layout.num_pages, dtype=torch.bool,
                                 device=probe.device)
        first_page[torch.searchsorted(layout.page_doc,
                                      torch.arange(layout.num_docs,
                                                   device=probe.device))] = True

        mass_by_step = []
        for trace in result.traces:
            steps_seen.append(trace.step)
            mass = true_page_mass(result, trace, probe)
            mass_by_step.append(mass)
            scores = page_scores(probe, result, trace, descriptors, generator)
            scores["oracle"] = mass

            doc_mass_all = None
            for name, score in scores.items():
                doc_score, doc_mass = to_documents(score, mass, layout.page_doc,
                                                   layout.num_docs)
                doc_mass_all = doc_mass
                for scope in ("all", "documents"):
                    if scope == "all":
                        page_args = (score, mass, page_len)
                        doc_args = (doc_score, doc_mass, doc_len)
                    else:
                        page_args = (score[..., keep], mass[..., keep],
                                     page_len[keep])
                        doc_args = (doc_score[..., 1:], doc_mass[..., 1:],
                                    doc_len[1:])
                    collected.setdefault((name, "page", scope), []).append(
                        recall_curve(*page_args).cpu())
                    collected.setdefault((name, "doc", scope), []).append(
                        recall_curve(*doc_args).cpu())
                    stripe = first_page if scope == "all" else first_page[keep]
                    collected.setdefault((name, "page_stripe", scope),
                                         []).append(
                        recall_curve(*page_args, forced=stripe).cpu())

            total_doc = doc_mass_all.sum(-1).clamp_min(1e-9)
            cached = trace.attention.sum(-1).clamp_min(1e-9)
            diagnostics.append({
                "example": example.example_id,
                "step": trace.step,
                "tail_mass": float(trace.tail_mass.mean()),
                "preamble_share": float((doc_mass_all[..., 0] / total_doc).mean()),
                "supporting_share": float(
                    (doc_mass_all[..., example.supporting_blocks].sum(-1) /
                     total_doc).mean()),
                "block_head_share": float(
                    (trace.attention[..., block_head].sum(-1) / cached).mean()),
                "num_pages": layout.num_pages,
                "num_tokens": layout.num_tokens,
            })
        overlaps.append(selection_overlap(mass_by_step))
        print(f"  [{index + 1}/{len(examples)}] {layout.num_docs} blocks, "
              f"{layout.num_pages} pages, {len(result.traces)} steps")

    return {"collected": collected, "diagnostics": diagnostics,
            "steps": torch.tensor(steps_seen), "overlaps": overlaps,
            "box_width": float(np.mean(box_widths))}


def report(name: str, out: dict) -> None:
    collected, step_index = out["collected"], out["steps"]

    def summarise(key, mask=None) -> np.ndarray:
        stacked = torch.stack(collected[key])
        if mask is not None:
            stacked = stacked[mask]
        return stacked.mean(dim=(0, 1, 2)).numpy()

    header = "  ".join(f"{b:>5.0%}" for b in BUDGETS)
    print(f"\n================ arm: {name} ================")
    print("Budget is a fraction of the *cached* tokens in scope; the value is "
          "the share of attention mass captured.")
    for scope in ("all", "documents"):
        for granularity in ("page", "doc"):
            print(f"\n--- {granularity}-level selection, scope={scope} ---")
            print(f"{'scorer':14s} {header}")
            for scorer in SCORERS:
                row = summarise((scorer, granularity, scope))
                print(f"{scorer:14s} " +
                      "  ".join(f"{value:5.3f}" for value in row))

    # Step 0 has just read the question and is where a router would first
    # commit; later steps are decoding an answer it has already found.
    print("\n--- page-level, scope=documents, by decode step ---")
    print(f"{'scorer/step':14s} {header}")
    for scorer in ("oracle", "quest"):
        for label, mask in (("step 0", step_index == 0),
                            ("step 1+", step_index > 0)):
            row = summarise((scorer, "page", "documents"), mask)
            print(f"{scorer + ' ' + label:14s} " +
                  "  ".join(f"{value:5.3f}" for value in row))

    # Per-layer divergence decides whether one routing decision can serve every
    # layer (P3.5) -- reported at the budget the curves are steepest around.
    focus = BUDGETS.index(0.15)
    print(f"\n--- per-layer recall at a {BUDGETS[focus]:.0%} token budget "
          f"(page-level, scope=documents) ---")
    for scorer in ("oracle", "quest", "centroid"):
        per_layer = torch.stack(collected[(scorer, "page",
                                           "documents")]).mean(dim=(0, 2))
        print(f"{scorer:9s} " +
              " ".join(f"{value:4.2f}" for value in per_layer[:, focus].numpy()))

    print("\n--- page-level, scope=documents, with a forced stripe "
          "(page 0 of every document kept, and charged for) ---")
    print(f"{'scorer':14s} {header}")
    for scorer in ("oracle", "quest", "random"):
        row = summarise((scorer, "page_stripe", "documents"))
        print(f"{scorer:14s} " + "  ".join(f"{value:5.3f}" for value in row))

    diagnostics = out["diagnostics"]
    mean = lambda key: float(np.mean([d[key] for d in diagnostics]))
    consecutive = float(np.nanmean([o[0] for o in out["overlaps"]]))
    from_first = float(np.nanmean([o[1] for o in out["overlaps"]]))
    print(f"\nmass outside the cache (query + generated tail): "
          f"{mean('tail_mass'):.3f}")
    print(f"share of cached mass on the preamble block:      "
          f"{mean('preamble_share'):.3f}")
    print(f"share of cached mass on supporting documents:    "
          f"{mean('supporting_share'):.3f}")
    print(f"share of cached mass on block-head tokens:       "
          f"{mean('block_head_share'):.3f}")
    print(f"mean descriptor box width (kmax - kmin):         "
          f"{out['box_width']:.3f}")
    print(f"top-25% page set overlap, consecutive steps:     {consecutive:.3f}")
    print(f"top-25% page set overlap, against step 0:        {from_first:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--sink-exclude", type=int, default=2,
                        help="tokens dropped from the *nosink* box statistics, "
                             "and the width of the block-head diagnostic")
    parser.add_argument("--arms", default="block,request_local",
                        help="block = the reusable cache under study; "
                             "request_local = the contextualised control")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = load_2wiki(limit=args.examples)
    probe = BlockProbe(args.model)

    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"\nrunning arm: {arm}")
        results[arm] = study(probe, examples, args,
                             request_local=(arm == "request_local"))
    for arm, out in results.items():
        report(arm, out)

    if "block" in results and "request_local" in results:
        print("\n================ adverse hypothesis ================")
        print("If independent prefill flattened key discriminability, the "
              "block arm would show a looser box and a wider oracle-to-Quest "
              "gap than the contextualised control.")
        focus = BUDGETS.index(0.25)
        print(f"{'arm':14s} {'box width':>10s} {'oracle':>8s} {'quest':>8s} "
              f"{'gap':>7s}   (page-level, scope=documents, "
              f"{BUDGETS[focus]:.0%} budget)")
        for arm, out in results.items():
            stack = lambda key: torch.stack(out["collected"][key]).mean(
                dim=(0, 1, 2)).numpy()[focus]
            oracle = stack(("oracle", "page", "documents"))
            quest = stack(("quest", "page", "documents"))
            print(f"{arm:14s} {out['box_width']:10.3f} {oracle:8.3f} "
                  f"{quest:8.3f} {oracle - quest:7.3f}")

    if args.out:
        payload = {"budgets": np.array(BUDGETS)}
        for arm, out in results.items():
            payload[f"{arm}|diagnostics"] = json.dumps(out["diagnostics"])
            for key, values in out["collected"].items():
                name = "|".join((arm, ) + key)
                payload[name] = torch.stack(values).numpy().astype(np.float16)
        np.savez_compressed(args.out, **payload)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
