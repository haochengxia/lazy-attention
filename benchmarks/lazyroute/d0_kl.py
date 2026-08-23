"""D0.4: what does dropping blocks do to the next token, and is it the softmax?

Recall says how much attention *mass* a selection keeps. It cannot say whether
keeping it matters, and it cannot separate two different failures:

  lost information   the dropped keys' values were contributing to the output
  broken denominator the dropped mass is re-spread over what was kept, scaling
                     every retained weight up by 1/retained -- an error even if
                     the dropped values carried nothing

They call for different remedies, and the stripe only addresses the first. So
each configuration is run twice:

  renorm=on   ordinary sparse attention: softmax over the retained keys
  renorm=off  dense-normalised weights, dropped keys contributing the zero
              vector -- identical to attaching a sink that absorbs the dropped
              mass and holds v=0, and it needs no KV at all

If renorm=off is much closer to dense, the damage is renormalisation and the
cheap fix is a denominator correction rather than more bytes.

The intervention is one step deep: the prefix is teacher-forced and dense, and
only the step under test is sparsified, so this measures the error a policy
injects rather than the drift it accumulates (that is G4-A's question).

    python benchmarks/lazyroute/d0_kl.py --examples 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from benchmarks.lazyroute.corpus import load_2wiki  # noqa: E402
from benchmarks.lazyroute.d0_recall import (build_descriptors,  # noqa: E402
                                            page_scores, true_page_mass)
from benchmarks.lazyroute.probe import BlockProbe  # noqa: E402

MODEL = "hxia7/Llama-3.2-1B-Block-FT"
BUDGETS = (0.10, 0.25, 0.50)
STRIPES = ("off", "selected", "all")

# The patched attention reads this; `mask` is None for a dense pass.
_STATE: dict = {"mask": None, "renorm": True, "retained": {}}
_ORIGINAL_ATTENTION = modeling_llama.eager_attention_forward


def patched_attention(module, query, key, value, attention_mask, scaling,
                      dropout=0.0, **kwargs):
    """Eager attention with a per-(layer, query head) mask over the cache region.

    Masking the *probabilities* rather than the logits is deliberate: it makes
    the renormalised and un-renormalised variants differ by exactly one
    division, so the comparison isolates the denominator and nothing else.
    """
    key_states = modeling_llama.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_llama.repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, :key_states.shape[-2]]
    probs = F.softmax(weights, dim=-1, dtype=torch.float32)

    keep = None if _STATE["mask"] is None else _STATE["mask"].get(
        module.layer_idx)
    if keep is not None:
        cache_len = keep.shape[-1]
        probs = probs.clone()
        probs[..., :cache_len] *= keep[None, :, None, :]
        retained = probs.sum(-1)
        _STATE["retained"][module.layer_idx] = retained.detach()
        if _STATE["renorm"]:
            probs = probs / retained.unsqueeze(-1).clamp_min(1e-9)

    output = torch.matmul(probs.to(query.dtype), value_states)
    return output.transpose(1, 2).contiguous(), probs.to(query.dtype)


def fill(scores: torch.Tensor, unit_tokens: torch.Tensor, budget_tokens: float,
         forced: torch.Tensor = None) -> torch.Tensor:
    """Take units in score order until the token budget runs out.

    Same accounting as the recall study: a forced unit is taken first and is
    charged for, so a stripe is never free.
    """
    if forced is not None:
        scores = scores.masked_fill(forced, float("inf"))
    order = scores.argsort(dim=-1, descending=True)
    cost = unit_tokens.expand_as(order).gather(-1, order).cumsum(-1)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(-1, order, cost <= budget_tokens)
    return selected


def selection_mask(scores: torch.Tensor, layout, budget: float, stripe: str,
                   group: int) -> torch.Tensor:
    """Page selection under a token budget -> per-query-head token mask."""
    page_len = layout.page_len.float()
    budget_tokens = max(budget * float(page_len.sum()), 1.0)
    first_page = torch.zeros(layout.num_pages, dtype=torch.bool,
                             device=scores.device)
    first_page[torch.searchsorted(
        layout.page_doc,
        torch.arange(layout.num_docs, device=scores.device))] = True

    if stripe == "all":
        selected = fill(scores, page_len, budget_tokens, first_page)
    elif stripe == "selected":
        # "If a block is read at all, read its head": a first pass says which
        # documents are worth touching, then their head pages are forced and
        # the budget is re-spent around them.
        touched = torch.zeros(scores.shape[:-1] + (layout.num_docs, ),
                              device=scores.device)
        touched.index_add_(-1, layout.page_doc,
                           fill(scores, page_len, budget_tokens).float())
        selected = fill(scores, page_len, budget_tokens,
                        first_page & (touched > 0)[..., layout.page_doc])
    else:
        selected = fill(scores, page_len, budget_tokens)

    token_keep = selected[..., layout.page_of_token].float()
    return token_keep.repeat_interleave(group, dim=1)  # kv heads -> query heads


@torch.no_grad()
def logits_for(probe, cache, token, position, mask, renorm) -> torch.Tensor:
    """One decode step under a policy, leaving the cache as it was found."""
    before = cache.get_seq_length()
    _STATE["mask"] = mask
    _STATE["renorm"] = renorm
    _STATE["retained"] = {}
    try:
        out = probe.model(input_ids=token.view(1, 1),
                          past_key_values=cache,
                          position_ids=torch.tensor([[position]],
                                                    device=probe.device),
                          use_cache=True)
    finally:
        _STATE["mask"] = None
        cache.crop(before)
    return out.logits[0, -1].float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    modeling_llama.eager_attention_forward = patched_attention
    probe = BlockProbe(args.model)
    generator = torch.Generator(device=probe.device).manual_seed(args.seed)
    examples = load_2wiki(limit=args.examples)

    records: dict[tuple, list[tuple[float, float, float]]] = {}
    for index, example in enumerate(examples):
        result = probe.run(example.blocks(), example.prompt(), example.answer,
                           max_steps=args.max_steps)
        layout = result.layout
        descriptors = {
            "quest": build_descriptors(result, probe.num_layers, 0),
            "centroid": build_descriptors(result, probe.num_layers, 0),
        }

        # Replay the trace, re-running each step under every policy. The cache
        # is rebuilt dense so that only the step under test is ever sparse.
        cache = probe.merge(result.documents, layout)
        prompt_ids = probe.tokenizer(example.prompt(), add_special_tokens=False,
                                     return_tensors="pt").input_ids.to(probe.device)
        if prompt_ids.shape[1] > 1:
            positions = torch.arange(layout.num_tokens,
                                     layout.num_tokens + prompt_ids.shape[1] - 1,
                                     device=probe.device)[None]
            probe.model(input_ids=prompt_ids[:, :-1], past_key_values=cache,
                        position_ids=positions, use_cache=True)

        for trace in result.traces:
            token = torch.tensor(trace.token_id, device=probe.device)
            dense = logits_for(probe, cache, token, trace.position, None, True)
            dense_logp = F.log_softmax(dense, dim=-1)
            dense_top1 = int(dense.argmax())

            mass = true_page_mass(result, trace, probe)
            scores = page_scores(probe, result, trace, descriptors, generator)
            scores["oracle"] = mass

            for scorer in ("oracle", "quest"):
                for budget in BUDGETS:
                    for stripe in STRIPES:
                        mask = selection_mask(scores[scorer], layout, budget,
                                              stripe, probe.group)
                        by_layer = {
                            layer: mask[layer]
                            for layer in range(probe.num_layers)
                        }
                        for renorm in (True, False):
                            sparse = logits_for(probe, cache, token,
                                                trace.position, by_layer,
                                                renorm)
                            kl = float(
                                F.kl_div(F.log_softmax(sparse, dim=-1),
                                         dense_logp,
                                         log_target=True,
                                         reduction="sum"))
                            retained = float(
                                torch.stack(list(
                                    _STATE["retained"].values())).mean())
                            records.setdefault(
                                (scorer, budget, stripe, renorm), []).append(
                                    (kl, float(int(sparse.argmax()) ==
                                               dense_top1), retained))
            # Advance the cache with the dense step.
            probe.model(input_ids=token.view(1, 1), past_key_values=cache,
                        position_ids=torch.tensor([[trace.position]],
                                                  device=probe.device),
                        use_cache=True)
        print(f"  [{index + 1}/{len(examples)}] {layout.num_pages} pages, "
              f"{len(result.traces)} steps")

    modeling_llama.eager_attention_forward = _ORIGINAL_ATTENTION

    print("\nKL(dense || sparse) of the next-token distribution, nats. "
          "One step sparsified, dense prefix.")
    for scorer in ("oracle", "quest"):
        print(f"\n--- scorer: {scorer} ---")
        print(f"{'stripe / renorm':22s} " +
              "  ".join(f"{b:>16.0%}" for b in BUDGETS))
        print(f"{'':22s} " +
              "  ".join(f"{'KL':>6s} {'top1':>4s} {'mass':>4s}"
                        for _ in BUDGETS))
        for stripe in STRIPES:
            for renorm in (True, False):
                cells = []
                for budget in BUDGETS:
                    values = np.array(records[(scorer, budget, stripe, renorm)])
                    cells.append(f"{values[:, 0].mean():6.3f} "
                                 f"{values[:, 1].mean():4.2f} "
                                 f"{values[:, 2].mean():4.2f}")
                label = f"{stripe}/{'renorm' if renorm else 'no-renorm'}"
                print(f"{label:22s} " + "  ".join(cells))

    print("\nKL against retained mass, every configuration. If the points lie "
          "on one curve,\nrenormalisation adds nothing beyond what recall "
          "already measures.")
    print(f"{'config':34s} {'mass':>6s} {'KL':>7s} {'p95 KL':>8s}")
    for key in sorted(records, key=lambda k: -np.mean(
            [row[2] for row in records[k]])):
        values = np.array(records[key])
        scorer, budget, stripe, renorm = key
        label = (f"{scorer}/{budget:.0%}/{stripe}/"
                 f"{'renorm' if renorm else 'no-renorm'}")
        print(f"{label:34s} {values[:, 2].mean():6.3f} "
              f"{values[:, 0].mean():7.3f} "
              f"{np.percentile(values[:, 0], 95):8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
