"""Does sparse decode change the answer? Free-running QA, paired against dense.

D0.4 sparsified one step against a dense prefix, which isolates injected error
but says nothing about drift. This generates the whole answer with the router
live at every step, which is the thing a system would actually do.

The comparison is **paired**: the same example is decoded dense and sparse, and
what is reported is how often the generation *changes* and how the win/loss
counts fall (McNemar). Comparing two accuracy rates would be hopeless here --
the 1B checkpoint scores ~24% on 2wiki at ten documents, so a rate difference
of a few points is inside the noise of any sample we can afford. "Sparse
produced a byte-identical answer on 88 of 100 examples" is a far stronger
statement than "accuracy fell by 2 points, +/- 6".

Selection is recomputed every step from that layer's own query, in the same
forward that consumes it: the q_proj hook fires, scores the pages, and leaves a
mask for the attention of that layer. That is a per-layer router -- stronger
than the planned global one, and the honest upper bound for a scorer study.

    python benchmarks/lazyroute/d0_e2e.py --examples 60
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from benchmarks.lazyroute.block_health import (contains_gold,  # noqa: E402
                                               degenerate)
from benchmarks.lazyroute.corpus import load_2wiki  # noqa: E402
from benchmarks.lazyroute.d0_kl import selection_mask  # noqa: E402
from benchmarks.lazyroute.d0_recall import build_descriptors  # noqa: E402
from benchmarks.lazyroute.probe import BlockProbe  # noqa: E402

MODEL = "hxia7/Llama-3.2-1B-Block-FT"
_ORIGINAL_ATTENTION = modeling_llama.eager_attention_forward
_ROUTER: dict = {"on": False, "masks": {}}


def patched_attention(module, query, key, value, attention_mask, scaling,
                      dropout=0.0, **kwargs):
    keys = modeling_llama.repeat_kv(key, module.num_key_value_groups)
    values = modeling_llama.repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, :keys.shape[-2]]
    probs = F.softmax(weights, dim=-1, dtype=torch.float32)

    keep = _ROUTER["masks"].get(module.layer_idx) if _ROUTER["on"] else None
    if keep is not None and probs.shape[-2] == 1:  # decode steps only
        probs = probs.clone()
        probs[..., :keep.shape[-1]] *= keep[None, :, None, :]
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)

    output = torch.matmul(probs.to(query.dtype), values)
    return output.transpose(1, 2).contiguous(), probs.to(query.dtype)


def install_router(probe, layout, descriptors, config, state):
    """Hook every q_proj so a layer routes on the query it is about to use."""
    handles = []
    for layer_idx, layer in enumerate(probe.model.model.layers):

        def hook(_module, _inputs, output, layer_idx=layer_idx):
            if not _ROUTER["on"] or output.shape[1] != 1:
                return
            query = output[0, -1].view(probe.num_q_heads, probe.head_dim)
            offsets = state["position"] - layout.frame_offset
            folded = query[:, None, :].expand(-1, layout.num_docs, -1)
            rotated = probe._rotate(folded.contiguous(), offsets).float()
            per_page = rotated[:, layout.page_doc]  # [q_heads, pages, dim]
            if config["scorer"] == "random":
                scores = torch.rand(probe.num_kv_heads, layout.num_pages,
                                    device=probe.device,
                                    generator=state["generator"])
            else:
                kmin = descriptors.kmin[layer_idx].repeat_interleave(
                    probe.group, dim=0)
                kmax = descriptors.kmax[layer_idx].repeat_interleave(
                    probe.group, dim=0)
                bound = torch.maximum(per_page * kmin, per_page * kmax).sum(-1)
                scores = bound.reshape(probe.num_kv_heads, probe.group,
                                       layout.num_pages).amax(dim=1)
            _ROUTER["masks"][layer_idx] = selection_mask(
                scores[None], layout, config["budget"], config["stripe"],
                probe.group)[0]

        handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
    return handles


@torch.no_grad()
def generate(probe, example, config, max_tokens, generator) -> str:
    documents = [probe.encode_document(block) for block in example.blocks()]
    from benchmarks.lazyroute.probe import build_layout
    layout = build_layout(documents, probe.device)
    cache = probe.merge(documents, layout)
    descriptors = None if config["scorer"] == "dense" else build_descriptors(
        type("R", (), {"documents": documents, "layout": layout})(),
        probe.num_layers, 0)

    prompt_ids = probe.tokenizer(example.prompt(), add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(probe.device)
    state = {"position": layout.num_tokens, "generator": generator}
    handles = []
    if descriptors is not None:
        handles = install_router(probe, layout, descriptors, config, state)

    _ROUTER["on"] = False
    positions = torch.arange(layout.num_tokens,
                             layout.num_tokens + prompt_ids.shape[1] - 1,
                             device=probe.device)[None]
    if prompt_ids.shape[1] > 1:
        probe.model(input_ids=prompt_ids[:, :-1], past_key_values=cache,
                    position_ids=positions, use_cache=True)
    position = layout.num_tokens + prompt_ids.shape[1] - 1
    token = prompt_ids[0, -1]

    _ROUTER["on"] = descriptors is not None
    generated = []
    try:
        for _ in range(max_tokens):
            state["position"] = position
            out = probe.model(input_ids=token.view(1, 1),
                              past_key_values=cache,
                              position_ids=torch.tensor([[position]],
                                                        device=probe.device),
                              use_cache=True)
            token = out.logits[0, -1].argmax()
            if int(token) in (probe.tokenizer.eos_token_id, ):
                break
            generated.append(int(token))
            position += 1
    finally:
        _ROUTER["on"] = False
        _ROUTER["masks"] = {}
        for handle in handles:
            handle.remove()
    return probe.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--examples", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = {
        "dense": {"scorer": "dense", "budget": 1.0, "stripe": "off"},
        "quest 50% +stripe": {"scorer": "quest", "budget": 0.50,
                              "stripe": "selected"},
        "quest 25% +stripe": {"scorer": "quest", "budget": 0.25,
                              "stripe": "selected"},
        "quest 25% no stripe": {"scorer": "quest", "budget": 0.25,
                                "stripe": "off"},
        "random 25% +stripe": {"scorer": "random", "budget": 0.25,
                               "stripe": "selected"},
    }

    modeling_llama.eager_attention_forward = patched_attention
    probe = BlockProbe(args.model)
    examples = load_2wiki(limit=args.examples)
    answers: dict[str, list[str]] = {name: [] for name in configs}

    for index, example in enumerate(examples):
        for name, config in configs.items():
            generator = torch.Generator(
                device=probe.device).manual_seed(args.seed + index)
            answers[name].append(
                generate(probe, example, config, args.max_tokens, generator))
        if (index + 1) % 10 == 0:
            print(f"  [{index + 1}/{len(examples)}]")
    modeling_llama.eager_attention_forward = _ORIGINAL_ATTENTION

    gold = [example.answer for example in examples]
    dense = answers["dense"]
    dense_hit = np.array([contains_gold(a, g) for a, g in zip(dense, gold)])

    print(f"\n2wiki dev, {len(examples)} examples, free-running greedy decode, "
          f"paired against dense.")
    print(f"{'config':22s} {'gold':>6s} {'degen':>6s} {'same':>6s} "
          f"{'D+S-':>5s} {'D-S+':>5s}")
    for name in configs:
        hit = np.array([contains_gold(a, g) for a, g in zip(answers[name], gold)])
        same = np.mean([a == b for a, b in zip(answers[name], dense)])
        bad = np.mean([degenerate(a) for a in answers[name]])
        lost = int(np.sum(dense_hit & ~hit))
        won = int(np.sum(~dense_hit & hit))
        print(f"{name:22s} {hit.mean():6.1%} {bad:6.1%} {same:6.1%} "
              f"{lost:5d} {won:5d}")
    print("\nsame = byte-identical generation to dense; D+S- = dense correct "
          "and sparse wrong; D-S+ = the reverse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
