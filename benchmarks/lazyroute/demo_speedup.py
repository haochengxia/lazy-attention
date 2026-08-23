"""A three-way token-stream race: dense vs Lazy-Attn vs LazyRoute, as a GIF.

The two savings this project claims land in different halves of a request, and
a single number hides that:

- **Lazy-Attn vs dense** is a *prefill* win. Documents are prefilled once and
  their KV is reused from whatever slot they land in, so a corpus retrieved in
  a new order costs nothing to re-read. Stock prefix caching matches only on a
  shared token prefix, so it recomputes from the first block that moved. This
  shows up as time-to-first-token.
- **LazyRoute vs Lazy-Attn** is a *decode* win. Every decode step reads a
  query-selected subset of the cached pages instead of all of them. This shows
  up as the inter-token interval -- the panel does not start sooner, it types
  faster.

So the demo has to show three panels and let the eye read both effects: two
panels start together and beat the third off the line, and one of those two
then pulls ahead while typing.

**Method.** Each arm is measured alone, in its own process, with the whole GPU
-- `demo_race.py`'s convention, and the only honest one on a single card, since
co-resident engines contend and the contention would be indistinguishable from
the effect. Timestamps come from driving `LLMEngine.step()` directly with
`RequestOutputKind.DELTA`, so every point on the timeline is a token that
actually arrived at that moment; nothing is interpolated from a mean rate.
Every arm is warmed on the natural document order first, then measured on a
reordered corpus: that is what makes the dense arm's cache miss real rather
than stipulated, and the lazy arms are handed the identical reordering.

    python benchmarks/lazyroute/demo_speedup.py --docs 50 --out analysis/lazyroute_demo.gif

`--arm` measures one side and writes its timeline; `--render` rebuilds the GIF
from three saved timelines without touching the GPU, which is what you want
while iterating on the drawing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

DENSE, LAZY, ROUTE = "dense", "lazy", "route"
ARMS = (DENSE, LAZY, ROUTE)

LABELS = {
    DENSE: "vLLM prefix caching",
    LAZY: "Lazy-Attn",
    ROUTE: "LazyRoute",
}
BLURBS = {
    DENSE: "re-prefills from the first moved doc",
    LAZY: "reuses every doc's KV, reads all of it",
    ROUTE: "reuses every doc's KV, reads ~1/3 of it",
}

MODEL = "hxia7/Llama-3.2-1B-block-FT"


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _configure_env(arm: str, route_layers: str = "4") -> None:
    """Set the engine switches this arm needs, before vLLM is imported.

    `VLLM_ENABLE_V1_MULTIPROCESSING=0` on the lazy arms is not a performance
    choice: it puts the worker in this process, which is the only way
    `get_sparse_router_stats()` can see the router that actually ran and report
    what fraction of the cache was read.
    """
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
    if arm == DENSE:
        os.environ.pop("VLLM_USE_LAZY_ATTENTION", None)
        os.environ.pop("LAZY_SPARSE", None)
        return
    os.environ["VLLM_USE_LAZY_ATTENTION"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
    if arm == ROUTE:
        os.environ["LAZY_SPARSE"] = "1"
        # Routing every layer is a net loss: the router is launch-bound, and a
        # sparse decode layer leaves only ~1 ms of GPU work to hide ~100 kernel
        # launches behind. Sharing one decision across a few layers is what
        # makes the saving visible, and at stride 4 it costs nothing measurable
        # in answer quality (PROJECT.md 9b).
        os.environ["LAZY_SPARSE_ROUTE_LAYERS"] = route_layers
    else:
        os.environ.pop("LAZY_SPARSE", None)


def _reorder(count: int, seed: int) -> list[int]:
    """A deterministic non-identity permutation, shared by all three arms.

    Recomputed rather than passed between processes, so the arms cannot drift
    apart when they are run separately -- which is the normal case here, since
    each one owns the GPU in turn.
    """
    natural = list(range(count))
    order = natural[:]
    rng = random.Random(seed + 7)
    while order == natural and count > 1:
        rng.shuffle(order)
    return order


def _build_llm(arm: str, args):
    from vllm import LLM
    kwargs = dict(model=args.model,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=True,
                  trust_remote_code=True,
                  enforce_eager=True,
                  max_num_seqs=1)
    if arm == DENSE:
        return LLM(**kwargs)
    from lazy.entrypoints.llm import LazyLLM
    return LazyLLM(**kwargs)


def _stream(llm, arm: str, blocks: list[str], tail: str,
            max_tokens: int, min_tokens: int) -> dict:
    """Run one request, timestamping each token as the engine emits it.

    Bypasses `LLM.generate`, which forces `RequestOutputKind.FINAL_ONLY` and so
    would collapse the whole timeline onto a single point at the end -- exactly
    the structure this figure exists to show.

    Events are keyed on emitted *token ids*, not on the detokenised delta: a
    step whose token renders as the empty string (EOS, and any special token)
    still cost a full decode, and dropping it both loses time from the timeline
    and -- for an answer short enough to be all-EOS -- leaves the request with
    no first token at all.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            min_tokens=min_tokens)
    params.output_kind = RequestOutputKind.DELTA

    if arm == DENSE:
        llm._add_request("".join(blocks) + tail, params)
    else:
        llm._add_request(tail, params, document_seq=blocks)

    start = time.perf_counter()
    events: list[list] = []
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            completion = output.outputs[0]
            if completion.token_ids:
                events.append([(time.perf_counter() - start) * 1e3,
                               completion.text])
    return {"events": events,
            "end_ms": (time.perf_counter() - start) * 1e3,
            "ttft_ms": events[0][0] if events else None}


def _warm(llm, arm: str, blocks: list[str], tail: str) -> None:
    """Populate the cache in natural order -- the state a served system is in.

    Without this the dense arm would be measured on a cold cache, which is a
    comparison Lazy-Attn wins for a reason that has nothing to do with it.
    """
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=1)
    if arm == DENSE:
        llm._add_request("".join(blocks) + tail, params)
    else:
        llm._add_request(tail, params, document_seq=blocks)
    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()


def measure(arm: str, args) -> dict:
    sys.path.insert(0, os.path.join(_REPO, "lazy_attn"))
    sys.path.insert(0, os.path.dirname(_HERE))
    sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
    _configure_env(arm, args.route_layers)

    if arm != DENSE:
        import lazy.__vllm__  # noqa: F401  (installs the patches)

    # The 1B Block-FT tokenizer predates `all_special_tokens_extended`; vLLM
    # 0.9.2 reads it unconditionally.
    import vllm.transformers_utils.tokenizer as vtok
    original = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or original(t)) if not hasattr(t, "all_special_tokens_extended") \
        else original(t)

    from lazyroute.corpus import load_2wiki, widen

    pool = load_2wiki(limit=max(args.examples * 4, args.docs * 2, 40))
    examples = [
        widen(ex, args.docs, pool) if args.docs > len(ex.documents) else ex
        for ex in pool[:args.examples]
    ]

    llm = _build_llm(arm, args)

    # Every arm decodes the same number of steps. Left free, each arm stops at
    # its own EOS -- and since the arms answer differently, the mean ITL would
    # then be taken over different sequence lengths, at which point it is not a
    # rate comparison at all. `-1` means "as many steps as `--max-tokens`".
    min_tokens = args.max_tokens if args.min_tokens < 0 else args.min_tokens

    timelines = []
    for index, example in enumerate(examples):
        blocks, tail = example.blocks(), example.prompt()
        _warm(llm, arm, blocks, tail)
        # Same documents, new order. Lazy reuses every block wherever it lands;
        # prefix caching matches only up to the first block that moved.
        # Reordering is what makes the dense arm's cache miss real. It is
        # optional because it is also a second, independent difficulty knob:
        # `widen()` keeps the answer documents at the front on purpose, and
        # shuffling moves them into the distractor mass.
        if args.reorder:
            order = _reorder(len(blocks) - 1, args.seed + index)
            shuffled = [blocks[0]] + [blocks[1 + j] for j in order]
        else:
            shuffled = blocks
        timeline = _stream(llm, arm, shuffled, tail,
                           args.max_tokens, min_tokens)
        timeline["question"] = example.question
        timeline["answer"] = example.answer
        timelines.append(timeline)
        ttft = timeline["ttft_ms"]
        print(f"[{arm}] example {index}: "
              f"TTFT {'n/a' if ttft is None else format(ttft, '.0f')} ms, "
              f"{len(timeline['events'])} tokens, "
              f"end {timeline['end_ms']:.0f} ms", flush=True)

    data = {"arm": arm, "label": LABELS[arm], "timelines": timelines,
            "docs": args.docs, "model": args.model,
            "max_tokens": args.max_tokens}
    if arm == ROUTE:
        from lazy.attention.backends.triton_attn import get_sparse_router_stats
        stats = get_sparse_router_stats()
        data["router"] = {k: float(v) for k, v in stats.items()}
        if not stats:
            print("[route] WARNING: the router never ran -- this arm is "
                  "measuring dense decode and the GIF would be a lie.",
                  flush=True)
        else:
            print(f"[route] kept_fraction {stats.get('kept_fraction', 0):.3f}",
                  flush=True)
    return data


# --------------------------------------------------------------------------
# summary statistics
# --------------------------------------------------------------------------

def summarise(data: dict) -> dict:
    """Median TTFT and mean inter-token interval across the measured requests.

    Median for TTFT because the first request of a run carries warmup the
    others do not; mean for ITL because it is a rate over many tokens and the
    per-token spread is what a user experiences as smoothness.
    """
    ttfts, itls = [], []
    for timeline in data["timelines"]:
        events = timeline["events"]
        if not events:
            continue
        ttfts.append(events[0][0])
        if len(events) > 1:
            itls.append((events[-1][0] - events[0][0]) / (len(events) - 1))
    ttfts.sort()
    return {
        "ttft_ms": ttfts[len(ttfts) // 2] if ttfts else float("nan"),
        "itl_ms": sum(itls) / len(itls) if itls else float("nan"),
        "tokens": (sum(len(t["events"]) for t in data["timelines"])
                   / max(len(data["timelines"]), 1)),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _fonts() -> tuple[str, str]:
    import glob
    pair = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    if os.path.exists(pair[0]):
        return pair[0], pair[1] if os.path.exists(pair[1]) else pair[0]
    hits = glob.glob("/usr/**/DejaVuSansMono.ttf", recursive=True)
    if hits:
        return hits[0], hits[0]
    raise SystemExit("no DejaVuSansMono.ttf found; apt-get install fonts-dejavu")


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        current = ""
        for word in para.split(" "):
            while len(word) > width:
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:width])
                word = word[width:]
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _text_at(events: list[list], t_ms: float) -> str:
    return "".join(chunk for (at, chunk) in events if at <= t_ms)


def render_gif(arms: dict, out_path: str, index: int, frames: int = 130,
               speed: float = 3.0) -> None:
    from PIL import Image, ImageDraw, ImageFont

    regular, bold = _fonts()
    f_title = ImageFont.truetype(bold, 24)
    f_sub = ImageFont.truetype(regular, 16)
    f_head = ImageFont.truetype(bold, 19)
    f_note = ImageFont.truetype(regular, 14)
    f_badge = ImageFont.truetype(bold, 16)
    f_body = ImageFont.truetype(regular, 15)
    f_clock = ImageFont.truetype(bold, 18)

    BG, PANEL, BORDER = (13, 17, 23), (22, 27, 34), (48, 54, 61)
    TEXT, DIM, WHITE = (201, 209, 217), (110, 118, 129), (240, 246, 252)
    GREY, GREEN, CYAN = (139, 148, 158), (63, 185, 80), (56, 189, 248)
    ACCENTS = {DENSE: GREY, LAZY: GREEN, ROUTE: CYAN}

    W, H, M = 1500, 620, 22
    panel_w = (W - 4 * M) // 3
    p_top, p_h = 118, 452
    pad, body_y = 16, 104
    line_h = 21
    body_top = p_top + body_y
    max_chars = int((panel_w - 2 * pad) / f_body.getlength("M"))
    max_lines = (p_h - body_y - 14) // line_h

    sides = [(arm, arms[arm], arms[arm]["timelines"][index]) for arm in ARMS]
    t_end = max(side["end_ms"] for _, _, side in sides)
    span = t_end + max(300.0, 0.08 * t_end)

    # Non-linear playback. The TTFT gap is decided in the first few hundred
    # milliseconds while decode runs for seconds, so linear time would spend
    # nine frames in ten on the part that is already settled. Most frames go to
    # the start-line race; the rest fast-forward through decode, where the
    # thing to see is a rate difference and a rate reads fine at speed.
    dense_ttft = arms[DENSE]["timelines"][index]["ttft_ms"] or t_end
    t_split = min(span, dense_ttft * 1.3 + 120.0)
    n_race = max(1, int(frames * 0.45))
    n_decode = max(1, frames - n_race)
    times = ([i / n_race * t_split for i in range(n_race)]
             + [t_split + (i + 1) / n_decode * (span - t_split)
                for i in range(n_decode)])
    hold = max(45, int(round(70 / max(speed, 1e-6) * 3)))

    stats = {arm: summarise(arms[arm]) for arm in ARMS}
    question = arms[LAZY]["timelines"][index]["question"]
    docs = arms[LAZY]["docs"]
    kept = arms[ROUTE].get("router", {}).get("kept_fraction")

    def draw(t_ms: float, frame_index: int, final: bool):
        image = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(image)
        d.text((M, 16), "LazyRoute: same answer, less of the cache read",
               font=f_title, fill=WHITE)
        subtitle = (f"{docs} cached documents, retrieved in a new order  ·  "
                    f"each engine measured alone on one GPU  ·  "
                    f"playback {speed:g}x slower than real time")
        d.text((M, 48), subtitle, font=f_sub, fill=DIM)
        d.text((M, 72), f"Q: {question[:118]}", font=f_sub, fill=(150, 160, 172))

        for slot, (arm, data, side) in enumerate(sides):
            px = M + slot * (panel_w + M)
            accent = ACCENTS[arm]
            d.rounded_rectangle([px, p_top, px + panel_w, p_top + p_h],
                                radius=10, fill=PANEL, outline=BORDER, width=1)
            d.rectangle([px, p_top, px + panel_w, p_top + 4], fill=accent)
            d.text((px + pad, p_top + 14), LABELS[arm], font=f_head, fill=accent)
            d.text((px + pad, p_top + 39), BLURBS[arm], font=f_note, fill=DIM)

            ttft = side["ttft_ms"]
            started = ttft is not None and t_ms >= ttft
            if started:
                emitted = sum(1 for at, _ in side["events"] if at <= t_ms)
                badge = f"first token @ {ttft:.0f} ms   ·   {emitted} tok"
                colour = accent
            else:
                badge = "prefilling …"
                colour = DIM
            d.text((px + pad, p_top + 62), badge, font=f_badge, fill=colour)

            lines = _wrap(_text_at(side["events"], t_ms), max_chars)[-max_lines:]
            for i, line in enumerate(lines):
                d.text((px + pad, body_top + i * line_h), line,
                       font=f_body, fill=TEXT)
            finished = t_ms >= side["end_ms"]
            if started and not finished and (frame_index // 3) % 2 == 0:
                last = lines[-1] if lines else ""
                cx = px + pad + f_body.getlength(last)
                cy = body_top + (max(len(lines), 1) - 1) * line_h
                d.rectangle([cx + 1, cy + 2, cx + 9, cy + 17], fill=accent)

        d.text((M, H - 34), f"t = {min(t_ms, t_end):6.0f} ms",
               font=f_clock, fill=WHITE)
        if final:
            ttft_gain = stats[DENSE]["ttft_ms"] / max(stats[LAZY]["ttft_ms"], 1e-9)
            itl_gain = stats[LAZY]["itl_ms"] / max(stats[ROUTE]["itl_ms"], 1e-9)
            kept_note = f" reading {kept:.0%} of the cache" if kept else ""
            message = (f"first token {ttft_gain:.1f}x sooner   ·   "
                       f"then {itl_gain:.2f}x faster per token{kept_note}")
            width = f_clock.getlength(message)
            d.text((W - M - width, H - 34), message, font=f_clock, fill=CYAN)
        return image

    images = [draw(t, i, final=False) for i, t in enumerate(times)]
    durations = [hold] * len(images)
    images.append(draw(span, len(times), final=True))
    durations.append(3000)

    palette = images[-1].convert("P", palette=Image.ADAPTIVE, colors=128)
    images = [im.quantize(palette=palette, dither=Image.NONE) for im in images]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    print(f"wrote {out_path}  ({len(images)} frames, "
          f"{sum(durations) / 1000:.1f}s)")


def report(arms: dict) -> None:
    stats = {arm: summarise(arms[arm]) for arm in ARMS}
    print(f"\n{'arm':<22}{'TTFT (ms)':>12}{'ITL (ms/tok)':>14}{'tok':>7}")
    for arm in ARMS:
        s = stats[arm]
        print(f"{LABELS[arm]:<22}{s['ttft_ms']:>12.1f}{s['itl_ms']:>14.2f}"
              f"{s['tokens']:>7.0f}")
    print(f"\nprefill: {LABELS[LAZY]} reaches the first token "
          f"{stats[DENSE]['ttft_ms'] / stats[LAZY]['ttft_ms']:.2f}x sooner "
          f"than {LABELS[DENSE]}")
    print(f"decode : {LABELS[ROUTE]} emits tokens "
          f"{stats[LAZY]['itl_ms'] / stats[ROUTE]['itl_ms']:.2f}x faster "
          f"than {LABELS[LAZY]}")
    kept = arms[ROUTE].get("router", {}).get("kept_fraction")
    if kept:
        print(f"         reading {kept:.1%} of the cached rows")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--docs", type=int, default=50,
                        help="cached documents per request; the decode saving "
                             "is proportional to the cache, so a small corpus "
                             "shows the router's overhead and not its point")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=-1,
                        help="-1 (default) pins it to --max-tokens so every "
                             "arm decodes the same number of steps")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--no-reorder", dest="reorder", action="store_false",
                        help="serve the documents in their cached order; the "
                             "dense arm then hits its prefix cache")
    parser.add_argument("--route-layers", default="4",
                        help="LAZY_SPARSE_ROUTE_LAYERS for the route arm: "
                             "'all', 'first', or a stride")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS, default="",
                        help="measure one arm and write its timeline")
    parser.add_argument("--render", action="store_true",
                        help="render from saved timelines, no GPU needed")
    parser.add_argument("--gif-example", type=int, default=0)
    parser.add_argument("--frames", type=int, default=130)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--out", default="analysis/lazyroute_demo.gif")
    parser.add_argument("--json-dir", default="analysis")
    args = parser.parse_args()

    paths = {arm: os.path.join(args.json_dir, f"lazyroute_demo_{arm}.json")
             for arm in ARMS}

    if args.arm:
        data = measure(args.arm, args)
        os.makedirs(args.json_dir, exist_ok=True)
        with open(paths[args.arm], "w") as handle:
            json.dump(data, handle)
        print(f"wrote {paths[args.arm]}")
        return 0

    if not args.render:
        # Each arm needs its engine switches set before vLLM is imported, and
        # only one 1B engine fits the card at a time, so the arms are separate
        # processes run in turn rather than threads.
        for arm in ARMS:
            print(f"\n=== measuring {LABELS[arm]} ===", flush=True)
            command = [sys.executable, os.path.abspath(__file__),
                       "--arm", arm, "--model", args.model,
                       "--docs", str(args.docs),
                       "--examples", str(args.examples),
                       "--max-tokens", str(args.max_tokens),
                       "--min-tokens", str(args.min_tokens),
                       "--gpu-memory-utilization",
                       str(args.gpu_memory_utilization),
                       "--seed", str(args.seed), "--json-dir", args.json_dir,
                       "--route-layers", args.route_layers]
        if not args.reorder:
            command.append("--no-reorder")
            result = subprocess.run(command, cwd=_REPO)
            if result.returncode != 0:
                raise SystemExit(f"arm {arm} failed ({result.returncode})")

    arms = {}
    for arm in ARMS:
        with open(paths[arm]) as handle:
            arms[arm] = json.load(handle)
    report(arms)
    render_gif(arms, args.out, args.gif_example,
               frames=args.frames, speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
