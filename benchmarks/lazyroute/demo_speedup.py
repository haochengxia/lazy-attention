"""A three-way token-stream race: dense vs Lazy-Attn vs LazyRoute, as a GIF.

The two savings this project claims land in different halves of a request, and
a single number hides that:

- **Lazy-Attn vs dense** is a *prefill* win. Documents are prefilled once and
  their KV is reused from whatever slot they land in, so a corpus retrieved in
  a new order costs nothing to re-read. Stock prefix caching matches only on a
  shared token prefix, so it recomputes from the first block that moved. This
  shows up as time-to-first-token.
- **LazyRoute vs Lazy-Attn** is a *read-volume* win: every decode step touches
  a query-selected subset of the cached pages instead of all of them. Whether
  that converts into wall-clock depends on what else is on the critical path.
  At batch 1 with the split decode kernel it does not -- the attention it
  removes is smaller than the router's own launch cost -- so this panel is
  drawn as what it measurably is, a fraction of the cache read, next to its
  honest per-token rate.

So the demo shows three panels of measured quantities: two start together and
beat the third off the line, and each reports its own decode rate and how much
of the cache it reads to get it. Nothing on screen is a projection.

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
    LAZY: "reuses every doc's KV · reads all of it",
    ROUTE: "reuses every doc's KV · reads what the query needs",
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
    # The serial decode kernel gives one CTA per (sequence, KV head) -- eight
    # of them at batch 1, on a card with seventy SMs. Measuring sparsity
    # against a kernel that leaves 90% of the GPU idle would credit the router
    # for removing work the hardware was never going to spend. Split-KV is what
    # the lazy arms should be judged as, so both of them get it.
    os.environ.setdefault("LAZY_SPLIT_KV", "1")
    os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
    if arm == ROUTE:
        os.environ["LAZY_SPARSE"] = "1"
        # The router's cost is linear in how many times it is called -- ~3.4 ms
        # each, almost all of it kernel-launch dispatch rather than arithmetic
        # -- so the number of calls per step is the only knob that moves it.
        # `first` is one call, and §9b measured it as costing nothing in answer
        # quality against routing all fourteen layers.
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
                  max_num_seqs=args.batch)
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


def _stream_many(llm, arm: str, requests: list, max_tokens: int,
                 min_tokens: int) -> list[dict]:
    """Run several requests concurrently, timestamping each one's tokens.

    Batch 1 is the wrong regime to judge sparsity in: once the decode kernel is
    fast, a single-sequence step is bound by the CPU issuing launches, so
    removing GPU work is invisible. Concurrency raises the GPU work behind each
    launch without raising the launch count, which is where a saving in bytes
    read should start to show as a saving in time.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            min_tokens=min_tokens)
    params.output_kind = RequestOutputKind.DELTA
    for blocks, tail in requests:
        if arm == DENSE:
            llm._add_request("".join(blocks) + tail, params)
        else:
            llm._add_request(tail, params, document_seq=blocks)

    start = time.perf_counter()
    events: dict = {}
    end_ms: dict = {}
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            now = (time.perf_counter() - start) * 1e3
            if output.outputs[0].token_ids:
                events.setdefault(output.request_id, []).append(
                    [now, output.outputs[0].text])
                end_ms[output.request_id] = now
    return [{
        "events": events.get(rid, []),
        "end_ms": end_ms.get(rid, (time.perf_counter() - start) * 1e3),
        "ttft_ms": events[rid][0][0] if events.get(rid) else None,
    } for rid in sorted(events, key=int)]


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
    if args.batch > 1:
        prepared = []
        for index, example in enumerate(examples[:args.batch]):
            blocks, tail = example.blocks(), example.prompt()
            _warm(llm, arm, blocks, tail)
            if args.reorder:
                order = _reorder(len(blocks) - 1, args.seed + index)
                blocks = [blocks[0]] + [blocks[1 + j] for j in order]
            prepared.append((blocks, tail))
        timelines = _stream_many(llm, arm, prepared, args.max_tokens,
                                 min_tokens)
        for index, timeline in enumerate(timelines):
            timeline["question"] = examples[index].question
            timeline["answer"] = examples[index].answer
            print(f"[{arm}] batch row {index}: "
                  f"{len(timeline['events'])} tokens, "
                  f"end {timeline['end_ms']:.0f} ms", flush=True)
        examples = examples[:args.batch]
    for index, example in ([] if args.batch > 1 else enumerate(examples)):
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

    Median for TTFT because a prefill that lands on a cold kernel cache or
    trips a preemption is several times slower than the same prefill warm, and
    one such outlier should not become the headline; mean for ITL because it is
    a rate over many tokens and the per-token spread is what a user experiences
    as smoothness. Run enough examples for the median to mean something -- at
    two it is the average of two, which an outlier still dominates.
    """
    import statistics

    ttfts, itls = [], []
    for timeline in data["timelines"]:
        events = timeline["events"]
        if not events:
            continue
        ttfts.append(events[0][0])
        if len(events) > 1:
            itls.append((events[-1][0] - events[0][0]) / (len(events) - 1))
    return {
        "ttft_ms": statistics.median(ttfts) if ttfts else float("nan"),
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


def render_gif(arms: dict, out_path: str, index: int, frames: int = 130,
               speed: float = 3.0) -> None:
    from PIL import Image, ImageDraw, ImageFont

    regular, bold = _fonts()
    f_title = ImageFont.truetype(bold, 24)
    f_sub = ImageFont.truetype(regular, 16)
    f_head = ImageFont.truetype(bold, 19)
    f_note = ImageFont.truetype(regular, 14)
    f_badge = ImageFont.truetype(bold, 16)
    f_cap = ImageFont.truetype(regular, 13)
    f_big = ImageFont.truetype(bold, 38)
    f_unit = ImageFont.truetype(regular, 15)
    f_clock = ImageFont.truetype(bold, 18)

    BG, PANEL, BORDER = (13, 17, 23), (22, 27, 34), (48, 54, 61)
    TRACK = (33, 39, 48)
    TEXT, DIM, WHITE = (201, 209, 217), (110, 118, 129), (240, 246, 252)
    GREY, GREEN, CYAN = (139, 148, 158), (63, 185, 80), (56, 189, 248)
    ACCENTS = {DENSE: GREY, LAZY: GREEN, ROUTE: CYAN}

    W, H, M = 1500, 620, 22
    panel_w = (W - 4 * M) // 3
    p_top, p_h = 118, 452
    pad = 18
    bar_w = panel_w - 2 * pad

    sides = [(arm, arms[arm], arms[arm]["timelines"][index]) for arm in ARMS]
    t_end = max(side["end_ms"] for _, _, side in sides)
    span = t_end + max(300.0, 0.08 * t_end)

    # Two playback rates, because the two arms live on different time scales:
    # the lazy arms answer in about a second while the dense arm is still
    # prefilling at seventy. One linear axis spends every frame but two on an
    # unchanging screen. So the first stretch of frames runs at the lazy arms'
    # own pace, up to the moment they both finish, and the rest fast-forwards
    # through the dense arm's prefill -- where the only thing that changes is
    # the clock, and the clock reads fine at speed.
    t_split = min(span, max(side["end_ms"] for arm, _, side in sides
                            if arm != DENSE) * 1.12)
    n_race = max(1, int(frames * 0.6))
    n_ff = max(1, frames - n_race)
    times = ([i / n_race * t_split for i in range(n_race)]
             + [t_split + (i + 1) / n_ff * (span - t_split)
                for i in range(n_ff)])
    warp = ((span - t_split) / n_ff) / max(t_split / n_race, 1e-9)
    hold = max(45, int(round(70 / max(speed, 1e-6) * 3)))

    stats = {arm: summarise(arms[arm]) for arm in ARMS}
    question = arms[LAZY]["timelines"][index]["question"]
    docs = arms[LAZY]["docs"]
    kept = arms[ROUTE].get("router", {}).get("kept_fraction") or 1.0
    read = {DENSE: 1.0, LAZY: 1.0, ROUTE: kept}
    total_tokens = max(len(side["events"]) for _, _, side in sides) or 1

    def rate_at(side: dict, t_ms: float) -> tuple[int, float | None]:
        """Tokens emitted by `t_ms`, and the ms/token implied so far.

        Measured between the *arrivals*, not from the request start, so prefill
        does not contaminate the decode rate -- and only once two tokens have
        landed, since one arrival is a timestamp, not an interval.
        """
        arrivals = [at for at, _ in side["events"] if at <= t_ms]
        if len(arrivals) < 2:
            return len(arrivals), None
        return len(arrivals), (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1)

    def draw(t_ms: float, frame_index: int, final: bool):
        image = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(image)

        def bar(x, y, width, height, frac, colour):
            d.rounded_rectangle([x, y, x + width, y + height],
                                radius=height // 2, fill=TRACK)
            filled = max(0.0, min(1.0, frac)) * width
            if filled >= height:
                d.rounded_rectangle([x, y, x + filled, y + height],
                                    radius=height // 2, fill=colour)

        d.text((M, 16), "One corpus, retrieved in a new order",
               font=f_title, fill=WHITE)
        subtitle = (f"{docs} documents prefilled once  ·  "
                    f"each engine measured alone on one GPU  ·  "
                    f"wall clock bottom-left, real measurements throughout")
        d.text((M, 48), subtitle, font=f_sub, fill=DIM)
        d.text((M, 72), f"Q: {question[:118]}", font=f_sub, fill=(150, 160, 172))

        for slot, (arm, data, side) in enumerate(sides):
            px = M + slot * (panel_w + M)
            bx = px + pad
            accent = ACCENTS[arm]
            d.rounded_rectangle([px, p_top, px + panel_w, p_top + p_h],
                                radius=10, fill=PANEL, outline=BORDER, width=1)
            d.rectangle([px, p_top, px + panel_w, p_top + 4], fill=accent)
            d.text((bx, p_top + 14), LABELS[arm], font=f_head, fill=accent)
            d.text((bx, p_top + 40), BLURBS[arm], font=f_note, fill=DIM)

            ttft = side["ttft_ms"]
            started = ttft is not None and t_ms >= ttft
            badge = (f"first token @ {ttft:,.0f} ms" if started
                     else "prefilling …")
            d.text((bx, p_top + 66), badge, font=f_badge,
                   fill=accent if started else DIM)

            emitted, live = rate_at(side, t_ms)
            d.text((bx, p_top + 108), "TOKENS GENERATED", font=f_cap, fill=DIM)
            count = f"{emitted} / {total_tokens}"
            d.text((bx + bar_w - f_cap.getlength(count), p_top + 108), count,
                   font=f_cap, fill=TEXT if emitted else DIM)
            bar(bx, p_top + 130, bar_w, 16, emitted / total_tokens, accent)

            # The live rate wobbles for the first few tokens and then settles;
            # once the arm is done, show the run's own mean rather than letting
            # a tail outlier stand as the headline number.
            finished = t_ms >= side["end_ms"]
            shown = stats[arm]["itl_ms"] if finished else live
            d.text((bx, p_top + 178), "MILLISECONDS PER TOKEN", font=f_cap,
                   fill=DIM)
            if shown is None:
                d.text((bx, p_top + 196), "—", font=f_big, fill=TRACK)
            else:
                text = f"{shown:.1f}"
                d.text((bx, p_top + 196), text, font=f_big, fill=WHITE)
                d.text((bx + f_big.getlength(text) + 8, p_top + 222), "ms/tok",
                       font=f_unit, fill=DIM)

            d.text((bx, p_top + 286), "KV CACHE READ EACH STEP", font=f_cap,
                   fill=DIM)
            share = f"{read[arm]:.0%}"
            d.text((bx + bar_w - f_cap.getlength(share), p_top + 286), share,
                   font=f_cap, fill=TEXT)
            bar(bx, p_top + 308, bar_w, 16, read[arm], accent)

            d.text((bx, p_top + 356), "TOTAL REQUEST TIME", font=f_cap,
                   fill=DIM)
            if finished:
                total = f"{side['end_ms'] / 1000:.2f}"
                d.text((bx, p_top + 374), total, font=f_big, fill=WHITE)
                d.text((bx + f_big.getlength(total) + 8, p_top + 400), "s",
                       font=f_unit, fill=DIM)
            else:
                running = f"{t_ms / 1000:.2f}"
                d.text((bx, p_top + 374), running, font=f_big, fill=DIM)
                d.text((bx + f_big.getlength(running) + 8, p_top + 400),
                       "s and counting", font=f_unit, fill=DIM)

        now = min(t_ms, t_end)
        clock = (f"t = {now:6.0f} ms" if t_end < 10_000
                 else f"t = {now / 1000:6.2f} s")
        d.text((M, H - 34), clock, font=f_clock, fill=WHITE)
        if t_ms > t_split and warp > 1.5 and not final:
            d.text((M + f_clock.getlength(clock) + 24, H - 32),
                   f"▶▶  fast-forwarding {warp:,.0f}x", font=f_note, fill=DIM)
        if final:
            ttft_gain = stats[DENSE]["ttft_ms"] / max(stats[LAZY]["ttft_ms"],
                                                      1e-9)
            message = (f"first token {ttft_gain:,.0f}x sooner   ·   "
                       f"{stats[LAZY]['itl_ms']:.1f} vs "
                       f"{stats[DENSE]['itl_ms']:.1f} ms/token   ·   "
                       f"LazyRoute reads {kept:.0%} of the cache")
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
    print(f"decode : {LABELS[LAZY]} vs {LABELS[DENSE]} "
          f"{stats[DENSE]['itl_ms'] / stats[LAZY]['itl_ms']:.2f}x, "
          f"{LABELS[ROUTE]} vs {LABELS[LAZY]} "
          f"{stats[LAZY]['itl_ms'] / stats[ROUTE]['itl_ms']:.2f}x "
          f"(>1 is a speedup)")
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
    parser.add_argument("--batch", type=int, default=1,
                        help="concurrent requests. Batch 1 decode is "
                             "CPU-launch-bound once the kernel is fast, which "
                             "hides any GPU work sparsity removes")
    parser.add_argument("--no-reorder", dest="reorder", action="store_false",
                        help="serve the documents in their cached order; the "
                             "dense arm then hits its prefix cache")
    parser.add_argument("--route-layers", default="first",
                        help="LAZY_SPARSE_ROUTE_LAYERS for the route arm: "
                             "'first', 'all', or a stride")
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
                       "--route-layers", args.route_layers,
                       "--batch", str(args.batch)]
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
