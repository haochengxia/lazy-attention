"""Where does a decode step's routing time actually go?

The engine measurement says the router costs far more than the attention it
saves, but "the router is slow" is not something you can fix. This breaks one
`Router.route` call into its phases and times each with CUDA events, at the
shapes a real request produces, so the cost lands on a specific tensor.

Phases are timed by calling the router's own methods -- not by a re-implementation
here -- so a change to `router.py` moves these numbers instead of quietly
diverging from them.

    python benchmarks/lazyroute/router_profile.py --docs 600
    python benchmarks/lazyroute/router_profile.py --docs 600 --ops   # aten-level

Run against `--layers 16` (the 1B checkpoint) to compare against a decode step:
every number printed is per layer, and a decode step pays all of them.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "lazy_attn"))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "lazy_attn", "tests", "sparse"))

import torch  # noqa: E402

from lazy.sparse.descriptors import DescriptorStore  # noqa: E402
from lazy.sparse.router import (Router, RouterConfig,  # noqa: E402
                                derotate_query, score_tile_blocks)


def _time(fn, iters: int, warmup: int = 5) -> float:
    """Milliseconds per call, GPU-side."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


class Bench:
    """One decode step's worth of router inputs, at a chosen corpus size."""

    def __init__(self, args):
        from conftest import Layout, rope_table

        self.args = args
        device = "cuda"
        self.layer = "layer.0"

        doc_lens = [args.doc_tokens] * args.docs
        self.layout = Layout(doc_lens, args.tail_tokens, block_size=16,
                             device=device)
        num_rows = self.layout.num_blocks + 1

        # Paged K in the backend's own layout: [blocks, kv_head, D//x, P, x].
        # The split around the block-position axis is what makes the descriptor
        # fill a permute rather than a reshape, so the shape has to be real.
        x = 8
        self.key_cache = torch.randn(
            num_rows + 8, args.kv_heads, args.head_size // x, 16, x,
            device=device, dtype=torch.bfloat16)

        self.store = DescriptorStore(dtype="bf16")
        block_ids = self.layout.block_ids
        valid_lens = torch.full((block_ids.numel(), ), 16,
                                dtype=torch.int32, device=device)
        self.store.fill(self.layer, self.key_cache, block_ids, valid_lens)

        self.router = Router(RouterConfig(granularity=args.granularity),
                             self.store)
        self.query = torch.randn(1, args.q_heads, args.head_size,
                                 device=device, dtype=torch.bfloat16)
        self.cos_sin = rope_table(args.head_size, 200_000, device=device)

        self.kwargs = self.layout.route_kwargs(
            layer_name=self.layer, query=self.query,
            cos_sin_cache=self.cos_sin, rotary_dim=args.head_size,
            num_kv_heads=args.kv_heads, max_blocks=self.layout.num_blocks)

        # The intermediates each phase needs, built once so a phase is timed on
        # its own work rather than on its predecessor's.
        self.router._ensure_counters(device)
        L = self.layout
        self.geometry = self.router._geometry(L.packed, L.seq_lens, L.doc_id,
                                              L.doc_offsets,
                                              L.num_doc_blocks_t, 16,
                                              L.num_blocks)
        self.phys = self.geometry.phys
        self.q_row = self.query
        self.q_by_doc = derotate_query(self.q_row, L.doc_offsets,
                                       self.cos_sin, args.head_size)
        self.scores = self.router._score_blocks(
            self.layer, self.q_by_doc, self.phys, L.doc_id, args.kv_heads,
            None)
        self.keep = self.router._select(self.scores, self.geometry)

        # Stands in for the step's attention metadata, which is what the
        # backend passes and what makes the geometry a once-per-step cost.
        self.step = type("Step", (), {})()
        self.step.lazy_route_geometry = self.geometry

    # -- phases -------------------------------------------------------------

    def phases(self) -> list[tuple[str, callable, str]]:
        a, L = self.args, self.layout
        return [
            ("derotate_query",
             lambda: derotate_query(self.q_row, L.doc_offsets, self.cos_sin,
                                    a.head_size),
             f"[1, {a.docs}, {a.q_heads}, {a.head_size}]"),
            ("_score_blocks",
             lambda: self.router._score_blocks(self.layer, self.q_by_doc,
                                               self.phys, L.doc_id,
                                               a.kv_heads, None),
             f"{L.num_blocks} blocks"),
            ("  store.boxes",
             lambda: self.store.boxes(self.layer, self.phys.reshape(-1)),
             f"[{L.num_blocks}, {a.kv_heads}, 2, {a.head_size}]"),
            ("  q gather",
             lambda: torch.gather(
                 self.q_by_doc, 1,
                 L.doc_id.clamp(min=0).long()[:, :, None, None].expand(
                     -1, -1, a.q_heads, a.head_size)),
             f"[1, {L.num_blocks}, {a.q_heads}, {a.head_size}]"),
            ("_select",
             lambda: self.router._select(self.scores, self.geometry),
             f"argsort over {L.num_blocks}"),
            ("  argsort only",
             lambda: self.scores.argsort(dim=1, descending=True),
             f"[1, {L.num_blocks}]"),
            ("compact",
             lambda: Router.compact(self.geometry.packed, self.keep,
                                    self.geometry.doc_mask,
                                    self.geometry.tail_len, 16),
             f"cumsum over {L.num_blocks}"),
            ("_geometry (per step)",
             lambda: self.router._geometry(L.packed, L.seq_lens, L.doc_id,
                                           L.doc_offsets, L.num_doc_blocks_t,
                                           16, L.num_blocks),
             "hoisted out of the per-layer path"),
            ("route (cold geom)", lambda: self.router.route(**self.kwargs),
             "first routed layer of a step"),
            ("route (warm geom)",
             lambda: self.router.route(step_cache=self.step, **self.kwargs),
             "every later layer"),
        ]

    def run(self) -> None:
        a = self.args
        print(f"docs={a.docs} blocks={self.layout.num_blocks} "
              f"q_heads={a.q_heads} kv_heads={a.kv_heads} "
              f"head_size={a.head_size} granularity={a.granularity}")
        print(f"scoring tile = "
              f"{score_tile_blocks(1, a.q_heads, a.head_size)} blocks\n")
        print(f"{'phase':<22}{'ms/layer':>10}{'ms/token':>11}   note")
        cold = warm = None
        for name, fn, shape in self.phases():
            ms = _time(fn, a.iters)
            if name == "route (cold geom)":
                cold = ms
            elif name == "route (warm geom)":
                warm = ms
            print(f"{name:<22}{ms:>10.3f}{ms * a.layers:>11.2f}   {shape}")
        if cold is not None and warm is not None:
            # A step routes `layers - dense_prefix` times, and pays the cold
            # geometry exactly once.
            routed = max(a.layers - 2, 1)
            total = cold + warm * (routed - 1)
            print(f"\nrouter adds {total:.1f} ms per decode token "
                  f"({routed} routed layers: 1 cold + {routed - 1} warm)")

    def graph(self) -> None:
        """Replay `route` from a CUDA graph -- the launch-overhead control.

        A captured graph runs the identical kernels on the identical buffers
        and differs only in that the CPU no longer issues them one at a time.
        Whatever the gap between this and the eager number is, that gap was
        never work; it was dispatch. This is the measurement that says whether
        fusing the router is worth doing before doing it.
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.router.route(step_cache=self.step, **self.kwargs)
        torch.cuda.current_stream().wait_stream(stream)

        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            self.router.route(step_cache=self.step, **self.kwargs)

        eager = _time(
            lambda: self.router.route(step_cache=self.step, **self.kwargs),
            self.args.iters)
        replay = _time(captured.replay, self.args.iters)
        layers = max(self.args.layers - 2, 1)
        print(f"\n{'':<22}{'ms/layer':>10}{'ms/token':>11}")
        print(f"{'eager':<22}{eager:>10.3f}{eager * layers:>11.2f}")
        print(f"{'cuda graph replay':<22}{replay:>10.3f}{replay * layers:>11.2f}")
        print(f"\n{eager / max(replay, 1e-9):.1f}x -- everything above 1x was "
              f"kernel-launch overhead, not work")

    def ops(self) -> None:
        """aten-level attribution -- what the phase table cannot decompose."""
        from torch.profiler import ProfilerActivity, profile
        for _ in range(5):
            self.router.route(**self.kwargs)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                self.router.route(**self.kwargs)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                        row_limit=18))

        # The launch budget. "The router is slow" and "the router issues too
        # many launches" look the same in the phase table above and call for
        # opposite fixes, so count them: host operations per call, the GPU work
        # they submit, and the ratio between the two.
        calls = 20
        host_ops = cuda_kernels = 0
        host_us = cuda_us = 0.0
        for event in prof.key_averages():
            if event.self_cpu_time_total:
                host_ops += event.count
                host_us += event.self_cpu_time_total
            if event.self_device_time_total:
                cuda_kernels += event.count
                cuda_us += event.self_device_time_total
        print(f"\nper route call: {host_ops / calls:.0f} host ops issuing "
              f"{cuda_kernels / calls:.0f} CUDA kernels")
        print(f"                {host_us / calls:.0f} us on the host, "
              f"{cuda_us / calls:.0f} us on the GPU "
              f"({host_us / max(cuda_us, 1e-9):.1f}x)")
        print(f"                {host_us / max(host_ops, 1):.1f} us of host "
              f"time per op -- the price of one dispatch on this machine")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=600)
    parser.add_argument("--doc-tokens", type=int, default=130)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--granularity", default="page")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--ops", action="store_true")
    parser.add_argument("--graph", action="store_true",
                        help="replay route() from a CUDA graph, to separate "
                             "launch overhead from real work")
    args = parser.parse_args()

    bench = Bench(args)
    bench.run()
    if args.graph:
        bench.graph()
    if args.ops:
        print()
        bench.ops()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
