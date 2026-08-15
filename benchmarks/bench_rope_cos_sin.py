"""LOAD vs COMPUTE cos/sin in the lazy decode kernel.

The kernel needs a cos/sin pair per rotation offset. It can read one row of the
model's `cos_sin_cache`, or evaluate the RoPE formula in-kernel
(`LAZY_DECODE_COMPUTE_COS_SIN=1`). This decides which is faster, and by how
much, instead of asserting it.

What is controlled for:

* **Only the constexpr changes.** Both variants launch the same kernel with the
  same tensors; `COMPUTE_COS_SIN` is the single difference, so the comparison
  cannot pick up a different input distribution.
* **Drift.** The two variants are interleaved inside every repetition, so clock
  and thermal drift lands on both. Reported numbers are medians over
  repetitions, with the full range, so a claim is only made when the ranges
  separate.
* **The variable that actually matters.** Q is re-rotated only when `q_offset`
  changes -- once per document, not per block -- so the cos/sin cost is
  amortised over a document's blocks. `--doc-lens` sweeps that density; a
  context of 8192 split into 128-token documents pays it 64x more often than
  the same context in one piece.
* **Compile-time effects.** `n_regs` / `n_spills` are read off each compiled
  kernel, since the standing explanation for the default is occupancy.

Correctness is checked too: the two paths must agree, or the faster one is
irrelevant.

    python benchmarks/bench_rope_cos_sin.py               # full sweep
    python benchmarks/bench_rope_cos_sin.py --quick       # one shape
    python benchmarks/bench_rope_cos_sin.py --json out.json
    python benchmarks/bench_rope_cos_sin.py --e2e         # model-level A/B

What it found (RTX 5070 Ti, sm_120, torch 2.7.0+cu128 / triton 3.4.0): compute
loses 36/36 sweep cases, median 1.10x, and wins only for large-batch decode at
head_size=128 (0.88x at 128 seqs), where the load path is itself register-bound
so the swap buys occupancy. Hence LOAD stays the default. docs/design.md 4.3.1
has the register and PTX numbers behind that.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import dataclass, field

import torch
import triton

from vllm.model_executor.layers.rotary_embedding import Llama3RotaryEmbedding

from lazy.attention.ops.models.llama_v1 import kernel_paged_attention_2d_llama
from lazy.model_executor.rope import rope_meta_from_layer

BLOCK_SIZE = 16
DTYPE = torch.bfloat16
# Llama-3 defaults; the RoPE table is built from these so the in-kernel formula
# and the table are the same function of position.
ROPE_BASE = 500000.0
ROPE_MAX_POSITION = 131072
ROPE_SCALING = dict(scaling_factor=8.0, low_freq_factor=1.0,
                    high_freq_factor=4.0, orig_max_position=8192)


@dataclass
class Shape:
    name: str
    num_query_heads: int
    num_kv_heads: int
    head_size: int


# The decode kernel is launched per (sequence, kv head) and its tile is
# [max(next_pow2(query_heads // kv_heads), 16), head_size] -- so what a shape
# costs in registers is set by head_size and by the GQA group size *once the
# group exceeds 16*, not by parameter count. Llama-3 8B/70B/405B all land on
# head_size=128 and a group of 4/8/16, i.e. the same compiled kernel; `70B` is
# here to make that checkable rather than asserted. The last two are the shapes
# that do move the register pressure.
SHAPES = [
    Shape("1B", num_query_heads=32, num_kv_heads=8, head_size=64),
    Shape("8B", num_query_heads=32, num_kv_heads=8, head_size=128),
    Shape("70B", num_query_heads=64, num_kv_heads=8, head_size=128),
    Shape("hs256", num_query_heads=32, num_kv_heads=8, head_size=256),
    Shape("mqa32", num_query_heads=32, num_kv_heads=1, head_size=128),
]

# The full sweep stays on the two shapes the project actually runs; the rest are
# opt-in via --shapes.
DEFAULT_SHAPES = ["1B", "8B"]


@dataclass
class Case:
    shape: Shape
    num_seqs: int
    context_len: int
    doc_len: int

    @property
    def blocks_per_seq(self) -> int:
        return self.context_len // BLOCK_SIZE

    @property
    def rotations_per_seq(self) -> int:
        """How often the kernel must fetch a new cos/sin pair."""
        return max(self.context_len // self.doc_len, 1)


@dataclass
class Timing:
    per_iter_ms: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.per_iter_ms)

    @property
    def lo(self) -> float:
        return min(self.per_iter_ms)

    @property
    def hi(self) -> float:
        return max(self.per_iter_ms)


def build_inputs(case: Case, device: torch.device):
    """Decode-shaped inputs: one query token per sequence, KV already cached."""
    shape = case.shape
    num_seqs = case.num_seqs
    x = 16 // DTYPE.itemsize

    # Distinct blocks per sequence -- sharing them would hand both variants an
    # unrealistically warm L2 and flatten the very cost being measured.
    num_blocks = num_seqs * case.blocks_per_seq + 1
    key_cache = torch.randn(num_blocks,
                            shape.num_kv_heads,
                            shape.head_size // x,
                            BLOCK_SIZE,
                            x,
                            dtype=DTYPE,
                            device=device)
    value_cache = torch.randn(num_blocks,
                              shape.num_kv_heads,
                              shape.head_size,
                              BLOCK_SIZE,
                              dtype=DTYPE,
                              device=device)

    query = torch.randn(num_seqs,
                        shape.num_query_heads,
                        shape.head_size,
                        dtype=DTYPE,
                        device=device)
    output = torch.empty_like(query)

    seq_lens = torch.full((num_seqs, ), case.context_len, dtype=torch.int32,
                          device=device)
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32,
                                   device=device)

    # Packed block table, exactly as LazyGPUModelRunner builds it:
    #   [physical_block_idx:32 | q_offset:16 | q_mask:16]
    block_ids = torch.arange(1, num_seqs * case.blocks_per_seq + 1,
                             dtype=torch.int64,
                             device=device).view(num_seqs,
                                                 case.blocks_per_seq)
    blocks_per_doc = max(case.doc_len // BLOCK_SIZE, 1)
    doc_index = (torch.arange(case.blocks_per_seq, device=device) //
                 blocks_per_doc)
    # Rotation offsets grow per document, +1-biased, as the scheduler emits
    # them; the exact values do not change the work, only how often it changes.
    q_offset = (doc_index * case.doc_len + 1).to(torch.int64)
    q_offset = q_offset.clamp(max=ROPE_MAX_POSITION - 1)
    q_mask = torch.zeros_like(q_offset)
    packed = (block_ids << 32) | (q_offset.unsqueeze(0) << 16) | q_mask

    rope = Llama3RotaryEmbedding(head_size=shape.head_size,
                                 rotary_dim=shape.head_size,
                                 max_position_embeddings=ROPE_MAX_POSITION,
                                 base=ROPE_BASE,
                                 is_neox_style=True,
                                 dtype=DTYPE,
                                 **ROPE_SCALING).to(device)

    return dict(
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=packed.contiguous(),
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=shape.head_size**-0.5,
        k_scale=torch.ones(1, dtype=torch.float32, device=device),
        v_scale=torch.ones(1, dtype=torch.float32, device=device),
        num_query_heads=shape.num_query_heads,
        num_queries_per_kv=shape.num_query_heads // shape.num_kv_heads,
        num_queries_per_kv_padded=max(
            triton.next_power_of_2(shape.num_query_heads //
                                   shape.num_kv_heads), 16),
        block_table_stride=packed.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        IN_PRECISION=None,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_SIZE=shape.head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(shape.head_size),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
        rotary_dim=shape.head_size,
        rotary_dim_pow2=triton.next_power_of_2(shape.head_size),
        is_neox_style=True,
        is_lazy_ptr=torch.ones(num_seqs, dtype=torch.bool, device=device),
        q_offset_ptr=q_offset.to(torch.int32).unsqueeze(0).repeat(num_seqs, 1),
        q_mask_ptr=q_mask.to(torch.int32).unsqueeze(0).repeat(num_seqs, 1),
        cos_sin_cache_ptr=rope.cos_sin_cache,
        IGNORE_Q_MASK=False,
        **rope_meta_from_layer(rope),
    )


def launch(kwargs, compute_cos_sin: bool, num_seqs: int, num_kv_heads: int):
    return kernel_paged_attention_2d_llama[(num_seqs, num_kv_heads)](
        **kwargs, COMPUTE_COS_SIN=compute_cos_sin)


def time_variant(kwargs, compute_cos_sin, case, iters) -> float:
    """Mean ms per launch over `iters`, measured with CUDA events."""
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(kwargs, compute_cos_sin, case.num_seqs,
               case.shape.num_kv_heads)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(case: Case, device, iters: int, reps: int, warmup: int) -> dict:
    kwargs = build_inputs(case, device)

    # Compile + warm caches for both variants before timing either.
    compiled = {}
    for compute in (False, True):
        for _ in range(warmup):
            compiled[compute] = launch(kwargs, compute, case.num_seqs,
                                       case.shape.num_kv_heads)
    torch.cuda.synchronize()

    # Agreement: same inputs, same kernel, different cos/sin source.
    outputs = {}
    for compute in (False, True):
        kwargs["output_ptr"].zero_()
        launch(kwargs, compute, case.num_seqs, case.shape.num_kv_heads)
        torch.cuda.synchronize()
        outputs[compute] = kwargs["output_ptr"].clone().float()
    diff = (outputs[False] - outputs[True]).abs()
    scale = outputs[False].abs().max().clamp(min=1e-6)

    timings = {False: Timing(), True: Timing()}
    for rep in range(reps):
        # Interleaved, so drift hits both variants inside one repetition, and
        # order-alternated, so whichever runs first does not keep the colder
        # caches every time.
        order = (False, True) if rep % 2 == 0 else (True, False)
        for compute in order:
            timings[compute].per_iter_ms.append(
                time_variant(kwargs, compute, case, iters))

    load, comp = timings[False], timings[True]
    result = dict(
        shape=case.shape.name,
        num_seqs=case.num_seqs,
        context_len=case.context_len,
        doc_len=case.doc_len,
        rotations_per_seq=case.rotations_per_seq,
        load_ms=load.median,
        load_range=[load.lo, load.hi],
        compute_ms=comp.median,
        compute_range=[comp.lo, comp.hi],
        ratio=comp.median / load.median,
        separated=load.hi < comp.lo or comp.hi < load.lo,
        max_abs_diff=diff.max().item(),
        max_rel_diff=(diff.max() / scale).item(),
        regs={
            "load": getattr(compiled[False], "n_regs", None),
            "compute": getattr(compiled[True], "n_regs", None),
        },
        spills={
            "load": getattr(compiled[False], "n_spills", None),
            "compute": getattr(compiled[True], "n_spills", None),
        },
    )
    del kwargs, outputs
    torch.cuda.empty_cache()
    return result


def run_e2e(model: str, rounds: int, max_tokens: int, num_prompts: int) -> int:
    """What the kernel difference is worth end to end, on a real model.

    Both variants run against one engine in one process -- the switch is read
    per call -- so weights, cache state and sampling are identical and only the
    kernel constexpr differs. Rounds alternate.
    """
    import os
    import time

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import lazy.__vllm__  # noqa: F401  patches vLLM
    from vllm import LLM, SamplingParams

    documents = [[
        f"Document {doc_idx}: " + "context words for retrieval augmented "
        "generation " * 12 for doc_idx in range(8)
    ] for _ in range(num_prompts)]
    prompts = ["Question: what do the documents describe? Answer:"
               ] * num_prompts

    llm = LLM(model=model, gpu_memory_utilization=0.6, max_model_len=4096,
              enforce_eager=True)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0,
                              ignore_eos=True)

    def one_round() -> tuple[float, int]:
        start = time.perf_counter()
        outputs = llm.generate(prompts=prompts, sampling_params=sampling,
                               document_seqs=documents, use_tqdm=False)
        elapsed = time.perf_counter() - start
        return elapsed, sum(len(o.outputs[0].token_ids) for o in outputs)

    results: dict[str, list[float]] = {"load": [], "compute": []}
    for round_idx in range(rounds + 1):
        for name in (("load", "compute")
                     if round_idx % 2 == 0 else ("compute", "load")):
            os.environ["LAZY_DECODE_COMPUTE_COS_SIN"] = (
                "1" if name == "compute" else "0")
            elapsed, tokens = one_round()
            if round_idx == 0:
                continue  # warmup round: JIT compile + cache warm
            results[name].append(tokens / elapsed)
    os.environ.pop("LAZY_DECODE_COMPUTE_COS_SIN", None)

    print(f"\nend to end ({model}, {max_tokens} tokens x {num_prompts} "
          f"prompts, {rounds} rounds)")
    for name, values in results.items():
        print(f"  {name:>8}: {statistics.median(values):8.1f} tok/s   "
              f"(range {min(values):.1f} - {max(values):.1f})")
    load_median = statistics.median(results["load"])
    compute_median = statistics.median(results["compute"])
    print(f"  compute throughput is {(compute_median / load_median - 1) * 100:+.1f}%"
          " vs load")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seqs", type=int, nargs="+",
                        default=[1, 8, 32])
    parser.add_argument("--context-lens", type=int, nargs="+",
                        default=[1024, 4096, 16384])
    parser.add_argument("--doc-lens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES,
                        choices=[shape.name for shape in SHAPES])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--quick", action="store_true",
                        help="one representative case")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--e2e", action="store_true",
                        help="model-level A/B instead of the kernel sweep")
    parser.add_argument("--e2e-model", default="hxia7/Llama-3.2-1B-Block-FT")
    parser.add_argument("--e2e-rounds", type=int, default=4)
    parser.add_argument("--e2e-max-tokens", type=int, default=128)
    parser.add_argument("--e2e-prompts", type=int, default=16)
    args = parser.parse_args()

    if args.e2e:
        return run_e2e(args.e2e_model, args.e2e_rounds, args.e2e_max_tokens,
                       args.e2e_prompts)

    if args.quick:
        args.shapes = ["8B"]
        args.num_seqs = [8]
        args.context_lens = [4096]
        args.doc_lens = [128]

    device = torch.device("cuda")
    print(f"device : {torch.cuda.get_device_name(device)}")
    print(f"torch  : {torch.__version__}   triton: {triton.__version__}")
    print(f"timing : {args.iters} launches x {args.reps} interleaved reps\n")

    shapes = [shape for shape in SHAPES if shape.name in args.shapes]
    cases = [
        Case(shape, num_seqs, context_len, doc_len)
        for shape, num_seqs, context_len, doc_len in itertools.product(
            shapes, args.num_seqs, args.context_lens, args.doc_lens)
        if doc_len <= context_len
    ]

    header = (f"{'shape':>5} {'seqs':>5} {'ctx':>6} {'doc':>5} {'rot':>5} "
              f"{'load ms':>9} {'compute ms':>11} {'ratio':>7} {'sep':>4} "
              f"{'maxdiff':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for case in cases:
        result = run_case(case, device, args.iters, args.reps, args.warmup)
        results.append(result)
        print(f"{result['shape']:>5} {result['num_seqs']:>5} "
              f"{result['context_len']:>6} {result['doc_len']:>5} "
              f"{result['rotations_per_seq']:>5} "
              f"{result['load_ms']:>9.4f} {result['compute_ms']:>11.4f} "
              f"{result['ratio']:>7.3f} "
              f"{'yes' if result['separated'] else 'no':>4} "
              f"{result['max_abs_diff']:>9.2e}")

    ratios = [result["ratio"] for result in results]
    print(f"\ncompute/load ratio: median {statistics.median(ratios):.3f}, "
          f"min {min(ratios):.3f}, max {max(ratios):.3f}")
    print(f"cases where compute wins: "
          f"{sum(1 for ratio in ratios if ratio < 1)}/{len(ratios)}")
    # Registers are a property of the compiled kernel, i.e. of the shape -- not
    # of the batch -- so report one line per shape rather than one for the run.
    seen = set()
    for result in results:
        if result["shape"] in seen:
            continue
        seen.add(result["shape"])
        regs, spills = result["regs"], result["spills"]
        print(f"{result['shape']:>5} registers/thread: load {regs['load']}, "
              f"compute {regs['compute']}   spills: load {spills['load']}, "
              f"compute {spills['compute']}")
    worst = max(results, key=lambda result: result["max_rel_diff"])
    print(f"worst |load - compute|: {worst['max_abs_diff']:.3e} abs, "
          f"{worst['max_rel_diff']:.3e} relative to peak output "
          f"(bfloat16 has ~8 mantissa bits, i.e. ~4e-3)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
