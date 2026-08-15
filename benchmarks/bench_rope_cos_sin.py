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


# q_offset occupies 16 bits of the packed block table, so a rotation offset
# beyond this would run into the physical-block field.
MAX_PACKED_Q_OFFSET = 0xFFFF
# bfloat16 keeps ~8 mantissa bits. The load path rounds cos/sin through a bf16
# table and the compute path does not, so they differ at about this level; more
# than this is a bug rather than rounding.
BF16_TOLERANCE = 1e-2


@dataclass
class Case:
    shape: Shape
    num_seqs: int
    context_len: int
    doc_len: int

    def validate(self) -> None:
        """Reject shapes the packed-table model cannot represent honestly.

        Each is a case where the benchmark would still run and still print a
        number, but the number would describe something other than the label.
        """
        if self.context_len % BLOCK_SIZE:
            raise ValueError(
                f"--context-lens must be a multiple of {BLOCK_SIZE}, got "
                f"{self.context_len}: the cache and the packed table would be "
                f"sized by flooring while the kernel walks ceil(seq_len / "
                f"{BLOCK_SIZE}) blocks, reading past both.")
        if self.doc_len % BLOCK_SIZE:
            raise ValueError(
                f"--doc-lens must be a multiple of {BLOCK_SIZE}, got "
                f"{self.doc_len}: documents are padded to whole blocks by the "
                f"scheduler, so an unaligned length would advance positions by "
                f"{self.doc_len} while changing offset every "
                f"{max(self.doc_len // BLOCK_SIZE, 1)} block(s) -- a "
                f"combination the real metadata never produces.")
        if self.context_len % self.doc_len:
            raise ValueError(
                f"--context-lens {self.context_len} is not a multiple of "
                f"--doc-lens {self.doc_len}: the last document would be a "
                f"partial one with a different rotation density, while the "
                f"reported document length and rotation count would describe "
                f"whole ones.")
        if self.max_q_offset > MAX_PACKED_Q_OFFSET:
            raise ValueError(
                f"context {self.context_len} with {self.doc_len}-token "
                f"documents needs a rotation offset of {self.max_q_offset}, "
                f"past the {MAX_PACKED_Q_OFFSET} the packed table's 16-bit "
                f"field holds; the excess bits would land in the block index.")

    @property
    def blocks_per_seq(self) -> int:
        return self.context_len // BLOCK_SIZE

    @property
    def num_docs(self) -> int:
        return max(self.context_len // self.doc_len, 1)

    @property
    def max_q_offset(self) -> int:
        """Largest +1-biased offset this case puts in the packed table."""
        return (self.num_docs - 1) * self.doc_len + 1 + self.num_seqs - 1

    @property
    def rotations_per_seq(self) -> int:
        """How many cos/sin pairs the kernel actually evaluates per sequence.

        The first document's offset is the `1` reset sentinel, which the kernel
        satisfies from Q_full without touching cos/sin -- so it is one fewer
        than the document count, and a single-document context evaluates none.
        """
        return max(self.num_docs - 1, 0)


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
    blocks_per_doc = case.doc_len // BLOCK_SIZE
    doc_index = (torch.arange(case.blocks_per_seq, device=device) //
                 blocks_per_doc)
    # Rotation offsets grow per document, +1-biased, as the scheduler emits
    # them. Sequences are given *different* absolute offsets: a single row
    # broadcast to every sequence would have the whole batch reading the same
    # handful of cos_sin_cache rows, which is an L2 hit rate the load path
    # would not get in a real batch -- the same reason the KV blocks above are
    # distinct per sequence. Document 0 keeps the `1` reset sentinel for every
    # sequence, so the number of rotations per sequence stays exactly
    # `case.rotations_per_seq` and only the addresses differ.
    q_offset = (doc_index * case.doc_len + 1).to(torch.int64)
    q_offset = q_offset.unsqueeze(0).repeat(num_seqs, 1)
    seq_shift = torch.arange(num_seqs, dtype=torch.int64,
                             device=device).unsqueeze(1)
    q_offset = torch.where(doc_index.unsqueeze(0) == 0, q_offset,
                           q_offset + seq_shift)
    assert int(q_offset.max()) <= MAX_PACKED_Q_OFFSET  # Case.validate()
    q_mask = torch.zeros_like(q_offset)
    packed = (block_ids << 32) | (q_offset << 16) | q_mask

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
        q_offset_ptr=q_offset.to(torch.int32),
        q_mask_ptr=q_mask.to(torch.int32),
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
    # A timing comparison between two paths that disagree is meaningless, so
    # this fails the run rather than printing the discrepancy underneath a
    # performance conclusion. The bound is bf16's own resolution: the two paths
    # are not expected to be bit-identical (one rounds through a bf16 table),
    # only to agree to within it.
    # Checked before the tolerance: a NaN anywhere makes `rel_diff` NaN, and
    # `nan > tol` is False -- the comparison below would wave through exactly
    # the failure it exists to catch.
    for name, output in (("load", outputs[False]), ("compute", outputs[True])):
        if not torch.isfinite(output).all():
            raise AssertionError(
                f"the {name} path produced non-finite output on "
                f"{case.shape.name} seqs={case.num_seqs} "
                f"ctx={case.context_len} doc={case.doc_len} "
                f"({int((~torch.isfinite(output)).sum())} of "
                f"{output.numel()} values) -- a correctness failure, not a "
                f"benchmark result.")
    rel_diff = (diff.max() / scale).item()
    if rel_diff > BF16_TOLERANCE:
        raise AssertionError(
            f"load and compute disagree by {rel_diff:.3e} relative "
            f"({diff.max().item():.3e} absolute) on {case.shape.name} "
            f"seqs={case.num_seqs} ctx={case.context_len} doc={case.doc_len}, "
            f"past the {BF16_TOLERANCE:.0e} bf16 tolerance -- this is a "
            f"correctness regression, not a benchmark result.")

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


def run_e2e(model: str, rounds: int, max_tokens: int, num_prompts: int,
            max_num_seqs: int, gpu_util: float) -> int:
    """What the kernel difference is worth end to end, on a real model.

    Both variants run against one engine in one process -- the switch is read
    per call -- so weights, cache state and sampling are identical and only the
    kernel constexpr differs. Rounds alternate.
    """
    import os
    import time

    # Must be in-process. The A/B flips LAZY_DECODE_COMPUTE_COS_SIN between
    # rounds and the kernel reads it per call; with a worker process the engine
    # forks before those mutations and both labelled variants would silently
    # run whichever value the worker inherited.
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
        print("note: forcing VLLM_ENABLE_V1_MULTIPROCESSING=0 -- the per-round "
              "switch does not reach a worker process")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import lazy.__vllm__  # noqa: F401  patches vLLM
    from vllm import LLM, SamplingParams

    documents = [[
        f"Document {doc_idx}: " + "context words for retrieval augmented "
        "generation " * 12 for doc_idx in range(8)
    ] for _ in range(num_prompts)]
    # Every measurement gets its own query set, for two reasons. Repeating a
    # query set makes the *second* variant of a round run against a prefix cache
    # the first one just filled -- the single largest confound here, worth more
    # than 2x in throughput. And identical queries within a set make every
    # request after the first a full prefix-cache hit (documents and query
    # alike), which trips `assert num_new_tokens > 0` in the lazy scheduler.
    # The documents are deliberately *not* varied: reusing a hot corpus across
    # requests is the case LazyAttention exists for.
    def query_set(tag: str) -> list[str]:
        return [
            f"Question {tag}-{idx}: what do the documents describe? Answer:"
            for idx in range(num_prompts)
        ]

    # max_num_seqs is the variable under test: the kernel-level win only
    # appears once the decode batch is large, so it has to be settable.
    llm = LLM(model=model, gpu_memory_utilization=gpu_util,
              max_model_len=4096, max_num_seqs=max_num_seqs,
              enforce_eager=True)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0,
                              ignore_eos=True)

    def one_round(tag: str) -> tuple[float, int]:
        start = time.perf_counter()
        outputs = llm.generate(prompts=query_set(tag), sampling_params=sampling,
                               document_seqs=documents, use_tqdm=False)
        elapsed = time.perf_counter() - start
        return elapsed, sum(len(o.outputs[0].token_ids) for o in outputs)

    results: dict[str, list[float]] = {"load": [], "compute": []}
    for round_idx in range(rounds + 1):
        for name in (("load", "compute")
                     if round_idx % 2 == 0 else ("compute", "load")):
            os.environ["LAZY_DECODE_COMPUTE_COS_SIN"] = (
                "1" if name == "compute" else "0")
            elapsed, tokens = one_round(f"r{round_idx}{name}")
            if round_idx == 0:
                continue  # warmup round: JIT compile + cache warm
            results[name].append(tokens / elapsed)
    os.environ.pop("LAZY_DECODE_COMPUTE_COS_SIN", None)

    print(f"\nend to end ({model}, {max_tokens} tokens x {num_prompts} "
          f"prompts, max_num_seqs={max_num_seqs}, {rounds} rounds)")
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
    parser.add_argument("--e2e-max-num-seqs", type=int, default=256)
    parser.add_argument("--e2e-gpu-util", type=float, default=0.6)
    args = parser.parse_args()

    if args.e2e:
        return run_e2e(args.e2e_model, args.e2e_rounds, args.e2e_max_tokens,
                       args.e2e_prompts, args.e2e_max_num_seqs,
                       args.e2e_gpu_util)

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
    for case in cases:
        case.validate()

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

    # Only cases whose timing ranges separate get a verdict. Counting a case
    # whose ranges overlap as a win for whichever median came out lower is how
    # noise gets reported as a result -- the rule this benchmark states in its
    # docstring, applied to its own summary.
    decided = [result for result in results if result["separated"]]
    undecided = len(results) - len(decided)
    if decided:
        ratios = [result["ratio"] for result in decided]
        print(f"\ncompute/load ratio over the {len(decided)} decided case(s): "
              f"median {statistics.median(ratios):.3f}, min {min(ratios):.3f}, "
              f"max {max(ratios):.3f}")
        print(f"compute wins {sum(1 for r in ratios if r < 1)}, "
              f"load wins {sum(1 for r in ratios if r > 1)}")
    else:
        print("\nno case separated: nothing measured here is conclusive")
    if undecided:
        print(f"inconclusive (ranges overlap): {undecided}/{len(results)}")
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
