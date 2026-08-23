"""The one place that decides which LazyAttention implementation runs.

Everything selectable at runtime is declared here, so switching variants is a
matter of reading this file rather than grepping for `os.environ` across the
kernels.

Attention variant -- `LAZY_ATTENTION_VARIANT` (aliases: `LAZY_ATTENTION_MODE`,
`VLLM_LAZY_VARIANT`):

    lazy   (default)  Deferred key rotation. RoPE rotates query and key
                      normally; the decode kernel re-rotates cached keys to
                      the slot they occupy.  -> lazy/attention/ops/
    mepic             Position-independent cache. RoPE rotates the query only,
                      because the kernel rotates the cached keys itself.
                      Aliases: `nope`, `position_independent`.
                      -> lazy/attention/old_ops/

A variant selects a matching kernel set, RoPE behaviour and scheduler rotation
metadata. The three consumers are:

    attention/backends/triton_attn.py            which kernel module to call
    model_executor/layers/rotary_embedding.py    rotate Q only, or Q and K
    core/sched/scheduler.py                      which q_offset/q_mask to emit

Tuning and profiling flags, boolean unless noted, accepting 1/true/yes/on:

    MEPIC_FIRST_BLOCK_RECOMPUTE     recompute each document's first block
                                    instead of reusing it (mepic only)
    MEPIC_FORCE_FP32_ROTARY         do the in-kernel rotation in fp32
    LAZY_SHARED_KV_PROFILE          log shared-KV scheduling stats
    LAZY_SHARED_KV_PROFILE_MIN_REQS int, default 32
    LAZY_PACKED_BLOCK_PROFILE       log packed block-table rebuild timings
    LAZY_PROFILE_ATTN_BACKEND       log per-layer attention timings

Decode-kernel switches, read per call so a benchmark can flip them between
runs. Each becomes a Triton constexpr, so a new value compiles a new kernel:

    NO_LAZY                       force every request onto the vanilla path
    LAZY_FORCE_SPLIT_DECODE       use the lazy-only decode kernel when every
                                  request in the batch is lazy
    LAZY_DECODE_IGNORE_Q_MASK     drop the document padding mask
    LAZY_DECODE_COMPUTE_COS_SIN   compute cos/sin in-kernel instead of loading
                                  them from cos_sin_cache; off by default,
                                  measured a win only for large-batch decode at
                                  head_size=128 (docs/design.md 4.3)
    LAZY_DECODE_WRAPPER_PROFILE   log decode wrapper timings

Sparse decode (LazyRoute). Off by default; `LAZY_SPARSE=1` turns the whole
family on and nothing below is read until it is. Sparsity is a *read-time
view*: these change which cached blocks a decode step walks, never how blocks
are allocated, hashed, evicted or packed.

    LAZY_SPARSE                   master switch, default off
    LAZY_SPARSE_BUDGET_TOKENS     float in (0, 1] -- share of the routable
                                  cached tokens a step may read -- or an
                                  integer > 1 for an absolute token count.
                                  Default 0.25. "Routable" excludes the
                                  preamble and the dense tail, which is the
                                  denominator the Phase-0 tables use.
    LAZY_SPARSE_KEEP_DOC0         default on -- document 0 is always read and
                                  never charged. This is the *preamble
                                  convention*: `benchmarks/lazyroute/corpus.py`
                                  submits the system preamble as document 0,
                                  and the Phase-0 recall tables exclude it from
                                  both budget and mass. Turn it off for a
                                  corpus that does not front-load a preamble
                                  (`lazy_block_infer.py`, for one) -- otherwise
                                  a real document gets a free pass and the
                                  budget denominator quietly shrinks.
    LAZY_SPARSE_BUDGET_DOCS       int, document budget at GRANULARITY=doc.
                                  Unset (0) means derive it from the token
                                  budget.
    LAZY_SPARSE_GRANULARITY       page (default) | prefix | doc. Page scores
                                  are the sufficient statistic for all three: a
                                  document scores as the max over its pages.
                                  `prefix` keeps every page of a document up to
                                  the highest one selected, so a late page
                                  never arrives without its document's head --
                                  which is where §9b measured 64.5% of the
                                  cached mass. It subsumes the sink stripe, and
                                  it spends past the nominal budget, so compare
                                  it by kept_fraction rather than by budget.
    LAZY_SPARSE_SCORER            quest (default) | oracle | centroid | random.
                                  oracle runs an auxiliary dense pass and is
                                  for evaluation only.
    LAZY_SPARSE_ROUTE_LAYERS      first (default) | all | int N. `first` routes
                                  once per step and shares the decision down
                                  the stack; `all` routes per layer with that
                                  layer's own query, which is what Phase 0
                                  measured and costs 14x as much for no
                                  measured gain (§9b).
    LAZY_SPARSE_FUSED_SELECT      on by default. Selection, compaction and the
                                  walk-table build as one kernel instead of
                                  ~470 dispatches; 0 falls back to the torch
                                  reference path.
    LAZY_SPARSE_DENSE_PREFIX_LAYERS  int, default 2. Leading layers left dense
                                  (Quest convention).
    LAZY_SPARSE_REFRESH           every (default) | onchange | int N -- how
                                  often selection is recomputed.
    LAZY_SPARSE_GQA_AGG           max (default) | sum -- across a GQA group.
    LAZY_SPARSE_SINK_STRIPE       selected (default) | off | all -- force each
                                  document's first page into the walk. `all`
                                  costs page_size/avg_doc_len of the corpus at
                                  any budget and measured harmful below 50%;
                                  it is kept only as an ablation arm.
    LAZY_DESC_DTYPE               bf16 (default) | fp8 -- descriptor storage.
"""
import os

LAZY_VARIANT_LAZY = 1
LAZY_VARIANT_MEPIC = 2

_VARIANT_ENV_KEYS = ("LAZY_ATTENTION_VARIANT", "LAZY_ATTENTION_MODE",
                     "VLLM_LAZY_VARIANT")
_VARIANT_CODES = {
    "lazy": LAZY_VARIANT_LAZY,
    "lazy_attn": LAZY_VARIANT_LAZY,
    "lazy_attention": LAZY_VARIANT_LAZY,
    "mepic": LAZY_VARIANT_MEPIC,
    "nope": LAZY_VARIANT_MEPIC,
    "position_independent": LAZY_VARIANT_MEPIC,
}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in _TRUTHY_ENV_VALUES


def _env_flag_default_on(name: str) -> bool:
    """A flag that is on unless explicitly turned off.

    For switches whose default *is* the shipped behaviour and which exist so a
    suspected kernel bug can be bisected against the reference path.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return True
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _env_choice(name: str, choices: tuple[str, ...], default: str) -> str:
    """One of `choices`, or a raised error naming them.

    Unlike the numeric readers below, a misspelled choice is not defaulted
    away: `LAZY_SPARSE_SCORER=quset` silently running the Quest scorer would
    produce a plausible number attributed to the wrong policy.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value not in choices:
        raise ValueError(f"Unsupported {name}='{value}'. Expected one of: "
                         f"{', '.join(choices)}.")
    return value


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_lazy_attention_variant_name() -> str:
    for key in _VARIANT_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value.strip().lower()
    return "lazy"


def get_lazy_attention_variant_code() -> int:
    variant = get_lazy_attention_variant_name()
    try:
        return _VARIANT_CODES[variant]
    except KeyError:
        raise ValueError(
            f"Unsupported lazy attention variant '{variant}'. Expected one "
            f"of: {', '.join(sorted(_VARIANT_CODES))}.") from None


def is_mepic_variant() -> bool:
    return get_lazy_attention_variant_code() == LAZY_VARIANT_MEPIC


def mepic_first_block_recompute_enabled() -> bool:
    return _env_flag("MEPIC_FIRST_BLOCK_RECOMPUTE")


def mepic_force_fp32_rotary_enabled() -> bool:
    return _env_flag("MEPIC_FORCE_FP32_ROTARY")


def lazy_shared_kv_profile_enabled() -> bool:
    return _env_flag("LAZY_SHARED_KV_PROFILE")


def lazy_shared_kv_profile_min_reqs() -> int:
    value = os.environ.get("LAZY_SHARED_KV_PROFILE_MIN_REQS")
    if value is None:
        return 32
    try:
        return max(int(value), 1)
    except ValueError:
        return 32


def lazy_packed_block_profile_enabled() -> bool:
    return _env_flag("LAZY_PACKED_BLOCK_PROFILE")


def lazy_profile_attn_backend_enabled() -> bool:
    return _env_flag("LAZY_PROFILE_ATTN_BACKEND")


# Read per call -- see the module docstring.
def no_lazy_enabled() -> bool:
    return _env_flag("NO_LAZY")


def lazy_force_split_decode_enabled() -> bool:
    return _env_flag("LAZY_FORCE_SPLIT_DECODE")


def lazy_decode_ignore_q_mask_enabled() -> bool:
    return _env_flag("LAZY_DECODE_IGNORE_Q_MASK")


def lazy_decode_compute_cos_sin_enabled() -> bool:
    return _env_flag("LAZY_DECODE_COMPUTE_COS_SIN")


def lazy_decode_wrapper_profile_enabled() -> bool:
    return _env_flag("LAZY_DECODE_WRAPPER_PROFILE")


def lazy_split_kv_enabled() -> bool:
    """Split the decode block walk across programs (flash-decoding).

    The default kernel launches `(num_seqs, num_kv_heads)` programs -- eight at
    batch 1 on this model, on a 70-SM card -- and each walks the whole block
    table serially. Off by default because it is a second decode kernel rather
    than a change to the first one, per design rule R1; `tests/kernels/
    test_split_decode.py` holds the two to each other.
    """
    return _env_flag("LAZY_SPLIT_KV")


# -- Sparse decode (LazyRoute) ------------------------------------------------
# Read once at import by the router and the descriptor store; `LAZY_SPARSE`
# itself gates whether any of the rest is consulted.

SPARSE_GRANULARITIES = ("page", "prefix", "doc")
SPARSE_SCORERS = ("quest", "oracle", "centroid", "random")
SPARSE_SINK_STRIPES = ("selected", "off", "all")
SPARSE_GQA_AGGS = ("max", "sum")
DESC_DTYPES = ("bf16", "fp8")


def lazy_sparse_enabled() -> bool:
    return _env_flag("LAZY_SPARSE")


def lazy_sparse_budget_tokens() -> float:
    """Share of routable cached tokens if <= 1, else an absolute token count.

    Routable excludes the preamble and the dense tail, matching the denominator
    the Phase-0 recall tables use -- an engine number computed against a
    different denominator cannot be compared to them.
    """
    return _env_float("LAZY_SPARSE_BUDGET_TOKENS", 0.25)


def lazy_sparse_keep_doc0() -> bool:
    """Whether document 0 is read free of charge (the preamble convention)."""
    value = os.environ.get("LAZY_SPARSE_KEEP_DOC0")
    if value is None or not value.strip():
        return True
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def lazy_sparse_budget_docs() -> int:
    """0 means derive the document budget from the token budget."""
    return _env_int("LAZY_SPARSE_BUDGET_DOCS", 0)


def lazy_sparse_granularity() -> str:
    return _env_choice("LAZY_SPARSE_GRANULARITY", SPARSE_GRANULARITIES, "page")


def lazy_sparse_scorer() -> str:
    return _env_choice("LAZY_SPARSE_SCORER", SPARSE_SCORERS, "quest")


def lazy_sparse_route_layer_stride() -> int:
    """How often to recompute selection down the layer stack.

    0 (`first`, the default) routes once per step and shares the decision with
    every later layer. 1 (`all`) routes in every sparse layer with that layer's
    own query -- the object Phase 0 measured, and 14x the cost. N routes every
    N layers.

    `first` was an ablation until §9b measured the transfer assumption at
    n=200: it matches dense on EM and breaks four dense-correct answers to
    `all`'s seven, so nothing is paid for sharing. Everything is saved -- the
    router's cost is linear in the call count, at ~3.4 ms each.
    """
    value = os.environ.get("LAZY_SPARSE_ROUTE_LAYERS")
    if value is None or not value.strip():
        return 0
    value = value.strip().lower()
    if value == "all":
        return 1
    if value == "first":
        return 0
    try:
        return max(int(value), 1)
    except ValueError:
        raise ValueError(
            f"Unsupported LAZY_SPARSE_ROUTE_LAYERS='{value}'. Expected "
            f"'all', 'first', or a positive integer stride.") from None


def lazy_sparse_fused_select_enabled() -> bool:
    """Run selection as one kernel instead of ~470 PyTorch operations.

    On by default. Set `LAZY_SPARSE_FUSED_SELECT=0` to fall back to the torch
    implementation, which is unchanged and is what
    `tests/sparse/test_selection.py` compares against -- so a disagreement can
    be bisected by flipping this rather than by editing either path.
    """
    return _env_flag_default_on("LAZY_SPARSE_FUSED_SELECT")


def lazy_sparse_dense_prefix_layers() -> int:
    """Leading layers left dense, Quest convention."""
    return _env_int("LAZY_SPARSE_DENSE_PREFIX_LAYERS", 2)


def lazy_sparse_refresh() -> tuple[str, int]:
    """('every', 1) | ('onchange', 1) | ('interval', N).

    `onchange` still scores every step; what it skips is rebuilding the walk
    table when the selected set did not move.
    """
    value = os.environ.get("LAZY_SPARSE_REFRESH")
    if value is None or not value.strip():
        return ("every", 1)
    value = value.strip().lower()
    if value in ("every", "onchange"):
        return (value, 1)
    try:
        return ("interval", max(int(value), 1))
    except ValueError:
        raise ValueError(
            f"Unsupported LAZY_SPARSE_REFRESH='{value}'. Expected 'every', "
            f"'onchange', or a positive integer.") from None


def lazy_sparse_gqa_agg() -> str:
    return _env_choice("LAZY_SPARSE_GQA_AGG", SPARSE_GQA_AGGS, "max")


def lazy_sparse_sink_stripe() -> str:
    return _env_choice("LAZY_SPARSE_SINK_STRIPE", SPARSE_SINK_STRIPES,
                       "selected")


def lazy_desc_dtype() -> str:
    return _env_choice("LAZY_DESC_DTYPE", DESC_DTYPES, "bf16")
