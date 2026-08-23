# LazyRoute — Research & Engineering Plan

Query-adaptive sparse decoding over independently prefilled, cross-request reusable KV blocks.
Substrate: `illinoisdata/lazy-attention` (vLLM 0.9.2 monkey-patch, Triton, Llama-family).

---

## 0. Claim, positioning, non-goals

**Claim (strict form).** For decoding over cross-request reusable KV objects (LazyAttention-style
position-agnostic block caches), each decode token needs only a small, query-dependent subset of the
reusable blocks. We route with descriptors that are **computed once at cache-build time in each
document's local frame, stored beside the paged KV, shared across requests, and coupled to
eviction** — the property request-local sparse decoders (Quest, SeerAttention-R, SentenceKV) do not
have, because their summaries are built on request-contextualized, position-committed keys during
each request's prefill.

**Positioning matrix** (paper Figure 1 companion):

|     | Dense decode | Sparse decode |
| --- | --- | --- |
| Request-local KV | Full attention | Quest / SeerAttention-R / SentenceKV / … |
| Cross-request reusable KV | Block-Attn / LazyAttention / PIC line | **this work** |

**Complementarity.** LazyAttention's measured gains are prefill-side (deferred rotation adds ~0.13%
to decode; the dominant win is document processing). Decode-side bytes/token is exactly the axis
left open. Sparse selection composes with their kernel unchanged.

**Non-goals (this paper).**

- Semantic segmentation as a contribution → one analysis figure only (§P2.5).
- On-policy distillation → gated contingency, likely paper #2 (§P4).
- Below-HBM hierarchical fetch → design-doc stretch only; in-repo cache is GPU-resident.
- Prefill sparsification; non-Llama models; upstreaming into vLLM mainline.

**"First" claim** — decide only after the Phase-0 literature re-sweep and a pre-submission sweep
(this area turns over monthly; the June-2026 sweep found the decode-sparse-over-PIC cell empty).

---

## 1. Verified substrate invariants (everything below depends on these)

Facts checked directly in the repo (dev branch, kernel dated 2025-09-15). If any breaks on a repo
update, stop and re-verify before proceeding.

1. **Packed block table** per KV-cache group, one `int64` per logical block:
  `[physical_block_idx:32 | q_offset:16 | q_mask:16]`, built in
  `lazy/worker/gpu_model_runner.py::_rebuild_packed_block_table`.
2. **Decode kernel** (`lazy/attention/ops/models/llama_v1.py::kernel_paged_attention_2d_llama`)
  launches per (sequence, KV head), loops `cdiv(seq_len, BLOCK_SIZE)` table rows, loads K/V by
  physical index, re-rotates Q from `Q_full` only when `q_offset` changes (once per document),
  masks per-row via `q_mask`. **No contiguity or logical-position assumption** beyond the
  `seq_offset < seq_len` tail mask (bites only in the unpadded query tail) — sliding window /
  alibi are off on this path.
3. **Documents are block-aligned and never share physical blocks** (right-padded to whole blocks);
  `q_mask` is nonzero only on a document's last block and is self-contained per row.
4. **`lazy` variant stores K rotated at document-local positions** (doc encoded standalone from
  position 0); the kernel de-rotates Q by per-doc Δ. `mepic` variant stores raw K. Consequence:
  scoring the de-rotated q against stored keys is *exactly* request-local Quest math inside each
  document's frame — mean-pooling is exact for mean score; min/max box is a valid max bound.
5. **Dispatch** `ops/chunked_prefill_paged_decode.py::chunked_prefill_paged_decode` accepts
  `packed_block_table` and `seq_lens` as per-call tensors → a per-step compacted view needs no
  signature change.
6. **Scheduler metadata is subset-invariant**: dropping a document while keeping every retained
  row's original Δ reproduces the dense layout's positions with holes ⇒ sparse output = exact
  subset attention of dense block attention. `scheduler.py::metadata_for_lazy_attention` needs
  zero changes.
7. **`MAX_PACKED_Q_OFFSET = 0xFFFF`** (`lazy/utils/rotation.py`), enforced at admission. Δ
  accumulates true doc lengths ⇒ ~65×1k-token docs overflow. Must widen before any large-M work.
8. Constraints: Llama-family only (`LlamaAttention.forward` is the patch point); Triton backend
  forced; prefix caching must stay enabled; vLLM pinned at 0.9.2 / torch 2.7.0+cu128 /
  transformers 4.53.2; runtime switches centralized in `lazy/utils/variants.py`.
9. Checkpoints: `hxia7/Llama-3.2-1B-block-FT` (authors' own, default in tests — cheap iteration
  instrument) and `ldsjmdy/Tulu3-Block-FT` (8B paper numbers). Neither has trained sink tokens
  ⇒ block-head keys are outliers ("lost in block head").
10. Benchmarks in-repo: `blockbench` (2wiki/HQA/NQ/TQA), `longbench`, `chatragbench`,
  `grade_accuracy`, `benchmark_rag_serving`, `decode_probe`, `hit_ratio_scale`,
  `shared_kv_scale`, `ttft_rate_scale`, exp1–exp7 slurm = paper experiments.

**Design rule R1 (governing invariant).** Sparsity is a **read-time view**: persistent packed
table, allocation, hashing, eviction, preemption are never modified. Each step gathers a subset of
rows into a walk table handed to the kernel.

**Design rule R2.** Descriptor granularity ≠ selection granularity. Descriptors are always
per-physical-block (16-token page), per KV head. Doc score = **max over its pages**. This fixes
length calibration, keeps page selection a later 50–80-LOC patch, and makes page scores the
sufficient statistic for every selection level.

**Design rule R3.** Every uncertain scientific question sits behind a named, measurable gate
(G0-A…G0-D, G2-A, G4-A). No gate, no build.

---

## 2. Phase 0 — Measurement gate (no engine changes) · ~wks 1–3

Instrument: 1B checkpoint via `lazy_block_infer.py` / `block_infer.py` reference path (or plain HF
block-diagonal implementation) with attention-mass logging per layer/head over doc blocks during
decode. Data: repo `benchmarks/block-attn-bench/data_setup` (2wiki, HQA, NQ, TQA) + LongBench
subsets including synthetic retrieval and summarization/aggregation stress. Spot-check key results
on 8B (`ldsjmdy/Tulu3-Block-FT`).

Deliverables:

- **D0.1 Oracle recall@budget** curves at doc *and* page granularity, per layer / head / task
  family. Adverse hypothesis explicitly tested: independent prefill may flatten cross-block key
  discriminability (report box-tightness and score-margin stats vs a request-local control run).
- **D0.2 Scorer ladder (offline)** from the same logged K: oracle vs Quest min/max box vs centroid
  (mean) vs random. Sub-ablation: exclude vs include the first 1–2 tokens of each doc (sink
  outliers) from box statistics.
- **D0.3 Temporal structure**: Jaccard overlap of oracle top-k sets between steps t and t+N (per
  layer, N ∈ {1,2,4,8,16,32}); staleness curve (freeze selection for N tokens → accuracy);
  selection-shift traces on multi-hop tasks (does the set flip at hop boundaries mid-generation?).
- **D0.4 Renormalization probe**: next-token KL, dense vs oracle-subset, with/without a sink
  stripe (always attend token 0 / page 0 of each doc). Decides the stripe policy.
- **D0.5 Literature re-sweep** (decode-sparse over position-independent caches; SeerAttention-R /
  SentenceKV / C²KV successors; anything post-June-2026).
- **D0.6 Author contact**: request the 1B Block-FT training recipe (and flag the collaboration /
  code-drift question) — they built the substrate; a 1B block-FT recipe de-risks Phase 4.

Gates (thresholds are defaults; fix them before running, then don't move them):

- **G0-A (existence).** Oracle doc-top-k using ≤25% of reusable tokens retains ≥99% of dense EM on
  the QA families and ≥95% on LongBench synthetic. Fail → pivot: the Phase-0 dataset becomes an
  analysis paper on independent-prefill attention geometry.
- **G0-B (scorer).** Quest-box recall within 2–3 pts of oracle at target budgets → training-free
  is the method, learned descriptor is future work. Materially worse → learned descriptor becomes
  the method (Phase 4 promoted), Quest becomes the baseline, and the gap is the motivating figure.
- **G0-C (granularity).** Page-vs-doc recall gap at matched token budget decides whether P3.1 is
  paper content or appendix. Expect ≈0 on short-passage RAG; expect it to open on long-doc corpora.
- **G0-D (temporal).** Overlap and staleness curves set refresh defaults (event-driven expected to
  dominate fixed-N). If sets are essentially static per request, reframe honestly as
  attention-native retrieval and make retrieve-then-read parity the bar.

---

## 3. Phase 1 — Engine v0: training-free doc-level sparse decode · ~wks 3–7

All work behind `LAZY_SPARSE=0|1` (default 0), switches registered in `utils/variants.py`
house-style: `LAZY_SPARSE_BUDGET_{DOCS,TOKENS}`, `LAZY_SPARSE_SCORER={oracle,quest,centroid,random}`,
`LAZY_SPARSE_REFRESH={every,onchange,N}`, `LAZY_SPARSE_GQA_AGG={max,sum}`,
`LAZY_SPARSE_SINK_STRIPE={off,selected,all}` (default `selected`, confirmed on
KL 2026-08-23 in §9b — `all` measurably hurts below a 50% budget, and
`selected` needs a budget guard: below ~2x its own cost it destroys more than
it saves).

**W1.1 Offset repack (prerequisite, do first).**
Widen `q_offset`: either repack to `[phys:32 | off:24 | mask:8]` (mask ≤ block_size ≤ 255 holds)
or split offsets into a parallel `int32` array. Touch: pack sites in `gpu_model_runner.py`
(the `<<16` at ~L328/338), kernel unpack, `rotation.py::MAX_PACKED_Q_OFFSET`, admission check in
`engine/processor.py`; update `tests/core/test_rotation_offset_bound.py`,
`tests/worker/test_q_offset_range.py`. ~60–120 LOC + tests.
*Exit: existing test suite green; a 200-doc synthetic request admits and answers correctly.*

**W1.2 Descriptor store.**
Tensor parallel to the paged KV, keyed by physical block id:
`desc[phys_blk, kv_head, {min,max}, head_size]` (bf16; `LAZY_DESC_DTYPE=fp8` option; optional
third `centroid` vector). Fill on document-request completion (doc requests are identifiable —
ids `f"{parent}_d{idx}"`): segmented amin/amax over `key_cache` rows, **excluding the first 1–2
tokens of each document** from statistics. Invalidate on free via `core/block_pool.py` hook.
Cross-request amortization is free (physical-id keyed; per-doc hashing/eviction already exists).
Overhead: 12.5% of K bytes at block_size 16 (6.25% of KV); fp8 halves it. ~100–150 LOC.
*Exit: descriptor tensor survives eviction/reuse round-trips (test T3); rebuild-on-respawn correct.*

**W1.3 Router v0 — global per-step, in `LazyGPUModelRunner`.**
Per step, per lazy sequence: gather per-doc Δ from the existing `lazy_offset` buffers → rotate q
once per doc via `cos_sin_cache` (same math the kernel uses, hoisted) → Quest bound per page per
KV head → doc score = max over pages → aggregate across the GQA group (flag: max/sum) → select:
**always keep doc 0** (preamble convention) **+ dense tail** (query + generated), then top-k docs
or greedy-until-token-budget → **event-driven rebuild**: score every token; diff selected set vs
current; on change, `gather` compacted walk table + compact `seq_lens`
(= selected_rows × B + true tail length; tail rows last). Pure torch, one launch region, no
per-layer host sync. Router cost budget: <1% of forward FLOPs, <5% of decode step wall-clock
(measure via `decode_probe`). ~150–250 LOC.

**W1.4 Dispatch plumbing.**
Pass the compacted `(walk_table, seq_lens)` pair through `chunked_prefill_paged_decode` — no
signature change, different tensors. Non-lazy and prefill paths untouched. ~40–60 LOC.

**W1.5 Oracle-in-engine mode.**
`SCORER=oracle` runs an auxiliary dense pass to compute true mass (debug/eval only). Replicates
Phase 0 inside the engine and powers CI equivalence.

**W1.6 Tests (mirror `tests/core` patterns).**

- **T1 Dense equivalence:** budget=∞ ⇒ logits match dense lazy within numeric tolerance. *The*
  trust anchor for everything after.
- **T2 Subset equivalence:** engine sparse output ≡ HF reference with the same doc subset masked
  (positions-with-holes check).
- **T3 Lifecycle:** eviction/preemption invalidate + regenerate descriptors; doc respawn correct.
- **T4 Mixed batches:** lazy-sparse + lazy-dense + non-lazy coexist.
- **T5 Repack bounds** (from W1.1).

*Phase-1 exit criteria:* T1–T5 green; quest-scorer accuracy at budget B on blockbench within ε of
the Phase-0 offline prediction (validates the engine implements the measured object); decode probe
shows HBM-bytes/token reduction ≈ proportional to token sparsity minus descriptor reads, router
overhead within budget.

---

## 4. Phase 2 — Evaluation & baselines (paper core) · ~wks 7–12

**P2.1 Quality.** Budget sweeps on blockbench (2wiki/HQA/NQ/TQA), LongBench (report synthetic +
summarization/aggregation stress honestly — these are where sparsity should fail), chatragbench;
`grade_accuracy` pipeline. Mitigation arms: threshold-adaptive k; entropy-triggered dense fallback;
small k-margin. 1B for sweeps, 8B for headline tables.

**P2.2 Baselines that can kill the paper (run them yourself).**

- Dense lazy reuse — quality ceiling, and the speed baseline.
- **Naive-transfer Quest** — rebuild page summaries per request after placement (position-committed
  view). Must either lose on quality/cost or show rebuild cost erasing the reuse benefit; its
  amortized rebuild cost vs our once-per-corpus descriptors is a headline systems number.
- **Retrieve-then-read** — embedding/Contriever top-k docs, then dense lazy over them. The
  deployed status quo; per-token routing must beat it where selection shifts mid-generation.
- Random selection at matched budget (sanity floor).
- Request-local sparse with full recompute (separates reuse effects from sparsity effects).

**P2.3 Systems.** Kernel level: `decode_probe` fork with sparse walk lengths (bytes/token, latency
vs budget). Serving level: `benchmark_rag_serving` + `hit_ratio_scale` / `shared_kv_scale` /
`ttft_rate_scale` under uniform and skewed traffic — the cross-request premise is only testable
with multi-request traces. **Large-M commitment:** a corpus-as-cache workload (hundreds of cached
docs; agentic/file-repo flavor per the Block-Attn appendix scenarios): TPOT and throughput vs M
and budget, with an honest router-overhead breakdown (score GEMV, top-k, gather, rebuild rate).

**P2.4 Multi-hop money figure.** Selection-shift trace on 2wiki/HQA + accuracy delta: per-token
routing vs selection frozen at step 0 (uses G0-D machinery). This is the figure that answers
"why not just run a retriever first."

**P2.5 Segmentation analysis (one figure, only if a long-doc regime is in scope).** Routing-unit
comparison at matched token budget and identical paged execution: boundary-oblivious fixed pages
vs bounded semantic docs (public Syon-Li segmenter) vs **random boundaries with matched length
distribution** — the control that attributes any win to semantics rather than unit size.

**Gate G2-A (ship bar).** At a budget giving ≥2× decode-bytes reduction at M≈50–200: quality
within ~1 pt of dense on QA families, TPOT improvement survives end-to-end with router overhead
accounted, and ≥1 regime where per-token routing beats retrieve-then-read. Miss → diagnose via the
ablation ladder before adding machinery.

---

## 5. Phase 3 — Refinements, each behind its gate · ~wks 10–14 (overlapping)

- **P3.1 Page-level selection** (gate: G0-C — material, decided 2026-08-23, so this is paper
  content). Within selected docs, rank pages by the already-computed descriptor scores; token
  budget = row count (uniform 16-token rows ⇒ flat top-k, no knapsack); `index_select` rows
  instead of doc ranges; **mandatory sink stripe** (keep page 0 of every selected doc; kept *in*
  the box statistics — the exclusion was measured harmful, §9b). Test: per-doc budget=all ≡
  doc-level bit-identical walk tables → verified refinement chain dense ⊇ doc ⊇ page. ~50–80 LOC.
- **P3.2 Per-KV-head walk tables** (gate: Phase-0 per-head divergence at page granularity).
  `[seq, kv_head, blocks]` table + one kernel indexing change — the only kernel edit on the
  roadmap. ~20–40 kernel LOC + runner plumbing.
- **P3.3 Router Triton kernel** (gate: host overhead > budget at large M). Fuse rotate+score(+topk).
- **P3.4 Adaptive granularity** (optional): within-doc score-profile peakedness switches
  pages-vs-whole-doc per document per step, at zero extra scoring cost.
- **P3.5 Per-layer routing** (gate: per-layer recall divergence in D0.1). Move scoring to
  `TritonAttentionImpl.forward` (q is in hand; attention is a CUDA-graph splitting op so the
  boundary exists); keep first 2 layers dense (Quest convention) either way.

---

## 6. Phase 4 — Learned components (contingent track)

**Trigger:** G0-B (material Quest↔oracle gap) or G2-A quality miss at target budgets.

- **P4.1 Block-FT reproduction.** Prefer the authors' 1B recipe (D0.6). Else the Tencent recipe:
  public segmenter + SemanticSeg; FlexAttention + Liger-Kernel; ~180k HotpotQA + segmenter-split
  ChatQA2; lr 2e-6→2e-7 cosine. Target Llama-3.2-1B/3B → Llama-3.1-8B (Qwen would need a new model
  patch — avoid). Cite honestly as "their released segmenter + our reimplementation"; original
  Block-Dist weights are unreleased and uncomparable.
- **P4.2 Routing descriptor head on the same run** (marginal cost once training): end-of-block
  summary token whose stored-frame key is the page/doc descriptor; evaluate as a scorer-swap
  against the Quest box on identical selection machinery.
- **P4.3 OPD — measure before building (gate G4-A).** Quantify autoregressive drift first:
  divergence between dense and sparse free-running rollouts; accuracy-vs-budget teacher-forced vs
  free-running. Try cheap mitigations (k-margin, periodic dense steps, threshold-adaptive k,
  entropy fallback). OPD only if drift is measured *and* mitigations fail — and then it is
  probably paper #2, not a fourth pillar of this one.

---

## 7. Risks & honest-outcome branches

| Risk | Signal | Branch |
| --- | --- | --- |
| No exploitable sparsity in this geometry | G0-A fails | Analysis paper: attention geometry under independent prefill (Phase-0 data is the paper) |
| Selection static per request | G0-D flat | Reframe: attention-native retrieval (exact bound, no separate retriever, per-layer granularity); retrieve-then-read parity is the bar |
| Quest ≈ oracle | G0-B tight | Training-free paper; contribution = cache-time lifecycle + systems; Phase 4 shelved (fine) |
| Host router overhead eats the win | decode_probe breakdown | P3.3 kernel; report breakdown regardless |
| Scooped | D0.5 / pre-submission sweep | Sharpen to the uncovered cell; "first" language decided last |
| Repo drift (0.9.2 pin, monkey-patch) | upstream bumps | Freeze env (`scripts/install.sh` pin); upstreaming out of scope |
| Checkpoint pathologies (no sinks, lost-in-block-head) | D0.2/D0.4 | Descriptor sink exclusion + stripe; don't attribute checkpoint artifacts to the method |
| 16-bit offset cap | admission refusals at large M | Fixed first (W1.1) |

---

## 8. Sequencing & resources

Weeks are relative and overlap-friendly; adjust to availability.

- **Wks 1–3:** Phase 0 (1B, single GPU) ∥ W1.1 repack ∥ D0.5 sweep ∥ D0.6 author contact.
- **Wks 3–7:** W1.2–W1.6; Phase-0 8B spot-checks (A100/H100).
- **Wks 7–12:** Phase 2 (serving runs on H100/GH200 — repo slurm scripts exist for both).
- **Wks 10–14:** gated Phase-3 items (P3.1 likely; P3.2 if evidence).
- **Wks 12–16:** writing; pre-submission literature sweep; ablation freeze.
- **Parallel from wk 6 if triggered:** Phase 4 (multi-GPU for 8B distillation; 1B feasible on a
  single large GPU).

Venue realism from late Aug 2026: ICML 2027 (abstracts ~late Jan 2027) fits this timeline;
MLSys 2027 as systems-flavored alternative if its deadline aligns; NeurIPS 2027 backstop.
Confirm exact deadlines when targeting.

---

## 9. Paper skeleton → evidence map

- **Fig 1** Method: cache-time descriptor lifecycle over the packed-table view (R1/R2 picture).
- **Fig 2** Oracle recall@budget, doc & page granularity (D0.1) — existence + adverse-hypothesis.
- **Fig 3** Scorer ladder: oracle / Quest / centroid / random, ± sink exclusion (D0.2).
- **Fig 4** Accuracy-vs-budget with baselines incl. naive-transfer Quest & retrieve-then-read (P2.1–2.2).
- **Fig 5** Multi-hop selection-shift trace + frozen-selection delta (P2.4).
- **Fig 6** Serving: TPOT/throughput vs M and budget; bytes/token; router-overhead breakdown (P2.3).
- **Table** Stress tasks (aggregation/summarization/counting) reported honestly + mitigation arms.
- **Ablations** sink stripe (D0.4), GQA aggregation, refresh policy (G0-D), granularity (G0-C),
  layer-dense prefix, descriptor dtype.
- **Appendix** offset repack; equivalence-test methodology; segmentation figure (P2.5) if in scope.

---

## 9b. Execution log (append-only; measurements, not intentions)

### 2026-08-23 — environment, W1.1, instrument health

**Environment.** RTX 5070 Ti (16 GB, sm_120), torch 2.7.0+cu128 / vLLM 0.9.2 /
transformers 4.53.2, per `scripts/install.sh`. Repo suite green on the 1B
checkpoint (66 passed, 3 skipped; the skips need a vLLM source checkout).
`tests/core/test_block_attn_preemption.py` needs the repo root on `PYTHONPATH`.
`lazy_block_infer.py --mode lazy` answers the two-hop example correctly.

**W1.1 offset repack — LANDED.** `[phys:32 | q_offset:16 | q_mask:16]` →
`[phys:32 | q_offset:24 | q_mask:8]`. `q_mask` holds a document's padding, at
most `block_size - 1`, so 8 bits cover every block size vLLM offers; block
sizes above 256 are now refused in `initialize_kv_cache`, and an out-of-range
mask is refused where an out-of-range offset already was. Field widths live in
`lazy/utils/rotation.py` and are read by the runner, the admission check and
both kernel unpack sites (as `tl.constexpr(...)` globals — Triton rejects the
annotated form). Cap: 65535 → 16777215, i.e. ~64k → ~16M tokens of reusable
cache per request. Suite green at 69 passed; new
`tests/core/test_packed_entry_roundtrip.py` walks scheduler metadata → runner
pack → kernel unpack at 200×400 and 1100×61 documents.
`scripts/large_m_admission.py` runs a 601-block / 66k-token request needing
offset 65954: refused at admission before, admits and decodes now.

**The 1B checkpoint is not a large-M instrument.** 2wiki dev, gold-string
containment / degeneracy (`benchmarks/lazyroute/block_health.py`):

| corpus | block path | full attention |
| --- | --- | --- |
| 2 supporting docs | 58% / 4% | — |
| native 10 docs (50 ex.) | 24% / 22% | 20% / 12% |
| widened to 20 docs | 0% / 45% | 5% / 80% |
| widened to 50 docs | 0% / 30% | 0% / 70% |

Two things follow, and they are separate. (i) **Block attention is not the
problem**: at every M it matches or beats full attention over the same text, so
degeneration past ~10 documents is the checkpoint, not the substrate — the 600
document run's word salad reproduces exactly under full attention. (ii)
**Answer-accuracy gates cannot be evaluated on 1B beyond ~10 documents**, where
the dense ceiling is already ~24%. Consequences:

- **G0-A as written is not measurable on the 1B checkpoint.** "≥99% of dense
  EM" against a ~24% dense baseline is noise at any sample size we can afford,
  and at M ≥ 20 the baseline is 0%. Phase 0 therefore runs its *existence*
  claim on 8B, and uses the 1B model only for attention-geometry statistics
  (D0.1–D0.3) and next-token KL (D0.4), which stay meaningful under a weak
  generator because they compare a model against itself.
- **P2.3's large-M workload needs 8B** (or a different base model), which does
  not fit this 16 GB card. Large-M work is A100/H100-only from here.
- Recheck whether the 8B `ldsjmdy/Tulu3-Block-FT` checkpoint holds up past 10
  documents before committing to the corpus-as-cache figure; nothing measured
  so far says it does.

**Template, recorded because it silently costs 30 points.** Both Block-FT
checkpoints want the Tulu markers `blockbench.py::make_blocks` rewrites into
(`<|user|>` / `<|assistant|>`), not the Llama-3 headers. Handed Llama-3
headers, the 1B model answers that the documents do not contain what is plainly
in them. `benchmarks/lazyroute/corpus.py` is now the single place that builds
blocks and prompts.

### 2026-08-23 — D0.1 / D0.2 first pass (1B, 2wiki, M=10)

Instrument: `benchmarks/lazyroute/probe.py` — documents prefilled standalone
(local-frame keys, as a cross-request cache would hold them), merged, then a
**teacher-forced** decode over the gold answer with per-layer attention and the
pre-RoPE query logged. Teacher-forced because ~22% of free-running generations
degenerate at this M and a repetition loop's attention is not the object of
study. Every scorer is measured against the oracle built from the same logged
tensors, and `check_equivalence` asserts that scoring the de-rotated query
against local-frame keys reproduces the model's own attention (max abs
difference 0.0075 post-softmax, i.e. bf16 noise) — the property the whole
descriptor argument rests on. Study: `benchmarks/lazyroute/d0_recall.py`,
20 examples × 9 steps × 16 layers × 8 KV heads.

**Attention mass captured, page-level, preamble excluded from budget and mass:**

| budget (share of cached tokens) | 10% | 15% | 25% | 35% | 50% |
| --- | --- | --- | --- | --- | --- |
| oracle | 0.641 | 0.837 | 0.965 | 0.987 | 0.994 |
| Quest box | 0.550 | 0.756 | 0.919 | 0.958 | 0.977 |
| centroid | 0.396 | 0.485 | 0.616 | 0.716 | 0.828 |
| random | 0.093 | 0.144 | 0.244 | 0.343 | 0.493 |

**The control that reframes all of it.** 64.5% of cached attention mass sits on
the **first two tokens of each document** — 2.1% of the cached tokens. These are
per-block sinks in a checkpoint with no trained sink token ("lost in block
head"). So the table above credits query-adaptive routing for finding
structure that a fixed, query-independent stripe finds for free. Forcing page 0
of every document into the selection *and charging its ~19% of budget for it*:

| budget | 25% | 35% | 50% |
| --- | --- | --- | --- |
| stripe + oracle | 0.952 | 0.985 | 0.993 |
| stripe + Quest | 0.937 | 0.973 | 0.986 |
| stripe + random | 0.833 | 0.867 | 0.897 |

The stripe alone is worth ~0.83 at ~19% of tokens. **Routing's real
contribution is the residual: +10.4 pts (Quest) / +11.9 pts (oracle) over
random at a matched 25% budget.** That is the number the paper has to defend,
not 0.965. Below ~20% the stripe does not fit and all three scorers coincide,
so those columns say nothing.

**Gate calls (thresholds not moved):**

- **G0-A — existence: pass, restated.** Sparsity is real at page granularity,
  but its honest form here is "routing beats a query-independent stripe by ~10
  pts of mass at matched budget", not an EM-retention number, which this
  checkpoint cannot support (§9b above).
- **G0-B — scorer: pass *with the stripe*, fail without.** Quest is 4.6 pts
  behind oracle at 25% with no stripe (the gate wanted 2–3), and 1.5 pts behind
  with it. Decision: **training-free Quest is the method, stripe default-on**;
  Phase 4 stays a contingency, not promoted. Centroid is not competitive
  (0.616 at 25%) except at layer 0 — mean pooling being exact for the *mean*
  score does not make it a useful *selector*.
- **G0-C — granularity: decided, and against the plan's expectation.**
  Document-level oracle reaches 0.409 at a 25% token budget against page-level
  0.965, and 0.576 vs 0.994 at 50%. Mass concentrates in a few pages *inside*
  many documents, so this is not only spend quantisation. **P3.1 is paper
  content, not appendix**, and the W1.3 doc-level router v0 should be treated
  as a scaffold to get the plumbing right, not as a configuration anyone would
  ship. (Caveat: the doc-level arm has no stripe variant yet.)
- **G0-D — temporal: preliminary only.** Top-25% page set overlap is 0.836
  between consecutive steps and 0.794 against step 0. Forced answers are 2–8
  tokens, so this is a weak probe of a question that needs long generations;
  it does say the set is neither static nor churning.
- **Adverse hypothesis (D0.1): rejected.** Independent prefill does not flatten
  cross-block key discriminability. Against a request-local contextualised
  control on the same tokens, box widths are equivalent (3.93 vs 3.85) and the
  oracle-to-Quest gap is *smaller* in the block arm (0.045 vs 0.139 at 25%) —
  the reusable cache is easier to route over, not harder. Only the within-arm
  gap is comparable across arms; the contextualised run puts 68.7% of cached
  mass on the preamble against the block arm's 17.6%, so absolute recall is not.

**Consequences for the build:**

- **W1.2 loses the sink exclusion.** Dropping each document's first 1–2 tokens
  from box statistics was measured harmful (0.460 vs 0.919 at 25%): without a
  stripe the scorer needs those keys in the box to find the page holding the
  mass, and with a stripe page 0 is taken unconditionally so its box is never
  consulted. Either way the exclusion buys nothing. Remove it and the
  `LAZY_DESC_*` surface shrinks.
- **`LAZY_SPARSE_SINK_STRIPE` defaults on, as `selected`** (page 0 of every
  *selected* document), and the ablation reports the stripe-off arm as the
  thing it is: a strictly worse policy, not a baseline. Three points behind
  that default:
  - The stripe is adopted for **scorer quality and per-document read
    integrity** — if a block is read at all, its head is read — not as a
    renormalisation fix. Keeping 3 of 10 documents still drops 7 documents'
    sinks, ~47% of cached mass at this M, whether or not the kept documents
    were striped.
  - The `all` variant measured above is **not shippable**: page 0 of every
    document costs `page_size / avg_doc_len` ≈ 19% of the corpus at any M,
    while the budgets this method targets at M≈200 are ~5%. It is affordable at
    M=10 and incoherent at the scale the paper is about.
  - Renormalisation therefore stays open for D0.4, with three candidate
    remedies at very different costs: an all-document *token-granular* stripe
    (2 tokens/doc ≈ 2.3% of corpus, but sub-page walk rows violate R1); a
    per-(layer, head) denominator correction, which needs no KV at all since
    sink mass is parked rather than informative; or nothing, if the measured KL
    with `selected` striping is small. Decide on the KL number, not on this
    one.
- **P3.5 (per-layer routing) is not motivated at this setting.** Oracle recall
  at a 15% budget is flat across all 16 layers (0.81–0.86); Quest 0.70–0.82.
  One routing decision can serve every layer here.
- **24.6% of decode attention lands outside the cache** (query + generated
  tail), which the router always keeps dense. Bytes/token claims are bounded by
  the remaining ~75% and should be quoted that way.

Open, in order: the same sweep on longer documents (LongBench) where page-vs-doc
should widen further; D0.4's KL probe, which is what actually licenses the
stripe on quality grounds rather than mass; a doc-level stripe arm; and G0-D on
long generations.

### 2026-08-23 — D0.4 renormalisation probe (1B, 2wiki, M=10)

`benchmarks/lazyroute/d0_kl.py`. One decode step is sparsified against a dense
teacher-forced prefix, so this is the error a policy *injects*, not the drift it
accumulates. Eager attention is patched to mask probabilities per (layer, query
head), which makes the renormalised and un-renormalised variants differ by
exactly one division. 20 examples × 7 steps × 36 configurations.
KL(dense ‖ sparse) in nats, with next-token top-1 agreement:

| budget | 10% | 25% | 50% |
| --- | --- | --- | --- |
| oracle, no stripe | 0.141 / 0.74 | 0.036 / 0.93 | 0.007 / 0.97 |
| Quest, no stripe | 0.649 / 0.64 | 0.207 / 0.82 | 0.090 / 0.90 |
| Quest, stripe=selected | 0.532 / 0.64 | 0.182 / 0.81 | 0.079 / 0.94 |
| Quest, stripe=all | 0.680 / 0.57 | 0.267 / 0.83 | 0.077 / 0.90 |

**Retained mass does not predict KL, and this is the important result.**
Oracle at a 10% budget retains 0.717 of the probability and lands at KL 0.141;
Quest at 25% retains 0.940 — 22 points *more* — and lands at 0.207, half again
worse. Which mass is kept dominates how much. Consequences:

- **Mass recall (D0.1/D0.2) cannot set budgets or rank policies.** It stays
  useful for ranking scorers inside one policy family, and it is what the
  router can actually optimise, but every gate and every headline number moves
  to KL / top-1 agreement.
- **G0-B is reopened, and my 2026-08-23 call on it was too generous.** In mass
  terms Quest trailed oracle by 4.6 pts at 25%; in KL it is 0.182 against 0.033
  (5.5×) with top-1 0.81 against 0.94. The scorer gap is materially larger than
  the mass gap made it look, which strengthens the Phase-4 case rather than
  shelving it. Not promoting Phase 4 yet — first check whether the gap survives
  on 8B and on long documents — but it is no longer "training-free wins".
- **The realistic operating point is weaker than the mass curves suggested.**
  Quest needs ~50% of cached tokens for 0.94 top-1 agreement; 25% costs ~1 step
  in 5 changing its argmax, and the tail is heavy (p95 KL 0.81 at 25%). At this
  setting that is a ~2× byte saving, the low end of G2-A's bar.

**Renormalisation: settled, against my hypothesis.** Letting dropped mass
vanish (dense-normalised weights, dropped keys contributing v=0 — the "sink
that needs no KV") is *worse* almost everywhere: 0.353 vs 0.141 for oracle at
10%, 0.202 vs 0.182 for Quest+stripe at 25%, never better than a tie. Zeroing
the dropped mass shrinks the attention output by the retained fraction, and
that magnitude error propagates through the residual stream. **Ordinary softmax
over retained keys is correct; do not build the denominator correction.** One
of the three candidate remedies is now eliminated on evidence.

**Stripe: `selected` confirmed as default, `all` rejected, and the mechanism is
not what we assumed.** The stripe is not repairing renormalisation — it barely
moves retained mass (0.940 → 0.946 at 25%). It is a *scorer crutch*: Quest's
misses are systematically the block-head pages, and forcing them corrects that.
Hence it helps Quest at every budget (−18% / −12% / −12% KL) and is
neutral-to-harmful for the oracle (0.033 vs 0.036 at 25%; 0.367 vs 0.141 at
10%, where the forced pages crowd out a selection that was already right).
Two consequences: the stripe needs a **budget guard** — below roughly twice its
own cost it destroys more than it saves, which is exactly the large-M regime —
and its value should be re-measured per scorer, since a better scorer needs it
less. `all` is harmful below a 50% budget (0.267 vs 0.207 for Quest at 25%),
confirming the budget-fit argument quantitatively.

Still open on renormalisation: whether the residual error at a fixed budget is
dominated by lost *value* content, which no denominator trick can recover. The
refutation above says the cheap fix does not work; it does not say what would.

### 2026-08-23 — end-to-end QA accuracy (1B, 2wiki, M=10): the metric is null

`benchmarks/lazyroute/d0_e2e.py`. Free-running greedy decode, 40 tokens, router
live at every step (per-layer: each layer's q_proj hook scores pages and leaves
a mask for that same layer's attention). Paired against dense on the same 60
examples, because comparing two accuracy *rates* off a ~24% baseline resolves
nothing at any sample size we can afford.

| config | gold | degenerate | identical to dense | D+S− | D−S+ |
| --- | --- | --- | --- | --- | --- |
| dense | 26.7% | 25.0% | 100% | – | – |
| Quest 50% +stripe | 23.3% | 20.0% | 25.0% | 2 | 0 |
| Quest 25% +stripe | 26.7% | 23.3% | 13.3% | 2 | 2 |
| Quest 25%, no stripe | 26.7% | 18.3% | 10.0% | 2 | 2 |
| **random 25% +stripe** | **30.0%** | 15.0% | 3.3% | 3 | 5 |

**Random selection scores highest.** It reproduces dense byte-for-byte on 3.3%
of examples — it is generating different text almost everywhere — and still
"beats" dense by 3.3 points, on McNemar counts of 3 losses to 5 wins, i.e.
noise. So end-to-end QA accuracy on this checkpoint cannot adjudicate anything:
"sparse matches dense accuracy" would be vacuous when *random* matches dense
accuracy. Part of the mechanism is visible in the degeneracy column — sparsity
falls from 25% to 15% because dropping keys breaks the repetition loops this
checkpoint falls into. That is an artifact rewarding the worst policy.

**What does carry signal is the identity rate**, and it ranks the policies in
exactly the KL order: 25.0% (Quest 50%) > 13.3% (Quest 25% +stripe) > 10.0%
(no stripe) > 3.3% (random). It also shows how expensive drift is: even at a
50% budget, three generations in four diverge from dense. Per-step top-1
agreement of 0.94 compounds over 40 steps.

Standing rule from this: **report KL and generation-identity, not EM, at 1B.**
Accuracy claims wait for 8B on hardware that fits it — the null here is a
property of the instrument, and it may well recover at 8B, which is precisely
why it cannot be assumed either way.

### 2026-08-23 — Phase 1 engine: W1.2–W1.4 and W1.6 landed

`lazy/sparse/{descriptors,router}.py`, behind `LAZY_SPARSE=0`. Suite green at
**95 passed, 3 skipped** (69 pre-existing, unchanged with the switch unset —
that is the R1 regression bar). Four departures from §3 as written, each forced
by the code rather than chosen:

- **W1.3's router cannot live in `LazyGPUModelRunner`.** `_prepare_inputs` runs
  *before* the forward pass, so the step's query does not exist yet. Routing
  moved to the patched `TritonAttentionImpl.forward`, where q is in hand — which
  is also where `d0_e2e.py` measured it, so the engine implements the object
  Phase 0 characterised. `LAZY_SPARSE_ROUTE_LAYERS={all,first,N}` keeps sharing
  available as an ablation; `all` is the default, because §9b's flat per-layer
  recall says each layer routes well *with its own query* and says nothing about
  whether layer 2's query routes for layer 15.
- **Page-level ships as the default, doc-level is the arm.** R2 makes page
  scores sufficient for both, so `LAZY_SPARSE_GRANULARITY={page,doc}` is one
  scorer and two selection rules. The doc arm takes documents *whole* (greedy by
  max-over-pages until the budget is spent); letting a token budget cut a
  document in half would make it neither level and useless as the G0-C
  comparison.
- **W1.2's eviction hook is impossible, and unnecessary.** `core/block_pool.py`
  runs in the EngineCore process; the descriptors are worker GPU tensors, so no
  call crosses. It is also not needed: a descriptor is a pure function of its
  block's contents, a document block is only ever written by a document
  request's prefill, and that path always fills. Freed-and-reused → refilled;
  prefix-cache hit → contents identical, so the box still holds; never described
  → `valid` is False and the router scores it `+inf`, i.e. reads it. A lifecycle
  bug degrades to dense, never to reading one document against another's
  statistics. The argument is guarded by `LAZY_SPARSE_DESC_VERIFY=1`, which
  recomputes and compares.
- **W1.4 needs a signature change after all.** `chunked_prefill_paged_decode`
  feeds one `seq_lens` to both `context_attention_fwd` and the decode kernel, so
  a mixed prefill+decode batch cannot share a compacted tensor. Added
  `decode_block_table` / `decode_seq_lens`, defaulted.

**A convention that was about to become a silent measurement bug.**
"Always keep doc 0" is only the *preamble* convention when the corpus puts the
preamble in block 0 — which `benchmarks/lazyroute/corpus.py` does and
`lazy_block_infer.py` does not. Left implicit, a corpus of the second kind
reads one real document free *and* shrinks the budget denominator, so its
numbers would not be comparable to §9b's. Now `LAZY_SPARSE_KEEP_DOC0`, default
on to match the Phase-0 tables.

**Correctness evidence, in increasing order of what it rules out:**

| check | result |
| --- | --- |
| budget=∞ walk table vs packed table | bit-identical, incl. `seq_lens` (T1) |
| de-rotation vs `llama_v1.py` transcribed elementwise | exact at offsets 1–513 |
| real Triton kernel over a walk table vs torch attention over exactly those keys | agrees at budgets 1.0 / 0.5 / 0.25 / 0.1 (T2) |
| `lazy_block_infer.py --mode lazy`, budget=1.0 | reproduces dense output verbatim |
| `lazy_block_infer.py`, budget 0.34 | still answers the two-hop question correctly |

T2 is the one that matters: it is invariant 6 ("sparse output = exact subset
attention") checked against the kernel rather than argued, and it fails if a
rotation, a `q_mask`, or the `seq_lens` arithmetic is off by anything.

**Two measurement bugs of my own, found before the numbers were believed.**
First pass reported gold containment 1/8 against dense 1/8 at 32 generated
tokens. Both were artefacts: these checkpoints restate the question before
answering, so 32 tokens measured *truncation*, and raw lowercase containment is
not the repo's metric. Fixed by generating 128 tokens (dense EM then lands at
0.285, i.e. §9b's ~24–27% ceiling, which is the sanity check that the
instrument is working) and by importing `blockbench.py::qa_em_score` — subspan
EM — rather than reimplementing it. `sparse_smoke.py` now also reports
`HIT_TOKEN_LIMIT`, so truncation cannot masquerade as a routing result again.

**Two performance bugs, both mine, and the second one dominated everything.**
The router first ran ~12× slower than dense decode. Host syncs (`.item()`,
`nonzero`, boolean-mask indexing) inside `_select`/`compact`, once per layer per
step, were part of it and are gone — selection is now a full sort plus a
scatter, and the statistics live on the device until a log line is due. But the
real cost was that **the packed block table is sized for `max_model_len`**:
8192 columns at this model's 131k context, against roughly a hundred a
ten-document request actually fills. The router scored the padding. Narrowing
to the populated prefix — a count the runner already tracks on the host for the
packed-table rebuild — took 200 examples from >37 minutes to ~70 seconds.
`tests/sparse` pins that narrowing is invisible to the walk.

**Engine-side results (1B, 2wiki, M=10, 200 examples, 128 tokens, batch 8).**
Costs are `kept_fraction` — the share of cached rows a decode step reads —
because the nominal budget is not comparable across granularities.

| arm | kept | subspan EM | identical to dense | dense-only / sparse-only |
| --- | --- | --- | --- | --- |
| dense | 1.000 | 57/200 (0.285) | — | — |
| page 0.25, stripe=selected | 0.352 | 54/200 (0.270) | 62/200 | **7** / 4 |
| prefix 0.16 | 0.337 | 59/200 (0.295) | 65/200 | **3** / 5 |
| prefix 0.17 | 0.348 | 62/200 (0.310) | 54/200 | **2** / 7 |
| doc 0.25 | 0.277 | 55/200 (0.275) | 42/200 | 7 / 5 |

Read the paired McNemar columns, not the rates: at 1B an EM *rate* adjudicates
nothing (§9b — random selection once "beat" dense by 3.3 points), and prefix
0.17 scoring above dense is that same artefact, not evidence prefix beats dense.
What does carry is **how many dense-correct answers a policy breaks**: page
breaks 7, prefix breaks 2–3 at equal or lower cost, consistently across two
budgets.

**A new granularity, and the mechanism behind it (`LAZY_SPARSE_GRANULARITY=prefix`).**
Proposed on the observation that pages of a document are not exchangeable:
if page b3 is worth reading, b1 and b2 should come with it. The failures page
selection produces say exactly why. `corpus.py::format_document` writes
`- Title: {title}` at the head of every block, so **a document's title lives in
its page 0** — and page-level selection happily keeps a body page while dropping
the head:

- gold `Tangled Destinies`: dense answers "*Tangled Destinies* is a 1954 3D
  Technicolor Western directed by William Castle"; sparse answers "*Jesse James
  vs. the Daltons* is a 1954 3D Technicolor Western directed by William Castle".
  Correct facts, wrong entity.
- gold `New York`: dense finds the director and his birthplace; sparse reports
  the director "is not explicitly mentioned".

Prefix closure prevents this structurally, and subsumes the sink stripe (page 0
is in every non-empty prefix). It also explains why the stripe did not already
cover it: `stripe_guard_trips` shows §9b's budget guard disabling the stripe on
~56% of requests — it withdraws precisely when the budget is tight, which is the
regime the method targets. Closure spends past the nominal budget, so arms are
only comparable at matched `kept_fraction`; the rows above are matched.

Not promoting `prefix` to the default on this alone: n=200 with 7-vs-3 flips is
directional, not decisive. What it deserves is the 8B check and a matched-cost
sweep, and it should be measured against mass recall and KL, where §9b's gates
actually live.

Two things this does not yet do. `LAZY_SPARSE_REFRESH` raises unless `every` —
the cadence is gated on G0-D, which §9b still marks preliminary, so it is not
worth freezing. And the router's host cost, though no longer catastrophic, is
still unmeasured against the <5% budget; that is P3.3's question. That
measurement, and the Phase-1 exit check against §9b's offline 0.919 / 0.937,
are the next things owed.

### 2026-08-23 — the decode kernel was the bottleneck, not the cache reads

Phase 1's premise — that reading a quarter of the cache makes decode faster — was
being validated against a decode kernel about ten times slower than it needed to
be. Fixing the kernel inverts the result, so this entry records the fix, the
overhead measurement §10.6 asked for, and what the two together say.

**The kernel.** `kernel_paged_attention_2d_llama` launches a grid of
`(num_seqs, num_kv_heads)`. At batch 1 on this 8-KV-head model that is **eight
thread blocks on a 70-SM card**, each walking the whole block table serially.
`llama_split.py` adds the flash-decoding form — the walk is cut into `NUM_SPLITS`
ranges whose partial softmaxes are merged by rescaling to a common max — selected
by `LAZY_SPLIT_KV`. `llama_v1.py` is untouched, so R1 holds. Per layer at 600
cached documents (~78k tokens):

| arm | serial | split | |
| --- | ---: | ---: | ---: |
| Lazy-Attn | 3.931 ms | 0.364 ms | 10.8x |
| LazyRoute | 1.322 ms | 0.182 ms | 7.3x |

Two lazy-specific details had to survive the cut and are the reason
`tests/kernels/test_split_decode.py` exists: each split restarts its
`prev_rot_offset`, so it re-rotates at its own first row rather than inheriting
the previous split's Q; and `q_mask` and the sequence boundary are indexed by the
**absolute** row, so split ranges pass absolute indices through. Eight cases hold
the split kernel to the serial one across ragged padding, exact blocks, one long
document, many short ones, walks shorter than the split count, and non-lazy rows.

Removed alongside it: `torch.all(is_lazy).item()` ran on **every lazy decode
layer** to compute a boolean that a default-off switch then discarded. That is a
device-to-host sync per layer per step, and it also made the decode path
uncapturable, since a sync is illegal inside a CUDA graph.

**Router overhead, the measurement §10.6 owed.** 1B, 600 documents, batch 1,
split kernel on, two examples forced to 96 tokens each so every arm decodes the
same number of steps (`demo_speedup.py --arm route --route-layers ...`):

| route calls / step | ms/token | over Lazy-Attn |
| --- | ---: | ---: |
| 0 (Lazy-Attn) | 10.6 | — |
| 1 (`first`) | 17.2 | +6.6 |
| 4 (stride 4) | 24.9 | +14.3 |
| 14 (`all`) | 61.3 | +50.7 |

Linear in the call count at ~3.4 ms per call over ~3.2 ms/step of fixed geometry.
Against the <5% budget the stride-4 default costs **+135% of the decode step** —
off by a factor of about 27, and off by 62% even at one call per step.

The cost is **not arithmetic**. `router_profile.py --docs 600` puts the same call
at 0.853 ms of GPU time warm, 1.489 ms cold; wall clock is ~4x that. This agrees
with the earlier control experiment — replaying the identical kernels from a
captured CUDA graph ran 8.2x faster at 10 documents and 1.7x at 600 — so what is
being paid for is dispatch, not work. Batch 4 and batch 8 were tried and did not
rescue it.

**The consequence.** Sparsity removes ~2.9 ms/token of GPU work at this corpus
size (14 routed layers x 0.182 ms saved). The router adds 14.3 ms/token of host
work to obtain it. With the serial kernel the saving was ~3.8 ms/*layer* and the
trade looked positive; with the kernel fixed, batch-1 decode is CPU-launch-bound
and the saving has nothing to buy. **At batch 1, on 1B, LazyRoute is a net loss
at every stride tested** — and it is a loss whose size is set by how many times
the router is invoked, not by how much cache it skips.

This does not touch the selection results. Mass recall, KL and the `prefix`
granularity findings above are statements about *which* pages are chosen and are
unaffected by what the walk costs. It does move the burden onto P3.3: the three
things that could change the arithmetic are CUDA-graph capture of the router (the
control experiment bounds the win at ~1.7x here), batch sizes past 8, and models
where per-layer attention is a larger share of the step than it is at 1B.

**Demo.** `benchmarks/lazyroute/demo_speedup.py` measures three arms — each alone
on the GPU, driving `LLMEngine.step()` with `RequestOutputKind.DELTA` so every
point is a token that actually arrived — and renders them side by side. At 600
documents served in a new order:

| arm | TTFT | ms/token | total |
| --- | ---: | ---: | ---: |
| vLLM prefix caching | 77,104 ms | 10.79 | 78.34 s |
| Lazy-Attn | 132 ms | 10.38 | 1.13 s |
| LazyRoute (kept 0.252) | 139 ms | 24.25 | 2.55 s |

**583x to the first token, and 1.04x per token after it.** The panels show token
counters, measured rates and cache-read fraction rather than generated text: at
600 documents the 1B checkpoint's output is degenerate (§9b — it is incoherent
past roughly 20 documents), and the timings are real whether or not the text is.

### 2026-08-23 — selection is shared across layers by default, and it is free

`LAZY_SPARSE_ROUTE_LAYERS=first` — route once per step, share the decision with
every layer below — was gated as an ablation because sharing "is only sound if
selection transfers across layers, and that is untested". It is now tested, and
it is now the default. 2wiki, 1B, 200 examples, 10 documents, budget 0.25
(`sparse_smoke.py --compare`), each arm paired against the same dense run:

| arm | route calls / step | subspan EM | identical to dense | broke | fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | — | 57/200 | — | — | — |
| `all` | 14 | 55/200 | 55/200 | 7 | 5 |
| stride 4 | 4 | 59/200 | 74/200 | 3 | 5 |
| **`first`** | **1** | **57/200** | 54/200 | 4 | 4 |

Sharing costs nothing against the per-layer object Phase 0 characterised: same
identity to dense (54 vs 55), fewer dense-correct answers broken (4 vs 7), EM
landing exactly on dense. This is consistent with §9b's earlier finding that
per-layer recall is *flat* — but note it does not follow from it, which is why
it needed measuring: flat per-layer recall says each layer routes well with its
**own** query, not that one layer's query routes well for another.

The stride-4 row is the curiosity: it is the most faithful of the three by
identity (74/200), beating both routing every layer and routing once. Worth an
explanation eventually — per-layer routing gives each layer a *different* mask,
so a page dropped at layer L returns at L+1, and the effective mask wanders down
the stack — but not worth blocking on. `first` is chosen because the router's
cost is linear in the call count and nothing in the accuracy data argues for
paying for more calls.

**Why the call count is the only knob.** `router_profile.py --ops` now reports
the launch budget directly. One route call at 600 documents:

```
per route call: 477 host ops issuing 221 CUDA kernels
                2783 us on the host, 785 us on the GPU (3.5x)
                5.8 us of host time per op
```

The host is an Intel Core Ultra 5 225F at 3.26 GHz and a trivial Triton launch
costs 10.8 us here — WSL2's paravirtualised driver is maybe 2x a native host,
which is a factor of two, not the factor of ten in question. The router is 477
dispatches because it is elementwise PyTorch over `[1, 5404]` tensors and each
line is its own kernel; the arithmetic is trivial (the fused Quest scorer does
all the scoring in 42 us). **Scoring is fused; selection is not** — top-k,
compaction and the walk-table build are ~470 of the 477 ops. That is the next
lever, and CUDA-graph capture is the other.

### 2026-08-23 — the speed result depends on corpus size, and 10 documents is the wrong end

With once-per-step routing and the split decode kernel, all three arms
re-measured (`demo_speedup.py`, each alone on the GPU, 96 forced tokens):

| | dense TTFT | lazy TTFT | dense ms/tok | lazy ms/tok | route ms/tok | kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 docs (n=9) | 43 ms | 11 ms | 8.11 | 9.61 | 13.69 | 33.0% |
| 600 docs (n=2) | 11,327 ms | 129 ms | 9.01 | 10.75 | 9.89 | 25.2% |

**LazyRoute vs Lazy-Attn is +4.08 ms/token at 10 documents and −0.86 ms/token at
600.** The router's cost is flat in corpus size — 477 dispatches whatever the
table length — while the attention it removes is proportional to it. Linear
through the two points puts break-even near **500 documents**. The earlier
estimate of ~80 was taken before the split kernel; making attention 10x cheaper
moved break-even out by roughly the same factor, because there is now 10x less
to save.

Two consequences worth stating plainly. **At 10 documents LazyRoute is a clear
loss** (0.70x), and 10 documents is where the 1B checkpoint is coherent and
where all the accuracy work above lives — so the regime that validates selection
quality and the regime that could show a speedup do not currently overlap on
this hardware. And **Lazy-Attn's decode is slower than stock vLLM's** at both
sizes (0.84x, ~1.5 ms/token), so its win is entirely prefill: at 10 documents it
pays back its decode penalty after ~22 generated tokens and loses on a 96-token
answer (0.80 s dense vs 1.10 s lazy); at 600 documents it pays back after ~6,400
and effectively always wins.

Also fixed here: `summarise` took `ttfts[len//2]` after sorting, which for an
even count is the *upper* of the two middle values, not the median — one cold
prefill at 600 documents (77 s against a steady-state 11.3 s) became the
reported TTFT and an inflated 583x headline. It now uses `statistics.median`,
and the corrected figure at 600 documents is **88x**.

### 2026-08-23 — long documents do not rescue it, and two earlier numbers were noise

**The idea.** The router's cost tracks blocks and documents; the model's
coherence tracks documents; §9b puts the 1B checkpoint's limit near 20. So ten
documents of six thousand tokens is the same cache as six hundred of one
hundred and thirty, with a document count the checkpoint can still answer over
— the shape where the accuracy regime and the speed regime finally overlap.
`corpus.py::lengthen` builds it, placing each supporting paragraph *inside* a
document rather than at its head, since head placement puts every answer in page
0, which the sink stripe and prefix closure both keep for free.

**Half of the mechanism is real.** One route call costs 1.095 ms at 10x6k
against 1.799 ms at 600x130 (`router_profile.py --doc-tokens`): 1.6x cheaper,
because `derotate_query` and the per-document tables scale with document count.
And sparsity works exactly as designed at this shape — the decode kernel drops
from 0.310 to 0.134 ms/layer (n=400, in-process CUDA events), a real 2.8
ms/token of GPU work removed.

**The other half is not.** Paired adjacent runs, three rounds, batch 1:

| shape | docs | tokens/doc | total | blocks | Lazy-Attn | LazyRoute | Δ ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 600 | 130 | 78k | 5,400 | 10.12 | 10.94 | **+0.8** |
| long | 10 | 6.1k | 61k | 3,794 | 10.30 | 12.11 | **+1.8** |
| longer | 10 | 12.1k | 121k | 7,576 | 13.65 | 18.21 | **+4.6** |

Longer documents are *worse*, and doubling the length again is worse still. The
per-document term the reshaping attacks is not the dominant one: the dominant
term is 477 host dispatches per call, and that scales with the block count,
which grows however you grow the cache. **There is no corpus shape that fixes
this** — fusing selection is the only lever, and it is now the gating item.

**Measurement protocol, learned the hard way.** This card boosts between 850 and
3090 MHz and cannot be clock-locked under WSL2 (`nvidia-smi -lgc` fails). The
LazyRoute arm's run-to-run spread is 17–36%; Lazy-Attn's is 1–11%, because the
router's cost is host time and host time is what varies. A single arm measured
back-to-back drifted from 1066 to 1827 ms across identical examples. Only
**paired adjacent runs** — the two arms measured next to each other, differenced
within a round — are trustworthy at these margins. `demo_speedup.py --rounds N`
now runs the arms round-robin for this reason.

**Two corrections.** Both of these were reported earlier today from
non-interleaved runs and both are wrong:

- "LazyRoute 9.89 against Lazy-Attn's 10.75 at 600 documents, the first net
  win." There is no win. Paired, it is 10.94 against 10.12 — a 0.8 ms/token
  **loss**. The apparent win was thermal drift between sequential arms.
- "LazyRoute is 0.55x at ten long documents." Paired, it is 0.85x. The 18.00
  ms/token behind that figure was a contaminated round; the same configuration
  measured 10.3 twenty minutes later.

The standing conclusion is unchanged and now better supported: **at batch 1,
LazyRoute is a net loss at every corpus shape measured**, by 0.8 to 4.6
ms/token, while removing 2.8 ms/token of real GPU work. The gap is host
dispatch, not arithmetic.

### 2026-08-23 — selection fused into one kernel; sparsity finally nets out positive

Scoring was already one launch. Everything after it — ranking candidates,
applying the budget and the sink stripe, compacting kept rows left, rebuilding
`seq_lens` — was ~470 elementwise PyTorch operations over `[R, W]` tensors, and
that dispatch, not the arithmetic, was the whole cost. `lazy/sparse/selection.py`
does all of it in one program per request.

**It collapses because the table has a hard width bound.** Sized for
`max_model_len`, at 131k context and block 16 that is 8192 columns, so a row's
scores are at most **32 KB of fp32** — small enough to live in one program's
registers. Selection without a sort: map the float to an order-preserving
unsigned key (sign-magnitude flip) and bisect it 32 times, each iteration a
masked count of "how many candidates score at least this". That lands exactly on
the budget-th largest key. Ties are broken by lower index; the torch path leaves
them to its radix sort, so the two agree whenever scores are distinct and both
are valid when they are not.

**The launch budget, same measurement as before:**

| | before | after |
| --- | ---: | ---: |
| host ops per route call | 477 | **96** |
| CUDA kernels per call | 221 | **34** |
| host time per call | 2783 us | **568 us** |
| GPU time per call | 785 us | 542 us |
| host / GPU | 3.5x | **1.0x** |
| route call (cold geom) | 1.799 ms | **0.431 ms** |

The router is no longer launch-bound. `derotate_query` is now the largest single
phase at 0.272 ms of the 0.431, and is the obvious next thing to fold into the
scorer.

**It is the same selection.** 24 cases in `tests/sparse/test_selection.py` hold
the kernel to the torch path across budgets, stripe and `keep_doc0` settings,
absolute token budgets, non-routable rows, the guard's discontinuity, and ties.
End to end: 100 2wiki examples at 10 documents, fused against torch,
**100/100 identical generations** and identical EM. `kept_fraction` is 0.252 on
both paths at 600 documents. The torch implementation is untouched and reachable
via `LAZY_SPARSE_FUSED_SELECT=0`, which is how a suspected kernel bug gets
bisected.

**And it flips the sign.** Paired adjacent runs, three rounds, 600 documents:

| round | Lazy-Attn | LazyRoute | Δ |
| --- | ---: | ---: | ---: |
| 1 | 11.58 | 9.15 | −2.43 |
| 2 | 10.38 | 9.97 | −0.41 |
| 3 | 11.07 | 9.94 | −1.13 |

Median **−1.13 ms/token, faster in three rounds of three**, against **+0.8**
under the same protocol before the fusion. This is the first configuration in
which reading a quarter of the cache is faster than reading all of it, and it
took fixing the decode kernel first (which removed the fake headroom sparsity
had been credited with) and then removing the router's dispatch.

At 10x6k the same change moves +1.8 to roughly a wash (−0.74 and +1.23 in the
two clean rounds; the third was thermally contaminated). Fewer blocks — 3,794
against 5,400 — means proportionally less to save against a router cost that is
now small but not zero. The break-even estimate of ~500 documents from the
previous entry no longer applies and has not been re-derived.

**The demo, re-measured** (`analysis/lazyroute_demo.gif`, 600 documents, three
round-robin rounds, medians):

| arm | TTFT | ms/token | total | cache read |
| --- | ---: | ---: | ---: | ---: |
| vLLM prefix caching | 11,158 ms | 9.05 | 11.95 s | 100% |
| Lazy-Attn | 131 ms | 10.53 | 1.13 s | 100% |
| LazyRoute | 134 ms | 9.36 | **1.01 s** | 25.2% |

**85x to the first token, then 1.13x per token reading a quarter of the cache** —
the framing this figure was originally built for in the morning and could not
honestly carry until now. Two rendering faults fixed with it, both of the same
kind: the panels printed each arm's median but filled their bars from example 0,
so a frame showed LazyRoute trailing while the final card claimed it was faster,
and "total request time" was that one example's while the rate above it was the
median, so the two did not multiply out. The animated example is now chosen as
the one closest to typical in every arm at once, and every printed number is a
median.

### 2026-08-23 — real text, at 8B, and what it costs LazyRoute

The figure could not show generated text because the 1B checkpoint stops
producing words at about twenty documents. Confirmed properly this time: at 50,
100, 200 and 600 documents it emits repetition, **dense included** (`Question`
repeated forty times), so it is the checkpoint and not the sparse path; long
documents do not help (10x60 paragraphs degenerates too), so the limit is
context length rather than document count; and a copy-one-string needle task
dies at the same place, so it is not task difficulty either.

`ldsjmdy/Tulu3-Block-FT` (8B) fixes it, and fits on a 16 GB card only through
bitsandbytes: fp8 quantises *after* the bf16 weights reach the GPU, so its peak
is the unquantised 15 GB and it OOMs in the embedding loader, while
bitsandbytes quantises during load — 5.65 GiB of weights, 7.66 GiB of KV cache,
62,784 tokens. At 300 documents it answers properly, with the three arms'
actual output streaming.

| arm | TTFT | ms/token | total | answer |
| --- | ---: | ---: | ---: | --- |
| vLLM prefix caching | 29,693 ms | 40.7 | 32.15 s | finds Xawery Zulawski |
| Lazy-Attn | 187 ms | 27.0 | 1.87 s | finds Xawery Zulawski |
| LazyRoute | 617 ms | 30.1 | 2.50 s | **"is not mentioned"** |

**159x to the first token, and on 8B Lazy-Attn also beats dense per token,
1.51x** — it did not at 1B (0.86x), because thirty-two layers of attention over
43k tokens is a large enough share of the step for the packed walk to pay.

**LazyRoute costs an answer here, and time.** At a 25% budget it drops the page
naming the director and confabulates a different film; on the needle task it
finds the right document and quotes its *first* sentence while dropping the page
holding the code — §9b's block-head finding reproducing at 8B, and an argument
for `prefix` closure that is now much stronger than the n=200 flip count was.
It is also 0.90x on decode and 3.3x worse on TTFT (617 against 187 ms), the
latter being descriptor construction across thirty-two layers at prefill, which
has never been measured and is not free.

**Two caveats against reading the 0.90x as the 8B verdict.** bitsandbytes 4-bit
dequantises every linear on every step, which inflates the non-attention part of
the decode step and so shrinks sparsity's share of it — the comparison is
distorted *against* routing by an amount not measured here. And 300 documents is
2,700 blocks against the 5,400 at which routing paid on the 1B. Neither is a
reason to disbelieve the accuracy result, which is the more important one.

So the 8B GIF is **not kept**: 4-bit is the wrong instrument for a speed claim
and this card has no way to run the comparison without it. The recipe is
recorded in `benchmarks/lazyroute/demo_speedup.py`'s docstring, both as run here
and as it should be re-run on an H200 — bf16 with no `--quantization`, 1,200
documents, three rounds — and the numbers above stand only as the accuracy
result plus a lower bound on the prefill win. `analysis/lazyroute_demo.gif` (1B,
600 documents) stays, being an undistorted measurement on this hardware.

## 10. Immediate next actions (this week)

1. ~~Freeze the environment per `scripts/install.sh`; run the repo test suite on the 1B model; run
  `lazy_block_infer.py` end-to-end (baseline sanity).~~ **Done 2026-08-23 (§9b).**
2. Fix G0-A/G0-B/G0-C/G0-D thresholds in writing (edit §2 defaults if needed — then stop moving them).
  **G0-A needs restating first**: it is an EM-retention gate and 1B has no usable EM ceiling past
  ~10 documents (§9b), so either it moves to 8B hardware or it is restated on recall/KL.
3. Start D0.1/D0.2 logging harness off `block_infer.py`; queue 2wiki + LongBench-synthetic first
  (fastest signal on both existence and the sink question). Corpora and prompt templates now come
  from `benchmarks/lazyroute/corpus.py`.
4. ~~Land W1.1 (offset repack) behind tests — it blocks every large-M experiment.~~
  **Done 2026-08-23 (§9b).**
5. Send the author email (D0.6): code-drift expectations, 1B Block-FT recipe, benchmark-harness
  reuse. Add a question about the 1B checkpoint's usable document count — §9b puts it near 10.
6. ~~Land W1.2–W1.4 + W1.6 behind `LAZY_SPARSE`.~~ **Done 2026-08-23 (§9b).** What that
  leaves owed, in order: the **Phase-1 exit check** (engine mass recall at budget 0.25 on
  2wiki M=10 against §9b's offline 0.919 / 0.937 — agreement is what says the engine
  implements the measured object, and a gap is a router bug rather than a new result);
  then W1.5's oracle arm at scale. ~~The router overhead breakdown against the <5%
  budget.~~ **Done 2026-08-23 (§9b): it fails the budget by ~27x at the stride-4
  default, the cost is dispatch rather than arithmetic, and with the decode kernel
  fixed sparsity is a net loss at batch 1.** That makes P3.3 — CUDA-graph capture of
  the router — the gating item for the whole speed claim, ahead of further selection
  work.
