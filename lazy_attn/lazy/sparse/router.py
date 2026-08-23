"""Per decode step, decide which cached pages are worth reading.

The output is a **walk table**: the persistent packed block table with rows
dropped, plus a matching `seq_lens`. Nothing else changes -- design rule R1
keeps allocation, hashing, eviction, preemption and the persistent table
untouched, so this file only ever produces a view.

Where this runs, and why not where the plan said. PROJECT.md's W1.3 places the
router in `LazyGPUModelRunner`, once per step for every sequence. That is not
implementable: the runner's `_prepare_inputs` runs *before* the forward pass, so
the step's query does not exist yet. The query first exists per layer, after
q_proj and RoPE, inside the patched `TritonAttentionImpl.forward` -- which is
also where `benchmarks/lazyroute/d0_e2e.py` measured routing, so running here
means the engine implements the object Phase 0 characterised. Sharing one
decision across layers is available as `LAZY_SPARSE_ROUTE_LAYERS=first`, but it
is an ablation rather than the default: §9b found per-layer recall *flat*, which
says each layer routes about as well with its own query, not that one layer's
query routes well for another.

The three invariants the compaction has to respect, all of them properties of
`kernel_paged_attention_2d_llama`:

1. **Rows stay in ascending original order** (documents, then tail). The kernel
   re-rotates Q only when `rot_offset` changes from one row to the next, so
   keeping a document's pages adjacent preserves that elision; score order would
   re-rotate on nearly every row.
2. **Tail rows keep their run.** Tail continuation blocks carry `q_offset == 0`,
   which the kernel reads as "keep the previous rotation" -- correct only after
   the `q_offset == 1` query block. Ascending order preserves the run, and every
   document row ahead of it carries an explicit nonzero offset, so dropping
   documents cannot disturb it.
3. **`seq_lens` is the walk length, not the sequence length.** The kernel walks
   `cdiv(seq_len, BLOCK_SIZE)` rows and masks the tail with
   `seq_offset < seq_len`, so a compacted row needs
   `kept_doc_rows * block_size + tail_len` -- which puts the boundary mask back
   in the final tail row, exactly where it was in the dense layout.

Budget convention: a share of the *routable* tokens, which excludes both the
preamble (document 0, kept free) and the dense tail. That is the denominator
§9b's recall tables use, and a Phase-1 exit criterion is that engine recall
matches those numbers -- a different denominator would silently break the
comparison rather than fail it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from lazy.sparse.descriptors import MAX, MEAN, MIN, DescriptorStore

# Ceiling on the elements materialised by one scoring tile, in fp32. The
# gather that broadcasts each document's query out to its pages is the router's
# only allocation that grows with *both* corpus size and batch size, so a tile
# measured in blocks is the wrong unit: 1024 blocks is 8 MB at batch 1 and
# 335 MB at batch 40, which on a 16 GB card is the difference between working
# and thrashing. Tiling on the product keeps the peak flat instead.
SCORE_TILE_ELEMENTS = 8 << 20  # 32 MB in fp32


def score_tile_blocks(num_reqs: int, num_q_heads: int, head_size: int) -> int:
    """Blocks per scoring tile, so the gather stays near `SCORE_TILE_ELEMENTS`."""
    per_block = max(num_reqs * num_q_heads * head_size, 1)
    return max(SCORE_TILE_ELEMENTS // per_block, 1)

# Large enough to dominate any Quest bound, small enough to stay finite so
# ordering among forced rows is still by score.
_FORCED = 1e30


@dataclass
class RouterConfig:
    budget_tokens: float = 0.25
    budget_docs: int = 0
    granularity: str = "page"
    scorer: str = "quest"
    route_layer_stride: int = 1
    dense_prefix_layers: int = 2
    gqa_agg: str = "max"
    sink_stripe: str = "selected"
    keep_doc0: bool = True

    def budget_rows(self, num_candidates: torch.Tensor,
                    block_size: int) -> torch.Tensor:
        """Rows a step may spend, per request.

        Rows are uniform 16-token pages, so a token budget *is* a row budget and
        no knapsack is involved -- which is why page selection is a flat top-k.
        A value above 1 is read as an absolute token count, at or below 1 as a
        share of the routable tokens.
        """
        if self.budget_tokens > 1.0:
            want = int(self.budget_tokens) // block_size
            return torch.minimum(torch.full_like(num_candidates, want),
                                 num_candidates)
        scaled = (num_candidates.float() * self.budget_tokens).round().int()
        return torch.minimum(scaled.clamp(min=1), num_candidates)


@dataclass
class WalkTable:
    """What the decode kernel is handed in place of the dense pair."""
    block_table: torch.Tensor  # [num_reqs, max_blocks] int64, packed entries
    seq_lens: torch.Tensor  # [num_reqs] int32
    rows_kept: Optional[torch.Tensor] = None  # [num_reqs] int32, for probes
    rows_dense: Optional[torch.Tensor] = None  # [num_reqs] int32, for probes


@dataclass
class RouteGeometry:
    """Everything about a step's layout that does not depend on the query.

    Which rows exist, which belong to documents, which are free, how many rows
    the budget buys, where the sink stripe would land: all of it is a function
    of the packed table and the step's document geometry, and none of it
    changes between layers. Recomputing it per layer was roughly forty kernel
    launches per layer per step spent producing sixteen identical answers --
    which matters here because the router is launch-bound, not bandwidth-bound
    (its CPU time was double its GPU time).

    Built once per step and cached on the step's attention metadata, the same
    object the walk table already caches on, and rebuilt with it.
    """
    packed: torch.Tensor  # narrowed to the populated prefix
    phys: torch.Tensor
    doc_id: torch.Tensor
    doc_offsets: torch.Tensor
    safe_doc_id: torch.Tensor  # doc_id clamped to >= 0, int64, for gathers
    index: torch.Tensor  # arange broadcast to the table's shape
    block_valid: torch.Tensor
    doc_mask: torch.Tensor
    candidate: torch.Tensor  # chargeable against the budget
    free_rows: torch.Tensor  # preamble | tail: kept regardless of score
    within: torch.Tensor  # rank < budget, for the selection scatter
    budget: torch.Tensor
    tail_len: torch.Tensor
    forced: Optional[torch.Tensor]  # sink-stripe bonus, or None if inactive
    block_size: int


def derotate_query(q: torch.Tensor, offsets: torch.Tensor,
                   cos_sin_cache: torch.Tensor,
                   rotary_dim: int) -> torch.Tensor:
    """Undo the query's global RoPE, per document, exactly as the kernel does.

    `q` is `[R, H, D]` as it reaches attention -- rotated at its global
    position. `offsets` is `[R, M]`, the packed `q_offset` of each document,
    which stores the absolute rotation position with a +1 bias (0 is a
    sentinel, 1 means "leave Q alone"). The result is `[R, M, H, D]`: the query
    as seen from inside each document's local frame, which is the frame the
    stored keys and therefore the descriptors live in.

    This mirrors `llama_v1.py` lines 223-248. Written out, the kernel's
    `q1`/`q2` construction broadcasts each half of Q across both halves, so the
    whole rotation collapses to a half-width pair of terms.

    Two shapes of this are load-bearing for speed, since this runs once per
    layer per decode step and the router is launch-bound rather than
    bandwidth-bound:

    * The halves are written into one preallocated buffer instead of being
      `torch.cat`ed. The cat was a single 121 us kernel at M=600 -- 10% of the
      router's entire GPU time -- spent copying tensors that were just built.
    * There is no `torch.where` for the identity case. `offsets <= 1` maps to
      table position 0, where a real RoPE table has cos=1 and sin=0, so the
      arithmetic below is already bit-exactly the identity there. The old
      explicit select was a second full-width pass to reproduce what the table
      already gives. `tests/sparse/conftest.py::rope_table` builds a real table
      for exactly this reason, and `test_derotation_is_identity_at_offset_one`
      pins it.
    """
    half = rotary_dim // 2
    positions = (offsets.long() - 1).clamp(min=0)
    table = cos_sin_cache[positions].float()  # [R, M, rotary_dim]
    cos = table[..., :half].unsqueeze(2)  # [R, M, 1, D/2]
    sin = table[..., half:].unsqueeze(2)

    a = q[:, None, :, :half].float()  # [R, 1, H, D/2]
    b = q[:, None, :, half:].float()
    out = torch.empty((q.shape[0], offsets.shape[1], q.shape[1], rotary_dim),
                      device=q.device,
                      dtype=torch.float32)
    torch.add(a * cos, b * sin, out=out[..., :half])
    torch.sub(b * cos, a * sin, out=out[..., half:])
    return out


def _quest_bound(q: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Upper bound of q.k over the box, for every (block, kv_head).

    `sum_d max(q_d*lo_d, q_d*hi_d)`, rewritten as `q.centre + |q|.half_width`
    so that each term is a contraction over the head dimension rather than an
    elementwise max over a materialised product.

    Both contractions are issued as batched matrix-vector products. Written the
    obvious way -- `(q * centre).sum(-1)` -- torch materialises a full
    `[blocks, group, heads, dim]` intermediate per term, which at M=600 is 44 MB
    written and read back twice; `aten::mul` and `aten::sum` together were 38%
    of the router's GPU time. Folding the broadcast into the GEMV removes the
    intermediate entirely.

    `q` is `[N, H, G, D]` (block, kv head, GQA group member, dim) -- contiguous,
    which is why `_score_blocks` no longer transposes into group-major order --
    and `box` is `[N, H, 2, D]`. The result is `[N, H, G]`.
    """
    n, h, g, d = q.shape
    lo, hi = box[:, :, MIN, :], box[:, :, MAX, :]
    centre = ((lo + hi) * 0.5).reshape(n * h, d, 1)
    half_width = ((hi - lo) * 0.5).reshape(n * h, d, 1)
    flat = q.reshape(n * h, g, d)
    bound = torch.baddbmm(torch.bmm(flat, centre), flat.abs(), half_width)
    return bound.reshape(n, h, g)


class Router:
    """Scores cached pages and compacts the packed block table.

    One instance per worker, held by the attention backend. State is only the
    descriptor store and the per-step cross-layer cache; nothing here survives
    a step, so batch reordering cannot stale it.
    """

    def __init__(self,
                 config: RouterConfig,
                 store: DescriptorStore,
                 profile_every: int = 0):
        self.config = config
        self.store = store
        self.profile_every = profile_every
        # Walk-length accounting. The decode kernel reads whole rows, so rows
        # kept over rows dense *is* the bytes/token ratio for the cached part
        # of attention -- the number P2.3 has to report, and the cheapest
        # possible check that routing is actually dropping anything.
        #
        # The totals live on the device and are read back only when a log line
        # is due. Accumulating them into Python ints would sync once per layer
        # per decode step, which costs more than everything this file does.
        self.stats = {"calls": 0, "rows_kept": 0, "rows_dense": 0,
                      "stripe_guard_trips": 0}
        self._stat_kept: Optional[torch.Tensor] = None
        self._stat_dense: Optional[torch.Tensor] = None
        self._stat_guard: Optional[torch.Tensor] = None

    def _ensure_counters(self, device: torch.device) -> None:
        if self._stat_kept is None:
            zero = lambda: torch.zeros((), dtype=torch.long, device=device)
            self._stat_kept, self._stat_dense, self._stat_guard = (zero(),
                                                                   zero(),
                                                                   zero())

    # -- scoring ------------------------------------------------------------

    def _score_blocks(self, layer_name: str, q_by_doc: torch.Tensor,
                      phys: torch.Tensor, doc_id: torch.Tensor,
                      num_kv_heads: int,
                      key_cache: Optional[torch.Tensor]) -> torch.Tensor:
        """`[R, MB]` score per block, aggregated across the GQA group.

        Undescribed blocks score `+inf`: a lifecycle bug degrades this into a
        dense read rather than into scoring one document's keys against
        another's statistics.
        """
        num_reqs, max_blocks = phys.shape
        group = q_by_doc.shape[2] // num_kv_heads
        scores = torch.full((num_reqs, max_blocks),
                            float("-inf"),
                            device=phys.device,
                            dtype=torch.float32)

        tile_blocks = score_tile_blocks(num_reqs, q_by_doc.shape[2],
                                        q_by_doc.shape[3])
        for start in range(0, max_blocks, tile_blocks):
            stop = min(start + tile_blocks, max_blocks)
            tile_phys = phys[:, start:stop]
            tile_doc = doc_id[:, start:stop].clamp(min=0).long()
            flat_phys = tile_phys.reshape(-1)

            # q for each block, via its document: [R, T, H_q, D]. Left in kv
            # head-major order `[N, H, G, D]` rather than transposed into
            # group-major: this is the layout `_quest_bound`'s GEMV wants, and
            # reaching it costs a reshape of an already-contiguous tensor
            # instead of a copy.
            q_tile = torch.gather(
                q_by_doc, 1,
                tile_doc[:, :, None, None].expand(-1, -1, q_by_doc.shape[2],
                                                  q_by_doc.shape[3]))
            t = q_tile.shape[1]
            q_tile = q_tile.reshape(num_reqs * t, num_kv_heads, group, -1)

            if self.config.scorer == "oracle":
                tile_score = self._oracle_score(key_cache, flat_phys, q_tile)
            elif self.config.scorer == "random":
                tile_score = torch.rand((num_reqs * t, num_kv_heads, group),
                                        device=phys.device)
            else:
                box, valid = self.store.boxes(layer_name, flat_phys)
                if box.numel() == 0:
                    scores[:, start:stop] = float("inf")
                    continue
                if self.config.scorer == "centroid":
                    n, h, g, d = q_tile.shape
                    tile_score = torch.bmm(
                        q_tile.reshape(n * h, g, d),
                        box[:, :, MEAN, :].reshape(n * h, d,
                                                   1)).reshape(n, h, g)
                else:
                    tile_score = _quest_bound(q_tile, box)

            # GQA aggregation over the group, then max over kv heads: a block is
            # worth reading if any head in the group wants it.
            if self.config.gqa_agg == "sum":
                per_head = tile_score.sum(-1)
            else:
                per_head = tile_score.amax(-1)
            tile_out = per_head.amax(-1).reshape(num_reqs, t)

            if self.config.scorer not in ("oracle", "random"):
                tile_out = torch.where(
                    valid.reshape(num_reqs, t), tile_out,
                    torch.full_like(tile_out, float("inf")))
            scores[:, start:stop] = tile_out
        return scores

    def _oracle_score(self, key_cache: torch.Tensor, flat_phys: torch.Tensor,
                      q_tile: torch.Tensor) -> torch.Tensor:
        """True max q.k over each block -- an auxiliary dense read, eval only.

        `q_tile` is `[N, H, G, D]`; the result is `[N, H, G]`, matching what
        `_quest_bound` returns so the aggregation below is scorer-agnostic.
        """
        keys = DescriptorStore._keys_by_row(key_cache, flat_phys).float()
        qk = torch.einsum("nhgd,nhpd->nhgp", q_tile, keys)
        return qk.amax(-1)

    # -- selection ----------------------------------------------------------

    def _geometry(self, packed: torch.Tensor, seq_lens: torch.Tensor,
                  doc_id: torch.Tensor, doc_offsets: torch.Tensor,
                  num_doc_blocks: torch.Tensor, block_size: int,
                  max_blocks: Optional[int]) -> RouteGeometry:
        """Derive a step's query-independent layout. See `RouteGeometry`.

        Kept free of charge: every tail row, and -- under the preamble
        convention, `LAZY_SPARSE_KEEP_DOC0` -- document 0. Charged against the
        budget: everything else in the document region, including forced
        sink-stripe rows.

        Whether document 0 is free is a property of how the *corpus* was
        submitted, not of the method: `benchmarks/lazyroute/corpus.py` sends the
        system preamble as document 0 and the Phase-0 tables exclude it from
        budget and mass, so the default matches those numbers. A corpus that
        keeps its preamble in the prompt template must turn the flag off, or a
        real document is read for free.
        """
        cfg = self.config
        # The block table is sized for `max_model_len`, not for what the batch
        # allocated. Narrowing it to the populated prefix is the single largest
        # lever in this file: at a 131k context the table is 8192 columns wide
        # and a ten-document request fills about a hundred of them. The kernel
        # reads `block_table_stride` off whatever tensor it is handed, so a
        # narrower table needs no padding back out.
        if max_blocks is not None:
            width = max(min(max_blocks, packed.shape[1]), 1)
            packed = packed[:, :width]
            doc_id = doc_id[:, :width]
            doc_offsets = doc_offsets[:, :width]

        phys = (packed >> 32).to(torch.int64)
        width = phys.shape[1]
        arange = torch.arange(width, device=phys.device)
        num_blocks = torch.div(seq_lens.int() + block_size - 1,
                               block_size,
                               rounding_mode="floor")
        block_valid = arange[None, :] < num_blocks[:, None]
        doc_mask = (arange[None, :] < num_doc_blocks[:, None]) & block_valid

        tail = block_valid & ~doc_mask
        if cfg.keep_doc0:
            preamble = doc_mask & (doc_id == 0)
            candidate = doc_mask & (doc_id > 0)
        else:
            preamble = torch.zeros_like(doc_mask)
            candidate = doc_mask

        budget = cfg.budget_rows(candidate.sum(1).int(), block_size)

        forced = None
        stripe = self._stripe_rows(doc_id, candidate)
        if stripe is not None:
            # §9b's budget guard: below roughly twice its own cost the stripe
            # destroys more than it saves, which is exactly the large-M regime.
            # Counted on the device and reported by `_maybe_log`; asking here
            # whether it tripped would sync once per layer per step.
            affordable = (stripe.sum(1).int() * 2) <= budget
            self._stat_guard += (~affordable).sum()
            forced = torch.where(stripe & affordable[:, None],
                                 torch.full_like(budget, _FORCED,
                                                 dtype=torch.float32)[:, None],
                                 torch.zeros((), dtype=torch.float32,
                                             device=packed.device))

        return RouteGeometry(
            packed=packed,
            phys=phys,
            doc_id=doc_id,
            doc_offsets=doc_offsets,
            safe_doc_id=doc_id.clamp(min=0).long(),
            index=arange.expand_as(phys),
            block_valid=block_valid,
            doc_mask=doc_mask,
            candidate=candidate,
            free_rows=preamble | tail,
            within=arange[None, :] < budget[:, None],
            budget=budget,
            tail_len=seq_lens.int() - num_doc_blocks * block_size,
            forced=forced,
            block_size=block_size)

    def _select(self, scores: torch.Tensor,
                geometry: RouteGeometry) -> torch.Tensor:
        """`[R, MB]` bool: which rows this step walks."""
        cfg = self.config
        if cfg.granularity == "doc":
            return geometry.free_rows | self._select_docs(scores, geometry)

        if geometry.forced is not None:
            scores = scores + geometry.forced
        priority = torch.where(geometry.candidate, scores,
                               torch.full_like(scores, float("-inf")))

        # Rank every candidate and keep each request's own prefix. A `topk`
        # would need `k` on the host, and masking its result with a boolean
        # index is data-dependent -- both sync. Sorting the full width and
        # scattering a per-request cutoff is the same selection without either.
        order = priority.argsort(dim=1, descending=True)
        chosen = torch.zeros_like(geometry.candidate)
        chosen.scatter_(1, order, geometry.within)
        # A row with fewer candidates than budget would otherwise pick up
        # whatever the sort put after the -inf entries.
        chosen &= geometry.candidate

        if cfg.granularity == "prefix":
            chosen = self._close_prefixes(chosen, geometry)

        return geometry.free_rows | chosen

    @staticmethod
    def _close_prefixes(chosen: torch.Tensor,
                        geometry: RouteGeometry) -> torch.Tensor:
        """Extend each document's selection down to its first page.

        If page 3 of a document is worth reading, pages 0-2 come with it. The
        motivation is §9b's block-head finding: 64.5% of cached attention mass
        sits on the first two tokens of each document, so a page-level pick that
        takes a late page and drops the document's head throws that mass away.
        Prefix closure subsumes the sink stripe -- page 0 is in every non-empty
        prefix -- which is why `sink_stripe` is redundant here.

        The pages of a document are contiguous and ascending in the block
        table, so "the prefix" is just "index <= the highest index chosen for
        this document".

        This *spends more than the nominal budget*: closure is applied after
        selection, so the budget is a floor rather than a ceiling. Compare arms
        by the reported `kept_fraction`, not by the budget they were asked for.
        """
        index, safe_id = geometry.index, geometry.safe_doc_id
        highest = torch.full_like(index, -1)
        highest.scatter_reduce_(1,
                                safe_id,
                                torch.where(chosen, index,
                                            torch.full_like(index, -1)),
                                reduce="amax",
                                include_self=True)
        return geometry.candidate & (index <= highest.gather(1, safe_id))

    def _select_docs(self, scores: torch.Tensor,
                     geometry: RouteGeometry) -> torch.Tensor:
        """Whole documents, greedily by score until the budget is spent.

        Design rule R2: a document scores as the max over its pages, so the
        page scores already computed are the sufficient statistic and no second
        scorer is needed. Selection takes documents *whole* -- letting the
        budget cut a document in half would make this arm neither doc-level nor
        page-level, and it is here to be the G0-C comparison.

        `LAZY_SPARSE_BUDGET_DOCS` overrides the derived budget with a document
        count, which is how the offline harness parameterises the same arm.
        """
        num_reqs, num_docs = scores.shape
        # One slot per block rather than per document: a document owns at least
        # one block, so this is an upper bound, and reading the true count off
        # the device would sync once per layer per step. The spare slots stay
        # at -inf and lose every comparison.
        safe_id = geometry.safe_doc_id
        candidate = geometry.candidate
        masked = torch.where(candidate, scores,
                             torch.full_like(scores, float("-inf")))
        per_doc = torch.full((num_reqs, num_docs),
                             float("-inf"),
                             device=scores.device)
        per_doc.scatter_reduce_(1, safe_id, masked, reduce="amax",
                                include_self=True)
        pages = torch.zeros((num_reqs, num_docs),
                            device=scores.device,
                            dtype=torch.int32)
        pages.scatter_add_(1, safe_id, candidate.int())

        order = per_doc.argsort(dim=1, descending=True)
        spend = pages.gather(1, order).cumsum(1)
        if self.config.budget_docs > 0:
            rank = torch.arange(num_docs, device=scores.device)
            fits = rank[None, :] < self.config.budget_docs
            fits = fits & (per_doc.gather(1, order) > float("-inf"))
        else:
            fits = spend <= geometry.budget[:, None]

        taken = torch.zeros_like(fits)
        taken.scatter_(1, order, fits)
        return candidate & taken.gather(1, safe_id)

    def _stripe_rows(self, doc_id: torch.Tensor,
                     candidate: torch.Tensor) -> Optional[torch.Tensor]:
        """Page 0 of each routable document, per `LAZY_SPARSE_SINK_STRIPE`.

        `selected` and `all` differ only in which documents get striped, and
        that difference is applied after selection for `selected`; forcing page
        0 up front is the same thing for a scorer whose misses are
        systematically the block-head pages (§9b), and it keeps selection a
        single top-k.
        """
        if self.config.sink_stripe == "off":
            return None
        first_of_doc = torch.ones_like(candidate)
        first_of_doc[:, 1:] = doc_id[:, 1:] != doc_id[:, :-1]
        return candidate & first_of_doc

    # -- compaction ---------------------------------------------------------

    @staticmethod
    def compact(packed: torch.Tensor, keep: torch.Tensor,
                doc_mask: torch.Tensor, tail_len: torch.Tensor,
                block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Move kept rows to the left, and rebuild `seq_lens` to match.

        Each row's destination is computed rather than sorted for: a kept row
        lands at its rank among kept rows, a dropped row after all of them. Both
        ranks are prefix sums, so the two together are a permutation and the
        scatter preserves the original order within each group -- which is
        invariants 1 and 2 in the module docstring, not an incidental property.

        A prefix sum rather than the `argsort` this used to do, and neither is
        `nonzero`: `nonzero`'s output size is data-dependent and so syncs, which
        is unaffordable once per layer per decode step, but the sort was doing
        `O(n log n)` comparisons to recover an order that is already known.
        """
        kept_rank = keep.cumsum(1)
        kept_total = kept_rank[:, -1:]
        dropped_rank = (~keep).cumsum(1) + kept_total
        # -1 because `cumsum` counts the row itself; both branches are therefore
        # 0-based and together cover [0, width).
        destination = torch.where(keep, kept_rank, dropped_rank) - 1
        walk = torch.empty_like(packed)
        walk.scatter_(1, destination, packed)

        kept_doc_rows = (keep & doc_mask).sum(1).int()
        return walk, kept_doc_rows * block_size + tail_len

    # -- entry point --------------------------------------------------------

    def route(self, layer_name: str, query: torch.Tensor,
              packed: torch.Tensor, seq_lens: torch.Tensor,
              doc_id: torch.Tensor, num_doc_blocks: torch.Tensor,
              doc_offsets: torch.Tensor, routable: torch.Tensor,
              query_start_loc: torch.Tensor, cos_sin_cache: torch.Tensor,
              rotary_dim: int, num_kv_heads: int, block_size: int,
              key_cache: Optional[torch.Tensor] = None,
              max_blocks: Optional[int] = None,
              step_cache: object = None) -> WalkTable:
        """Compact `packed`/`seq_lens` for every routable row; pass others through.

        `routable` marks lazy rows taking a single-token decode step. Prefill
        rows and non-lazy rows keep their dense table untouched -- prefill
        sparsification is an explicit non-goal, and a non-lazy row has no
        document region to route over.

        `step_cache` is the step's attention metadata, used to memoise the
        query-independent geometry across the layers of one step. Passing
        nothing is correct and simply recomputes it, which is what the tests do.

        Every row is scored and then discarded by `torch.where` if it was not
        routable, rather than gathered up front. Selecting rows would mean
        `nonzero`, whose output size is data-dependent and so syncs. A
        non-routable row is harmless to score: its geometry says zero document
        blocks, so nothing is indexed out of range and its result is thrown away
        below.
        """
        self._ensure_counters(packed.device)
        geometry = getattr(step_cache, "lazy_route_geometry", None)
        if geometry is None:
            geometry = self._geometry(packed, seq_lens, doc_id, doc_offsets,
                                      num_doc_blocks, block_size, max_blocks)
            if step_cache is not None:
                step_cache.lazy_route_geometry = geometry
        packed = geometry.packed

        q = query[query_start_loc[:packed.shape[0]].long()]  # [R, H_q, D]
        q_by_doc = derotate_query(q, geometry.doc_offsets, cos_sin_cache,
                                  rotary_dim)

        # Not masked to `doc_mask` here: `_select` admits only `candidate`,
        # which is a subset of it, so the mask would be a second full-width pass
        # to enforce what selection enforces anyway.
        scores = self._score_blocks(layer_name, q_by_doc, geometry.phys,
                                    geometry.doc_id, num_kv_heads, key_cache)

        keep = self._select(scores, geometry)
        sub_walk, sub_lens = self.compact(packed, keep, geometry.doc_mask,
                                          geometry.tail_len, block_size)

        walk = torch.where(routable[:, None], sub_walk, packed)
        out_lens = torch.where(routable, sub_lens, seq_lens.int())

        rows_kept = torch.where(routable, keep.sum(1).int(),
                                geometry.block_valid.sum(1).int())
        rows_dense = geometry.block_valid.sum(1).int()
        self.stats["calls"] += 1
        self._stat_kept += rows_kept.sum()
        self._stat_dense += rows_dense.sum()
        self._maybe_log()
        return WalkTable(walk, out_lens, rows_kept, rows_dense)

    def sync_stats(self) -> dict:
        """Materialise the device counters into `self.stats` and return it.

        Syncs, so this is for probes and log lines -- never the hot path. It is
        separate from `_maybe_log` because the counters used to be read back
        only when a log line was due: with `LAZY_SPARSE_PROFILE` unset, a
        caller reading `stats` got `rows_kept=0, rows_dense=0` from a router
        that had routed thousands of steps, which reads as "sparsity is off"
        rather than "nobody asked for the number yet".
        """
        if self._stat_kept is not None:
            self.stats["rows_kept"] = int(self._stat_kept)
            self.stats["rows_dense"] = int(self._stat_dense)
            self.stats["stripe_guard_trips"] = int(self._stat_guard)
        return self.stats

    def _maybe_log(self) -> None:
        """Report walk lengths from inside the worker.

        The router lives in the EngineCore's worker process, so a caller in the
        parent cannot read `self.stats` unless the worker is in-process
        (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). Logging is the channel that
        always crosses, which is why this exists alongside the accessor.
        """
        if not self.profile_every:
            return
        if self.stats["calls"] % self.profile_every:
            return
        dense = self.sync_stats()["rows_dense"]
        # `print`, not `logger`, and deliberately: this runs in the worker
        # process, and the repo's other worker-side probe (`LazyDecodeProfile`)
        # prints for the same reason.
        print(
            f"LazySparseRouter calls={self.stats['calls']} "
            f"rows_kept={self.stats['rows_kept']} rows_dense={dense} "
            f"kept_fraction="
            f"{(self.stats['rows_kept'] / dense) if dense else float('nan'):.4f} "
            f"stripe_guard_trips={self.stats['stripe_guard_trips']}",
            flush=True)
