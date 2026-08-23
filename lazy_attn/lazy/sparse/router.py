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
from lazy.utils.rotation import MAX_PACKED_Q_OFFSET, PACKED_Q_OFFSET_SHIFT

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
    """
    half = rotary_dim // 2
    positions = (offsets.long() - 1).clamp(min=0)
    table = cos_sin_cache[positions]  # [R, M, rotary_dim]
    cos = table[..., :half].unsqueeze(2).float()  # [R, M, 1, D/2]
    sin = table[..., half:].unsqueeze(2).float()

    a = q[:, None, :, :half].float()  # [R, 1, H, D/2]
    b = q[:, None, :, half:].float()
    rotated = torch.cat([a * cos + b * sin, -a * sin + b * cos], dim=-1)
    # offset 1 (and the 0 sentinel) mean "already in the right frame".
    identity = (offsets <= 1)[:, :, None, None]
    return torch.where(identity, q[:, None, :, :].float(), rotated)


def _quest_bound(q: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Upper bound of q.k over the box, for every (block, kv_head).

    `max(q*lo, q*hi)` summed over the head dimension, rewritten as
    `q.centre + |q|.half_width` so it is two contractions instead of an
    elementwise max over a materialised `[blocks, heads, dim]` product.

    `q` is `[T, G, H, D]` (tile, GQA group member, kv head, dim) and `box` is
    `[T, H, 2, D]`; the result is `[T, G, H]`.
    """
    lo, hi = box[:, :, MIN, :], box[:, :, MAX, :]
    centre = ((lo + hi) * 0.5).unsqueeze(1)  # [T, 1, H, D]
    half_width = ((hi - lo) * 0.5).unsqueeze(1)
    return (q * centre).sum(-1) + (q.abs() * half_width).sum(-1)


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

            # q for each block, via its document: [R, T, H_q, D] -> grouped
            q_tile = torch.gather(
                q_by_doc, 1,
                tile_doc[:, :, None, None].expand(-1, -1, q_by_doc.shape[2],
                                                  q_by_doc.shape[3]))
            t = q_tile.shape[1]
            q_tile = q_tile.reshape(num_reqs * t, num_kv_heads, group,
                                    -1).transpose(1, 2)

            if self.config.scorer == "oracle":
                tile_score = self._oracle_score(key_cache, flat_phys, q_tile)
            elif self.config.scorer == "random":
                tile_score = torch.rand((num_reqs * t, group, num_kv_heads),
                                        device=phys.device)
            else:
                box, valid = self.store.boxes(layer_name, flat_phys)
                if box.numel() == 0:
                    scores[:, start:stop] = float("inf")
                    continue
                if self.config.scorer == "centroid":
                    tile_score = (q_tile *
                                  box[:, :, MEAN, :].unsqueeze(1)).sum(-1)
                else:
                    tile_score = _quest_bound(q_tile, box)

            # GQA aggregation, then per-kv-head max: a block is worth reading
            # if any head in the group wants it.
            if self.config.gqa_agg == "sum":
                per_head = tile_score.sum(1)
            else:
                per_head = tile_score.amax(1)
            tile_out = per_head.amax(-1).reshape(num_reqs, t)

            if self.config.scorer not in ("oracle", "random"):
                tile_out = torch.where(
                    valid.reshape(num_reqs, t), tile_out,
                    torch.full_like(tile_out, float("inf")))
            scores[:, start:stop] = tile_out
        return scores

    def _oracle_score(self, key_cache: torch.Tensor, flat_phys: torch.Tensor,
                      q_tile: torch.Tensor) -> torch.Tensor:
        """True max q.k over each block -- an auxiliary dense read, eval only."""
        keys = DescriptorStore._keys_by_row(key_cache, flat_phys).float()
        n, h, p, d = keys.shape
        group = q_tile.shape[1]
        qk = torch.einsum("nghd,nhpd->nghp", q_tile.reshape(n, group, h, d),
                          keys)
        return qk.amax(-1)

    # -- selection ----------------------------------------------------------

    def _select(self, scores: torch.Tensor, doc_id: torch.Tensor,
                block_valid: torch.Tensor, doc_mask: torch.Tensor,
                block_size: int) -> torch.Tensor:
        """`[R, MB]` bool: which rows this step walks.

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
        tail = block_valid & ~doc_mask
        if cfg.keep_doc0:
            preamble = doc_mask & (doc_id == 0)
            candidate = doc_mask & (doc_id > 0)
        else:
            preamble = torch.zeros_like(doc_mask)
            candidate = doc_mask

        num_candidates = candidate.sum(1).int()
        budget = cfg.budget_rows(num_candidates, block_size)

        if cfg.granularity == "doc":
            return preamble | tail | self._select_docs(
                scores, doc_id, candidate, budget)

        priority = torch.where(candidate, scores,
                               torch.full_like(scores, float("-inf")))

        stripe = self._stripe_rows(doc_id, candidate)
        if stripe is not None:
            # §9b's budget guard: below roughly twice its own cost the stripe
            # destroys more than it saves, which is exactly the large-M regime.
            # Counted on the device and reported by `_maybe_log`; asking here
            # whether it tripped would sync once per layer per step.
            affordable = (stripe.sum(1).int() * 2) <= budget
            stripe = stripe & affordable[:, None]
            self._stat_guard += (~affordable).sum()
            priority = torch.where(stripe, priority + _FORCED, priority)

        # Rank every candidate and keep each request's own prefix. A `topk`
        # would need `k` on the host, and masking its result with a boolean
        # index is data-dependent -- both sync. Sorting the full width and
        # scattering a per-request cutoff is the same selection without either.
        order = priority.argsort(dim=1, descending=True)
        within = (torch.arange(order.shape[1], device=scores.device)[None, :]
                  < budget[:, None])
        chosen = torch.zeros_like(candidate)
        chosen.scatter_(1, order, within)
        # A row with fewer candidates than budget would otherwise pick up
        # whatever the sort put after the -inf entries.
        chosen &= candidate

        if cfg.granularity == "prefix":
            chosen = self._close_prefixes(chosen, doc_id, candidate)

        return preamble | tail | chosen

    @staticmethod
    def _close_prefixes(chosen: torch.Tensor, doc_id: torch.Tensor,
                        candidate: torch.Tensor) -> torch.Tensor:
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
        width = chosen.shape[1]
        index = torch.arange(width, device=chosen.device).expand_as(chosen)
        safe_id = doc_id.clamp(min=0).long()

        highest = torch.full_like(index, -1)
        highest.scatter_reduce_(1,
                                safe_id,
                                torch.where(chosen, index,
                                            torch.full_like(index, -1)),
                                reduce="amax",
                                include_self=True)
        return candidate & (index <= highest.gather(1, safe_id))

    def _select_docs(self, scores: torch.Tensor, doc_id: torch.Tensor,
                     candidate: torch.Tensor,
                     budget: torch.Tensor) -> torch.Tensor:
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
        safe_id = doc_id.clamp(min=0).long()
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
            fits = spend <= budget[:, None]

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
                doc_mask: torch.Tensor, seq_lens: torch.Tensor,
                num_doc_blocks: torch.Tensor,
                block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather kept rows to the left, and rebuild `seq_lens` to match.

        `nonzero` yields indices in ascending (row, column) order, so the
        scatter below preserves the original row order within each request --
        which is invariant 1 and 2 in the module docstring, not an incidental
        property of the implementation.
        """
        # Kept rows sort ahead of dropped ones while both keep their original
        # order, because the sort key is the row index itself, shifted by the
        # table width for anything dropped. `argsort` rather than `nonzero`
        # deliberately: `nonzero`'s output size is data-dependent, so it forces
        # a device-to-host sync -- and this runs once per layer per decode
        # step, where a sync costs far more than the sort it saves.
        width = keep.shape[1]
        index = torch.arange(width, device=keep.device).expand_as(keep)
        rank = torch.where(keep, index, index + width).argsort(dim=1)
        walk = packed.gather(1, rank)

        kept_doc_rows = (keep & doc_mask).sum(1).int()
        tail_len = seq_lens.int() - num_doc_blocks * block_size
        return walk, kept_doc_rows * block_size + tail_len

    # -- entry point --------------------------------------------------------

    def route(self, layer_name: str, query: torch.Tensor,
              packed: torch.Tensor, seq_lens: torch.Tensor,
              doc_id: torch.Tensor, num_doc_blocks: torch.Tensor,
              doc_offsets: torch.Tensor, routable: torch.Tensor,
              query_start_loc: torch.Tensor, cos_sin_cache: torch.Tensor,
              rotary_dim: int, num_kv_heads: int, block_size: int,
              key_cache: Optional[torch.Tensor] = None,
              max_blocks: Optional[int] = None) -> WalkTable:
        """Compact `packed`/`seq_lens` for every routable row; pass others through.

        `routable` marks lazy rows taking a single-token decode step. Prefill
        rows and non-lazy rows keep their dense table untouched -- prefill
        sparsification is an explicit non-goal, and a non-lazy row has no
        document region to route over.
        """
        self._ensure_counters(packed.device)
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
        # Every row is scored and then discarded by `torch.where` if it was not
        # routable, rather than gathered up front. Selecting rows would mean
        # `nonzero`, whose output size is data-dependent and so syncs -- and
        # this runs per layer per decode step. A non-routable row is harmless
        # to score: its geometry says zero document blocks, so nothing is
        # indexed out of range and its result is thrown away below.
        phys = (packed >> 32).to(torch.int64)
        offsets = ((packed >> PACKED_Q_OFFSET_SHIFT)
                   & MAX_PACKED_Q_OFFSET).to(torch.int32)
        max_blocks = phys.shape[1]

        arange = torch.arange(max_blocks, device=phys.device)
        num_blocks = torch.div(seq_lens.int() + block_size - 1,
                               block_size,
                               rounding_mode="floor")
        block_valid = arange[None, :] < num_blocks[:, None]
        doc_mask = (arange[None, :] < num_doc_blocks[:, None]) & block_valid

        q = query[query_start_loc[:packed.shape[0]].long()]  # [R, H_q, D]
        q_by_doc = derotate_query(q, doc_offsets, cos_sin_cache, rotary_dim)

        scores = self._score_blocks(layer_name, q_by_doc, phys, doc_id,
                                    num_kv_heads, key_cache)
        scores = torch.where(doc_mask, scores,
                             torch.full_like(scores, float("-inf")))

        keep = self._select(scores, doc_id, block_valid, doc_mask, block_size)
        sub_walk, sub_lens = self.compact(packed, keep, doc_mask, seq_lens,
                                          num_doc_blocks, block_size)

        walk = torch.where(routable[:, None], sub_walk, packed)
        out_lens = torch.where(routable, sub_lens, seq_lens.int())

        rows_kept = torch.where(routable, keep.sum(1).int(),
                                block_valid.sum(1).int())
        rows_dense = block_valid.sum(1).int()
        self.stats["calls"] += 1
        self._stat_kept += rows_kept.sum()
        self._stat_dense += rows_dense.sum()
        self._maybe_log()
        return WalkTable(walk, out_lens, rows_kept, rows_dense)

    def _maybe_log(self) -> None:
        """Report walk lengths from inside the worker.

        The router lives in the EngineCore's worker process, so a caller in the
        parent cannot read `self.stats` -- it would see a router that never ran.
        Logging is the only channel that crosses, which is why this exists
        rather than an accessor.
        """
        if not self.profile_every:
            return
        if self.stats["calls"] % self.profile_every:
            return
        # The only sync in the router, and only every `profile_every` calls.
        self.stats["rows_kept"] = int(self._stat_kept)
        self.stats["rows_dense"] = int(self._stat_dense)
        self.stats["stripe_guard_trips"] = int(self._stat_guard)
        dense = self.stats["rows_dense"]
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
