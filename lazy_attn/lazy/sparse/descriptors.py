"""Min/max boxes over cached keys, one box per (layer, physical block, KV head).

The box is what the router scores against: for a query q, `sum_d max(q_d*min_d,
q_d*max_d)` upper-bounds q.k over every key in the block, so a block whose bound
is low cannot hold the step's attention mass and need not be read. This is
Quest's summary statistic, computed here in each document's *local* frame --
which is the whole point of the LazyRoute claim, since that frame is
request-independent and so the box is computed once per corpus rather than once
per request.

Two things about this file are less obvious than they look.

**Padding rows are excluded, and that is correctness, not tuning.** Documents
are right-padded to whole blocks with real pad tokens, so a document's last
block holds `block_size - len` genuine key vectors for `<pad>`. The decode
kernel masks them out via `q_mask`; folding them into the box would widen it
toward keys no query will ever attend to, and the widening lands on exactly one
page per document. (This is a different exclusion from the *sink* exclusion the
plan originally carried -- dropping each document's first 1-2 tokens -- which
was measured harmful and removed; see PROJECT.md §9b.)

**There is no invalidation hook, on purpose.** The plan called for one in
`core/block_pool.py`, but the block pool lives in the EngineCore process and
these tensors live on the worker's GPU, so no direct call can cross. It turns
out none is needed. A descriptor is a pure function of its block's contents, and
a document block is only ever written by a document request's prefill, which
always runs the fill path below. So:

  * block freed, re-allocated to another document request -> `fill` reruns and
    overwrites; fresh.
  * block freed, re-allocated to a non-document request -> its contents are no
    longer in any lazy request's document region, so the router never scores it.
  * block retained and hit by prefix caching -> contents are unchanged (blocks
    are content-hashed and immutable while cached), so the descriptor written by
    the first writer is still exactly right. This is where cross-request
    amortisation comes from for free.
  * block never described at all -> `valid` is False, and the router scores it
    `+inf`, i.e. reads it. A lifecycle bug degrades to dense, never to reading a
    block against another document's statistics.

The argument is load-bearing, so it is tested rather than trusted:
`tests/sparse/test_descriptor_lifecycle.py`, plus `LAZY_SPARSE_DESC_VERIFY=1`,
which recomputes boxes for the blocks a step actually selected and asserts they
match what is stored.
"""
from __future__ import annotations

import torch

_STORE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

# Index of the vectors in the descriptor's third axis. MEAN is only allocated
# when the centroid scorer asks for it: (min+max)/2 is the box centre, not the
# mean, and scoring a centroid against the box centre would measure neither.
MIN = 0
MAX = 1
MEAN = 2


def newly_complete_blocks(num_computed_before: int, num_scheduled: int,
                          block_size: int) -> tuple[int, int]:
    """The half-open range of block indices this step finishes filling.

    Chunked prefill can stop mid-block, so "blocks written this step" is not
    "blocks this step's tokens touched": the block a chunk ends inside is only
    complete once a later chunk covers the rest of it. Counting whole blocks
    from each end handles both -- a partially written block is simply not yet
    in range, and the chunk that completes it picks it up.

    Document prompts are padded to a whole number of blocks, so the final block
    of a document is never a partial one.
    """
    start = num_computed_before // block_size
    end = (num_computed_before + num_scheduled) // block_size
    return start, max(end, start)


def block_valid_lens(block_indices: torch.Tensor, num_prompt_tokens: int,
                     true_len: int, block_size: int) -> torch.Tensor:
    """How many rows of each block hold real (non-padding) keys.

    Every block is full except the document's last, which carries the padding
    that `q_mask` masks out in the kernel.
    """
    last_block = num_prompt_tokens // block_size - 1
    padding = num_prompt_tokens - true_len
    lens = torch.full_like(block_indices, block_size)
    return torch.where(block_indices == last_block, block_size - padding, lens)


class DescriptorStore:
    """Per-layer descriptor tensors, allocated on first use.

    Shapes are read off the paged key cache rather than reconstructed from
    config, so the store cannot disagree with the cache it describes:
    `key_cache` is `[num_blocks, num_kv_heads, head_size//x, block_size, x]`.

    Storage is `[num_blocks, num_kv_heads, 2, head_size]` per layer -- two
    vectors per 16-token page, i.e. 12.5% of K bytes and 6.25% of KV.
    """

    def __init__(self,
                 dtype: str = "bf16",
                 verify: bool = False,
                 with_mean: bool = False):
        self.dtype = _STORE_DTYPES[dtype]
        self.verify = verify
        self.slots = 3 if with_mean else 2
        self._desc: dict[str, torch.Tensor] = {}
        self._valid: dict[str, torch.Tensor] = {}

    def _tensors(self, layer_name: str,
                 key_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        desc = self._desc.get(layer_name)
        if desc is None:
            num_blocks, num_kv_heads, dx, _, x = key_cache.shape
            desc = torch.zeros((num_blocks, num_kv_heads, self.slots, dx * x),
                               dtype=self.dtype,
                               device=key_cache.device)
            valid = torch.zeros((num_blocks, ),
                                dtype=torch.bool,
                                device=key_cache.device)
            self._desc[layer_name] = desc
            self._valid[layer_name] = valid
        return desc, self._valid[layer_name]

    @staticmethod
    def _keys_by_row(key_cache: torch.Tensor,
                     block_ids: torch.Tensor) -> torch.Tensor:
        """`[N, kv_head, block_size, head_size]` for the given physical blocks.

        The paged layout splits head_size into `(head_size//x, x)` around the
        block-position axis, which is what the kernel's `k_offset` arithmetic
        reassembles; the permute below is the same reassembly in torch.
        """
        k = key_cache[block_ids]  # [N, H, D//x, P, x]
        n, h, dx, p, x = k.shape
        return k.permute(0, 1, 3, 2, 4).reshape(n, h, p, dx * x)

    def fill(self, layer_name: str, key_cache: torch.Tensor,
             block_ids: torch.Tensor, valid_lens: torch.Tensor) -> None:
        """Describe `block_ids`, counting only the first `valid_lens` rows."""
        if block_ids.numel() == 0:
            return
        desc, valid = self._tensors(layer_name, key_cache)
        keys = self._keys_by_row(key_cache, block_ids).float()
        rows = torch.arange(keys.shape[2], device=keys.device)
        live = (rows[None, :] < valid_lens[:, None])[:, None, :, None]

        desc[block_ids, :, MIN] = keys.masked_fill(
            ~live, float("inf")).amin(dim=2).to(desc.dtype)
        desc[block_ids, :, MAX] = keys.masked_fill(
            ~live, float("-inf")).amax(dim=2).to(desc.dtype)
        if self.slots > MEAN:
            live_counts = valid_lens.clamp(min=1)[:, None, None].float()
            desc[block_ids, :, MEAN] = (
                keys.masked_fill(~live, 0.0).sum(dim=2) /
                live_counts).to(desc.dtype)
        # A block with no live rows would have stored +-inf; refuse to call it
        # described so the router reads it instead of scoring against infinity.
        valid[block_ids] = valid_lens > 0

    def boxes(self, layer_name: str,
              block_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`([N, kv_head, 2, head_size], [N])` -- boxes and their validity.

        A layer that has never been filled returns everything invalid, which
        the router reads as "select these", so an un-warmed store is dense.
        """
        desc = self._desc.get(layer_name)
        if desc is None:
            return (torch.empty(0), torch.zeros(block_ids.shape,
                                                dtype=torch.bool,
                                                device=block_ids.device))
        return desc[block_ids].float(), self._valid[layer_name][block_ids]

    def check(self, layer_name: str, key_cache: torch.Tensor,
              block_ids: torch.Tensor, valid_lens: torch.Tensor) -> None:
        """Recompute and compare -- `LAZY_SPARSE_DESC_VERIFY=1` only.

        This is what holds the no-invalidation argument above to account: if a
        physical block ever carries a descriptor belonging to different
        contents, this is where it surfaces.
        """
        if not self.verify or block_ids.numel() == 0:
            return
        stored, valid = self.boxes(layer_name, block_ids)
        described = valid.nonzero(as_tuple=True)[0]
        if described.numel() == 0:
            return
        ids = block_ids[described]
        keys = self._keys_by_row(key_cache, ids).float()
        rows = torch.arange(keys.shape[2], device=keys.device)
        live = (rows[None, :] < valid_lens[described][:, None])[:, None, :,
                                                               None]
        want_min = keys.masked_fill(~live, float("inf")).amin(dim=2)
        want_max = keys.masked_fill(~live, float("-inf")).amax(dim=2)
        got = stored[described]
        torch.testing.assert_close(got[:, :, MIN],
                                   want_min.to(self.dtype).float(),
                                   rtol=0,
                                   atol=0,
                                   msg=lambda m: f"stale descriptor min on "
                                   f"layer {layer_name}, blocks "
                                   f"{ids.tolist()}: {m}")
        torch.testing.assert_close(got[:, :, MAX],
                                   want_max.to(self.dtype).float(),
                                   rtol=0,
                                   atol=0,
                                   msg=lambda m: f"stale descriptor max on "
                                   f"layer {layer_name}, blocks "
                                   f"{ids.tolist()}: {m}")
