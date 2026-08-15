"""
Rotary Positional Embeddings.

Changed by Haocheng at 2025/09/14
"""
import math
from typing import Optional, Tuple, Union

import torch

from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding,
    _apply_rotary_emb_torch,
)
from lazy.utils.variants import (
    LAZY_VARIANT_MEPIC,
    get_lazy_attention_variant_code,
)


USE_MEPIC_Q_ONLY_ROTARY = (
    get_lazy_attention_variant_code() == LAZY_VARIANT_MEPIC
)


def _should_use_q_only_rotary() -> bool:
    return USE_MEPIC_Q_ONLY_ROTARY


class _LazyRotaryMixin:
    """Shared behaviour for every RoPE class LazyAttention substitutes in.

    Two things, both required by the deferred key rotation:

    1. `inv_freq` is materialised as a buffer. The model forward hands
       `self.rotary_emb.inv_freq` to the attention kernel, but vLLM's base
       RotaryEmbedding only keeps the derived `cos_sin_cache`.
    2. Under the MEPIC variant, `forward` rotates the query only. The MEPIC
       attention kernel rotates the cached keys itself, so rotating them here
       as well would double-rotate them.

    Both apply regardless of which concrete RoPE class the checkpoint selects,
    so they live here rather than on one subclass -- a checkpoint with
    `rope_scaling: null` (e.g. the block-fine-tuned models) builds the plain
    RotaryEmbedding, not Llama3RotaryEmbedding.
    """

    def _register_inv_freq(self, base: Union[int, float]) -> None:
        inv_freq = self._compute_inv_freq(base).to(torch.bfloat16)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not _should_use_q_only_rotary():
            return super().forward_native(positions, query, key, offsets)

        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin,
                                            self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not _should_use_q_only_rotary():
            return super().forward_cuda(positions, query, key, offsets)

        from vllm import _custom_ops as ops

        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)

        # The kernel writes both operands in place; give it a throwaway key so
        # the real one reaches the attention kernel unrotated.
        key_scratch = key.clone()
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key_scratch,
                                         self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style,
                                         self.rotary_dim, offsets)
        else:
            ops.rotary_embedding(positions, query, key_scratch,
                                 self.head_size,
                                 self.cos_sin_cache,
                                 self.is_neox_style)
        return query, key


class LazyRotaryEmbedding(_LazyRotaryMixin, RotaryEmbedding):
    """Plain RoPE, for checkpoints with `rope_scaling: null`."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        self._register_inv_freq(base)


class Llama3RotaryEmbedding(_LazyRotaryMixin, RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        self._register_inv_freq(base)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor
                      ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = 0

        # print("wave_len:", wave_len)
        # print("high_freq_wavelen:", high_freq_wavelen)
        # print("low_freq_wavelen:", low_freq_wavelen)

        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor +
                smooth * inv_freqs,
            ),
        )
        return new_freqs
