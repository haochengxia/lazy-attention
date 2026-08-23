"""Paged decode split over the sequence, so the GPU is not mostly idle.

`kernel_paged_attention_2d_llama` launches a grid of
`(num_seqs, num_kv_heads)`. At batch 1 on an 8-KV-head model that is **eight
thread blocks on a 70-SM card**, each walking the whole block table in a serial
loop. Measured at a 78k-token context it costs 3.81 ms per layer against stock
vLLM's 11.7 ms per *token* across all sixteen layers -- roughly six times off
the pace, and the gap is parallelism, not arithmetic.

This is flash-decoding: the block walk is cut into `NUM_SPLITS` contiguous
ranges, each handled by its own program, and a second pass merges the partial
softmaxes. The merge is the standard one -- each split reports its running max
`m`, its denominator `l` and its *unnormalised* accumulator, and the combine
rescales everything to a common max before dividing.

`llama_v1.py` is untouched. Design rule R1 keeps the decode kernel off-limits,
and this does not edit it: it is a second kernel selected by
`LAZY_SPLIT_KV`, with `tests/kernels/test_split_decode.py` holding the two to
each other. The split path also has to reproduce two lazy-specific details
exactly, and both are easy to get wrong:

* **The rotation elision.** The original re-rotates Q only when `rot_offset`
  changes between adjacent rows. A split starts with `prev_rot_offset = -1`, so
  it re-rotates once at its own first row -- more work than the serial walk,
  but the same answer, and bounded by one rotation per split.
* **`q_mask` and the sequence boundary** are indexed by the *absolute* row `j`,
  not by an offset within the split, so the split ranges pass absolute indices
  through unchanged.

Splits that fall past the end of a sequence write `m = -inf`, which the combine
turns into a zero weight; that is why an over-estimated split count is merely
wasteful rather than wrong.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from lazy.model_executor.rope import rope_cos_sin_from_freqs, rope_freqs
from lazy.utils.rotation import (MAX_PACKED_Q_MASK, MAX_PACKED_Q_OFFSET,
                                 PACKED_Q_OFFSET_SHIFT)

Q_OFFSET_SHIFT = tl.constexpr(PACKED_Q_OFFSET_SHIFT)
Q_OFFSET_MASK = tl.constexpr(MAX_PACKED_Q_OFFSET)
Q_MASK_MASK = tl.constexpr(MAX_PACKED_Q_MASK)

# Enough programs to fill the card several times over without making the
# partial buffers large. Past this the merge starts to cost more than the
# parallelism buys.
MAX_SPLITS = 16
# Below this a split does not earn its own launch and its share of the merge.
MIN_BLOCKS_PER_SPLIT = 8


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_decode_split(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        alibi_slopes_ptr,
        scale,
        k_scale,
        v_scale,
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        num_queries_per_kv_padded: tl.constexpr,
        block_table_stride: tl.int64,
        query_stride_0: tl.int64,
        query_stride_1: tl.int64,
        stride_pa_0: tl.int64,
        stride_pa_1: tl.int64,
        stride_pa_2: tl.int64,
        stride_pm_0: tl.int64,
        stride_pm_1: tl.int64,
        stride_pm_2: tl.int64,
        IN_PRECISION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        USE_ALIBI_SLOPES: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        x: tl.constexpr,
        stride_k_cache_0: tl.int64,
        stride_k_cache_1: tl.int64,
        stride_k_cache_2: tl.int64,
        stride_k_cache_3: tl.int64,
        stride_k_cache_4: tl.int64,
        stride_v_cache_0: tl.int64,
        stride_v_cache_1: tl.int64,
        stride_v_cache_2: tl.int64,
        stride_v_cache_3: tl.int64,
        filter_by_query_len: tl.constexpr,
        query_start_len_ptr,
        rotary_dim: tl.constexpr,
        is_lazy_ptr,
        cos_sin_cache_ptr,
        NUM_SPLITS: tl.constexpr,
        ROPE_TYPE: tl.constexpr = 0,
        BASE: tl.constexpr = 10000.0,
        SCALING_FACTOR: tl.constexpr = 1.0,
        LOW_FACTOR: tl.constexpr = 1.0,
        HIGH_FACTOR: tl.constexpr = 1.0,
        ORIG_MAX_POSITION: tl.constexpr = 8192,
        PI_VALUE: tl.constexpr = 3.141592653589793,
        IGNORE_Q_MASK: tl.constexpr = False,
        COMPUTE_COS_SIN: tl.constexpr = False,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    if filter_by_query_len:
        start_index = tl.load(query_start_len_ptr + seq_idx)
        stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        if stop_index - start_index > 1:
            return
    else:
        start_index = seq_idx

    is_lazy = tl.load(is_lazy_ptr + seq_idx)

    query_head_idx = (kv_head_idx * num_queries_per_kv +
                      tl.arange(0, num_queries_per_kv_padded))
    query_offset = (start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)
    embed_dim: tl.constexpr = HEAD_SIZE // 2
    mask_q1 = tl.arange(0, HEAD_SIZE) < embed_dim

    Q_full = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )
    Q_rotated = Q_full
    block_table_offset = seq_idx * block_table_stride

    if COMPUTE_COS_SIN:
        rope_freq = rope_freqs(ROPE_TYPE, HEAD_SIZE, BASE, SCALING_FACTOR,
                               LOW_FACTOR, HIGH_FACTOR, ORIG_MAX_POSITION,
                               PI_VALUE)

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx,
                              mask=head_mask,
                              other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
    # Contiguous ranges, so a document's pages stay adjacent within a split and
    # the rotation elision below still fires for all but the first row.
    blocks_per_split = cdiv_fn(num_blocks, NUM_SPLITS)
    split_start = split_idx * blocks_per_split
    split_stop = tl.minimum(split_start + blocks_per_split, num_blocks)

    prev_rot_offset = tl.full([], -1, dtype=tl.int32)
    for j in range(split_start, split_stop):
        packed_val = tl.load(block_tables_ptr + block_table_offset + j)
        physical_block_idx = (packed_val >> 32).to(tl.int32)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[None, :] * stride_v_cache_2 +
                    offs_n[:, None] * stride_v_cache_3)
        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None],
                         other=0.0)
        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q_full.dtype)
        else:
            K = K_load

        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :],
                         other=0.0)
        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q_full.dtype)
        else:
            V = V_load

        q_mask_val = 0
        if is_lazy:
            rot_offset_val = ((packed_val >> Q_OFFSET_SHIFT)
                              & Q_OFFSET_MASK).to(tl.int32)
            if not IGNORE_Q_MASK:
                q_mask_val = (packed_val & Q_MASK_MASK).to(tl.int32)
            if rot_offset_val != prev_rot_offset:
                if rot_offset_val == 1:
                    Q_rotated = Q_full
                elif rot_offset_val != 0:
                    if COMPUTE_COS_SIN:
                        cos_val, sin_val = rope_cos_sin_from_freqs(
                            rot_offset_val - 1, rope_freq)
                    else:
                        cache_cols = tl.arange(0, HEAD_SIZE) % embed_dim
                        cache_base = (rot_offset_val - 1) * rotary_dim
                        cos_val = tl.load(cos_sin_cache_ptr + cache_base +
                                          cache_cols)
                        sin_val = tl.load(cos_sin_cache_ptr + cache_base +
                                          embed_dim + cache_cols)
                    rev = (tl.arange(0, HEAD_SIZE_PADDED) +
                           (HEAD_SIZE_PADDED // 2)) % HEAD_SIZE_PADDED
                    Q_rev = tl.load(
                        query_ptr + query_offset + rev[None, :],
                        mask=dim_mask[None, :] & head_mask[:, None],
                        other=0.0,
                    )
                    q1 = tl.where(mask_q1[None, :], Q_full, Q_rev)
                    q2 = tl.where(mask_q1[None, :], Q_rev, Q_full)
                    q1_new = q1 * cos_val + q2 * sin_val
                    q2_new = -q1 * sin_val + q2 * cos_val
                    Q_rotated = tl.where(mask_q1[None, :],
                                         q1_new.to(Q_full.dtype),
                                         q2_new.to(Q_full.dtype))
                prev_rot_offset = rot_offset_val

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary
        seq_mask = seq_mask & (tl.arange(0, BLOCK_SIZE) <
                               (BLOCK_SIZE - q_mask_val))

        qk = tl.dot(Q_rotated, K, input_precision=IN_PRECISION)
        S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                     float("-inf")).to(tl.float32)
        S += scale * qk

        context_len = seq_len - 1
        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)
        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)

    # Unnormalised on purpose: the combine divides once, after rescaling every
    # split to a common max. Dividing here would lose the cross-split weights.
    lane = tl.arange(0, num_queries_per_kv_padded)
    partial_base = (seq_idx * stride_pm_0 + kv_head_idx * stride_pm_1 +
                    split_idx * stride_pm_2)
    tl.store(partial_m_ptr + partial_base + lane, M)
    tl.store(partial_l_ptr + partial_base + lane, L)
    tl.store(partial_acc_ptr + seq_idx * stride_pa_0 +
             kv_head_idx * stride_pa_1 + split_idx * stride_pa_2 +
             lane[:, None] * HEAD_SIZE_PADDED +
             tl.arange(0, HEAD_SIZE_PADDED)[None, :],
             acc)


@triton.jit
def kernel_paged_decode_combine(
        output_ptr,
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_start_len_ptr,
        output_stride_0: tl.int64,
        output_stride_1: tl.int64,
        stride_pa_0: tl.int64,
        stride_pa_1: tl.int64,
        stride_pa_2: tl.int64,
        stride_pm_0: tl.int64,
        stride_pm_1: tl.int64,
        stride_pm_2: tl.int64,
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        num_queries_per_kv_padded: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        filter_by_query_len: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        start_index = tl.load(query_start_len_ptr + seq_idx)
        stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        if stop_index - start_index > 1:
            return
    else:
        start_index = seq_idx

    lane = tl.arange(0, num_queries_per_kv_padded)
    dims = tl.arange(0, HEAD_SIZE_PADDED)
    query_head_idx = kv_head_idx * num_queries_per_kv + lane
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)
    dim_mask = dims < HEAD_SIZE

    m_base = (seq_idx * stride_pm_0 + kv_head_idx * stride_pm_1)
    a_base = (seq_idx * stride_pa_0 + kv_head_idx * stride_pa_1)

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    for s in range(NUM_SPLITS):
        M = tl.maximum(M, tl.load(partial_m_ptr + m_base + s * stride_pm_2 +
                                  lane))

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                   dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        m_s = tl.load(partial_m_ptr + m_base + s * stride_pm_2 + lane)
        l_s = tl.load(partial_l_ptr + m_base + s * stride_pm_2 + lane)
        a_s = tl.load(partial_acc_ptr + a_base + s * stride_pa_2 +
                      lane[:, None] * HEAD_SIZE_PADDED + dims[None, :])
        # A split with no rows left `m` at -inf. Guarding on that rather than
        # letting `exp(-inf - M)` run keeps an all-empty lane from producing
        # `exp(nan)`, which no later mask would clean up.
        weight = tl.where(m_s > float("-inf"), tl.exp(m_s - M), 0.0)
        L += l_s * weight
        acc += a_s * weight[:, None]

    acc = acc / L[:, None]
    tl.store(output_ptr + start_index * output_stride_0 +
             query_head_idx[:, None] * output_stride_1 + dims[None, :],
             acc,
             mask=head_mask[:, None] & dim_mask[None, :])


def choose_splits(num_seqs: int, num_kv_heads: int, max_blocks: int,
                  num_sms: int) -> int:
    """How many ways to cut the walk.

    Enough programs to give the card a few waves, but never so many that splits
    are shorter than they are worth. `max_blocks` is an upper bound derived from
    the dense `max_seq_len`, so a compacted walk table over-estimates it and
    over-splits; empty splits cost a launch and a zero weight, which is the
    cheap direction to be wrong in.
    """
    if max_blocks <= MIN_BLOCKS_PER_SPLIT:
        return 1
    occupancy = max(1, (4 * num_sms) // max(num_seqs * num_kv_heads, 1))
    by_length = max(1, max_blocks // MIN_BLOCKS_PER_SPLIT)
    return max(1, min(MAX_SPLITS, occupancy, by_length))


def allocate_partials(num_seqs: int, num_kv_heads: int, splits: int,
                      lanes: int, head_size_padded: int,
                      device: torch.device):
    acc = torch.empty((num_seqs, num_kv_heads, splits, lanes,
                       head_size_padded),
                      dtype=torch.float32,
                      device=device)
    stats = torch.empty((2, num_seqs, num_kv_heads, splits, lanes),
                        dtype=torch.float32,
                        device=device)
    return acc, stats[0], stats[1]
