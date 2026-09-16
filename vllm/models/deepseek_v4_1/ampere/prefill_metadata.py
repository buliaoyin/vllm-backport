# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere prefill index construction without device-to-host synchronization."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _combine_kernel(
    topk_ptr,
    query_start_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    output_ptr,
    lengths_ptr,
    topk_stride,
    row_stride,
    M,
    N,
    NUM_REQS: tl.constexpr,
    BLOCK_REQS: tl.constexpr,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    token = tl.program_id(0)
    query_base = tl.load(query_start_ptr)
    query = token + query_base
    reqs = tl.arange(0, BLOCK_REQS)
    ends = tl.load(query_start_ptr + reqs + 1, mask=reqs < NUM_REQS, other=0)
    request = tl.sum(((reqs < NUM_REQS) & (query >= ends)).to(tl.int32), 0)
    query_start = tl.load(query_start_ptr + request)
    query_end = tl.load(query_start_ptr + request + 1)
    seq_len = tl.load(seq_lens_ptr + request)
    gather_len = tl.load(gather_lens_ptr + request)
    position = seq_len - (query_end - query_start) + query - query_start
    if TOP_K > 0:
        topk_len = tl.minimum((position + 1) // COMPRESS_RATIO, TOP_K)
        topk_len = tl.maximum(topk_len, 0)
    else:
        topk_len = 0
    swa_len = tl.minimum(tl.maximum(position + 1, 0), WINDOW_SIZE)
    columns = tl.arange(0, BLOCK_COLS)
    topk = tl.load(
        topk_ptr + token.to(tl.int64) * topk_stride + columns,
        mask=columns < topk_len,
        other=-1,
    ).to(tl.int32)
    base = M.to(tl.int64) * request
    topk = tl.where((topk >= 0) & (topk < N), topk + base, -1)
    swa = base + N + columns - topk_len + position - swa_len + 1 - seq_len + gather_len
    value = tl.where(
        columns < topk_len,
        topk,
        tl.where(columns < topk_len + swa_len, swa, -1),
    )
    tl.store(
        output_ptr + token.to(tl.int64) * row_stride + columns,
        value,
        mask=columns < row_stride,
    )
    tl.store(lengths_ptr + token, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the dense top-k/SWA rows, including padding, in one launch."""
    num_tokens = topk_indices.shape[0]
    combined_width = triton.cdiv(topk + window_size, 128) * 128
    indices = torch.empty(
        (num_tokens, combined_width), device=topk_indices.device, dtype=torch.int32
    )
    lengths = torch.empty(num_tokens, device=topk_indices.device, dtype=torch.int32)
    if not num_tokens:
        return indices, lengths
    topk_indices = topk_indices.reshape(num_tokens, -1).contiguous()
    logical_topk = min(topk, topk_indices.shape[1])
    assert not logical_topk or compress_ratio > 0
    num_reqs = seq_lens.numel()
    _combine_kernel[(num_tokens,)](
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        indices,
        lengths,
        topk_indices.stride(0),
        combined_width,
        M,
        N,
        NUM_REQS=num_reqs,
        BLOCK_REQS=triton.next_power_of_2(num_reqs),
        TOP_K=logical_topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        BLOCK_COLS=triton.next_power_of_2(combined_width),
        num_warps=4,
    )
    return indices, lengths
