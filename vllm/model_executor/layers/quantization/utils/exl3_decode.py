# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch expert decode with DP4A for sparse experts and M32 for reused weights."""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _hot_routes(
    Ids,
    Weights,
    Counts,
    HotCounts,
    Tokens,
    SortedWeights,
    Slots: tl.constexpr,
    TopK: tl.constexpr,
    Experts: tl.constexpr,
    Threshold: tl.constexpr,
    Block: tl.constexpr,
    EBlock: tl.constexpr,
    ColdTasks=None,
    ColdCount=None,
    Compact: tl.constexpr = False,
):
    expert = tl.program_id(0)
    count = tl.load(Counts + expert)
    hot = count >= Threshold
    tl.store(HotCounts + expert, tl.where(hot, count, 0))
    if hot:
        e = tl.arange(0, EBlock)
        counts = tl.load(Counts + e, e < Experts, 0)
        offset = tl.sum(tl.where((e < expert) & (counts >= Threshold), counts, 0), 0)
        i = tl.arange(0, Block)
        ids = tl.load(Ids + i, i < Slots, -1)
        selected = (ids == expert) & (i < Slots)
        positions = offset + tl.cumsum(selected.to(tl.int32), 0) - 1
        weights = tl.load(Weights + i, i < Slots, 0)
        tl.store(Tokens + positions, i // TopK, selected)
        tl.store(SortedWeights + positions, weights, selected)

    if Compact and expert == Experts:
        i = tl.arange(0, Block)
        ids = tl.load(Ids + i, i < Slots, 0)
        cold = (i < Slots) & (tl.load(Counts + ids) < Threshold)
        positions = tl.cumsum(cold.to(tl.int32), 0) - 1
        tl.store(ColdTasks + positions, i, cold)
        tl.store(ColdCount, tl.sum(cold.to(tl.int32), 0))


@triton.jit
def _combine(
    Down,
    Hot,
    Ids,
    Counts,
    Weights,
    Output,
    Hidden: tl.constexpr,
    TopK: tl.constexpr,
    Threshold: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * Block + tl.arange(0, Block)
    total = tl.load(Hot + row * Hidden + col, col < Hidden, 0)
    for j in tl.static_range(TopK):
        offset = row * TopK + j
        expert = tl.load(Ids + offset)
        cold = tl.load(Counts + expert) < Threshold
        weight = tl.load(Weights + offset).to(tl.float32)
        value = tl.load(Down + offset * Hidden + col, (col < Hidden) & cold, 0)
        total += value * weight
    tl.store(Output + row * Hidden + col, total, col < Hidden)


def is_decode_batch() -> bool:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return False
    context = get_forward_context()
    # Piecewise graphs are reused for prefill and mixed batches.
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        return False
    metadata: object = context.attn_metadata
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values()), None)
    prefills = getattr(metadata, "num_prefills", None)
    if not isinstance(prefills, int) or prefills != 0:
        return False
    return any(
        isinstance(value, int) and value > 0
        for value in (
            getattr(metadata, "num_decodes", 0),
            getattr(metadata, "num_spec_decodes", 0),
        )
    )


def moe_batched_decode(
    x,
    weights,
    ids,
    ptrs,
    workspace,
    locks,
    scratch,
    limit,
    residual=False,
    compact=True,
):
    """Use caller-owned scratch on the input device and its current stream."""
    with torch.accelerator.device_index(x.device.index):
        return _moe_batched_decode(
            x, weights, ids, ptrs, workspace, locks, scratch, limit, residual, compact
        )


def _moe_batched_decode(
    x, weights, ids, ptrs, workspace, locks, scratch, limit, residual, compact
):
    from vllm.model_executor.layers.quantization.exl3 import _extension

    logger.info_once(
        "Using EXL3 batched expert decode (sparse %s / hot FP16).",
        "residual INT8" if residual else "plain INT8",
    )
    rows, hidden = x.shape
    topk, slots = ids.shape[1], ids.numel()
    experts = ptrs[0].numel()
    intermediate = workspace[2].shape[-1]
    threshold = 3
    compact = (
        compact
        and not residual
        and 9 <= rows <= 128
        and hidden == 4096
        and intermediate == 2048
        and topk in (1, 2, 4, 8)
    )
    cold_tasks = (
        torch.empty(slots, dtype=torch.int32, device=x.device) if compact else None
    )
    cold_count = torch.empty(1, dtype=torch.int32, device=x.device) if compact else None
    hidden_x = x.to(torch.float16).contiguous()
    indices = ids.to(torch.int64).contiguous().flatten()
    routing = weights.to(torch.float16).contiguous().flatten()
    counts = torch.zeros(experts + 1, dtype=torch.int64, device=x.device)
    counts.scatter_add_(0, indices, torch.ones_like(indices))
    hot_counts = torch.empty_like(counts)
    tokens = torch.empty_like(indices)
    sorted_weights = torch.empty_like(routing)
    _hot_routes[(experts + 1,)](
        indices,
        routing,
        counts,
        hot_counts,
        tokens,
        sorted_weights,
        slots,
        topk,
        experts,
        threshold,
        triton.next_power_of_2(slots),
        triton.next_power_of_2(experts),
        cold_tasks,
        cold_count,
        compact,
    )
    hot_result = torch.zeros((rows, hidden), dtype=torch.float32, device=x.device)
    torch.ops._exl3_C.moe_m32_decode(
        hidden_x,
        hot_result,
        hot_counts,
        tokens,
        sorted_weights,
        *workspace,
        *ptrs,
        locks,
        limit,
    )
    gate = torch.empty((slots, intermediate), dtype=torch.float16, device=x.device)
    up, activated = torch.empty_like(gate), torch.empty_like(gate)

    def project(input, pointers, output, input_group):
        if compact:
            torch.ops._exl3_C.expert_gemv_compact(
                input,
                *pointers,
                indices,
                output,
                scratch,
                cold_tasks,
                cold_count,
                rows,
                input_group,
            )
        else:
            torch.ops._exl3_C.expert_gemv_cold(
                input,
                *pointers,
                indices,
                output,
                scratch,
                counts,
                rows,
                input_group,
                threshold,
                residual,
            )

    project(hidden_x, ptrs[:3], gate, topk)
    project(hidden_x, ptrs[3:6], up, topk)
    _extension().silu_mul(gate, up, activated, limit)
    down = torch.empty((slots, hidden), dtype=torch.float32, device=x.device)
    project(activated, ptrs[6:9], down, 1)
    output = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    _combine[(rows, triton.cdiv(hidden, 256))](
        down,
        hot_result,
        indices,
        counts,
        routing,
        output,
        hidden,
        topk,
        threshold,
        256,
    )
    return output
