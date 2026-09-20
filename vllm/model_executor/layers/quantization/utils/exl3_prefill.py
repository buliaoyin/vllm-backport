# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row quantization and grouped INT8 GEMM with affine EXL3 codebook scaling."""

import struct

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton

INT8_MAX_ROWS = 6144
ALPHA = struct.unpack("<e", bytes.fromhex("ee1e"))[0]
ALPHA4 = 4 * ALPHA
BETA = 1534 * ALPHA + struct.unpack("<e", bytes.fromhex("31c9"))[0]


def allocate_workspace(
    device, experts, hidden, intermediate, capacity, topk, experts_per_group=0
):
    """Allocate once per device/shape before KV cache memory profiling."""
    if experts_per_group < 0:
        raise ValueError("EXL3 experts per group must be nonnegative (0 means all)")
    group_size = min(experts_per_group or experts, experts)
    slots = capacity * topk
    return [
        torch.empty(
            group_size * hidden * intermediate, device=device, dtype=torch.int8
        ),
        torch.empty((slots, hidden), device=device, dtype=torch.float16),
        torch.empty((slots, intermediate), device=device, dtype=torch.float16),
        torch.empty((slots, intermediate), device=device, dtype=torch.float16),
        torch.empty(slots * max(hidden, intermediate), device=device, dtype=torch.int8),
        torch.empty(slots, device=device, dtype=torch.float32),
        torch.empty(slots, device=device, dtype=torch.float32),
    ]


def moe_int8(x, topk_weights, topk_ids, ptrs, workspace, limit):
    """Use one temporary INT8 expert projection for all rows in the batch.

    The caller serializes access to the shared workspace on the current stream.
    Router IDs must be valid, unique expert indices within each row.
    """
    with torch.accelerator.device_index(x.device.index):
        return _moe_int8(x, topk_weights, topk_ids, ptrs, workspace, limit)


def _moe_int8(x, topk_weights, topk_ids, ptrs, workspace, limit):
    weight, stage, gate, up, quantized, scales, sums = workspace
    rows, hidden = x.shape
    experts, topk = ptrs[0].numel(), topk_ids.shape[1]
    slots = rows * topk
    intermediate = gate.shape[1]
    group_size = weight.numel() // (hidden * intermediate)
    if slots > stage.shape[0]:
        raise ValueError("EXL3 INT8 prefill workspace capacity exceeded")
    stage, gate, up = stage[:slots], gate[:slots], up[:slots]
    inp = x.to(torch.float16).contiguous()
    ids = topk_ids.to(torch.int64).contiguous().flatten()
    routing = topk_weights.to(torch.float16).contiguous().flatten()
    sorted_ids, expert_ids, padded = moe_align_block_size(
        topk_ids, 64, experts, pad_sorted_ids=True
    )
    output = torch.zeros((rows, hidden), device=x.device, dtype=torch.float32)
    num_groups = triton.cdiv(experts, group_size)
    bounds = padded
    if num_groups > 1:
        bounds = torch.empty(num_groups + 1, dtype=torch.int32, device=x.device)
        group_bounds[(num_groups + 1,)](
            expert_ids,
            padded,
            bounds,
            group_size,
            experts,
            triton.next_power_of_2(expert_ids.numel()),
        )

    ops = torch.ops._exl3_C

    def gemm(a, table, c):
        k, n = a.shape[1], c.shape[1]
        q = quantized[: slots * k].view(slots, k)
        quantize[(slots,)](
            a, q, scales, sums, k, triton.next_power_of_2(k), num_warps=4
        )
        for expert_start in range(0, experts, group_size):
            expert_count = min(group_size, experts - expert_start)
            b = weight[: expert_count * k * n].view(expert_count, k, n)
            ops.prefill_reconstruct(
                table[expert_start : expert_start + expert_count], b
            )
            grid_m = sorted_ids.numel() // 64
            if num_groups > 1:
                grid_m = triton.cdiv(grid_m * expert_count, experts)
            grouped[(grid_m * triton.cdiv(n, 128),)](
                q,
                b,
                c,
                scales,
                sums,
                sorted_ids,
                expert_ids,
                padded,
                bounds,
                expert_start // group_size,
                slots,
                n,
                k,
                sorted_ids.numel(),
                expert_start,
                expert_count,
                num_groups > 1,
                ALPHA4,
                BETA,
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_N=128,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8,
                num_warps=4,
                num_stages=3,
            )

    for i, destination in ((0, gate), (1, up)):
        ops.prefill_gather(inp, ids, ptrs[i * 3 + 1], stage, topk)
        gemm(stage, ptrs[i * 3], destination)
    ops.prefill_activate(gate, up, ids, ptrs[2], ptrs[5], ptrs[7], limit)
    gemm(gate, ptrs[6], stage)
    ops.prefill_scatter(stage, ids, routing, ptrs[8], output, topk)
    return output.to(x.dtype)


@triton.jit
def quantize(A, Q, Scales, Sums, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    values = tl.load(A + row * K + cols, mask=cols < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 127.0, 1e-20)
    q = tl.extra.cuda.libdevice.nearbyint(values / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + cols, q, mask=cols < K)
    tl.store(Scales + row, scale)
    tl.store(Sums + row, tl.sum(values, 0))


@triton.jit
def group_bounds(
    Experts,
    Padded,
    Bounds,
    GROUP: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    group = tl.program_id(0)
    boundary = tl.minimum(group * GROUP, EXPERTS)
    blocks = tl.arange(0, BLOCKS)
    valid = blocks < tl.load(Padded) // 64
    expert = tl.load(Experts + blocks, valid, other=EXPERTS)
    offset = tl.sum((valid & (expert < boundary)).to(tl.int32))
    tl.store(Bounds + group, offset)


@triton.jit
def grouped(
    A,
    B,
    C,
    Scales,
    Sums,
    Sorted,
    Experts,
    Padded,
    Bounds,
    group_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    expert_start,
    EXPERT_COUNT: tl.constexpr,
    GROUPED: tl.constexpr,
    ALPHA4: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    first_m = 0
    grid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iterations = 1
    if GROUPED:
        first_m = tl.load(Bounds + group_index)
        grid_m = tl.load(Bounds + group_index + 1) - first_m
        iterations = tl.cdiv(grid_m * grid_n - tl.program_id(0), tl.num_programs(0))
    for iteration in range(iterations):
        pid = tl.program_id(0) + iteration * tl.num_programs(0)
        nm, nn = grid_m, grid_n
        group_id = pid // (GROUP_SIZE_M * nn)
        first = group_id * GROUP_SIZE_M
        group_size = tl.minimum(nm - first, GROUP_SIZE_M)
        pm = first_m + first + pid % group_size
        pn = (pid % (GROUP_SIZE_M * nn)) // group_size
        if pm * BLOCK_SIZE_M < tl.load(Padded):
            expert = tl.load(Experts + pm).to(tl.int64)
            if expert >= expert_start and expert < expert_start + EXPERT_COUNT:
                slots = tl.load(
                    Sorted + pm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                ).to(tl.int64)
                valid = slots < M
                cols = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                ks = tl.arange(0, BLOCK_SIZE_K)
                ap = A + slots[:, None] * K + ks[None, :]
                bp = (
                    B
                    + (expert - expert_start) * K * N
                    + ks[:, None] * N
                    + cols[None, :]
                )
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int32)
                for block in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    a = tl.load(
                        ap,
                        mask=valid[:, None] & (ks[None, :] + block * BLOCK_SIZE_K < K),
                        other=0,
                    )
                    b = tl.load(
                        bp,
                        mask=(cols[None, :] < N)
                        & (ks[:, None] + block * BLOCK_SIZE_K < K),
                        other=0,
                    )
                    acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                    ap += BLOCK_SIZE_K
                    bp += BLOCK_SIZE_K * N
                scale = tl.load(Scales + slots, mask=valid, other=0)
                row_sum = tl.load(Sums + slots, mask=valid, other=0)
                result = (
                    acc.to(tl.float32) * (scale[:, None] * ALPHA4)
                    + row_sum[:, None] * BETA
                )
                tl.store(
                    C + slots[:, None] * N + cols[None, :],
                    result.to(tl.float16),
                    mask=valid[:, None] & (cols[None, :] < N),
                )
