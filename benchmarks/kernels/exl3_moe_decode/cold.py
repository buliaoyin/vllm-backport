# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route sparse experts to DP4A and reuse FP16 Tensor Cores for hot experts."""

import ctypes as ct

import torch

from benchmarks.kernels.exl3_moe_decode.launcher import Decode
from vllm.triton_utils import tl, triton


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


class ColdDecode(Decode):
    def __init__(self, extension, library, locks, threshold=3, grid=8):
        super().__init__(extension, library, False, grid)
        self.locks, self.threshold = locks, threshold
        for suffix in ("half", "float"):
            factory = getattr(self.lib, f"exl3_cold_plain_{suffix}")
            factory.argtypes, factory.restype = [], ct.c_void_p
            self.functions[suffix] = factory()
            self.check(
                self.cuda.cudaFuncSetAttribute(self.functions[suffix], 8, 90 * 1024)
            )
            self.check(self.cuda.cudaFuncSetAttribute(self.functions[suffix], 9, 100))

    def project_cold(self, x, ptrs, ids, counts, output, input_group):
        k, n = x.shape[-1], output.shape[-1]
        rows = k // 16
        r = (rows * (n // 256) + self.grid - 1) // self.grid
        per = min(max((max(r, min(2 * r, 32)) + 7) & ~7, 16), 512, (rows + 7) & ~7)
        split = (rows + per - 1) // per
        stride = 5120 + split * n
        key = (x.device, ids.numel(), stride)
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(
                (ids.numel(), stride), device=x.device, dtype=torch.int32
            )
        suffix = "float" if output.dtype == torch.float32 else "half"
        self.launch(
            self.functions[suffix],
            [x, ptrs[0], output, ptrs[1], ptrs[2], ids, self.buffers[key], counts],
            [k, n, 8, input_group, stride, self.threshold],
            (self.grid, 1, ids.numel()),
            256,
            per * 96 + 1024,
        )

    def __call__(self, x, weights, ids, ptrs, workspace, bits, flags, limit):
        rows, hidden = x.shape
        topk, slots = ids.shape[1], ids.numel()
        experts = ptrs[0].numel()
        intermediate = workspace[2].shape[-1]
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
            self.threshold,
            triton.next_power_of_2(slots),
            triton.next_power_of_2(experts),
        )
        hot_result = torch.zeros((rows, hidden), dtype=torch.float32, device=x.device)
        torch.ops._exl3_C.moe_m32(
            hidden_x,
            hot_result,
            hot_counts,
            tokens,
            sorted_weights,
            *workspace,
            *ptrs,
            self.locks,
            limit,
        )
        gate = torch.empty((slots, intermediate), dtype=torch.float16, device=x.device)
        up, activated = torch.empty_like(gate), torch.empty_like(gate)
        self.project_cold(hidden_x, ptrs[:3], indices, counts, gate, topk)
        self.project_cold(hidden_x, ptrs[3:6], indices, counts, up, topk)
        self.ext.silu_mul(gate, up, activated, limit)
        down = torch.empty((slots, hidden), dtype=torch.float32, device=x.device)
        self.project_cold(activated, ptrs[6:9], indices, counts, down, 1)
        output = torch.empty_like(x)
        _combine[(rows, triton.cdiv(hidden, 256))](
            down,
            hot_result,
            indices,
            counts,
            routing,
            output,
            hidden,
            topk,
            self.threshold,
            256,
        )
        return output
