# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental small-row expert grouping with DP4A weight reuse."""

import ctypes as ct

import torch

from benchmarks.kernels.exl3_moe_decode.launcher import Decode
from vllm.triton_utils import tl, triton


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _plan(
    Ids,
    Experts,
    Counts,
    Inverse,
    Slots: tl.constexpr,
    Tile: tl.constexpr,
    Block: tl.constexpr,
):
    i = tl.arange(0, Block)
    ids = tl.load(Ids + i, i < Slots, 0).to(tl.int32)
    key = tl.where(i < Slots, ids * Block + i, 0x7FFFFFFF)
    key = tl.sort(key, descending=False)
    expert = key // Block
    source = key % Block
    previous = tl.gather(expert, tl.maximum(i - 1, 0), 0)
    head = (i == 0) | (expert != previous)
    start = tl.associative_scan(tl.where(head, i, 0), 0, _maximum)
    within = i - start
    group_head = (within % Tile == 0) & (i < Slots)
    group = tl.cumsum(group_head.to(tl.int32), 0) - 1
    following = tl.gather(expert, tl.minimum(i + 1, Block - 1), 0)
    last = (within % Tile == Tile - 1) | (expert != following) | (i == Slots - 1)
    tl.store(Experts + group, expert.to(tl.int64), group_head)
    tl.store(Counts + group, within % Tile + 1, last & (i < Slots))
    tl.store(Inverse + source, group * Tile + within % Tile, i < Slots)


@triton.jit
def _gather(
    X, Inverse, Packed, Hidden: tl.constexpr, TopK: tl.constexpr, Block: tl.constexpr
):
    slot = tl.program_id(0)
    col = tl.program_id(1) * Block + tl.arange(0, Block)
    target = tl.load(Inverse + slot)
    values = tl.load(X + (slot // TopK) * Hidden + col, col < Hidden, 0)
    tl.store(Packed + target * Hidden + col, values, col < Hidden)


@triton.jit
def _combine(
    Down,
    Inverse,
    Weights,
    Output,
    Hidden: tl.constexpr,
    TopK: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * Block + tl.arange(0, Block)
    total = tl.full((Block,), 0, tl.float32)
    for j in tl.static_range(TopK):
        offset = row * TopK + j
        source = tl.load(Inverse + offset)
        weight = tl.load(Weights + offset).to(tl.float16).to(tl.float32)
        value = tl.load(Down + source * Hidden + col, col < Hidden, 0)
        total += value * weight
    tl.store(Output + row * Hidden + col, total, col < Hidden)


class GroupedDecode(Decode):
    def __init__(self, extension, library, tile=4, grid=8):
        # The ordinary prototype provides runtime bindings and launch checking.
        super().__init__(extension, library, False, grid)
        self.tile = tile
        for suffix in ("half", "float"):
            factory = getattr(self.lib, f"exl3_grouped_plain_{suffix}_m{tile}")
            factory.argtypes, factory.restype = [], ct.c_void_p
            self.functions[suffix] = factory()
            self.check(
                self.cuda.cudaFuncSetAttribute(self.functions[suffix], 8, 90 * 1024)
            )
            self.check(self.cuda.cudaFuncSetAttribute(self.functions[suffix], 9, 100))

    def project_grouped(self, x, ptrs, ids, counts, output):
        k, n = x.shape[-1], output.shape[-1]
        rows = k // 16
        r = (rows * (n // 256) + self.grid - 1) // self.grid
        maximum = min(512, ((80 * 1024) // (32 + 64 * self.tile)) & ~7)
        per = min(max((max(r, min(2 * r, 32)) + 7) & ~7, 16), maximum, (rows + 7) & ~7)
        split = (rows + per - 1) // per
        stride = 5120 + split * self.tile * n
        key = (x.device, ids.numel(), stride)
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(
                (ids.numel(), stride), device=x.device, dtype=torch.int32
            )
        suffix = "float" if output.dtype == torch.float32 else "half"
        self.launch(
            self.functions[suffix],
            [x, ptrs[0], output, ptrs[1], ptrs[2], ids, self.buffers[key], counts],
            [k, n, stride],
            (self.grid, 1, ids.numel()),
            256,
            per * (32 + 64 * self.tile) + 1024 * self.tile,
        )

    def __call__(self, x, weights, ids, ptrs, workspace, bits, flags, limit):
        rows, hidden = x.shape
        topk, slots = ids.shape[1], ids.numel()
        groups = min(slots, ptrs[0].numel() + slots // self.tile)
        intermediate = workspace[2].shape[-1]
        group_ids = torch.empty(groups, dtype=torch.int64, device=x.device)
        counts = torch.zeros(groups, dtype=torch.int32, device=x.device)
        inverse = torch.empty(slots, dtype=torch.int32, device=x.device)
        _plan[(1,)](
            ids,
            group_ids,
            counts,
            inverse,
            slots,
            self.tile,
            triton.next_power_of_2(slots),
        )
        packed = torch.empty(
            (groups * self.tile, hidden), dtype=torch.float16, device=x.device
        )
        _gather[(slots, triton.cdiv(hidden, 256))](
            x, inverse, packed, hidden, topk, 256
        )
        gate = torch.empty(
            (groups * self.tile, intermediate), dtype=torch.float16, device=x.device
        )
        up, activated = torch.empty_like(gate), torch.empty_like(gate)
        self.project_grouped(packed, ptrs[:3], group_ids, counts, gate)
        self.project_grouped(packed, ptrs[3:6], group_ids, counts, up)
        self.ext.silu_mul(gate, up, activated, limit)
        down = torch.empty(
            (groups * self.tile, hidden), dtype=torch.float32, device=x.device
        )
        self.project_grouped(activated, ptrs[6:9], group_ids, counts, down)
        output = torch.empty_like(x)
        _combine[(rows, triton.cdiv(hidden, 256))](
            down, inverse, weights, output, hidden, topk, 256
        )
        return output
