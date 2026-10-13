# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded native R8 staging for bit-preserving Marlin expert refills."""

from threading import Lock
from weakref import WeakValueDictionary

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _r8_weights(
    raw,
    gate_up,
    down,
    HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ints: tl.constexpr = HIDDEN * INTERMEDIATE // 8
    matrix_bytes: tl.constexpr = HIDDEN * INTERMEDIATE * 17 // 32
    is_down = index >= 2 * ints
    local = index - tl.where(is_down, 2 * ints, 0)
    size_k = tl.where(is_down, INTERMEDIATE, HIDDEN)
    size_n = tl.where(is_down, HIDDEN, 2 * INTERMEDIATE)
    tiles = size_n // 64
    tile = local // 128
    lane = local % 128 // 4
    warp = local % 4
    n0 = tile % tiles * 64 + warp * 16 + lane // 4
    k0 = tile // tiles * 16 + lane % 4 * 2
    packed = tl.full((BLOCK,), 0, tl.uint32)
    for j in tl.static_range(8):
        n = n0 + (8 if j >= 4 else 0)
        k = k0 + j % 4 // 2 * 8 + j % 2
        part = tl.where(
            is_down, matrix_bytes, tl.where(n >= INTERMEDIATE, 2 * matrix_bytes, 0)
        )
        n = tl.where((~is_down) & (n >= INTERMEDIATE), n - INTERMEDIATE, n)
        block = n // 8 * (size_k // 32) + k // 32
        position = k % 16
        offset = block * 136 + 8 + position // 4 * 32 + n % 8 * 4 + position % 4
        byte = tl.load(raw + part + offset, index < 3 * ints, other=0).to(tl.uint32)
        nibble = (byte >> (k % 32 // 16 * 4)) & 15
        packed |= nibble << ((j % 2 * 4 + j // 2) * 4)
    tl.store(gate_up + local, packed, (index < 3 * ints) & (~is_down))
    tl.store(down + local, packed, (index < 3 * ints) & is_down)


@triton.jit
def _r8_scales(
    raw,
    gate_up,
    down,
    HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scales: tl.constexpr = HIDDEN * INTERMEDIATE // 32
    matrix_bytes: tl.constexpr = scales * 17
    is_down = index >= 2 * scales
    local = index - tl.where(is_down, 2 * scales, 0)
    size_k = tl.where(is_down, INTERMEDIATE, HIDDEN)
    size_n = tl.where(is_down, HIDDEN, 2 * INTERMEDIATE)
    pre = local // 4 * 4 + local % 2 * 2 + local % 4 // 2
    old = pre // 64 * 64 + pre % 8 * 8 + pre % 64 // 8
    n, k = old % size_n, old // size_n
    part = tl.where(
        is_down, matrix_bytes, tl.where(n >= INTERMEDIATE, 2 * matrix_bytes, 0)
    )
    n = tl.where((~is_down) & (n >= INTERMEDIATE), n - INTERMEDIATE, n)
    offset = (n // 8 * (size_k // 32) + k) * 136 + n % 8
    byte = tl.load(raw + part + offset, index < 3 * scales, other=0)
    tl.store(gate_up + local, byte, (index < 3 * scales) & (~is_down))
    tl.store(down + local, byte, (index < 3 * scales) & is_down)


class R8RefillWorkspace:
    pools: WeakValueDictionary[tuple[int, int, int], "R8RefillWorkspace"] = (
        WeakValueDictionary()
    )
    pool_lock = Lock()

    @classmethod
    def get(cls, device, hidden, intermediate):
        key = (device.index, hidden, intermediate)
        with cls.pool_lock:
            pool = cls.pools.get(key)
            if pool is None:
                pool = cls(device, hidden, intermediate)
                cls.pools[key] = pool
            return pool

    def __init__(self, device, hidden, intermediate):
        self.lock = Lock()
        self.bytes = 3 * hidden * intermediate * 17 // 32
        with torch.inference_mode(False), torch.accelerator.device_index(device.index):
            self.host = [
                torch.empty(self.bytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(2)
            ]
            self.raw = [torch.empty_like(x, device=device) for x in self.host]
            self.copy_stream = torch.cuda.Stream(device=device)
            self.copied = [torch.cuda.Event() for _ in range(2)]
            self.read = [torch.cuda.Event() for _ in range(2)]
        self.used = [False, False]
        self.next = 0


class R8Refill:
    def __init__(self, backend, device):
        self.backend = backend
        self.device = device
        self.hidden, self.intermediate = backend.hidden_size, backend.intermediate_size
        if self.hidden % 64 or self.intermediate % 64:
            raise ValueError(
                "Marlin R8 refill requires 64-aligned projection dimensions"
            )
        self.workspace = R8RefillWorkspace.get(device, self.hidden, self.intermediate)

    def pack(self, expert):
        with self.workspace.lock:
            return self._pack(expert)

    def _pack(self, expert):
        w = self.workspace
        slot = w.next
        w.next = 1 - slot
        if w.used[slot]:
            w.copied[slot].synchronize()
        self.backend.stage_expert_r8(expert, w.host[slot])
        with torch.cuda.stream(w.copy_stream):
            if w.used[slot]:
                w.copy_stream.wait_event(w.read[slot])
            w.raw[slot].copy_(w.host[slot], non_blocking=True)
            w.copied[slot].record(w.copy_stream)
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(w.copied[slot])
        h, i = self.hidden, self.intermediate
        outputs = (
            torch.empty((1, h // 16, 4 * i), dtype=torch.int32, device=self.device),
            torch.empty((1, i // 16, 2 * h), dtype=torch.int32, device=self.device),
            torch.empty(
                (1, h // 32, 2 * i), dtype=torch.float8_e8m0fnu, device=self.device
            ),
            torch.empty(
                (1, i // 32, h), dtype=torch.float8_e8m0fnu, device=self.device
            ),
        )
        _r8_weights[(triton.cdiv(3 * h * i // 8, 256),)](
            w.raw[slot],
            outputs[0],
            outputs[1],
            h,
            i,
            256,
            num_warps=4,
        )
        _r8_scales[(triton.cdiv(3 * h * i // 32, 256),)](
            w.raw[slot],
            outputs[2].view(torch.uint8),
            outputs[3].view(torch.uint8),
            h,
            i,
            256,
            num_warps=4,
        )
        w.read[slot].record(stream)
        w.used[slot] = True
        return outputs
