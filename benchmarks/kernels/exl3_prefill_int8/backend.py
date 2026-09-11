# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary INT8 expert weights, row quantization and grouped IMMA."""

import ctypes as ct
import struct

import torch

from benchmarks.kernels.exl3_prefill_fp16.backend import Backend as Fp16Backend
from benchmarks.kernels.exl3_prefill_int8.kernels import grouped, quantize
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import triton

ALPHA = struct.unpack("<e", bytes.fromhex("ee1e"))[0]
ALPHA4 = 4 * ALPHA
BETA = 1534 * ALPHA + struct.unpack("<e", bytes.fromhex("31c9"))[0]
CONFIGS = {
    f"m{m}n128k{k}": dict(
        BLOCK_SIZE_M=m,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=k,
        GROUP_SIZE_M=8,
        num_warps=4 if m == 64 else 8,
        num_stages=3,
    )
    for m in (64, 128)
    for k in (64, 128)
}


class Backend(Fp16Backend):
    def __init__(self, library, variant="m64n128k64"):
        super().__init__(library)
        self.variant = variant
        self.config = CONFIGS[variant]
        self.quant_workspaces = {}
        self.gemm_metadata = {}

    def __call__(
        self,
        x,
        topk_weights,
        topk_ids,
        ptrs,
        workspace,
        bits,
        flags,
        limit,
        m32_locks=None,
    ):
        if tuple(bits) != (4, 4, 4) or tuple(flags) != (False, True) * 3:
            raise ValueError("Requires uniform 4-bit mul1 experts")
        if torch.cuda.get_device_capability(x.device) != (8, 0):
            raise ValueError("This experimental library targets SM80")
        if torch.accelerator.current_device_index() != x.device.index:
            raise ValueError("Make the input device current")
        rows, hidden = x.shape
        intermediate = workspace[2].shape[2]
        experts, topk = ptrs[0].numel(), topk_ids.shape[1]
        capacity = max(rows, workspace[0].shape[1])
        slots = rows * topk
        key = (x.device, experts, hidden, intermediate, capacity, topk)
        if key not in self.workspaces:
            self.workspaces[key] = (
                torch.empty(
                    experts * hidden * intermediate,
                    device=x.device,
                    dtype=torch.int8,
                ),
                torch.empty(
                    (capacity * topk, hidden), device=x.device, dtype=torch.float16
                ),
                torch.empty(
                    (capacity * topk, intermediate),
                    device=x.device,
                    dtype=torch.float16,
                ),
                torch.empty(
                    (capacity * topk, intermediate),
                    device=x.device,
                    dtype=torch.float16,
                ),
            )
        weight, stage, gate, up = self.workspaces[key]
        stage, gate, up = stage[:slots], gate[:slots], up[:slots]
        inp = x.to(torch.float16).contiguous()
        ids = topk_ids.to(torch.int64).contiguous().flatten()
        routing = topk_weights.to(torch.float16).contiguous().flatten()
        sorted_ids, expert_ids, padded = moe_align_block_size(
            topk_ids, self.config["BLOCK_SIZE_M"], experts, pad_sorted_ids=True
        )
        output = torch.zeros((rows, hidden), device=x.device, dtype=torch.float32)

        def pointers(*tensors):
            return [ct.c_void_p(t.data_ptr()) for t in tensors]

        if key not in self.quant_workspaces:
            self.quant_workspaces[key] = (
                torch.empty(
                    capacity * topk * max(hidden, intermediate),
                    device=x.device,
                    dtype=torch.int8,
                ),
                torch.empty(capacity * topk, device=x.device, dtype=torch.float32),
                torch.empty(capacity * topk, device=x.device, dtype=torch.float32),
            )
        quantized, scales, sums = self.quant_workspaces[key]

        def gemm(a, b, c):
            k, n = b.shape[1:]
            q = quantized[: slots * k].view(slots, k)
            quantize[(slots,)](
                a, q, scales, sums, k, triton.next_power_of_2(k), num_warps=4
            )
            bm, bn = self.config["BLOCK_SIZE_M"], self.config["BLOCK_SIZE_N"]
            grid = ((sorted_ids.numel() // bm) * triton.cdiv(n, bn),)
            compiled = grouped[grid](
                q,
                b,
                c,
                scales,
                sums,
                sorted_ids,
                expert_ids,
                padded,
                slots,
                n,
                k,
                sorted_ids.numel(),
                ALPHA4,
                BETA,
                **self.config,
            )
            if (k, n) not in self.gemm_metadata:
                ptx = compiled.asm["ptx"]
                self.gemm_metadata[k, n] = {
                    "registers": compiled.n_regs,
                    "spills": compiled.n_spills,
                    "shared_bytes": compiled.metadata.shared,
                    "imma": "mma.sync" in ptx and ".s32.s8.s8.s32" in ptx,
                }

        for i, destination in ((0, gate), (1, up)):
            self.launch(
                "gather",
                pointers(inp, ids, stage, ptrs[i * 3 + 1])
                + [ct.c_int(slots), ct.c_int(hidden), ct.c_int(topk)],
                ((slots * (hidden // 128) + 7) // 8, 1, 1),
            )
            self.launch(
                "reconstruct",
                pointers(weight, ptrs[i * 3])
                + [ct.c_int(intermediate // 16), ct.c_int(hidden)],
                (intermediate // 128, hidden // 16, experts),
            )
            gemm(stage, weight.view(experts, hidden, intermediate), destination)
        self.launch(
            "activate",
            pointers(gate, up, ids, ptrs[2], ptrs[5], ptrs[7])
            + [ct.c_int(slots), ct.c_int(intermediate), ct.c_float(limit)],
            ((slots * (intermediate // 128) + 7) // 8, 1, 1),
        )
        self.launch(
            "reconstruct",
            pointers(weight, ptrs[6])
            + [ct.c_int(hidden // 16), ct.c_int(intermediate)],
            (hidden // 128, intermediate // 16, experts),
        )
        gemm(gate, weight.view(experts, intermediate, hidden), stage)
        self.launch(
            "scatter",
            pointers(stage, output, ids, routing, ptrs[8])
            + [ct.c_int(slots), ct.c_int(hidden), ct.c_int(topk)],
            ((slots * (hidden // 128) + 7) // 8, 1, 1),
            shared=4096,
        )
        self.calls += 1
        return output.to(x.dtype)

    def metadata(self):
        result = super().metadata()
        result["activation_quantization_bytes"] = sum(
            t.numel() * t.element_size()
            for group in self.quant_workspaces.values()
            for t in group
        )
        result["weight_dtype"] = "int8"
        result["gemm_kernels"] = {
            f"{k}x{n}": v for (k, n), v in self.gemm_metadata.items()
        }
        return result
