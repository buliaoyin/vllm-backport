# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary full-projection reconstruction followed by grouped FP16 GEMMs."""

import ctypes as ct
from pathlib import Path

import torch

from benchmarks.kernels.exl3_m32.launcher import Dim3
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl

CONFIGS = {
    "m64n128k32": dict(
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=3,
    ),
    "m64n128k64": dict(
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=3,
    ),
    "m128n128k32": dict(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_warps=8,
        num_stages=3,
    ),
    "m128n128k64": dict(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_warps=8,
        num_stages=3,
    ),
}


class Backend:
    def __init__(self, library, variant="m64n128k64"):
        self.variant = variant
        self.config = CONFIGS[variant]
        self.binary = ct.CDLL(str(Path(library).resolve()))
        self.kernels = {}
        for name in ("reconstruct", "gather", "activate", "scatter"):
            factory = getattr(self.binary, "fp16_" + name)
            factory.argtypes, factory.restype = [], ct.c_void_p
            self.kernels[name] = factory()
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart.so" in line
        }
        if len(paths) != 1:
            raise RuntimeError(f"Expected one CUDA runtime: {paths}")
        self.cuda = ct.CDLL(paths.pop())
        self.cuda.cudaLaunchKernel.argtypes = [
            ct.c_void_p,
            Dim3,
            Dim3,
            ct.POINTER(ct.c_void_p),
            ct.c_size_t,
            ct.c_void_p,
        ]
        self.cuda.cudaLaunchKernel.restype = ct.c_int
        self.cuda.cudaGetErrorString.argtypes = [ct.c_int]
        self.cuda.cudaGetErrorString.restype = ct.c_char_p
        self.workspaces = {}
        self.calls = 0

    def launch(self, name, values, grid, shared=0):
        pointers = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
        status = self.cuda.cudaLaunchKernel(
            self.kernels[name],
            Dim3(*grid),
            Dim3(256, 1, 1),
            pointers,
            shared,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(self.cuda.cudaGetErrorString(status).decode())

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
                    dtype=torch.float16,
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

        def gemm(a, b, c):
            invoke_fused_moe_triton_kernel(
                a,
                b.transpose(1, 2),
                c.unsqueeze(1),
                None,
                None,
                None,
                sorted_ids,
                expert_ids,
                padded,
                False,
                1,
                self.config,
                tl.float16,
                False,
                False,
                False,
                False,
                False,
            )

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
        return {
            "variant": self.variant,
            "config": self.config,
            "calls": self.calls,
            "workspace_bytes": sum(
                t.numel() * t.element_size()
                for group in self.workspaces.values()
                for t in group
            ),
        }
