# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in multi-expert DP4A experiment; one stream owns its workspace."""

import ctypes as ct
import importlib.metadata
from pathlib import Path

import torch

from benchmarks.kernels.exl3_m32.launcher import Dim3


class Decode:
    def __init__(self, extension, library, residual=False, grid=None):
        if importlib.metadata.version("exllamav3") != "1.4.8":
            raise ValueError("This experiment requires ExLlamaV3 1.4.8")
        if torch.cuda.get_device_capability() not in ((8, 0), (12, 0)):
            raise ValueError("The experiment supports SM80 and SM120")
        if grid is not None and grid not in (4, 8, 16, 32, 64):
            raise ValueError("Unsupported grid size")
        self.ext = extension
        self.library = str(Path(library).resolve())
        self.lib = ct.CDLL(self.library)
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart.so" in line
        }
        if len(paths) != 1:
            raise RuntimeError(f"Expected one CUDA runtime, found {paths}")
        self.cuda = ct.CDLL(paths.pop())
        self.cuda.cudaGetErrorString.argtypes = [ct.c_int]
        self.cuda.cudaGetErrorString.restype = ct.c_char_p
        self.cuda.cudaLaunchKernel.argtypes = [
            ct.c_void_p,
            Dim3,
            Dim3,
            ct.POINTER(ct.c_void_p),
            ct.c_size_t,
            ct.c_void_p,
        ]
        self.cuda.cudaLaunchKernel.restype = ct.c_int
        self.cuda.cudaFuncSetAttribute.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
        self.cuda.cudaFuncSetAttribute.restype = ct.c_int
        self.residual, self.grid = residual, grid
        self.functions, self.buffers, self.calls = {}, {}, {}
        self.mode = "residual" if residual else "plain"
        for name in (self.mode + "_half", self.mode + "_float", "combine"):
            factory = getattr(self.lib, "exl3_decode_" + name)
            factory.argtypes, factory.restype = [], ct.c_void_p
            self.functions[name] = factory()
            if name != "combine":
                self.check(
                    self.cuda.cudaFuncSetAttribute(self.functions[name], 8, 90 * 1024)
                )
                self.check(self.cuda.cudaFuncSetAttribute(self.functions[name], 9, 100))

    def check(self, status):
        if status:
            raise RuntimeError(self.cuda.cudaGetErrorString(status).decode())

    def launch(self, kernel, tensors, ints, grid, threads, shared):
        values = [ct.c_void_p(t.data_ptr()) for t in tensors]
        values.extend(ct.c_int(v) for v in ints)
        args = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
        self.check(
            self.cuda.cudaLaunchKernel(
                kernel,
                Dim3(*grid),
                Dim3(threads, 1, 1),
                args,
                shared,
                torch.cuda.current_stream().cuda_stream,
            )
        )

    def grid_for(self, rows):
        if self.grid is not None:
            return self.grid
        if torch.cuda.get_device_capability() == (8, 0):
            return 32 if rows <= 4 else 16
        return 64 if rows == 1 or rows > 4 else 32 if rows == 2 else 16

    def project(self, x, ptrs, ids, output, input_group, grid):
        slots, k, n = ids.numel(), x.shape[-1], output.shape[-1]
        rows = k // 16
        r = (rows * (n // 256) + grid - 1) // grid
        rows_max = min(
            512, ((80 * 1024) // (32 + 64 * (2 if self.residual else 1))) & ~7
        )
        per = min(max((max(r, min(2 * r, 32)) + 7) & ~7, 16), rows_max, (rows + 7) & ~7)
        split = (rows + per - 1) // per
        stride = 5120 + split * n * (2 if self.residual else 1)
        key = (x.device, slots, stride)
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(
                slots * stride, device=x.device, dtype=torch.int32
            )
        shared = per * (32 + 64 * (2 if self.residual else 1)) + 1024
        suffix = "_float" if output.dtype == torch.float32 else "_half"
        self.launch(
            self.functions[self.mode + suffix],
            [x, ptrs[0], output, ptrs[1], ptrs[2], ids, self.buffers[key]],
            [k, n, 8, input_group, stride],
            (grid, 1, slots),
            256,
            shared,
        )

    def __call__(self, x, weights, ids, ptrs, workspace, bits, flags, limit):
        if tuple(bits) != (4, 4, 4) or tuple(flags) != (False, True) * 3:
            raise ValueError("INT8 decode requires uniform 4-bit mul1 experts")
        if x.device.index != torch.accelerator.current_device_index():
            raise ValueError("The input CUDA device must be current")
        if ids.device != x.device or weights.device != x.device:
            raise ValueError("Routes and weights must be on the input device")
        rows, k = x.shape
        n = workspace[2].shape[-1]
        if x.dtype != torch.bfloat16 or not 1 <= rows <= 8:
            raise ValueError("INT8 decode requires 1 to 8 BF16 rows")
        if any(d % 256 or not 256 <= d <= 8192 for d in (k, n)):
            raise ValueError(
                "Dimensions must be multiples of 256, between 256 and 8192"
            )
        if (
            ids.ndim != 2
            or ids.shape != weights.shape
            or ids.shape[0] != rows
            or not 1 <= ids.shape[1] <= 8
        ):
            raise ValueError(
                "Expected matching routes with at most eight experts per row"
            )
        if len(ptrs) != 9 or any(
            t.device != x.device or t.dtype != torch.int64 or not t.is_contiguous()
            for t in ptrs
        ):
            raise ValueError(
                "Expected nine contiguous pointer tables on the input device"
            )
        topk, slots = ids.shape[1], ids.numel()
        grid = self.grid_for(rows)
        self.calls[rows] = self.calls.get(rows, 0) + 1
        hidden = x.to(torch.float16).contiguous()
        indices = ids.to(torch.int64).contiguous().flatten()
        routing = weights.to(torch.float16).contiguous().flatten()
        gate = torch.empty((slots, n), dtype=torch.float16, device=x.device)
        up, activated = torch.empty_like(gate), torch.empty_like(gate)
        self.project(hidden, ptrs[:3], indices, gate, topk, grid)
        self.project(hidden, ptrs[3:6], indices, up, topk, grid)
        self.ext.silu_mul(gate, up, activated, limit)
        down = torch.empty((slots, k), dtype=torch.float32, device=x.device)
        self.project(activated, ptrs[6:9], indices, down, 1, grid)
        output = torch.empty((rows, k), dtype=x.dtype, device=x.device)
        self.launch(
            self.functions["combine"],
            [down, routing, output],
            [rows, topk, k],
            ((rows * k + 255) // 256, 1, 1),
            256,
            0,
        )
        return output
