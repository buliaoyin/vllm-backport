# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only launcher for ExLlamaV3 1.4.8's private MoE/DevCtx ABI.

Supports SM80, uniform 4-bit mul1 experts and dimensions divisible by 256.
"""

import ctypes as ct
import importlib.metadata
from pathlib import Path

import torch


class Dim3(ct.Structure):
    _fields_ = [("x", ct.c_uint), ("y", ct.c_uint), ("z", ct.c_uint)]


class Launcher:
    def __init__(self, extension, library, variant="m32_predicated"):
        if importlib.metadata.version("exllamav3") != "1.4.8":
            raise ValueError("This experiment requires ExLlamaV3 1.4.8")
        if variant not in ("m16", "m32", "m32_predicated"):
            raise ValueError(f"Unknown row kernel: {variant}")
        self.original = extension.exl3_moe
        self.library = str(Path(library).resolve())
        self.variant = variant
        self.native = ct.CDLL(extension.__file__)
        self.kernels = ct.CDLL(self.library)
        factory = getattr(self.kernels, "exl3_rows_" + variant)
        factory.argtypes, factory.restype = [], ct.c_void_p
        self.kernel = factory()
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart.so" in line
        }
        if len(paths) != 1:
            raise RuntimeError(f"Expected one loaded CUDA runtime, found {paths}")
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
        self.cuda.cudaFuncSetAttribute.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
        self.cuda.cudaFuncSetAttribute.restype = ct.c_int
        self.cuda.cudaOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ct.POINTER(ct.c_int),
            ct.c_void_p,
            ct.c_int,
            ct.c_size_t,
        ]
        self.cuda.cudaOccupancyMaxActiveBlocksPerMultiprocessor.restype = ct.c_int
        self.cuda.cudaGetErrorString.argtypes = [ct.c_int]
        self.cuda.cudaGetErrorString.restype = ct.c_char_p
        instance = self.native._ZN6DevCtx8instanceEv
        instance.argtypes, instance.restype = [], ct.c_void_p
        self.context = instance()
        self.get_locks = self.native._ZN6DevCtx9get_locksEi
        self.get_locks.argtypes, self.get_locks.restype = (
            [ct.c_void_p, ct.c_int],
            ct.c_void_p,
        )
        self.plans = {}

    def check(self, status):
        if status:
            raise RuntimeError(self.cuda.cudaGetErrorString(status).decode())

    def plan(self, device, bits, flags):
        if tuple(bits) != (4, 4, 4) or tuple(flags) != (False, True) * 3:
            raise ValueError("Row kernels require uniform 4-bit mul1 experts")
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError("Row kernels are compiled and validated for SM80 only")
        if device not in self.plans:
            shared = 90 * 1024
            # cudaFuncAttributeMaxDynamicSharedMemorySize = 8.
            self.check(self.cuda.cudaFuncSetAttribute(self.kernel, 8, shared))
            active = ct.c_int()
            self.check(
                self.cuda.cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    ct.byref(active), self.kernel, 512, shared
                )
            )
            self.plans[device] = {
                "threads": 512,
                "shared_bytes": shared,
                "active_blocks_per_sm": active.value,
                "sms": torch.cuda.get_device_properties(device).multi_processor_count,
                "locks": self.get_locks(self.context, device),
            }
        return self.plans[device]

    def __call__(self, *args):
        if len(args) != 30:
            raise ValueError("Unexpected ExLlamaV3 MoE ABI")
        hidden, output, counts, tokens, routing = args[:5]
        workspace, activation, bits = args[5:9], args[9], args[10:13]
        ptrs, flags, limit, hint = args[13:22], args[22:28], args[28], args[29]
        if hidden.dtype != torch.float16 or output.dtype != torch.float32:
            raise ValueError("Expected FP16 inputs and FP32 output")
        if hidden.shape != output.shape or not hidden.is_contiguous():
            raise ValueError("Expected contiguous inputs and matching output shape")
        if hidden.shape[1] % 256 or workspace[2].shape[-1] % 256:
            raise ValueError("Row kernels require dimensions divisible by 256")
        device = hidden.device.index
        plan = self.plan(device, bits, flags)
        groups = min(workspace[0].shape[0], 64)
        width = 8
        if hint > 0:
            groups = min(groups, hint)
            width = min(plan["sms"] // groups, 32)
        if groups * width > plan["sms"] * plan["active_blocks_per_sm"]:
            raise ValueError("The cooperative expert grid must remain fully resident")
        tensors = [hidden, *workspace, output, *ptrs, counts, tokens, routing]
        if any(t.device != hidden.device or not t.is_contiguous() for t in tensors):
            raise ValueError("All kernel tensors must be contiguous on one device")
        values = [ct.c_void_p(t.data_ptr()) for t in tensors]
        dimensions = [
            hidden.shape[1],
            workspace[2].shape[-1],
            counts.numel() - 1,
            tokens.numel() // hidden.shape[0],
            workspace[0].shape[1],
            groups,
        ]
        values.extend(ct.c_int(value) for value in dimensions)
        values.extend(
            [
                ct.c_float(limit),
                ct.c_int(activation),
                *(ct.c_int(value) for value in bits),
                ct.c_void_p(plan["locks"]),
            ]
        )
        pointers = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
        self.check(
            self.cuda.cudaLaunchKernel(
                self.kernel,
                Dim3(width, 1, groups),
                Dim3(512, 1, 1),
                pointers,
                plan["shared_bytes"],
                torch.cuda.current_stream(device).cuda_stream,
            )
        )
