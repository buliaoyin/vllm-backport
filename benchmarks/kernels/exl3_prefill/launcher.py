# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark launcher with caller-owned locks and occupancy-sized workspaces."""

import ctypes as ct
import json
from pathlib import Path

import torch

from benchmarks.kernels.exl3_m32.launcher import Dim3


class Launcher:
    def __init__(self, library, variant):
        self.library = str(Path(library).resolve())
        metadata = json.loads(Path(self.library).with_name("build.json").read_text())
        self.config = metadata["variants"][variant]
        self.variant = variant
        self.binary = ct.CDLL(self.library)
        factory = getattr(self.binary, "exl3_prefill_" + variant)
        factory.argtypes, factory.restype = [], ct.c_void_p
        self.kernel = factory()
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart.so" in line
        }
        if len(paths) != 1:
            raise RuntimeError(f"Expected one CUDA runtime, got {paths}")
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
        self.binary.exl3_initialize_lookup.argtypes = [ct.c_void_p, ct.c_void_p]
        self.binary.exl3_initialize_lookup.restype = ct.c_int
        self.plans = {}
        self.workspaces = {}

    def check(self, status):
        if status:
            raise RuntimeError(self.cuda.cudaGetErrorString(status).decode())

    def plan(self, device):
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError("Prefill experiments require SM80")
        if device not in self.plans:
            threads, shared = self.config["threads"], self.config["shared_bytes"]
            self.check(self.cuda.cudaFuncSetAttribute(self.kernel, 8, shared))
            # Prefer L1, subject to the requested shared memory and residency.
            self.check(self.cuda.cudaFuncSetAttribute(self.kernel, 9, 0))
            active = ct.c_int()
            self.check(
                self.cuda.cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    ct.byref(active),
                    self.kernel,
                    threads,
                    shared,
                )
            )
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            groups = min(64, active.value * sms // 8)
            if groups <= 0:
                raise RuntimeError("No fully resident expert group")
            table = None
            if self.config["lookup"]:
                table = torch.empty(65536, dtype=torch.float16, device=device)
                self.check(
                    self.binary.exl3_initialize_lookup(
                        table.data_ptr(),
                        torch.cuda.current_stream(device).cuda_stream,
                    )
                )
            self.plans[device] = {
                "active_blocks_per_sm": active.value,
                "sms": sms,
                "groups": groups,
                "width": 8,
                "lookup": table,
            }
        return self.plans[device]

    def __call__(self, *args):
        if len(args) != 20:
            raise ValueError("Expected the native moe_m32 operator contract")
        hidden, output, counts, tokens, routing = args[:5]
        old, ptrs, original_locks, limit = args[5:9], args[9:18], args[18], args[19]
        device = hidden.device.index
        if torch.accelerator.current_device_index() != device:
            raise ValueError("Make the tensor device current before launching")
        if hidden.dtype != torch.float16 or output.dtype != torch.float32:
            raise ValueError("Expected FP16 input and FP32 output")
        k, n, capacity = hidden.shape[1], old[2].shape[2], old[0].shape[1]
        if any(width < 256 or width > 8192 or width % 256 for width in (k, n)):
            raise ValueError("Dimensions must be multiples of 256 in [256, 8192]")
        if hidden.shape != output.shape or hidden.shape[0] > capacity:
            raise ValueError("Output shape or workspace capacity mismatch")
        plan = self.plan(device)
        key = (device, plan["groups"], capacity, k, n)
        if key not in self.workspaces:
            self.workspaces[key] = (
                [
                    torch.empty(
                        (plan["groups"], capacity, width),
                        dtype=torch.float16,
                        device=hidden.device,
                    )
                    for width in (k, k, n, n)
                ],
                torch.zeros_like(original_locks),
            )
        workspace, locks = self.workspaces[key]
        tensors = [hidden, *workspace, output, *ptrs, counts, tokens, routing]
        if any(t.device != hidden.device or not t.is_contiguous() for t in tensors):
            raise ValueError("All tensors must be contiguous on the same device")
        values = [ct.c_void_p(t.data_ptr()) for t in tensors]
        values += [
            ct.c_int(x)
            for x in (
                k,
                n,
                counts.numel() - 1,
                tokens.numel() // hidden.shape[0],
                capacity,
                plan["groups"],
            )
        ]
        values += [
            ct.c_float(limit),
            ct.c_int(0),
            ct.c_int(4),
            ct.c_int(4),
            ct.c_int(4),
            ct.c_void_p(locks.data_ptr()),
        ]
        pointers = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
        self.check(
            self.cuda.cudaLaunchKernel(
                self.kernel,
                Dim3(8, 1, plan["groups"]),
                Dim3(self.config["threads"], 1, 1),
                pointers,
                self.config["shared_bytes"],
                torch.cuda.current_stream(device).cuda_stream,
            )
        )

    def metadata(self):
        return {
            "variant": self.variant,
            "config": self.config,
            "plans": {
                str(k): {x: y for x, y in p.items() if x != "lookup"}
                for k, p in self.plans.items()
            },
            "workspace_bytes": sum(
                sum(t.numel() * t.element_size() for t in tensors)
                + locks.numel() * locks.element_size()
                for tensors, locks in self.workspaces.values()
            ),
        }
