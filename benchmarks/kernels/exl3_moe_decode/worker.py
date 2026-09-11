# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install experimental decode before model warmup and CUDA Graph capture."""

import hashlib
import os
from pathlib import Path

import torch
from kernels.exl3_m32.worker import Exl3RowsWorkerExtension

from benchmarks.kernels.exl3_m32.launcher import Launcher
from benchmarks.kernels.exl3_moe_decode.launcher import Decode
from vllm.model_executor.layers.quantization import exl3

_MODE = os.environ.get("VLLM_EXL3_EXPERIMENTAL_DECODE", "native")
_LIBRARY = os.environ.get("VLLM_EXL3_DECODE_LIBRARY")
_LAUNCHERS = {}
if _MODE not in ("native", "plain", "residual", "hybrid"):
    raise ValueError(f"Unknown expert decode mode: {_MODE}")
if _MODE != "native" and not _LIBRARY:
    raise ValueError("VLLM_EXL3_DECODE_LIBRARY is required for experimental decode")


def _decode(x, weights, ids, ptrs, workspace, bits, flags, limit):
    device = x.device.index
    if device not in _LAUNCHERS:
        residual = _MODE == "residual" or (
            _MODE == "hybrid" and torch.cuda.get_device_capability() == (12, 0)
        )
        _LAUNCHERS[device] = Decode(exl3._extension(), _LIBRARY, residual)
    return _LAUNCHERS[device](x, weights, ids, ptrs, workspace, bits, flags, limit)


if _MODE != "native":
    exl3._exl3_moe_decode = _decode


class PrefillLauncher(Launcher):
    variants = (
        *Launcher.variants,
        "base_m32_k32_n256",
        "nobar_m32_k32_n256",
        "half_m32_k32_n256",
    )


class Exl3DecodeWorkerExtension(Exl3RowsWorkerExtension):
    launcher_class = PrefillLauncher

    def get_exl3_runtime_state(self):
        result = super().get_exl3_runtime_state()
        launcher = _LAUNCHERS.get(torch.accelerator.current_device_index())
        result["expert_decode"] = {
            "mode": _MODE,
            "activation_mode": launcher.mode if launcher else None,
            "library": _LIBRARY,
            "sha256": hashlib.sha256(Path(_LIBRARY).read_bytes()).hexdigest()
            if _LIBRARY
            else None,
            "python_launch_calls_by_rows": launcher.calls if launcher else {},
            "grid_by_rows": {r: launcher.grid_for(r) for r in range(1, 9)}
            if launcher
            else {},
        }
        return result
