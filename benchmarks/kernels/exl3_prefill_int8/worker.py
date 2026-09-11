# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in full-projection INT8 prefill for large SM80 token batches."""

import os
from functools import wraps

from benchmarks.kernels.exl3_prefill.worker import WorkerExtension as BaseWorker
from benchmarks.kernels.exl3_prefill_int8.backend import Backend
from vllm.model_executor.layers.quantization import exl3

_BACKEND = None
_MIN_ROWS = int(os.environ.get("VLLM_EXL3_BATCHED_INT8_MIN_ROWS", "4096"))
_FALLBACK_CALLS = 0
_ORIGINAL_LOAD = exl3.Exl3MoEMethod.process_weights_after_loading
_ORIGINAL_FUSED = exl3._exl3_moe_fused


@wraps(_ORIGINAL_FUSED)
def _fused(x, *args, **kwargs):
    global _FALLBACK_CALLS
    if x.shape[0] >= _MIN_ROWS:
        return _BACKEND(x, *args, **kwargs)
    _FALLBACK_CALLS += 1
    return _ORIGINAL_FUSED(x, *args, **kwargs)


@wraps(_ORIGINAL_LOAD)
def _load(method, layer):
    global _BACKEND
    result = _ORIGINAL_LOAD(method, layer)
    if method.m32_locks is not None and _BACKEND is None:
        if _MIN_ROWS <= 8:
            raise ValueError("INT8 prefill threshold must exceed decode batch sizes")
        _BACKEND = Backend(
            os.environ["VLLM_EXL3_BATCHED_INT8_LIBRARY"],
            os.environ.get("VLLM_EXL3_BATCHED_INT8_CONFIG", "m64n128k64"),
        )
        exl3._exl3_moe_fused = _fused
    return result


exl3.Exl3MoEMethod.process_weights_after_loading = _load


class WorkerExtension(BaseWorker):
    def get_exl3_runtime_state(self):
        result = super().get_exl3_runtime_state()
        result["batched_int8_prefill"] = (
            {
                **_BACKEND.metadata(),
                "minimum_rows": _MIN_ROWS,
                "fallback_calls": _FALLBACK_CALLS,
            }
            if _BACKEND is not None
            else None
        )
        return result
