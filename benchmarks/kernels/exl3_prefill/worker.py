# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional prefill variants and calibrated CUDA-event layer timelines."""

import os
import time
from functools import wraps

import torch

from benchmarks.exl3_profile_worker import Exl3ProfileWorkerExtension

_LINEAR_CACHE = None
if os.environ.get("VLLM_EXL3_LINEAR_CACHE_GIB"):
    from benchmarks.kernels.exl3_prefill import linear_cache

    linear_cache.install(float(os.environ["VLLM_EXL3_LINEAR_CACHE_GIB"]))
    _LINEAR_CACHE = linear_cache.STATE

_VARIANT = os.environ.get("VLLM_EXL3_EXPERIMENTAL_PREFILL")
if _VARIANT:
    from benchmarks.kernels.exl3_prefill.launcher import Launcher
    from vllm.model_executor.layers.quantization import exl3

    _ORIGINAL_LOAD = exl3.Exl3MoEMethod.process_weights_after_loading

    @wraps(_ORIGINAL_LOAD)
    def _load(method, layer):
        result = _ORIGINAL_LOAD(method, layer)
        if method.m32_locks is not None and not hasattr(exl3, "_prefill_experiment"):
            launcher = Launcher(os.environ["VLLM_EXL3_PREFILL_LIBRARY"], _VARIANT)
            torch.ops._exl3_C.moe_m32 = launcher
            exl3._prefill_experiment = launcher
        return result

    exl3.Exl3MoEMethod.process_weights_after_loading = _load


class WorkerExtension(Exl3ProfileWorkerExtension):
    def get_exl3_runtime_state(self):
        from vllm.model_executor.layers.quantization import exl3

        result = super().get_exl3_runtime_state()
        launcher = getattr(exl3, "_prefill_experiment", None)
        result["prefill_experiment"] = launcher.metadata() if launcher else None
        result["linear_cache"] = _LINEAR_CACHE
        return result

    def install_exl3_event_profile(self):
        result = super().install_exl3_event_profile()
        state = self._exl3_event_profile
        samples = []
        for _ in range(8):
            anchor = torch.cuda.Event(enable_timing=True)
            torch.accelerator.synchronize()
            before = time.perf_counter_ns()
            anchor.record()
            anchor.synchronize()
            after = time.perf_counter_ns()
            samples.append((after - before, before, after, anchor))
        span, before, after, anchor = min(samples, key=lambda row: row[0])
        state["anchor"] = anchor
        state["anchor_host_midpoint_ms"] = (before + after) / 2e6
        state["anchor_uncertainty_ms"] = span / 2e6
        names = {row["name"] for row in self.get_exl3_runtime_state()["decoder_layers"]}

        def wrap(module, name):
            original = module.forward
            state["originals"].append((module, "forward", original))

            @wraps(original)
            def measured(*args, **kwargs):
                index = len(state["events"])
                if index >= len(state["pool"]):
                    raise RuntimeError("CUDA event pool exhausted")
                start, end = state["pool"][index]
                state["events"].append(("layer:" + name, start, end))
                start.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    end.record()

            module.forward = measured

        for name, module in self.get_model().named_modules():
            if name in names:
                wrap(module, name)
        result["anchor_uncertainty_ms"] = state["anchor_uncertainty_ms"]
        return result

    def collect_exl3_event_profile(self):
        state = self._exl3_event_profile
        torch.accelerator.synchronize()
        anchor = state["anchor"]
        timeline = [
            {
                "label": label,
                "start_ms": anchor.elapsed_time(start),
                "end_ms": anchor.elapsed_time(end),
            }
            for label, start, end in state["events"]
        ]
        midpoint = state["anchor_host_midpoint_ms"]
        uncertainty = state["anchor_uncertainty_ms"]
        result = super().collect_exl3_event_profile()
        result.update(
            timeline=timeline,
            anchor_host_midpoint_ms=midpoint,
            anchor_uncertainty_ms=uncertainty,
        )
        return result
