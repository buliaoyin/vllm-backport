# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded decode diagnostics on an existing local development server."""

import json
import os
import time
from pathlib import Path


class DecodeBenchmarkWorker:
    def dsv41_decode_control(self, payload):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        settings = json.loads(payload)
        action = settings["action"]
        runner = self.model_runner
        state = runner.model_state
        modules = {
            name: module
            for name, module in runner.model.named_modules()
            if isinstance(module, CPUExpertModule)
        }
        torch.accelerator.synchronize()
        if action == "configure":
            if getattr(state, "hybrid_requests", None):
                raise RuntimeError("Change benchmark controls between requests")
            if "threads" in settings:
                threads = settings["threads"]
                if len(threads) != 2 or any(
                    type(n) is not int or n < 1 for n in threads
                ):
                    raise ValueError("Expected two positive CPU thread counts")
                state.cpu_phase_threads = threads
                for module in modules.values():
                    module.backend.set_num_threads(threads[1])
            if "freeze_cache" in settings:
                for module in modules.values():
                    if module.gpu_cache is not None:
                        module.gpu_cache.dynamic_enabled = not settings["freeze_cache"]
            if "selections" in settings:
                for name, module in modules.items():
                    if module.gpu_cache is not None:
                        module.gpu_cache.select(
                            settings["selections"][name], preserve_slots=False
                        )
            if "synchronize_stats" in settings:
                if not hasattr(self, "dsv41_unsynchronized_postprocess"):
                    self.dsv41_unsynchronized_postprocess = state.postprocess_state
                original = self.dsv41_unsynchronized_postprocess
                if settings["synchronize_stats"]:

                    def synchronized(*args, **kwargs):
                        result = original(*args, **kwargs)
                        torch.cuda.current_stream().synchronize()
                        if len(args) > 1 and isinstance(args[1], torch.Tensor):
                            args[1].sum().item()
                        return result

                    state.postprocess_state = synchronized
                else:
                    state.postprocess_state = original
            if "executor" in settings:
                for module in modules.values():
                    module.backend.set_execution_mode(settings["executor"])
            return {"pid": os.getpid(), "threads": state.cpu_phase_threads}
        if action == "snapshot":
            config = runner.vllm_config
            model = runner.model.language_model.model
            return {
                "pid": os.getpid(),
                "time_ns": time.monotonic_ns(),
                "device": torch.accelerator.current_device_index(),
                "layers": [model.start_layer, model.end_layer],
                "threads": state.cpu_phase_threads,
                "max_concurrent_batches": config.max_concurrent_batches,
                "use_ubatching": config.parallel_config.use_ubatching,
                "cpu": {name: m.backend.cuda_stats() for name, m in modules.items()},
                "cache": {
                    name: {
                        "device": str(m.gpu_cache.device),
                        "selected": m.gpu_cache.selected,
                        "dynamic": m.gpu_cache.dynamic_enabled,
                    }
                    for name, m in modules.items()
                    if m.gpu_cache is not None
                },
            }
        if action == "torch_profile_start":
            if modules:
                self.dsv41_torch_profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                self.dsv41_torch_profiler.start()
            return {"pid": os.getpid(), "enabled": bool(modules)}
        if action == "torch_profile_finish":
            profiler = getattr(self, "dsv41_torch_profiler", None)
            if profiler is not None:
                profiler.stop()
                destination = Path(settings["directory"]) / f"worker-{os.getpid()}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(destination))
                del self.dsv41_torch_profiler
                return {"pid": os.getpid(), "trace": str(destination)}
            return {"pid": os.getpid()}
        if action == "profile_start":
            interval = int(settings.get("interval", 16))
            if interval < 1:
                raise ValueError("Expected positive profile interval")
            for module in modules.values():
                module.backend.set_profile(interval)
            return {"pid": os.getpid(), "interval": interval}
        if action == "profile_finish":
            directory = Path(settings["directory"])
            result = {}
            for name, module in modules.items():
                result[name] = module.backend.profile_stats(directory / f"{name}.bin")
                module.backend.set_profile(0)
            return result
        if action == "timing_start":
            if hasattr(self, "dsv41_decode_timing"):
                raise RuntimeError("Decode timing already enabled")
            records, restores = [], []
            current = {}
            original_prepare = state.prepare_inputs
            restores.append((state, "prepare_inputs", original_prepare))

            def prepare(batch, requests):
                current.update(
                    num_reqs=batch.num_reqs,
                    num_tokens=batch.num_tokens,
                    prefill=batch.has_prefill,
                )
                return original_prepare(batch, requests)

            state.prepare_inputs = prepare

            def wrap(obj, method, label):
                original = getattr(obj, method)
                restores.append((obj, method, original))

                def timed(*args, **kwargs):
                    if len(records) >= 20000:
                        return original(*args, **kwargs)
                    begin, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    begin.record()
                    start = time.monotonic_ns()
                    result = original(*args, **kwargs)
                    finish = time.monotonic_ns()
                    end.record()
                    records.append((dict(current), label, start, finish, begin, end))
                    return result

                setattr(obj, method, timed)

            wrap(runner.cudagraph_manager, "run_fullgraph", "target_graph")
            wrap(state, "postprocess_state", "postprocess")
            speculator = getattr(runner, "speculator", None)
            draft = getattr(speculator, "query_cudagraph_manager", None)
            if draft is not None:
                wrap(draft, "run_fullgraph", "draft_graph")
            self.dsv41_decode_timing = records, restores
            return {"pid": os.getpid()}
        if action == "timing_finish":
            records, restores = self.dsv41_decode_timing
            for obj, method, original in reversed(restores):
                setattr(obj, method, original)
            del self.dsv41_decode_timing
            return {
                "pid": os.getpid(),
                "records": [
                    {
                        **batch,
                        "label": label,
                        "start_ns": start,
                        "end_ns": finish,
                        "host_ms": (finish - start) / 1e6,
                        "gpu_ms": begin.elapsed_time(end),
                    }
                    for batch, label, start, finish, begin, end in records
                ],
            }
        raise ValueError(f"Unknown decode diagnostic action: {action}")
