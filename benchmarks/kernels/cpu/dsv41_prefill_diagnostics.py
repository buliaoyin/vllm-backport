# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in prefill diagnostics and request-boundary chunk sweeps."""

import json
import os
import time
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class PrefillBenchmarkScheduler(AsyncScheduler):
    def add_request(self, request):
        control = json.loads(Path(os.environ["DSV41_PREFILL_CONTROL"]).read_text())
        chunk = control["chunk_tokens"]
        assert 16 <= chunk <= self.scheduler_config.max_num_batched_tokens
        assert not self.running and not self.waiting
        self.max_num_scheduled_tokens = chunk
        super().add_request(request)


class PrefillTimer:
    """CUDA-event intervals include stream waits; nested intervals overlap."""

    def __init__(self, worker):
        import torch

        from vllm.models.deepseek_v4_1 import ced

        self.events = []
        self.epoch = torch.cuda.Event(enable_timing=True)
        self.epoch_low_ns = time.perf_counter_ns()
        self.epoch.record()
        self.epoch.synchronize()
        self.epoch_high_ns = time.perf_counter_ns()
        self.restores = []
        self.active = False
        self.chunk = -1
        self.chunk_tokens = 0
        model = worker.model_runner.model.language_model.model

        def wrap(obj, attr, label, root=False):
            original = getattr(obj, attr)
            self.restores.append((obj, attr, original))

            def timed(*args, **kwargs):
                if root:
                    positions = kwargs.get("positions")
                    if positions is None and len(args) > 1:
                        positions = args[1]
                    self.active = positions is not None and positions.shape[0] > 16
                    if self.active:
                        self.chunk += 1
                        self.chunk_tokens = positions.shape[0]
                if not self.active:
                    return original(*args, **kwargs)
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                host_start = time.perf_counter_ns()
                result = original(*args, **kwargs)
                host_end = time.perf_counter_ns()
                end.record()
                self.events.append(
                    (
                        label,
                        self.chunk,
                        self.chunk_tokens,
                        start,
                        end,
                        host_start,
                        host_end,
                    )
                )
                if root:
                    self.active = False
                return result

            setattr(obj, attr, timed)

        wrap(model, "forward", "model", root=True)
        for index in range(model.start_layer, model.end_layer):
            layer = model.layers[index]
            prefix = f"layer{index}"
            wrap(layer, "forward", prefix)
            wrap(layer.attn, "forward", f"{prefix}.attention")
            wrap(layer.ffn, "forward", f"{prefix}.ffn")
            for module in layer.ffn.modules():
                cache = getattr(module, "gpu_cache", None)
                if cache is not None:
                    wrap(cache, "launch", f"{prefix}.gpu_cache.launch")
                    wrap(cache, "join", f"{prefix}.gpu_cache.join")
            for method in (
                "_run_parallel_input_projections",
                "_forward_prefill",
                "_combine_prefill_indices",
                "_o_proj",
            ):
                wrap(layer.attn, method, f"{prefix}.attention.{method}")
            if layer.attn.indexer is not None:
                wrap(layer.attn.indexer.indexer_op, "forward", f"{prefix}.indexer")
            if layer.engram is not None:
                wrap(layer.engram, "forward", f"{prefix}.engram")
                wrap(layer.engram, "prepare_embeddings", f"{prefix}.engram_lookup")
        if model.ced_prefill is not None:
            wrap(ced, "publish_decoder_kv", "publish_decoder_kv")

    def finish(self):
        import torch

        torch.accelerator.synchronize()
        for obj, attr, original in reversed(self.restores):
            setattr(obj, attr, original)
        return [
            {
                "label": label,
                "chunk": chunk,
                "chunk_tokens": tokens,
                "gpu_ms": start.elapsed_time(end),
                "host_ms": (host_end - host_start) / 1e6,
                "host_start_ns": host_start,
                "host_end_ns": host_end,
                "gpu_start_ns": (self.epoch_low_ns + self.epoch_high_ns) // 2
                + int(self.epoch.elapsed_time(start) * 1e6),
                "gpu_end_ns": (self.epoch_low_ns + self.epoch_high_ns) // 2
                + int(self.epoch.elapsed_time(end) * 1e6),
                "clock_uncertainty_ns": self.epoch_high_ns - self.epoch_low_ns,
            }
            for label, chunk, tokens, start, end, host_start, host_end in self.events
        ]


class ExpertRouteHistogram:
    """Count original expert routes and weight reuse without changing inputs."""

    def __init__(self, num_experts, device):
        import torch

        self.num_experts = num_experts
        self.routes = torch.zeros((2, num_experts), device=device, dtype=torch.int64)
        self.groups = torch.zeros_like(self.routes)
        self.tokens = torch.zeros(2, device=device, dtype=torch.int64)
        self.calls = torch.zeros_like(self.tokens)

    def record(self, ids, padding=None, actual_tokens=None):
        import torch

        phase = int(ids.shape[0] > 16)
        valid = (ids >= 0) & (ids < self.num_experts)
        if padding is not None:
            valid &= ~padding[: ids.shape[0], None]
        if actual_tokens is not None:
            valid &= (torch.arange(ids.shape[0], device=ids.device) < actual_tokens)[
                :, None
            ]
        counts = torch.zeros_like(self.routes[phase])
        counts.scatter_add_(
            0,
            ids.clamp(0, self.num_experts - 1).flatten().long(),
            valid.flatten().long(),
        )
        self.routes[phase].add_(counts)
        self.groups[phase].add_(counts != 0)
        self.tokens[phase].add_(valid.any(dim=1).sum())
        self.calls[phase].add_(1)

    def snapshot(self):
        return {
            name: {
                "route_counts": self.routes[index].tolist(),
                "expert_calls": self.groups[index].tolist(),
                "tokens": self.tokens[index].item(),
                "calls": self.calls[index].item(),
            }
            for index, name in enumerate(("decode", "prefill"))
        }


class ExpertRouteRecorder:
    """Eager diagnostic hooks observe routes before cache membership masking."""

    def __init__(self, worker, prefill_only=False):
        from vllm.forward_context import get_forward_context
        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        self.restores = []
        self.histograms = {}

        def wrap(module, histogram):
            original = module.forward
            self.restores.append((module, original))

            def recorded(hidden, ids, routes):
                if not prefill_only or ids.shape[0] > 16:
                    context = get_forward_context()
                    actual = None
                    if ids.shape[0] > 16 and isinstance(context.attn_metadata, dict):
                        actual = next(
                            (
                                item.num_actual_tokens
                                for item in context.attn_metadata.values()
                                if hasattr(item, "num_actual_tokens")
                            ),
                            None,
                        )
                    histogram.record(ids, context.is_padding, actual)
                return original(hidden, ids, routes)

            module.forward = recorded

        for name, module in worker.model_runner.model.named_modules():
            if isinstance(module, CPUExpertModule):
                cache = module.gpu_cache
                if cache is None:
                    raise ValueError("Route diagnostics require a GPU expert cache")
                histogram = ExpertRouteHistogram(cache.num_experts, cache.origin)
                self.histograms[name] = histogram
                wrap(module, histogram)

    def finish(self):
        for module, original in reversed(self.restores):
            module.forward = original
        return {
            name: histogram.snapshot() for name, histogram in self.histograms.items()
        }
