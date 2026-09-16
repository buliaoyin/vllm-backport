# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routed experts with GPU-resident routing and shared experts."""

from __future__ import annotations

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4 import (
    CPUMoEConfig,
    CPUMXFP4Experts,
)
from vllm.v1.metrics.stats import ExpertCacheStats

logger = init_logger(__name__)


def max_callback_tokens(vllm_config: VllmConfig) -> int:
    """Bound callbacks by graph shapes the decode dispatcher can execute."""
    compilation = vllm_config.compilation_config
    capacity = compilation.max_cudagraph_capture_size or 16
    if getattr(compilation, "cudagraph_mode", None) == CUDAGraphMode.FULL_DECODE_ONLY:
        capacity = min(
            capacity,
            vllm_config.scheduler_config.max_num_seqs
            * vllm_config.uniform_decode_query_len,
        )
    return capacity


def cpu_moe_config(vllm_config: VllmConfig, layer: int) -> CPUMoEConfig | None:
    extra = vllm_config.additional_config
    raw = extra.get("cpu_moe") if isinstance(extra, dict) else None
    if raw is None:
        return None
    raw = dict(raw)
    selections = raw.pop("gpu_cache_selections", None)
    devices = raw.pop("gpu_cache_devices", None)
    dynamic = raw.pop("gpu_cache_dynamic", None)
    if raw.get("start_layer", 20) <= layer < raw.get("end_layer", 40):
        if selections is not None:
            if str(layer) not in selections:
                raise ValueError(f"Missing static expert selection for layer {layer}")
            raw["gpu_cache_static_experts"] = tuple(selections[str(layer)])
            raw["gpu_cache_experts"] = len(raw["gpu_cache_static_experts"])
        if devices is not None:
            if str(layer) not in devices:
                raise ValueError(f"Missing GPU cache device for layer {layer}")
            raw["gpu_cache_device"] = devices[str(layer)]
        if dynamic is not None:
            raw["gpu_cache_dynamic"] = dynamic.get(str(layer))
    config = CPUMoEConfig(**raw)
    hf = vllm_config.model_config.hf_config
    if config.end_layer > hf.num_hidden_layers:
        raise ValueError("CPU MoE layer range exceeds the target model")
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.enable_expert_parallel:
        raise ValueError("CPU MoE currently requires TP1 without expert parallelism")
    if parallel.enable_eplb or parallel.use_ubatching:
        raise ValueError("CPU MoE does not support EPLB or overlapping microbatches")
    if getattr(hf, "expert_dtype", "fp4") != "fp4":
        raise ValueError("CPU MoE requires native MXFP4 expert weights")
    compilation = vllm_config.compilation_config
    full_callbacks = (
        config.cuda_library_path is not None
        and compilation.cudagraph_mode.has_full_cudagraphs()
        and max_callback_tokens(vllm_config) <= 128
    )
    if compilation.cudagraph_mode != CUDAGraphMode.NONE and not (
        full_callbacks
        or (
            is_breakable_cudagraph_enabled()
            and compilation.cudagraph_mode == CUDAGraphMode.PIECEWISE
        )
    ):
        raise ValueError(
            "CPU MoE needs --enforce-eager, or VLLM_USE_BREAKABLE_CUDAGRAPH=1 "
            "with PIECEWISE graphs"
        )
    return config if config.start_layer <= layer < config.end_layer else None


class CPUExpertModule(nn.Module):
    def __init__(self, config: CPUMoEConfig, vllm_config: VllmConfig):
        super().__init__()
        hf = vllm_config.model_config.hf_config
        self.backend = CPUMXFP4Experts(
            config,
            hf.n_routed_experts,
            hf.hidden_size,
            hf.moe_intermediate_size,
            hf.num_experts_per_tok,
            float(hf.swiglu_limit or 0),
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        self.gpu_cache = None
        self._eager_gpu_hits = 0
        self._eager_cpu_misses = 0
        self._last_cache_stats = ExpertCacheStats()
        self.use_prefill_cache = config.gpu_cache_prefill
        if config.gpu_cache_experts:
            from .expert_cache import GPUExpertCache

            if config.gpu_cache_experts >= hf.n_routed_experts:
                raise ValueError(
                    "GPU expert cache must be smaller than the expert bank"
                )
            library = self.backend._library
            if (
                not hasattr(library, "dsv41_moe_supports_cached_routes")
                or not (library.dsv41_moe_supports_cached_routes())
                or not hasattr(
                    self.backend._cuda_library, "dsv41_cuda_task_cached_routes"
                )
            ):
                raise RuntimeError(
                    "GPU expert cache requires updated CPU and CUDA bridges"
                )
            self.gpu_cache = GPUExpertCache(
                config.gpu_cache_experts,
                hf.n_routed_experts,
                config.gpu_cache_device,
                float(hf.swiglu_limit or 0),
                hf.num_experts_per_tok,
                weight_source=self.backend.export_expert
                if self.backend.supports_export
                else None,
            )
        self.pending: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.num_loaded = 0
        self.async_host: tuple[torch.Tensor, ...] | None = None
        self.async_tokens = max(
            16,
            min(128, max_callback_tokens(vllm_config)),
        )
        self.hybrid_has_prefill = False
        self.host_tokens = 0
        self.last_tokens = 0
        self.hidden_size = hf.hidden_size
        self.top_k = hf.num_experts_per_tok
        logger.info(
            "CPU routed experts: backend=%s, threads=%d, experts=%d, H=%d, I=%d",
            config.backend,
            config.num_threads,
            hf.n_routed_experts,
            hf.hidden_size,
            hf.moe_intermediate_size,
        )

    def load_weight(
        self, expert: int, projection: int, kind: str, weight: torch.Tensor
    ) -> None:
        pair = self.pending.setdefault((expert, projection), {})
        if kind in pair:
            raise ValueError("Duplicate CPU expert tensor")
        pair[kind] = weight
        if "weight" in pair and "weight_scale" in pair:
            self.backend.load_expert(
                expert, projection, pair["weight"], pair["weight_scale"]
            )
            if self.gpu_cache is not None:
                self.gpu_cache.load_weight(
                    expert, projection, pair["weight"], pair["weight_scale"]
                )
            del self.pending[expert, projection]
            self.num_loaded += 1
            if self.num_loaded == 3 * self.backend.num_experts:
                self.backend.prepare()

    def finalize(self) -> None:
        if self.pending:
            raise RuntimeError("Unpaired CPU expert weights or scales")
        self.backend.prepare()
        if self.gpu_cache is not None:
            if (
                self.backend.config.gpu_cache_static_experts is not None
                and self.gpu_cache.packed is None
            ):
                self.gpu_cache.select_static(
                    self.backend.config.gpu_cache_static_experts
                )
            self.gpu_cache.finalize(self.async_tokens)
            if self.backend.config.gpu_cache_dynamic is not None:
                self.gpu_cache.prepare_dynamic(self.backend.config.gpu_cache_dynamic)
        if self.backend.config.cuda_library_path and self.async_host is None:
            kwargs = {"device": "cpu", "pin_memory": True}
            self.async_host = (
                torch.empty(
                    (self.async_tokens, self.hidden_size), dtype=torch.float32, **kwargs
                ),
                torch.empty(
                    (self.async_tokens, self.top_k), dtype=torch.int32, **kwargs
                ),
                torch.empty(
                    (self.async_tokens, self.top_k), dtype=torch.float32, **kwargs
                ),
                torch.empty(
                    (self.async_tokens, self.hidden_size), dtype=torch.float32, **kwargs
                ),
            )

    def take_cache_stats(self) -> ExpertCacheStats:
        """Collect deltas after the existing postprocess stream synchronization."""
        gpu_hits, cpu_misses = self.backend.cuda_route_counts()
        stats = ExpertCacheStats(
            gpu_hits=gpu_hits + self._eager_gpu_hits,
            cpu_misses=cpu_misses + self._eager_cpu_misses,
        )
        if self.gpu_cache is not None:
            cache = self.gpu_cache
            stats.updates = cache.reload_updates
            stats.decode_checks = cache.dynamic_stats.get("decode_observations", 0)
            stats.experts_reloaded = cache.reloaded_experts
            stats.reload_seconds = cache.reload_seconds
            stats.host_lru_hits = cache.dynamic_stats["lru_hits"]
            stats.repacked_experts = cache.dynamic_stats["repacked_experts"]
        delta = stats.delta(self._last_cache_stats)
        self._last_cache_stats = stats
        return delta

    def _forward_async(self, hidden, ids, routes):
        tokens = hidden.shape[0]
        assert self.async_host is not None
        hidden_cpu, ids_cpu, routes_cpu, output_cpu = self.async_host
        padding = get_forward_context().is_padding
        if padding is not None:
            ids = ids.masked_fill(padding[:tokens, None], -1)
        hidden_cpu[:tokens].copy_(hidden.float(), non_blocking=True)
        ids_cpu[:tokens].copy_(ids.int(), non_blocking=True)
        routes_cpu[:tokens].copy_(routes, non_blocking=True)
        self.backend.enqueue_cuda(
            tokens,
            hidden_cpu,
            ids_cpu,
            routes_cpu,
            output_cpu,
            torch.cuda.current_stream().cuda_stream,
        )
        return (
            output_cpu[:tokens]
            .to(device=hidden.device, non_blocking=True)
            .to(hidden.dtype)
        )

    @eager_break_during_capture
    def _forward_cpu(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        routes: torch.Tensor,
        output: torch.Tensor,
        tokens_override: int | None = None,
    ) -> None:
        tokens = hidden.shape[0] if tokens_override is None else tokens_override
        if tokens_override is None and is_forward_context_available():
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                tokens = next(
                    (
                        item.num_actual_tokens
                        for item in metadata.values()
                        if hasattr(item, "num_actual_tokens")
                    ),
                    tokens,
                )
        if not 0 <= tokens <= hidden.shape[0]:
            raise ValueError("CPU MoE token count exceeds the physical batch")
        # Attention can leave padding rows nonfinite; TP1 packs real tokens first.
        if tokens < hidden.shape[0]:
            output.zero_()
        self.last_tokens = tokens
        if not tokens:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CPU MoE must execute outside CUDA graph capture")
        valid_rows = None
        if tokens_override is None and is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                valid_rows = ~padding[:tokens].cpu()
                if not valid_rows.any():
                    output.zero_()
                    self.last_tokens = 0
                    return
                if valid_rows.all():
                    valid_rows = None
        if tokens > self.host_tokens:
            self.host_tokens = tokens
            kwargs = {"device": "cpu", "pin_memory": True}
            self.host_hidden = torch.empty(
                (tokens, self.hidden_size), dtype=torch.bfloat16, **kwargs
            )
            self.host_ids = torch.empty(
                (tokens, self.top_k), dtype=torch.int64, **kwargs
            )
            self.host_routes = torch.empty(
                (tokens, self.top_k), dtype=torch.float32, **kwargs
            )
            self.host_output = torch.empty_like(self.host_hidden, **kwargs)
        hidden_cpu = self.host_hidden[:tokens]
        ids_cpu = self.host_ids[:tokens]
        routes_cpu = self.host_routes[:tokens]
        hidden_cpu.copy_(hidden[:tokens], non_blocking=True)
        ids_cpu.copy_(ids[:tokens], non_blocking=True)
        routes_cpu.copy_(routes[:tokens], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        if valid_rows is not None:
            hidden_cpu = hidden_cpu[valid_rows]
            ids_cpu = ids_cpu[valid_rows]
            routes_cpu = routes_cpu[valid_rows]
        result = self.backend.forward(hidden_cpu, ids_cpu, routes_cpu)
        # The CPU copies already exist; zero-weight padding is not a route.
        active = routes_cpu != 0
        self._eager_gpu_hits += int(((ids_cpu == -2) & active).sum())
        self._eager_cpu_misses += int(((ids_cpu >= 0) & active).sum())
        if valid_rows is None:
            self.host_output[:tokens].copy_(result)
        else:
            self.host_output[:tokens].zero_()
            self.host_output[:tokens][valid_rows] = result.to(self.host_output.dtype)
        output[:tokens].copy_(self.host_output[:tokens], non_blocking=True)
        if self.gpu_cache is not None and ids_cpu.shape[0] >= 128:
            self.gpu_cache.calibrate(ids_cpu)

    def forward(self, hidden, ids, routes):
        if (
            self.async_host is not None
            and hidden.shape[0] <= self.async_tokens
            and not (self.hybrid_has_prefill and hidden.shape[0] > 16)
        ):
            if self.gpu_cache is None:
                return self._forward_async(hidden, ids, routes)
            padding = get_forward_context().is_padding
            if self.gpu_cache.decode_feedback is not None:
                self.gpu_cache.decode_feedback.record(ids, padding)
            if padding is not None:
                hidden = hidden.masked_fill(padding[: hidden.shape[0], None], 0)
                routes = routes.masked_fill(padding[: hidden.shape[0], None], 0)
            cached = self.gpu_cache.launch(hidden, ids, routes)
            cpu_ids = ids.masked_fill(self.gpu_cache.membership[ids.long()], -2)
            output = self._forward_async(hidden, cpu_ids, routes)
            self.gpu_cache.join(cached)
            return output.add_(cached)
        output = torch.empty_like(hidden)
        if (
            self.use_prefill_cache
            and self.gpu_cache is not None
            and not self.gpu_cache.calibrate_from_prompt
            and hidden.shape[0] > 16
        ):
            padding = get_forward_context().is_padding
            if padding is not None:
                hidden = hidden.masked_fill(padding[: hidden.shape[0], None], 0)
                routes = routes.masked_fill(padding[: hidden.shape[0], None], 0)
            if self.gpu_cache.dynamic_enabled:
                assert self.gpu_cache.dynamic_policy is not None
                metadata = get_forward_context().attn_metadata
                actual = (
                    next(
                        (
                            m.num_actual_tokens
                            for m in metadata.values()
                            if hasattr(m, "num_actual_tokens")
                        ),
                        0,
                    )
                    if isinstance(metadata, dict)
                    else 0
                )
                if not 0 <= actual <= ids.shape[0]:
                    raise ValueError("Dynamic cache token count exceeds the batch")
                if actual >= self.gpu_cache.dynamic_policy.min_tokens:
                    observed = ids[:actual]
                    if padding is not None:
                        observed = observed.masked_fill(padding[:actual, None], -1)
                    self.gpu_cache.adapt(observed)
            for start in range(0, hidden.shape[0], 128):
                end = min(start + 128, hidden.shape[0])
                chunk_ids = ids[start:end]
                cached = self.gpu_cache.launch(
                    hidden[start:end], chunk_ids, routes[start:end]
                )
                cpu_ids = chunk_ids.masked_fill(
                    self.gpu_cache.membership[chunk_ids.long()], -2
                )
                self._forward_cpu(
                    hidden[start:end],
                    cpu_ids,
                    routes[start:end],
                    output[start:end],
                    tokens_override=end - start,
                )
                self.gpu_cache.join(cached)
                output[start:end].add_(cached)
            return output
        self._forward_cpu(hidden, ids, routes, output)
        return output
