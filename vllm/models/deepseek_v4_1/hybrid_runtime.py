# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor-owned resources and post-KV expert allocation for hybrid serving."""

import os
import random
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4_numa import physical_cpus
from vllm.utils.torch_utils import set_default_torch_num_threads

from .host_memory import check_host_headroom
from .hybrid import GiB, hybrid_settings, plan_expert_cache

logger = init_logger(__name__)


@contextmanager
def preserve_hybrid_cpu_affinity(config):
    """Keep communication initialization from narrowing CPU expert resources."""
    affinity = os.sched_getaffinity(0) if hybrid_settings(config) is not None else None
    try:
        yield
    finally:
        if affinity is not None and os.sched_getaffinity(0) != affinity:
            os.sched_setaffinity(0, affinity)
            logger.info(
                "Restored hybrid CPU affinity after communication initialization: "
                "%d allowed logical CPUs",
                len(affinity),
            )


class HybridExecutorResources:
    def __init__(self, config):
        settings = hybrid_settings(config)
        self.path = None
        self.previous = {}
        if settings is None:
            return
        hf = config.model_config.hf_config
        expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
        resident = (hf.num_hidden_layers - 20) * hf.n_routed_experts * expert_bytes
        resident += sum(hf.engram_num_embeddings) * (
            hf.engram_head_dim + hf.engram_head_dim // 32
        )
        check_host_headroom(
            resident + int(settings.get("host_cache_gib", 12) * GiB) + 16 * GiB
        )
        cpus = physical_cpus()
        phases = config.additional_config["cpu_phase_threads"]
        if max(phases) > len(cpus):
            raise ValueError(
                f"cpu_threads exceeds the {len(cpus)} available physical cores"
            )
        os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
        os.environ.setdefault("GOMP_SPINCOUNT", "10000000")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        if config.parallel_config.pipeline_parallel_size == 1:
            return
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
            logger.info(
                "Hybrid workers use the explicitly configured CUDA MPS instance"
            )
            return
        binary = shutil.which("nvidia-cuda-mps-control")
        if binary is None:
            raise RuntimeError(
                "Cross-process hybrid GPU experts require nvidia-cuda-mps-control"
            )
        self.path = Path(tempfile.mkdtemp(prefix="dsv41-mps-"))
        for name in ("pipe", "log"):
            (self.path / name).mkdir(mode=0o700)
        updates = {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.path / "pipe"),
            "CUDA_MPS_LOG_DIRECTORY": str(self.path / "log"),
        }
        # The daemon remaps device ordinals. UUIDs have the same meaning in clients.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and all(part.strip().isdigit() for part in visible.split(",")):
            rows = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
            )
            mapping = dict(line.strip().split(", ") for line in rows.splitlines())
            updates["CUDA_VISIBLE_DEVICES"] = ",".join(
                mapping[p.strip()] for p in visible.split(",")
            )
        self.previous = {key: os.environ.get(key) for key in updates}
        os.environ.update(updates)
        try:
            subprocess.run([binary, "-d"], check=True, capture_output=True, timeout=30)
        except Exception:
            self.close()
            raise
        logger.info("Started executor-owned CUDA MPS instance: %s", self.path)

    def close(self):
        if self.path is None:
            return
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            text=True,
            capture_output=True,
            timeout=30,
            env=os.environ | {"CUDA_MPS_PIPE_DIRECTORY": str(self.path / "pipe")},
        )
        logger.info(
            "Executor-owned CUDA MPS shutdown returned %d: %s",
            result.returncode,
            self.path,
        )
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.path, ignore_errors=True)
        self.path = None


@torch.inference_mode()
def initialize_hybrid_devices(worker):
    """Account for remote expert contexts before profiling available memory."""
    if hybrid_settings(worker.vllm_config) is None:
        return
    group = get_pp_group()
    if group.world_size == 1:
        return
    if group.is_last_rank:
        origin = worker.device
        origin_stream = torch.cuda.current_stream(origin)
        with torch.cuda.stream(origin_stream):
            source = torch.zeros(1, device=origin)
            for device in range(group.world_size - 1):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    remote = source.to(device=device, non_blocking=True)
                stream.synchronize()
                source.copy_(remote, non_blocking=True)
                origin_stream.synchronize()
                del remote
                with torch.accelerator.device_index(device):
                    torch.accelerator.empty_cache()
        logger.info(
            "Initialized hybrid expert CUDA contexts and bidirectional copies "
            "on %d devices before loading weights",
            group.world_size,
        )
    torch.distributed.barrier(group=group.cpu_group)


def initialize_hybrid_cache(worker):
    config = worker.vllm_config
    settings = hybrid_settings(config)
    if settings is None:
        return
    from .cpu_moe import CPUExpertModule
    from .expert_cache import GPUExpertCache

    group = get_pp_group()
    runner = worker.model_runner
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    num_tokens = min(
        config.scheduler_config.max_num_batched_tokens,
        config.model_config.max_model_len,
    )
    # Attention workspaces depend on both query length and request count.
    shapes = {(num_tokens, 1), (num_tokens, min(num_tokens, runner.max_num_reqs))}
    decode_tokens = runner.max_num_reqs * runner.decode_query_len
    if decode_tokens <= num_tokens:
        shapes.add((decode_tokens, runner.max_num_reqs))
    with torch.inference_mode():
        for tokens, requests in sorted(shapes, reverse=True):
            runner._dummy_run(
                tokens,
                num_reqs=requests,
                context_len=config.model_config.max_model_len
                - (tokens + requests - 1) // requests,
                skip_eplb=True,
            )
            torch.accelerator.synchronize()
            logger.info(
                "Hybrid workspace profile: tokens=%d requests=%d peak=%.3f GiB",
                tokens,
                requests,
                (torch.accelerator.max_memory_allocated() - before) / GiB,
            )
    torch.accelerator.synchronize()
    transient = torch.accelerator.max_memory_allocated() - before
    torch.accelerator.empty_cache()
    free, total = torch.accelerator.get_memory_info()
    # KVCacheTensor entries are views of one shared backing allocation.
    kv_sizes = {tensor.size for tensor in runner.kv_cache_config.kv_cache_tensors}
    assert len(kv_sizes) <= 1
    kv_bytes = next(iter(kv_sizes), 0)
    graph_reserve = 0
    if not config.model_config.enforce_eager:
        graph_reserve = max(
            GiB, transient, getattr(worker, "cudagraph_memory_estimate", 0)
        )
    reserve = GiB // 2 + graph_reserve
    profiled = getattr(worker, "available_kv_cache_memory_bytes", free) - kv_bytes
    reserved_for_others = int(total * (1 - config.cache_config.gpu_memory_utilization))
    available = max(0, min(profiled, free - reserved_for_others - transient) - reserve)
    logger.info(
        "Hybrid cache budget: KV=%.3f GiB, full-context transient=%.3f GiB, "
        "graph reserve=%.3f GiB, free=%.3f GiB, expert budget=%.3f GiB",
        kv_bytes / GiB,
        transient / GiB,
        graph_reserve / GiB,
        free / GiB,
        available / GiB,
    )
    stats: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(
        stats, (available, free, transient, kv_bytes), group=group.cpu_group
    )
    if group.is_last_rank:
        hf = config.model_config.hf_config
        modules = [
            m
            for m in worker.model_runner.model.modules()
            if isinstance(m, CPUExpertModule)
        ]
        expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
        capacities = [int(item[0] // expert_bytes) for item in stats]
        plan = plan_expert_cache(
            capacities, len(modules), hf.n_routed_experts, group.rank_in_group
        )
        host_budget = int(settings.get("host_cache_gib", 12) * GiB)
        lru_slots = min(32, host_budget // max(1, len(modules) * expert_bytes))
        enabled = config.speculative_config is not None
        for layer, (module, (device, count)) in enumerate(zip(modules, plan), start=20):
            if not count:
                continue
            if not module.backend.supports_export:
                raise RuntimeError(
                    "Automatic caching requires resident CPU expert export"
                )
            cache = GPUExpertCache(
                count,
                hf.n_routed_experts,
                device,
                float(hf.swiglu_limit or 0),
                hf.num_experts_per_tok,
                weight_source=module.backend.export_expert,
            )
            cache.weight_shape = (hf.moe_intermediate_size, hf.hidden_size // 2)
            cache.loaded = set(module.backend._loaded)
            selected = random.Random(20260915 + layer).sample(
                range(hf.n_routed_experts), count
            )
            # Packing alternates native export with small Torch CPU copies.
            # Keep those copies serial instead of repeatedly resizing the team.
            with set_default_torch_num_threads(1):
                cache.select_static(selected)
            cache.finalize(module.async_tokens)
            cache.warmup()
            sm = torch.cuda.get_device_capability(device)[0]
            cache.prepare_dynamic(
                {
                    "mutable_experts": tuple(selected),
                    "group_size": config.uniform_decode_query_len,
                    "cpu_call_ms": 0.3 if enabled else 0.25,
                    "transfer_ms": 1.0 if sm >= 12 else 3.2,
                    "repack_transfer_ms": 4.6 if sm >= 12 else 6.4,
                    "max_swaps": 16,
                    "budget_ms": 24.0,
                    "feedback_weight": 0.9,
                    "feedback_decay": 0.95,
                    "feedback_debias": True,
                    "feedback_horizon_requests": 8.0,
                    "expected_tokens_per_step": 2.5 if enabled else 1.0,
                    "prepack_resident": False,
                    "host_lru_experts": lru_slots,
                    "host_cache_bytes": lru_slots * expert_bytes,
                }
            )
            cache.dynamic_enabled = False
            assert cache.decode_feedback is not None
            cache.decode_feedback.set_enabled(False)
            module.gpu_cache = cache
            module.use_prefill_cache = True
            logger.info(
                "Automatic GPU expert cache: layer=%d device=%d slots=%d "
                "bytes=%d host_lru=%d",
                layer,
                device,
                count,
                count * expert_bytes,
                lru_slots,
            )
        cached_experts = sum(count for _, count in plan)
        total_experts = len(modules) * hf.n_routed_experts
        logger.info(
            "GPU expert cache coverage=%.2f%% (%d/%d routed experts, %.3f GiB); "
            "Hybrid GPU cache plan=%s, host LRU %.3f GiB; post-KV budgets=%s",
            100 * cached_experts / max(1, total_experts),
            cached_experts,
            total_experts,
            cached_experts * expert_bytes / (1 << 30),
            plan,
            lru_slots * len(modules) * expert_bytes / GiB,
            stats,
        )
    torch.distributed.barrier(group=group.cpu_group)


def activate_hybrid_cache(worker):
    """Begin request learning only after kernel warmup and graph capture."""
    if hybrid_settings(worker.vllm_config) is None:
        return
    from .cpu_moe import CPUExpertModule

    torch.accelerator.synchronize()
    for module in worker.model_runner.model.modules():
        if isinstance(module, CPUExpertModule) and module.gpu_cache is not None:
            cache = module.gpu_cache
            assert cache.decode_feedback is not None
            cache.decode_feedback.counts.zero_()
            cache.decode_feedback.set_enabled(False)
            cache.dynamic_enabled = True
    worker.model_runner.model_state.hybrid_ready = True
