# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS ownership and post-KV expert allocation for hybrid serving."""

import gc
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import psutil
import torch

from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4_numa import physical_cpus
from vllm.utils.torch_utils import set_default_torch_num_threads

from .host_memory import check_host_headroom
from .hybrid import (
    GiB,
    calibrated_expert_capacities,
    expand_expert_cache,
    hybrid_settings,
    plan_engram_cache,
    plan_expert_cache,
)

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


class HybridMPS:
    """Own a private MPS instance independently of EngineCore's lifetime."""

    def __init__(self):
        self.path: Path | None = None
        self.previous: dict[str, str | None] = {}

    def start(self, config):
        if (
            hybrid_settings(config) is None
            or config.parallel_config.pipeline_parallel_size == 1
        ):
            return
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
            logger.info("Hybrid workers use the configured CUDA MPS instance")
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
            # MPS otherwise defaults to two work queues. PP communication and
            # remote experts need independent queues to overlap with prefill.
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get(
                "CUDA_DEVICE_MAX_CONNECTIONS", "16"
            ),
        }
        try:
            # UUIDs retain their meaning after the daemon remaps device ordinals.
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
            subprocess.run([binary, "-d"], check=True, capture_output=True, timeout=30)
        except BaseException:
            self.close()
            raise
        logger.info(
            "Started vLLM-owned CUDA MPS instance: %s (CUDA_DEVICE_MAX_CONNECTIONS=%s)",
            self.path,
            updates["CUDA_DEVICE_MAX_CONNECTIONS"],
        )

    def restore_environment(self):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.previous.clear()

    def _processes(self) -> list[psutil.Process]:
        assert self.path is not None
        result = []
        for proc in psutil.process_iter(["name", "uids"]):
            uids = proc.info["uids"]
            if (
                proc.info["name"]
                not in (
                    "nvidia-cuda-mps-control",
                    "nvidia-cuda-mps-server",
                )
                or uids is None
                or uids.real != os.getuid()
            ):
                continue
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                if proc.environ().get("CUDA_MPS_PIPE_DIRECTORY") == str(
                    self.path / "pipe"
                ):
                    result.append(proc)
        return result

    def close(self):
        self.restore_environment()
        if self.path is None:
            return
        processes = self._processes()
        try:
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit -t 10\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=50,
                env=os.environ | {"CUDA_MPS_PIPE_DIRECTORY": str(self.path / "pipe")},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("CUDA MPS quit failed for %s: %s", self.path, exc)

        # Keep process identities across quit, and find servers created while it ran.
        processes = list(set(processes + self._processes()))
        _, alive = psutil.wait_procs(processes, timeout=1)
        if alive:
            logger.warning(
                "Force stopping private CUDA MPS instance %s: pids=%s",
                self.path,
                [proc.pid for proc in alive],
            )
            for proc in alive:
                with suppress(psutil.NoSuchProcess):
                    proc.kill()
            _, alive = psutil.wait_procs(alive, timeout=5)
            remaining = []
            for proc in alive:
                with suppress(psutil.NoSuchProcess):
                    if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                        remaining.append(proc.pid)
            if remaining:
                raise RuntimeError(
                    f"CUDA MPS processes still alive: {remaining}; "
                    f"retaining {self.path} for cleanup"
                )
        logger.info("Stopped vLLM-owned CUDA MPS instance: %s", self.path)
        shutil.rmtree(self.path, ignore_errors=True)
        self.path = None


class HybridExecutorResources(HybridMPS):
    def __init__(self, config):
        super().__init__()
        settings = hybrid_settings(config)
        if settings is None:
            return
        hf = config.model_config.hf_config
        expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
        resident = (hf.num_hidden_layers - 20) * hf.n_routed_experts * expert_bytes
        engram_bytes = sum(hf.engram_num_embeddings) * (
            hf.engram_head_dim + hf.engram_head_dim // 32
        )
        if settings.get("engram_storage", "ram") == "ram":
            resident += engram_bytes
        else:
            resident += sum(
                plan_engram_cache(
                    settings, hf.engram_num_embeddings, hf.engram_head_dim
                )
            )
            if settings.get("engram_cache_gib", 1) > engram_bytes / GiB:
                logger.info(
                    "Engram SSD cache request exceeds table size; "
                    "limiting total host row cache to %.3f GiB",
                    engram_bytes / GiB,
                )
        check_host_headroom(
            resident + int(settings.get("host_cache_gib", 0) * GiB) + 16 * GiB
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
        self.start(config)


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
    group = get_pp_group()
    runner = worker.model_runner
    worker._hybrid_calibration_started = time.monotonic()
    worker._hybrid_graph_calibration = (
        worker.use_v2_model_runner
        and not config.model_config.enforce_eager
        and config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    )
    _zero_hybrid_profile_kv(worker)
    if worker._hybrid_graph_calibration:
        from vllm.v1.worker.workspace import current_workspace_manager

        from .cache_budget import (
            CacheBudgetProfile,
            budget_fingerprint,
            cached_expert_capacities,
            fit_cached_plan,
        )

        memory = _hybrid_memory(worker)
        worker._hybrid_cache_start = memory
        kv_layouts: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(
            kv_layouts,
            (
                runner.kv_cache_config.num_blocks,
                [tensor.size for tensor in runner.kv_cache_config.kv_cache_tensors],
                len(current_workspace_manager().get_workspace_sizes()),
            ),
            group=group.cpu_group,
        )
        saved = None
        worker._hybrid_budget_profile = None
        if group.is_last_rank:
            hf = config.model_config.hf_config
            profile = CacheBudgetProfile(
                budget_fingerprint(config, memory, kv_layouts),
                [total for _, total, _ in memory],
                hf.num_hidden_layers - 20,
                hf.n_routed_experts,
                [entry[2] for entry in kv_layouts],
            )
            worker._hybrid_budget_profile = profile
            saved = profile.load()
        entries: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(entries, saved, group=group.cpu_group)
        saved = entries[-1]
        worker._hybrid_cached_budget = saved
        worker._hybrid_validated_headrooms = (
            saved["headroom"] if saved is not None else None
        )
        if saved is not None:
            hf = config.model_config.hf_config
            expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
            capacities = cached_expert_capacities(
                memory, saved, expert_bytes, config.cache_config.gpu_memory_utilization
            )
            plan = fit_cached_plan(saved["plan"], capacities, hf.n_routed_experts)
            if plan is not None:
                from vllm.v1.worker.workspace import current_workspace_manager

                current_workspace_manager().reserve_workspace_sizes(
                    saved["workspaces"][group.rank_in_group]
                )
                _initialize_hybrid_prefill_staging(worker)
                worker._hybrid_cache_stats = [
                    (0, free, 0, max(sizes, default=0))
                    for (free, _, _), (_, sizes, _) in zip(memory, kv_layouts)
                ]
                worker._hybrid_cache_initial_slots = sum(count for _, count in plan)
                logger.info(
                    "Reusing hybrid expert cache startup budget: "
                    "slots=%s; skipping full-context workspace profiling",
                    [
                        sum(n for d, n in plan if d == i)
                        for i in range(group.world_size)
                    ],
                )
                _allocate_hybrid_cache(worker, plan)
                return
            worker._hybrid_cached_budget = None
    staging_bytes = _initialize_hybrid_prefill_staging(worker)
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
            _profile_hybrid_serving_step(
                worker,
                tokens,
                requests,
                decode=tokens == decode_tokens and requests == runner.max_num_reqs,
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
    reserve = graph_reserve
    profiled = (
        getattr(worker, "available_kv_cache_memory_bytes", free)
        - kv_bytes
        - staging_bytes
    )
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
    if settings.get("r8_refill", True):
        hf = config.model_config.hf_config
        raw_workspace = 6 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
        available = max(0, available - raw_workspace)
        logger.info("R8 refill staging reserve: %.3f GiB", raw_workspace / GiB)
    stats: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(
        stats, (available, free, transient, kv_bytes), group=group.cpu_group
    )
    worker._hybrid_cache_stats = stats
    hf = config.model_config.hf_config
    expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
    plan = plan_expert_cache(
        [int(item[0] // expert_bytes) for item in stats],
        hf.num_hidden_layers - 20,
        hf.n_routed_experts,
        group.world_size - 1,
    )
    worker._hybrid_cache_initial_slots = sum(count for _, count in plan)
    if worker._hybrid_graph_calibration:
        plan = [
            (device, max(8, count // 16 * 8) if count else 0) for device, count in plan
        ]
    _allocate_hybrid_cache(worker, plan, probe=worker._hybrid_graph_calibration)
    if not worker._hybrid_graph_calibration:
        _log_hybrid_cache_plan(worker)


def _zero_hybrid_profile_kv(worker):
    """Give synthetic computed prefixes finite KV data before startup profiles."""
    seen = set()
    with torch.inference_mode():
        for tensor in worker.model_runner.kv_caches:
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            view.set_(storage, 0, (storage.nbytes(),)).zero_()


def _initialize_hybrid_prefill_staging(worker):
    if get_pp_group().is_last_rank or worker.model_runner.max_num_tokens < 1024:
        return 0
    from vllm.v1.worker.gpu.pp_utils import PPSendStaging

    runner = worker.model_runner
    model = runner.model.language_model.model
    specs = {
        "hidden_states": (
            (runner.max_num_tokens, model.hc_mult, model.config.hidden_size),
            worker.vllm_config.model_config.dtype,
        ),
        "pre_mix": ((runner.max_num_tokens, model.hc_mult), torch.float32),
    }
    if (
        model.pp_shared_kv is not None
        and model.pp_shared_kv.plan.outgoing_source is not None
    ):
        specs.update(
            shared_kv_rows=((runner.max_num_tokens, 584), torch.uint8),
            shared_topk_indices=(
                (runner.max_num_tokens, model.config.index_topk),
                torch.int32,
            ),
        )
    worker._pp_send_staging = PPSendStaging(specs, worker.device)
    logger.info(
        "PP prefill send buffers: 2 slots, %.3f GiB",
        worker._pp_send_staging.nbytes / GiB,
    )
    return worker._pp_send_staging.nbytes


def _allocate_hybrid_cache(worker, plan, *, probe=False):
    from .cpu_moe import CPUExpertModule
    from .expert_cache import GPUExpertCache

    config = worker.vllm_config
    settings = hybrid_settings(config)
    group = get_pp_group()
    runner = worker.model_runner
    worker._hybrid_cache_plan = plan
    if group.is_last_rank:
        hf = config.model_config.hf_config
        modules = [
            m
            for m in worker.model_runner.model.modules()
            if isinstance(m, CPUExpertModule)
        ]
        expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
        pools = {}
        if "adaptive_cache" in settings:
            from .cache_capacity import CapacityPolicy, SharedExpertPool

            policy = CapacityPolicy(**settings["adaptive_cache"])
            for device in range(group.world_size):
                capacity = sum(count for d, count in plan if d == device)
                if capacity:
                    pools[device] = SharedExpertPool(capacity, policy)
            runner.model_state.hybrid_capacity_pools = tuple(pools.values())
        host_budget = int(settings.get("host_cache_gib", 0) * GiB)
        lru_slots = min(32, host_budget // max(1, len(modules) * expert_bytes))
        enabled = config.speculative_config is not None
        from .cache_profile import ExpertCacheProfile

        profile = ExpertCacheProfile.from_config(
            settings,
            config.model_config,
            hf.n_routed_experts,
            range(20, 20 + len(modules)),
        )
        learned = {}
        if profile is not None:
            logger.info("Expert cache profile: %s", profile.path)
            learned = profile.load()
        runner.model_state.hybrid_expert_profile = profile
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
            if settings.get("r8_refill", True):
                if module.backend.supports_r8_staging:
                    from .r8_refill import R8Refill

                    cache.r8_refill = R8Refill(module.backend, cache.device)
                elif layer == 20:
                    logger.warning(
                        "Native R8 staging is unavailable; rebuild libdsv41_ik. "
                        "Using checkpoint-layout expert refills."
                    )
            cache.weight_shape = (hf.moe_intermediate_size, hf.hidden_size // 2)
            cache.loaded = set(module.backend._loaded)
            selected = random.Random(20260915 + layer).sample(
                range(hf.n_routed_experts), count
            )
            entry = learned.get(str(layer))
            if entry is not None:
                assert profile is not None
                selected = profile.selection(entry, count)
            if device in pools:
                pools[device].register(cache, layer, selected)
                cache.cpu_stats = module.backend.cuda_stats
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
                    "decode_interval": 16 if cache.r8_refill is not None else 64,
                    "decode_budget_ms": 16.0 if cache.r8_refill is not None else 8.0,
                }
            )
            cache.dynamic_enabled = False
            assert cache.decode_feedback is not None
            cache.decode_feedback.set_enabled(False)
            if entry is not None:
                cache.restore_learning_state(entry)
            module.gpu_cache = cache
            module.use_prefill_cache = True
            logger.info(
                "%s GPU expert cache: layer=%d device=%d slots=%d bytes=%d host_lru=%d",
                "Probe" if probe else "Calibrating",
                layer,
                device,
                count,
                count * expert_bytes,
                lru_slots,
            )
        for device, pool in pools.items():
            logger.info(
                "Calibrating adaptive expert capacity: device=%d slots=%d layers=%s",
                device,
                pool.capacity,
                [cache.layer_idx for cache in pool.caches],
            )
        cached_experts = sum(count for _, count in plan)
        if learned:
            assert profile is not None
            logger.info(
                "Restored expert cache profile %s (%d layers)",
                profile.path,
                len(learned),
            )
        total_experts = len(modules) * hf.n_routed_experts
        logger.info(
            "Calibrating GPU expert cache coverage=%.2f%% "
            "(%d/%d routed experts, %.3f GiB); "
            "Hybrid GPU cache plan=%s, host LRU %.3f GiB; post-KV budgets=%s",
            100 * cached_experts / max(1, total_experts),
            cached_experts,
            total_experts,
            cached_experts * expert_bytes / (1 << 30),
            plan,
            lru_slots * len(modules) * expert_bytes / GiB,
            worker._hybrid_cache_stats,
        )
    torch.distributed.barrier(group=group.cpu_group)


def _log_hybrid_cache_plan(worker):
    group = get_pp_group()
    if not group.is_last_rank:
        return
    hf = worker.vllm_config.model_config.hf_config
    expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
    plan = worker._hybrid_cache_plan
    for layer, (device, count) in enumerate(plan, start=20):
        logger.info(
            "Automatic GPU expert cache: layer=%d device=%d slots=%d bytes=%d",
            layer,
            device,
            count,
            count * expert_bytes,
        )
    for device in range(group.world_size):
        logger.info(
            "Hybrid final expert capacity: device=%d slots=%d",
            device,
            sum(count for owner, count in plan if owner == device),
        )
    logger.info(
        "Hybrid final GPU cache plan=%s, total=%d slots, "
        "initial estimate=%d (%.3f GiB)",
        plan,
        sum(count for _, count in plan),
        worker._hybrid_cache_initial_slots,
        sum(count for _, count in plan) * expert_bytes / GiB,
    )


def _hybrid_memory(worker, *, reset_peak=False, empty_cache=True):
    """Sample every allocator owner after all PP and remote GPU streams finish."""
    group = get_pp_group()
    devices = range(group.world_size) if group.is_last_rank else [group.rank_in_group]
    for device in devices:
        torch.accelerator.synchronize(device)
    torch.distributed.barrier(group=group.cpu_group)
    if empty_cache:
        for device in devices:
            with torch.accelerator.device_index(device):
                torch.accelerator.empty_cache()
    torch.distributed.barrier(group=group.cpu_group)
    local = {}
    for device in devices:
        with torch.accelerator.device_index(device):
            if reset_peak:
                torch.accelerator.reset_peak_memory_stats()
            free, total = torch.accelerator.get_memory_info()
            allocated = torch.accelerator.memory_allocated()
            peak = torch.accelerator.max_memory_allocated()
            reserved = torch.accelerator.memory_reserved()
            peak_reserved = torch.accelerator.max_memory_reserved()
            initial_headroom = 0
            if (
                device == group.rank_in_group
                and group.is_first_rank
                and worker.model_runner.model_state.supports_mm_inputs
            ):
                initial_headroom = max(
                    getattr(
                        worker, "_hybrid_cache_stats", [(0, 0, 0, 0)] * group.world_size
                    )[device][2],
                    getattr(worker, "peak_activation_memory", 0)
                    - getattr(worker, "cudagraph_memory_estimate", 0),
                )
            local[device] = (
                free,
                total,
                allocated,
                peak,
                reserved,
                peak_reserved,
                initial_headroom,
            )
    samples: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(samples, local, group=group.cpu_group)
    return _aggregate_hybrid_memory(samples)


def _aggregate_hybrid_memory(samples):
    memory = []
    for device in range(len(samples)):
        free, total, *_ = samples[device][device]
        headroom = sum(
            max(
                0,
                owner[device][3] - owner[device][2],
                owner[device][6],
            )
            for owner in samples
            if device in owner
        )
        memory.append((free, total, headroom))
    return memory


def _profile_hybrid_serving_step(worker, tokens, requests, *, decode=False):
    """Exercise CED, sampler and PP buffers with normal scheduler inputs."""
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        KpoolTailSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.gpu.warmup import _warmup_block_counter

    runner = worker.model_runner
    config = worker.vllm_config
    query = (tokens + requests - 1) // requests
    if not config.cache_config.enable_prefix_caching:
        runner._dummy_run(
            tokens,
            num_reqs=requests,
            uniform_decode=decode,
            context_len=config.model_config.max_model_len - query,
            skip_eplb=True,
        )
        return
    width = config.model_config.hf_config.sliding_window
    seed_query = max(2 * width, runner.decode_query_len + 1) if decode else query
    lookahead = max(config.num_lookahead_tokens, runner.decode_query_len * 2)
    specs = [group.kv_cache_spec for group in runner.kv_cache_config.kv_cache_groups]
    counter = _warmup_block_counter(runner)

    def shared_count(context, spec):
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if isinstance(spec, (CircularBufferSpec, SlidingWindowSpec)):
            return 0
        return context // spec.block_size

    def skipped_count(context, spec):
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if isinstance(spec, KpoolTailSpec):
            return 0
        if isinstance(spec, SlidingWindowSpec):
            skipped = max(
                0, context - spec.sliding_window + 1 - spec.extra_retained_tokens
            )
            return skipped // spec.block_size
        return 0

    def counts(context):
        shared = [shared_count(context, spec) for spec in specs]
        full = [counter(context + seed_query + lookahead, spec) for spec in specs]
        return shared, [
            n - prefix - skipped_count(context, spec)
            for n, prefix, spec in zip(full, shared, specs)
        ]

    low, high = 0, max(0, runner.max_model_len - seed_query - lookahead) // width
    while low < high:
        middle = (low + high + 1) // 2
        shared, unique = counts(middle * width)
        if sum(shared) + requests * sum(unique) < runner.kv_cache_config.num_blocks:
            low = middle
        else:
            high = middle - 1
    limits: list[Any] = [None] * get_pp_group().world_size
    torch.distributed.all_gather_object(limits, low, group=get_pp_group().cpu_group)
    # Computed prefix hits are block-aligned and cannot supply a partial
    # R128 encoder state to the first replay window.
    context = min(limits) * width
    shared, unique = counts(context)
    skipped = [skipped_count(context, spec) for spec in specs]
    next_block = 1

    def blocks(count):
        nonlocal next_block
        values = list(range(next_block, next_block + count))
        next_block += count
        return values

    prefix_blocks = [blocks(count) for count in shared]
    vocab_size = config.model_config.get_vocab_size()
    prompt = [i % vocab_size for i in range(context + seed_query)]
    logger.info(
        "Hybrid serving memory profile: requests=%d query=%d context=%d mode=%s",
        requests,
        query,
        context,
        "decode" if decode else "CED prefill",
    )
    ids = [f"_hybrid_memory_{i}" for i in range(requests)]
    params = SamplingParams.for_sampler_warmup()
    params.prompt_logprobs = None
    new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt,
            prefill_token_ids=prompt,
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=tuple(
                [0] * nulls + prefix + blocks(count)
                for nulls, prefix, count in zip(skipped, prefix_blocks, unique)
            ),
            num_computed_tokens=context,
            lora_request=None,
        )
        for req_id in ids
    ]
    max_tokens = config.scheduler_config.max_num_batched_tokens
    seed_batch_size = max(1, max_tokens // seed_query)
    if seed_query * requests > max_tokens:
        logger.info(
            "Hybrid serving seed prefill split: requests=%d query=%d "
            "max_batch_tokens=%d max_batch_requests=%d",
            requests,
            seed_query,
            max_tokens,
            seed_batch_size,
        )
    cleanup = SchedulerOutput.make_empty()
    cleanup.finished_req_ids = set(ids)
    adaptive = runner.adaptive_verification
    runner.adaptive_verification = None
    runner.kv_connector.set_disabled(True)
    try:
        for start in range(0, requests, seed_batch_size):
            batch = new_reqs[start : start + seed_batch_size]
            batch_ids = [req.req_id for req in batch]
            chunk_size = max_tokens // len(batch)
            for offset in range(0, seed_query, chunk_size):
                chunk_tokens = min(chunk_size, seed_query - offset)
                output = SchedulerOutput.make_empty()
                output.num_spec_tokens_to_schedule = runner.num_speculative_steps
                output.num_common_prefix_blocks = shared
                if offset == 0:
                    output.scheduled_new_reqs = batch
                else:
                    cached = CachedRequestData.make_empty()
                    cached.req_ids = batch_ids
                    cached.num_computed_tokens = [context + offset] * len(batch)
                    cached.num_output_tokens = [0] * len(batch)
                    cached.new_block_ids = [None] * len(batch)
                    output.scheduled_cached_reqs = cached
                output.num_scheduled_tokens = dict.fromkeys(batch_ids, chunk_tokens)
                output.total_num_scheduled_tokens = chunk_tokens * len(batch)
                worker.execute_model(output)
                sampled = worker.sample_tokens(None)
                if sampled is not None and hasattr(sampled, "get_output"):
                    sampled.get_output()
                torch.accelerator.synchronize()
                get_pp_group().barrier()
        if decode:
            cached = CachedRequestData.make_empty()
            cached.req_ids = ids
            cached.num_computed_tokens = [context + seed_query] * requests
            cached.num_output_tokens = [1] * requests
            cached.new_block_ids = [None] * requests
            output = SchedulerOutput.make_empty()
            output.scheduled_cached_reqs = cached
            output.num_spec_tokens_to_schedule = runner.num_speculative_steps
            output.num_common_prefix_blocks = shared
            output.num_scheduled_tokens = {req_id: query for req_id in ids}
            output.total_num_scheduled_tokens = query * requests
            if runner.num_speculative_steps:
                output.scheduled_spec_decode_tokens = {
                    req_id: [0] * (query - 1) for req_id in ids
                }
            worker.execute_model(output)
            sampled = worker.sample_tokens(None)
            if sampled is not None and hasattr(sampled, "get_output"):
                sampled.get_output()
            torch.accelerator.synchronize()
            get_pp_group().barrier()
    finally:
        worker.execute_model(cleanup)
        runner.kv_connector.set_disabled(False)
        runner.adaptive_verification = adaptive


def _profile_hybrid_runtime_memory(worker, *, decode_only=False):
    runner = worker.model_runner
    config = worker.vllm_config
    baseline = _hybrid_memory(worker, reset_peak=True)
    group = get_pp_group()
    minimum = [free for free, _, _ in baseline]
    phase_minima: dict[tuple[int, int, bool], list[int]] = {}
    active_phase: list[tuple[int, int, bool] | None] = [None]
    finished = threading.Event()
    errors = []

    def sample():
        try:
            while not finished.is_set():
                for device in range(group.world_size):
                    free = torch.accelerator.get_memory_info(device)[0]
                    minimum[device] = min(minimum[device], free)
                    phase = active_phase[0]
                    if phase is not None:
                        phase_minima[phase][device] = min(
                            phase_minima[phase][device], free
                        )
                finished.wait(0.002)
        except Exception as error:
            errors.append(error)

    monitor = None
    if group.is_last_rank:
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
    torch.distributed.barrier(group=group.cpu_group)
    tokens = min(runner.max_num_tokens, config.model_config.max_model_len)
    shapes = {(tokens, 1), (tokens, min(tokens, runner.max_num_reqs))}
    decode_tokens = runner.max_num_reqs * runner.decode_query_len
    if decode_tokens <= tokens:
        shapes.add((decode_tokens, runner.max_num_reqs))
    if decode_only:
        shapes = {(min(tokens, decode_tokens), min(tokens, runner.max_num_reqs))}
    try:
        with torch.inference_mode():
            for tokens, requests in sorted(shapes, reverse=True):
                decode = tokens == decode_tokens and requests == runner.max_num_reqs
                phase = (tokens, requests, decode)
                phase_minima[phase] = [total for _, total, _ in baseline]
                active_phase[0] = phase
                _profile_hybrid_serving_step(
                    worker,
                    tokens,
                    requests,
                    decode=decode,
                )
                if group.is_last_rank:
                    for device in range(group.world_size):
                        free = torch.accelerator.get_memory_info(device)[0]
                        minimum[device] = min(minimum[device], free)
                        phase_minima[phase][device] = min(
                            phase_minima[phase][device], free
                        )
        retained = _hybrid_memory(worker, empty_cache=False)
        if group.is_last_rank:
            minimum = [min(low, row[0]) for low, row in zip(minimum, retained)]
    finally:
        finished.set()
        if monitor is not None:
            monitor.join()
    if errors:
        raise RuntimeError("Cannot sample hybrid physical memory") from errors[0]
    minima: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(
        minima, (minimum, phase_minima), group=group.cpu_group
    )
    minimum, phase_minima = minima[-1]
    memory = _hybrid_memory(worker)
    observations: list[dict[str, Any]] = []
    for (tokens, requests, decode), free_min in phase_minima.items():
        observations.append(
            {
                "tokens": tokens,
                "requests": requests,
                "decode": decode,
                "minimum_free": free_min,
                "physical_headroom": [
                    max(0, free - low) for (free, _, _), low in zip(memory, free_min)
                ],
            }
        )
    worker._hybrid_runtime_observations = observations
    prefill = [row for row in observations if not row["decode"]]
    worker._hybrid_prefill_headroom = [
        max(floor, max(row["physical_headroom"][device] for row in prefill))
        if prefill
        else 0
        for device, (_, _, floor) in enumerate(memory)
    ]
    result = []
    for device, (free, total, floor) in enumerate(memory):
        physical = max(0, free - minimum[device])
        result.append((free, total, max(floor, physical)))
        if device == group.rank_in_group:
            logger.info(
                "Hybrid runtime physical peak: device=%d headroom=%.3f GiB, "
                "allocation floor=%.3f GiB, profile=%s",
                device,
                physical / GiB,
                floor / GiB,
                "decode check" if decode_only else "full context",
            )
    return result


def _clear_hybrid_graphs_and_cache(worker):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.workspace import unlock_workspace

    _hybrid_memory(worker)
    runner = worker.model_runner
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    manager = runner.cudagraph_manager
    manager.graphs.clear()
    manager._graphs_captured = False
    manager.hidden_states = None
    manager.aux_hidden_states.clear()
    manager.intermediate_tensors = None
    manager.breakable_cg_runner = None
    if runner.model_state.supports_mm_inputs:
        runner.model_state.encoder_runner.clear()
    if runner.speculator is not None:
        speculator = runner.speculator
        speculator.query_cudagraph_manager = None
        speculator._draft_managers.clear()
        speculator.hidden_states = None
        speculator.init_cudagraph_manager(
            worker.vllm_config.compilation_config.cudagraph_mode
        )
    runner.model_state.hybrid_capacity_pools = ()
    for module in runner.model_state.cpu_expert_modules:
        module.gpu_cache = None
        module.use_prefill_cache = False
    gc.collect()
    unlock_workspace()
    _hybrid_memory(worker)


def _save_hybrid_budget(worker, memory):
    from vllm.v1.worker.workspace import current_workspace_manager

    group = get_pp_group()
    sizes: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(
        sizes, current_workspace_manager().get_workspace_sizes(), group=group.cpu_group
    )
    if not group.is_last_rank:
        return
    profile = worker._hybrid_budget_profile
    hf = worker.vllm_config.model_config.hf_config
    expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
    growth = [
        max(
            0,
            start[0]
            - free
            - sum(
                count for owner, count in worker._hybrid_cache_plan if owner == device
            )
            * expert_bytes,
        )
        for device, (start, (free, _, _)) in enumerate(
            zip(worker._hybrid_cache_start, memory)
        )
    ]
    payload = {
        "version": 2,
        "fingerprint": profile.fingerprint,
        "growth": growth,
        "headroom": [row[2] for row in memory],
        "prefill_headroom": worker._hybrid_prefill_headroom,
        "capture_free": [row[0] for row in memory],
        "plan": worker._hybrid_cache_plan,
        "workspaces": sizes,
        "observations": getattr(worker, "_hybrid_runtime_observations", []),
    }
    profile.save(payload)
    worker._hybrid_budget_payload = payload


def capture_hybrid_graphs(worker):
    """Calibrate per-device cache budgets against complete Graph captures."""
    runner = worker.model_runner
    if not getattr(worker, "_hybrid_graph_calibration", False):
        return runner.capture_model()
    from vllm.compilation.counter import compilation_counter

    from .hybrid import minimum_cache_reserve

    group = get_pp_group()
    config = worker.vllm_config
    hf = config.model_config.hf_config
    expert_bytes = 3 * hf.hidden_size * hf.moe_intermediate_size * 17 // 32
    utilization = config.cache_config.gpu_memory_utilization
    counters = (
        compilation_counter.num_cudagraph_captured,
        compilation_counter.num_gpu_runner_capture_triggers,
    )
    fit_limits = [None] * group.world_size
    saved = worker._hybrid_cached_budget
    validated_headrooms = worker._hybrid_validated_headrooms
    verified_plan = None
    verified_headrooms = None
    for attempt in range(5):
        before = _hybrid_memory(worker)
        graph_bytes = runner.capture_model()
        captured = _hybrid_memory(worker)
        reuse_prefill = saved is not None and worker._hybrid_cache_plan == [
            tuple(entry) for entry in saved["plan"]
        ]
        memory = _profile_hybrid_runtime_memory(worker, decode_only=reuse_prefill)
        if reuse_prefill:
            worker._hybrid_prefill_headroom = saved["prefill_headroom"]
        if validated_headrooms is not None:
            memory = [
                (free, total, max(peak, validated))
                for (free, total, peak), validated in zip(memory, validated_headrooms)
            ]
        if saved is not None and any(
            free - peak < minimum_cache_reserve(total, utilization)
            for free, total, peak in memory
        ):
            logger.info(
                "Reused expert budget failed the Graph memory check; "
                "profiling full context"
            )
            assert validated_headrooms is not None
            if reuse_prefill:
                memory = _profile_hybrid_runtime_memory(worker)
                memory = [
                    (free, total, max(peak, validated))
                    for (free, total, peak), validated in zip(
                        memory, validated_headrooms
                    )
                ]
            saved = None
        plan = worker._hybrid_cache_plan
        measured = calibrated_expert_capacities(memory, plan, expert_bytes, utilization)
        safe = True
        for device, (free, total, headroom) in enumerate(memory):
            slots = sum(count for owner, count in plan if owner == device)
            planned = int(total * (1 - utilization)) + headroom
            required = minimum_cache_reserve(total, utilization) + headroom
            safe &= free >= required
            if free < required:
                limit = max(0, slots - 8)
                fit_limits[device] = (
                    min(fit_limits[device], limit)
                    if fit_limits[device] is not None
                    else limit
                )
            if device == group.rank_in_group:
                logger.info(
                    "Hybrid Graph calibration: pass=%d/5 device=%d capture growth=%.3f "
                    "GiB, free=%.3f GiB, runtime headroom=%.3f GiB, "
                    "planned free=%.3f GiB, required free=%.3f GiB "
                    "(40%% reserve tolerance), slots=%d capacity=%d, reused=%s",
                    attempt + 1,
                    device,
                    (before[device][0] - captured[device][0]) / GiB,
                    free / GiB,
                    headroom / GiB,
                    planned / GiB,
                    required / GiB,
                    slots,
                    measured[device],
                    saved is not None,
                )
        if safe:
            verified_plan = list(plan)
            verified_headrooms = [row[2] for row in memory]
        capacities = [
            min(capacity, limit) if limit is not None else capacity
            for capacity, limit in zip(measured, fit_limits)
        ]
        next_plan = (
            expand_expert_cache(plan, capacities, hf.n_routed_experts)
            if safe and (attempt or saved is not None)
            else plan_expert_cache(
                capacities,
                len(plan),
                hf.n_routed_experts,
                group.world_size - 1,
            )
        )
        if safe and (next_plan == plan or attempt >= 3):
            if next_plan != plan:
                logger.warning(
                    "Hybrid cache calibration reached its capture limit; "
                    "keeping the verified safe plan"
                )
            _save_hybrid_budget(worker, memory)
            _log_hybrid_cache_plan(worker)
            logger.info(
                "Hybrid cache startup finished: captures=%d elapsed=%.2f seconds, "
                "reused=%s",
                attempt + 1,
                time.monotonic() - worker._hybrid_calibration_started,
                saved is not None,
            )
            return graph_bytes
        if attempt >= 3:
            if attempt == 3 and verified_plan is not None:
                logger.warning(
                    "Final expert cache expansion exceeded the memory budget; "
                    "restoring and rechecking the last verified layout"
                )
                next_plan = verified_plan
                validated_headrooms = verified_headrooms
                saved = None
            else:
                raise RuntimeError(
                    "Hybrid Graph cache calibration could not preserve runtime headroom"
                )
        _clear_hybrid_graphs_and_cache(worker)
        compilation_counter.num_cudagraph_captured = counters[0]
        compilation_counter.num_gpu_runner_capture_triggers = counters[1]
        _allocate_hybrid_cache(worker, next_plan)
    raise AssertionError("Unreachable hybrid cache calibration state")


def warmup_hybrid_input_kernels(worker):
    """Load metadata kernels before asynchronous PP receives can be pending."""
    if hybrid_settings(worker.vllm_config) is None:
        return
    from vllm.v1.attention.ops.common import fill_token_to_req_indices
    from vllm.v1.worker.gpu.input_batch import expand_idx_mapping

    runner = worker.model_runner
    device = worker.device
    buffer = torch.empty(17, dtype=torch.int32, device=device)
    for requests in range(1, runner.max_num_reqs + 1):
        query = torch.arange(requests + 1, dtype=torch.int32, device=device)
        for tokens in (1, 16, 17):
            fill_token_to_req_indices(query, buffer, tokens)
    mapping = torch.zeros(1, dtype=torch.int64, device=device)
    query = torch.tensor([0, 1], dtype=torch.int32, device=device)
    for exponent in range(runner.decode_query_len.bit_length() + 1):
        expand_idx_mapping(mapping, 1, query, 1 << exponent)
    torch.accelerator.synchronize()
    get_pp_group().barrier()


def activate_hybrid_cache(worker):
    """Begin request learning only after kernel warmup and graph capture."""
    if hybrid_settings(worker.vllm_config) is None:
        return
    if get_pp_group().is_last_rank and getattr(worker, "_hybrid_budget_payload", None):
        from .cache_budget import RuntimeBudgetObserver

        try:
            worker.model_runner.model_state.hybrid_budget_observer = (
                RuntimeBudgetObserver(
                    worker._hybrid_budget_profile,
                    worker._hybrid_budget_payload,
                    [
                        torch.cuda.get_device_properties(device).uuid
                        for device in range(get_pp_group().world_size)
                    ],
                )
            )
        except Exception as error:
            logger.warning("Cannot track expert cache serving peaks: %s", error)
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
