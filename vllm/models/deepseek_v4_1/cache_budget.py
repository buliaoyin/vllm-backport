# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured startup budgets, independent of expert selection history."""

import copy
import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

import torch

from vllm import envs
from vllm.logger import init_logger

from .cache_profile import model_fingerprint

logger = init_logger(__name__)


def budget_fingerprint(config, memory, kv_sizes):
    normalized = copy.copy(config)
    normalized.additional_config = copy.deepcopy(config.additional_config)
    settings = normalized.additional_config["deepseek_v41_hybrid"]
    for key in (
        "expert_profile",
        "expert_profile_interval",
        "route_tail_skip",
        "cpu_threads",
        "host_cache_gib",
        "engram_cache_gib",
    ):
        settings.pop(key, None)
    cpu_moe = normalized.additional_config.get("cpu_moe", {})
    for key in ("route_tail_skip", "num_threads"):
        cpu_moe.pop(key, None)
    normalized.additional_config.pop("cpu_phase_threads", None)
    adaptive = settings.get("adaptive_cache")
    if adaptive is True:
        adaptive = settings["adaptive_cache"] = {}
    elif adaptive is False:
        settings.pop("adaptive_cache")
    if isinstance(adaptive, dict):
        for key in (
            "interval",
            "max_swaps",
            "decay",
            "resize",
            "capacity_only",
            "history",
            "online",
            "horizon",
            "safety_factor",
        ):
            adaptive.pop(key, None)
        adaptive.setdefault("min_experts", 8)
    normalized.cache_config = copy.copy(config.cache_config)
    if (
        normalized.cache_config.kv_offloading_backend == "native"
        and normalized.cache_config.kv_offloading_size
    ):
        normalized.cache_config.kv_offloading_size = 1.0
    root = Path(__file__).parents[2]
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *Path(__file__).parent.joinpath("nvidia").glob("*.py"),
        *root.joinpath("v1/worker/gpu/spec_decode/dspark").glob("*.py"),
        *root.joinpath("v1/worker/gpu").rglob("*.py"),
        *root.joinpath("v1/attention/backends/mla").glob("*.py"),
        root / "v1/worker/workspace.py",
        root / "v1/worker/gpu_worker.py",
        root / "v1/worker/utils.py",
        root / "utils/mem_utils.py",
        root / "sampling_params.py",
        root / "v1/worker/gpu/model_runner.py",
        root / "v1/worker/gpu/cudagraph_utils.py",
    ]
    source = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
    native = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.glob("*.so"))
    ]
    factors = {
        "version": 2,
        "model": model_fingerprint(config.model_config),
        "config": normalized.compute_hash(),
        "memory": [total for _, total, _ in memory],
        "kv_sizes": kv_sizes,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mps_active": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        "connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source": source.hexdigest(),
        "native": native,
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "prefix_caching": config.cache_config.enable_prefix_caching,
        "decode_query_len": config.uniform_decode_query_len,
        "lookahead_tokens": config.num_lookahead_tokens,
        "max_model_len": config.model_config.max_model_len,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
    }
    with suppress(OSError):
        factors["driver"] = Path("/proc/driver/nvidia/version").read_text()
    return hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()


class CacheBudgetProfile:
    def __init__(self, fingerprint, totals, layers, bank_size, workspace_slots=None):
        self.fingerprint = fingerprint
        self.totals = totals
        self.layers = layers
        self.bank_size = bank_size
        self.workspace_slots = workspace_slots
        self.path = (
            Path(envs.VLLM_CACHE_ROOT)
            / "startup_plan"
            / f"dsv41_experts_{fingerprint}.json"
        )

    def _validate(self, payload):
        if payload["version"] != 2 or payload["fingerprint"] != self.fingerprint:
            raise ValueError("cache budget identity differs")
        plan = payload["plan"]
        if len(plan) != self.layers:
            raise ValueError("cache budget layer count differs")
        for device, count in plan:
            if (
                type(device) is not int
                or not 0 <= device < len(self.totals)
                or type(count) is not int
                or not 0 <= count < self.bank_size
                or count % 8
            ):
                raise ValueError("invalid cache budget slots")
        for key in ("growth", "headroom", "prefill_headroom", "capture_free"):
            values = payload[key]
            if len(values) != len(self.totals) or any(
                type(n) is not int or not 0 <= n <= total
                for n, total in zip(values, self.totals)
            ):
                raise ValueError("invalid cache budget memory")
        if any(
            prefill > peak
            for prefill, peak in zip(payload["prefill_headroom"], payload["headroom"])
        ):
            raise ValueError("prefill reserve exceeds the validated peak")
        workspaces = payload["workspaces"]
        if len(workspaces) != len(self.totals) or any(
            not isinstance(sizes, list)
            or not 1 <= len(sizes) <= 64
            or any(type(n) is not int or not 0 <= n <= total for n in sizes)
            or sum(sizes) > total
            for sizes, total in zip(workspaces, self.totals)
        ):
            raise ValueError("invalid cache budget workspaces")
        if (
            self.workspace_slots is not None
            and [len(s) for s in workspaces] != self.workspace_slots
        ):
            raise ValueError("cache budget workspace layout differs")
        return payload

    def load(self):
        try:
            with self.path.open("rb") as file:
                data = file.read(65537)
            if len(data) > 65536:
                raise ValueError("cache budget exceeds its byte limit")
            return self._validate(json.loads(data))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            logger.warning("Ignoring expert cache budget %s: %s", self.path, error)
            return None

    def save(self, payload):
        temporary = None
        try:
            data = json.dumps(self._validate(payload), separators=(",", ":"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(data)
            os.replace(temporary, self.path)
            logger.info("Saved expert cache startup budget: %s", self.path)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            logger.warning("Cannot save expert cache budget %s: %s", self.path, error)
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def record_runtime_peak(self, payload, minimum_free):
        if len(minimum_free) != len(self.totals) or any(
            type(free) is not int or not 0 <= free <= total
            for free, total in zip(minimum_free, self.totals)
        ):
            raise ValueError("invalid serving memory observations")
        updated = copy.deepcopy(payload)
        updated["headroom"] = [
            max(peak, baseline - free)
            for peak, baseline, free in zip(
                payload["headroom"], payload["capture_free"], minimum_free
            )
        ]
        if updated["headroom"] != payload["headroom"]:
            self.save(updated)
            logger.info(
                "Updated expert budget from serving peaks: headroom=%s",
                updated["headroom"],
            )
        return updated


class RuntimeBudgetObserver:
    """Learn GPU-wide high-water marks without synchronizing request streams."""

    def __init__(self, profile, payload, device_uuids):
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        self.reader = pynvml.nvmlDeviceGetMemoryInfo
        self.handles = [pynvml.nvmlDeviceGetHandleByUUID(str(u)) for u in device_uuids]
        self.profile = profile
        self.payload = payload
        self.minimum = list(payload["capture_free"])
        self.active = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while True:
            self.active.wait()
            try:
                readings = [self.reader(handle).free for handle in self.handles]
                with self.lock:
                    self.minimum = [
                        min(old, free) for old, free in zip(self.minimum, readings)
                    ]
            except Exception as error:
                logger.warning("Cannot observe expert cache serving memory: %s", error)
                return
            time.sleep(0.01)

    def begin(self):
        self.active.set()

    def finish(self):
        self.active.clear()
        with self.lock:
            minimum = self.minimum.copy()
        self.payload = self.profile.record_runtime_peak(self.payload, minimum)


def cached_expert_capacities(memory, saved, expert_bytes, utilization):
    """Apply the same retention threshold to recorded serving peaks on restart."""
    from .hybrid import calibrated_expert_capacities

    projected = [
        (
            free
            - growth
            - sum(count for owner, count in saved["plan"] if owner == device)
            * expert_bytes,
            total,
            peak,
        )
        for device, ((free, total, _), growth, peak) in enumerate(
            zip(memory, saved["growth"], saved["headroom"])
        )
    ]
    return calibrated_expert_capacities(
        projected, saved["plan"], expert_bytes, utilization
    )


def fit_cached_plan(plan, capacities, bank_size):
    """Resize slots on the learned devices without changing cache IO owners."""
    from .hybrid import expand_expert_cache

    plan = [tuple(entry) for entry in plan]
    for device, capacity in enumerate(capacities):
        layers = [
            i for i, (owner, count) in enumerate(plan) if owner == device and count
        ]
        if capacity < len(layers) * 8:
            return None
        excess = sum(plan[i][1] for i in layers) - capacity
        while excess > 0:
            layer = max(layers, key=lambda i: plan[i][1])
            plan[layer] = (device, plan[layer][1] - 8)
            excess -= 8
    return expand_expert_cache(plan, capacities, bank_size)
