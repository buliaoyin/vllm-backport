# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped defaults and memory planning for CPU hybrid serving."""

import math
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)
GiB = 1024**3


def hybrid_settings(config):
    extra = config.additional_config
    return extra.get("deepseek_v41_hybrid") if isinstance(extra, dict) else None


def apply_hybrid_defaults(args):
    settings = hybrid_settings(args)
    if settings is None or getattr(args, "_dsv41_hybrid_defaults", False):
        return
    if not isinstance(settings, dict):
        raise ValueError("deepseek_v41_hybrid must be an object")
    unknown = set(settings) - {
        "pipeline_layers",
        "cpu_threads",
        "host_cache_gib",
        "engram_storage",
        "engram_cache_gib",
        "expert_profile",
        "expert_profile_interval",
        "expert_allocation",
    }
    if unknown:
        raise ValueError(f"Unknown DeepSeek hybrid options: {sorted(unknown)}")
    if "expert_profile" in settings and (
        not isinstance(settings["expert_profile"], str)
        or not settings["expert_profile"]
    ):
        raise ValueError("expert_profile must be a nonempty path")
    if "expert_profile_interval" in settings and (
        "expert_profile" not in settings
        or type(settings["expert_profile_interval"]) is not int
        or settings["expert_profile_interval"] < 1
    ):
        raise ValueError(
            "expert_profile_interval requires a profile and positive count"
        )
    allocation = settings.get("expert_allocation", "fair")
    if allocation not in ("fair", "profile") or (
        allocation == "profile" and "expert_profile" not in settings
    ):
        raise ValueError(
            "expert_allocation must be fair or profile with expert_profile"
        )
    storage = settings.get("engram_storage", "ram")
    if storage not in ("ram", "ssd"):
        raise ValueError("engram_storage must be 'ram' or 'ssd'")
    if storage == "ssd":
        cache = settings.get("engram_cache_gib", 1)
        if (
            type(cache) not in (int, float)
            or cache < 0
            or (isinstance(cache, float) and not math.isfinite(cache))
        ):
            raise ValueError("engram_cache_gib must be a finite non-negative number")
        if args.load_format not in ("auto", "safetensors") or (
            args.safetensors_load_strategy not in (None, "lazy")
        ):
            raise ValueError("Engram SSD requires lazy safetensors loading")
        args.safetensors_load_strategy = "lazy"
    elif "engram_cache_gib" in settings:
        raise ValueError("engram_cache_gib requires engram_storage='ssd'")
    layers = settings.get("pipeline_layers")
    if (
        not isinstance(layers, list)
        or len(layers) != args.pipeline_parallel_size
        or any(type(n) is not int or n < 1 for n in layers)
        or sum(layers) != 40
        or sum(layers[:-1]) > 19
    ):
        raise ValueError(
            "pipeline_layers must partition 40 layers across PP stages, "
            "with layers 19–39 on the last stage"
        )
    threads = settings.get("cpu_threads", 22)
    threads = [threads, threads] if type(threads) is int else threads
    if (
        not isinstance(threads, list)
        or len(threads) != 2
        or any(type(n) is not int or n < 1 for n in threads)
    ):
        raise ValueError("cpu_threads must be positive or [prefill, decode]")
    budget = settings.get("host_cache_gib", 12)
    if type(budget) not in (int, float) or not 0 <= budget <= 64:
        raise ValueError("host_cache_gib must be between 0 and 64")
    if (
        args.tensor_parallel_size != 1
        or args.data_parallel_size != 1
        or args.nnodes != 1
    ):
        raise ValueError("DeepSeek CPU hybrid requires TP1 and DP1 on one host")
    if args.distributed_executor_backend not in (None, "mp"):
        raise ValueError("DeepSeek CPU hybrid currently requires the local mp executor")
    for key in ("cpu_moe", "ced_prefill", "pp_kv_transfer", "cpu_phase_threads"):
        if key in args.additional_config:
            raise ValueError(f"{key} cannot be combined with deepseek_v41_hybrid")
    if args.max_num_seqs is None:
        args.max_num_seqs = 1
    if args.enable_prefix_caching is None:
        args.enable_prefix_caching = False
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = 2048
    if not args.limit_mm_per_prompt:
        args.limit_mm_per_prompt = {"image": 1}
    if not args.reasoning_parser:
        args.reasoning_parser = "deepseek_v41"
    from vllm.config import CUDAGraphMode

    if args.compilation_config.cudagraph_mode is None:
        args.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    if args.speculative_config is not None:
        spec = args.speculative_config
        if spec.get("method") != "dspark":
            raise ValueError("DeepSeek CPU hybrid supports DSpark speculation only")
        spec.setdefault("num_speculative_tokens", 3)
        spec.setdefault(
            "dspark_num_query_tokens", max(5, int(spec["num_speculative_tokens"]))
        )
        spec.setdefault("use_local_argmax_reduction", True)
    libraries = native_libraries()
    args.additional_config.update(
        cpu_moe={
            "backend": "ik",
            "num_threads": threads[0],
            "library_path": str(libraries[0]),
            "cuda_library_path": str(libraries[1]),
        },
        ced_prefill=True,
        pp_kv_transfer=True,
        cpu_phase_threads=threads,
    )
    args._dsv41_hybrid_defaults = True


def plan_engram_cache(settings, num_embeddings, head_dim):
    """Split the host cache budget evenly, capped by FP8 table and scale bytes."""
    sizes = [rows * (head_dim + head_dim // 32) for rows in num_embeddings]
    total = sum(sizes)
    requested = settings.get("engram_cache_gib", 1)
    remaining = total if requested >= total / GiB else int(requested * GiB)
    result = [0] * len(sizes)
    for rank, index in enumerate(sorted(range(len(sizes)), key=sizes.__getitem__)):
        result[index] = min(sizes[index], remaining // (len(sizes) - rank))
        remaining -= result[index]
    return tuple(result)


def plan_expert_cache(
    capacities,
    num_layers=20,
    bank_size=384,
    local_device=None,
    *,
    scores=None,
):
    """Allocate bounded slots, optionally favoring learned expert demand.

    Profile allocation preserves the fair plan's device placement and assigns
    each next group of eight slots by its expected saved expert calls.
    """
    remaining = list(capacities)
    result = []
    for layer in range(num_layers):
        target = min(bank_size - 1, sum(remaining) // (num_layers - layer))
        if target < 8:
            result.append((local_device or 0, 0))
            continue
        target = target // 8 * 8
        device = max(range(len(remaining)), key=lambda d: remaining[d])
        if local_device is not None and remaining[local_device] >= target:
            device = local_device
        count = min(target, remaining[device]) // 8 * 8
        result.append((device, count))
        remaining[device] -= count
    if scores is None or len(scores) != num_layers or any(s is None for s in scores):
        return result
    if any(
        len(s) != bank_size or any(not math.isfinite(value) or value < 0 for value in s)
        for s in scores
    ):
        raise ValueError("Expert demand scores must be finite non-negative banks")
    ranked = [sorted(s, reverse=True) for s in scores]
    if not any(any(s) for s in ranked):
        return result
    remaining = list(capacities)
    counts = [0] * num_layers
    while True:
        candidates = []
        for layer, (device, _) in enumerate(result):
            count = counts[layer]
            if count + 8 >= bank_size or remaining[device] < 8:
                continue
            saving = sum(ranked[layer][count : count + 8])
            if saving > 0:
                candidates.append((saving, -layer))
        if not candidates:
            break
        _, selected = max(candidates)
        layer = -selected
        device = result[layer][0]
        remaining[device] -= 8
        counts[layer] += 8
    learned = [(device, counts[layer]) for layer, (device, _) in enumerate(result)]
    learned_saving = sum(
        sum(ranked[layer][:count]) for layer, (_, count) in enumerate(learned)
    )
    fair_saving = sum(
        sum(ranked[layer][:count]) for layer, (_, count) in enumerate(result)
    )
    return learned if learned_saving > fair_saving else result


def native_libraries():
    import vllm

    libdir = Path(vllm.__file__).parent
    libraries = [libdir / f"libdsv41_{name}.so" for name in ("ik", "cuda")]
    if any(not library.is_file() for library in libraries):
        raise RuntimeError(
            "DeepSeek hybrid native libraries are not installed. Install this "
            "branch from source with VLLM_USE_PRECOMPILED unset, or build and "
            "install the libdsv41_ik and libdsv41_cuda CMake targets. Upstream "
            "precompiled wheels do not contain this branch's hybrid backends."
        )
    return libraries
