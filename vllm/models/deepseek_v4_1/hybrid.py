# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped defaults and memory planning for CPU hybrid serving."""

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
    unknown = set(settings) - {"pipeline_layers", "cpu_threads", "host_cache_gib"}
    if unknown:
        raise ValueError(f"Unknown DeepSeek hybrid options: {sorted(unknown)}")
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
    if args.enable_prefix_caching:
        raise ValueError("DeepSeek CPU hybrid requires prefix caching disabled")
    for key in ("cpu_moe", "ced_prefill", "pp_kv_transfer", "cpu_phase_threads"):
        if key in args.additional_config:
            raise ValueError(f"{key} cannot be combined with deepseek_v41_hybrid")
    if args.max_num_seqs is None:
        args.max_num_seqs = 1
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
        spec.setdefault("dspark_num_query_tokens", 5)
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


def plan_expert_cache(capacities, num_layers=20, bank_size=384, local_device=None):
    """Distribute bounded slots fairly, preferring local execution when it fits."""
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
    return result


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
