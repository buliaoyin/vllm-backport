# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional native MXFP4 CPU experts for GPU/CPU hybrid execution."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4_numa import cpu_numa_nodes

logger = init_logger(__name__)


@dataclass(frozen=True)
class CPUMoEConfig:
    backend: Literal["kt", "llama", "ik"] = "kt"
    start_layer: int = 20
    end_layer: int = 40
    num_threads: int = 28
    library_path: str | None = None
    cuda_library_path: str | None = None
    gpu_cache_experts: int = 0
    gpu_cache_device: int | None = None
    gpu_cache_static_experts: tuple[int, ...] | None = None
    gpu_cache_prefill: bool = False
    gpu_cache_dynamic: dict[str, Any] | None = None

    def __post_init__(self):
        if self.backend not in ("kt", "llama", "ik"):
            raise ValueError(f"Unknown CPU MoE backend: {self.backend}")
        if self.start_layer < 0 or self.end_layer <= self.start_layer:
            raise ValueError("CPU MoE requires a nonempty, nonnegative layer range")
        if self.cuda_library_path and self.backend == "kt":
            raise ValueError("CUDA host callbacks currently support llama/ik backends")
        if self.gpu_cache_experts < 0:
            raise ValueError("GPU expert cache capacity must be nonnegative")
        if self.gpu_cache_experts and (
            self.backend != "ik"
            or not self.cuda_library_path
            or self.gpu_cache_device is None
            or self.gpu_cache_device < 0
        ):
            raise ValueError(
                "GPU expert caching requires IK callbacks and a CUDA device"
            )
        if self.gpu_cache_dynamic is not None and (
            self.gpu_cache_static_experts is None or not self.gpu_cache_prefill
        ):
            raise ValueError(
                "Dynamic caching requires static slots and the prefill cache"
            )
        if self.num_threads < 1:
            raise ValueError("CPU MoE num_threads must be positive")
        if self.gpu_cache_static_experts is not None and (
            not self.gpu_cache_experts
            or len(self.gpu_cache_static_experts) != self.gpu_cache_experts
            or len(set(self.gpu_cache_static_experts)) != self.gpu_cache_experts
            or min(self.gpu_cache_static_experts) < 0
        ):
            raise ValueError("Static GPU expert selection must fill each slot once")


def _cpu_contiguous(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError("CPU MoE backend inputs must be on CPU")
    return tensor.to(dtype=dtype).contiguous()


def _weight_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or tensor.element_size() != 1:
        raise ValueError("MXFP4 weights and E8M0 scales must be CPU byte tensors")
    return tensor.view(torch.uint8).contiguous()


@cache
def _native_library(path: str):
    library = ctypes.CDLL(str(Path(path).resolve()))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    library.dsv41_moe_error.restype = ctypes.c_char_p
    library.dsv41_moe_create.argtypes = [integer] * 5 + [ctypes.c_float]
    library.dsv41_moe_create.restype = pointer
    library.dsv41_moe_destroy.argtypes = [pointer]
    library.dsv41_moe_destroy.restype = None
    library.dsv41_moe_load.argtypes = [pointer, integer, integer, pointer, pointer]
    library.dsv41_moe_load.restype = integer
    if hasattr(library, "dsv41_moe_supports_export"):
        library.dsv41_moe_supports_export.restype = integer
        library.dsv41_moe_export.argtypes = [
            pointer,
            integer,
            integer,
            pointer,
            pointer,
        ]
        library.dsv41_moe_export.restype = integer
    library.dsv41_moe_forward.argtypes = (
        pointer,
        integer,
        pointer,
        pointer,
        pointer,
        pointer,
    )
    library.dsv41_moe_forward.restype = integer
    if hasattr(library, "dsv41_moe_configure_numa"):
        library.dsv41_moe_configure_numa.argtypes = [
            pointer,
            integer,
            pointer,
            pointer,
            pointer,
        ]
        library.dsv41_moe_configure_numa.restype = integer
    if hasattr(library, "dsv41_moe_numa_pages"):
        library.dsv41_moe_numa_pages.argtypes = [pointer, pointer]
        library.dsv41_moe_numa_pages.restype = integer
    if hasattr(library, "dsv41_moe_numa_expert_pages"):
        library.dsv41_moe_numa_expert_pages.argtypes = [pointer, integer, pointer]
        library.dsv41_moe_numa_expert_pages.restype = integer
    if hasattr(library, "dsv41_moe_set_threads"):
        library.dsv41_moe_set_threads.argtypes = [pointer, integer]
        library.dsv41_moe_set_threads.restype = integer
    if hasattr(library, "dsv41_moe_set_execution_mode"):
        library.dsv41_moe_set_execution_mode.argtypes = [pointer, integer]
        library.dsv41_moe_set_execution_mode.restype = integer
    if hasattr(library, "dsv41_moe_set_schedule"):
        library.dsv41_moe_set_schedule.argtypes = [pointer, integer]
        library.dsv41_moe_set_schedule.restype = integer
    if hasattr(library, "dsv41_moe_set_profile"):
        library.dsv41_moe_set_profile.argtypes = [pointer, integer]
        library.dsv41_moe_set_profile.restype = integer
        library.dsv41_moe_get_profile.argtypes = [pointer, pointer, integer]
        library.dsv41_moe_get_profile.restype = integer
        library.dsv41_moe_dump_trace.argtypes = [pointer, ctypes.c_char_p]
        library.dsv41_moe_dump_trace.restype = integer
    if hasattr(library, "dsv41_moe_supports_cached_routes"):
        library.dsv41_moe_supports_cached_routes.restype = integer
    return library


@cache
def _cuda_library(path: str):
    library = ctypes.CDLL(str(Path(path).resolve()))
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    library.dsv41_cuda_task_create.argtypes = (
        [pointer] * 3 + [integer] * 3 + [pointer] * 4
    )
    library.dsv41_cuda_task_create.restype = pointer
    library.dsv41_cuda_task_enqueue.argtypes = [pointer, pointer]
    library.dsv41_cuda_task_enqueue.restype = integer
    library.dsv41_cuda_task_error.argtypes = [pointer]
    library.dsv41_cuda_task_error.restype = ctypes.c_char_p
    library.dsv41_cuda_task_destroy.argtypes = [pointer]
    library.dsv41_cuda_task_destroy.restype = None
    if hasattr(library, "dsv41_cuda_tasks_route_counts"):
        library.dsv41_cuda_tasks_route_counts.argtypes = [pointer, integer, pointer]
        library.dsv41_cuda_tasks_route_counts.restype = None
    if hasattr(library, "dsv41_cuda_task_stats"):
        library.dsv41_cuda_task_stats.argtypes = [pointer, pointer]
        library.dsv41_cuda_task_stats.restype = None
    if hasattr(library, "dsv41_cuda_task_thread_info"):
        library.dsv41_cuda_task_thread_info.argtypes = [pointer, pointer]
        library.dsv41_cuda_task_thread_info.restype = None
    if hasattr(library, "dsv41_cuda_task_cached_routes"):
        library.dsv41_cuda_task_cached_routes.argtypes = [pointer]
        library.dsv41_cuda_task_cached_routes.restype = ctypes.c_uint64
    return library


@cache
def _kt_extension(path: str | None):
    if path is None:
        from kt_kernel import kt_kernel_ext

        return kt_kernel_ext
    spec = importlib.util.spec_from_file_location("kt_kernel_ext", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load KT extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _kt_pool(path: str | None, threads: int):
    return _kt_extension(path).CPUInfer(threads)


class CPUMXFP4Experts:
    """Execute routed experts; the caller supplies already-scaled route weights.

    Inputs and outputs are CPU tensors. Each instance is used serially. Weight
    loading preserves E2M1 values and E8M0 scales; GGML backends quantize
    activations to their native Q8 formats, while KT consumes BF16 inputs.
    """

    def __init__(
        self,
        config: CPUMoEConfig,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        swiglu_limit: float,
        max_tokens: int = 2048,
    ):
        if (
            num_experts <= 0
            or hidden_size <= 0
            or intermediate_size <= 0
            or hidden_size % 32
            or intermediate_size % 32
            or not 0 < top_k <= num_experts
            or max_tokens < 1
            or swiglu_limit < 0
        ):
            raise ValueError("Invalid CPU MXFP4 MoE dimensions")
        self.config = config
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.swiglu_limit = swiglu_limit
        self._handle = None
        self._numa_enabled = self._numa_logged = False
        self._cuda_tasks: dict[tuple[int, ...], int] = {}
        self._cuda_library = (
            _cuda_library(config.cuda_library_path)
            if config.cuda_library_path
            else None
        )
        self._weights: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._loaded: set[tuple[int, int]] = set()
        self._kt_moe: Any = None
        if config.backend != "kt":
            if config.library_path is None:
                raise ValueError("llama/ik CPU MoE requires library_path")
            self._library = _native_library(config.library_path)
            self._handle = self._library.dsv41_moe_create(
                num_experts,
                hidden_size,
                intermediate_size,
                top_k,
                config.num_threads,
                swiglu_limit,
            )
            if not self._handle:
                raise RuntimeError(self._library.dsv41_moe_error().decode())
            if config.backend == "ik":
                self._configure_numa()

    def _configure_numa(self) -> None:
        nodes = cpu_numa_nodes()
        if not nodes or nodes[0].node_id < 0:
            return
        if not hasattr(self._library, "dsv41_moe_configure_numa"):
            if len(nodes) > 1:
                raise RuntimeError(
                    "This IK library lacks NUMA sharding. Rebuild libdsv41_ik "
                    "from this branch before running on multiple NUMA nodes."
                )
            return
        self._numa_enabled = len(nodes) > 1
        cpus = [cpu for node in nodes for cpu in node.cpus]
        offsets = [0]
        for node in nodes:
            offsets.append(offsets[-1] + len(node.cpus))
        integers = lambda values: (ctypes.c_int * len(values))(*values)
        result = self._library.dsv41_moe_configure_numa(
            self._handle,
            len(nodes),
            integers([node.node_id for node in nodes]),
            integers(offsets),
            integers(cpus),
        )
        if result < 0:
            raise RuntimeError(self._library.dsv41_moe_error().decode())
        if result:
            logger.warning_once(
                "CPU expert local memory policy unavailable (%s); using pinned "
                "first-touch with the inherited memory policy.",
                os.strerror(result),
                scope="process",
            )
        logger.info_once(
            "CPU expert NUMA: nodes/physical CPUs=%s, threads=%d, "
            "row sharding=%s, placement=%s; explicit membind preserved; "
            "one resident weight copy",
            tuple((node.node_id, node.cpus) for node in nodes),
            self.config.num_threads,
            len(nodes) > 1,
            "pinned first-touch" if self._numa_enabled else "inherited",
            scope="process",
        )

    def numa_page_counts(
        self, expert: int | None = None
    ) -> tuple[int, int, int, int] | None:
        """Sample total/local/remote/unknown pages for a layer or one expert."""
        if expert is not None and (
            type(expert) is not int or not 0 <= expert < self.num_experts
        ):
            raise ValueError("Invalid expert index for NUMA page query")
        if not self._numa_enabled:
            return None
        counts = (ctypes.c_uint64 * 4)()
        if expert is None:
            result = self._library.dsv41_moe_numa_pages(self._handle, counts)
        else:
            query = getattr(self._library, "dsv41_moe_numa_expert_pages", None)
            if query is None:
                return None
            result = query(self._handle, expert, counts)
        if result:
            logger.warning_once(
                "Cannot query CPU expert NUMA page placement: %s",
                os.strerror(result),
                scope="process",
            )
            return None
        return counts[0], counts[1], counts[2], counts[3]

    def set_num_threads(self, threads: int) -> None:
        """Change CPU parallelism after all forwards and CUDA replays complete."""
        if threads < 1:
            raise ValueError("CPU MoE num_threads must be positive")
        if self.config.backend == "kt" or not hasattr(
            self._library, "dsv41_moe_set_threads"
        ):
            raise RuntimeError("This CPU backend cannot change its thread count")
        if self._library.dsv41_moe_set_threads(self._handle, threads):
            raise RuntimeError(self._library.dsv41_moe_error().decode())
        self.config = replace(self.config, num_threads=threads)

    def set_execution_mode(self, mode: Literal["graph", "compact"]) -> None:
        """Select the IK executor after all forwards and CUDA replays complete."""
        if mode not in ("graph", "compact"):
            raise ValueError("CPU execution mode must be graph or compact")
        if self.config.backend != "ik" or not hasattr(
            self._library, "dsv41_moe_set_execution_mode"
        ):
            raise RuntimeError("This CPU backend cannot change its execution mode")
        if self._library.dsv41_moe_set_execution_mode(self._handle, mode == "compact"):
            raise RuntimeError(self._library.dsv41_moe_error().decode())

    def set_schedule(self, flags: int) -> None:
        """Select auto (-1) or flags after all forwards and replays complete."""
        if not -1 <= flags <= 31:
            raise ValueError(
                "CPU schedule must be -1 (automatic) or flags between 0 and 31"
            )
        if self.config.backend != "ik" or not hasattr(
            self._library, "dsv41_moe_set_schedule"
        ):
            raise RuntimeError("This CPU backend cannot select a schedule")
        if self._library.dsv41_moe_set_schedule(self._handle, flags):
            raise RuntimeError(self._library.dsv41_moe_error().decode())

    def set_profile(self, interval: int) -> None:
        """Reset optional diagnostics after all in-flight forwards complete."""
        if interval < 0:
            raise ValueError("CPU profile interval must be nonnegative")
        if self.config.backend != "ik" or not hasattr(
            self._library, "dsv41_moe_set_profile"
        ):
            raise RuntimeError("This CPU backend does not expose diagnostics")
        if self._library.dsv41_moe_set_profile(self._handle, interval):
            raise RuntimeError(self._library.dsv41_moe_error().decode())

    def profile_stats(self, trace_path: Path) -> dict:
        """Read diagnostic counters and save bounded real-input traces."""
        values = (ctypes.c_uint64 * 64)()
        if self._library.dsv41_moe_get_profile(self._handle, values, len(values)):
            raise RuntimeError(self._library.dsv41_moe_error().decode())
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self._library.dsv41_moe_dump_trace(self._handle, str(trace_path).encode()):
            raise RuntimeError(self._library.dsv41_moe_error().decode())
        names = (
            "calls",
            "tokens",
            "cpu_routes",
            "expert_groups",
            "all_cached_calls",
            "sampled_calls",
            "sampled_workers",
            "sampled_forward_ns",
            "setup_ns",
            "fork_join_ns",
            "input_work_thread_ns",
            "input_wait_thread_ns",
            "up_work_thread_ns",
            "up_wait_thread_ns",
            "down_work_thread_ns",
            "down_wait_thread_ns",
            "up_gemm_thread_ns",
            "activation_quant_thread_ns",
            "down_gemm_thread_ns",
            "reduce_thread_ns",
            "input_stage_ns",
            "up_stage_ns",
            "down_stage_ns",
            "input_ready_stage_ns",
            "up_ready_stage_ns",
            "down_ready_stage_ns",
            "dispatch_join_ns",
        )
        result: dict = {name: values[i] for i, name in enumerate(names)}
        result["expert_group_size_histogram"] = list(values[33:49])
        result["trace_path"] = str(trace_path)
        return result

    def close(self) -> None:
        for task in getattr(self, "_cuda_tasks", {}).values():
            assert self._cuda_library is not None
            self._cuda_library.dsv41_cuda_task_destroy(task)
        self._cuda_tasks = {}
        if getattr(self, "_handle", None) is not None:
            self._library.dsv41_moe_destroy(self._handle)
            self._handle = None
        self._kt_moe = None

    def __del__(self):
        self.close()

    def load_expert(
        self,
        expert: int,
        projection: int,
        weight: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        """Load w1/w2/w3 as projection 0/1/2 in checkpoint byte layout."""
        if not 0 <= expert < self.num_experts or projection not in (0, 1, 2):
            raise ValueError("Invalid expert or projection index")
        rows, cols = self.intermediate_size, self.hidden_size
        if projection == 1:
            rows, cols = cols, rows
        if weight.shape != (rows, cols // 2) or scales.shape != (rows, cols // 32):
            raise ValueError("MXFP4 tensor shape does not match the expert projection")
        weight, scales = _weight_bytes(weight), _weight_bytes(scales)
        if self.config.backend == "kt":
            if self._kt_moe is not None:
                raise RuntimeError("KT expert weights cannot change after preparation")
            exponent = scales.to(torch.int32) - 127
            bf16_scales = (
                torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
                .masked_fill(scales == 255, float("nan"))
                .to(torch.bfloat16)
            )
            self._weights[expert, projection] = (weight, bf16_scales)
        else:
            result = self._library.dsv41_moe_load(
                self._handle, expert, projection, weight.data_ptr(), scales.data_ptr()
            )
            if result:
                raise RuntimeError(self._library.dsv41_moe_error().decode())
        self._loaded.add((expert, projection))

    @property
    def supports_export(self) -> bool:
        return (
            self.config.backend == "ik"
            and hasattr(self._library, "dsv41_moe_supports_export")
            and bool(self._library.dsv41_moe_supports_export())
        )

    def export_expert(self, expert: int, projection: int):
        """Copy resident R8 bytes to checkpoint layout without file access."""
        if not self.supports_export:
            raise RuntimeError("Expert export requires an updated IK backend")
        if (expert, projection) not in self._loaded:
            raise ValueError("Invalid or unloaded expert projection")
        rows, cols = self.intermediate_size, self.hidden_size
        if projection == 1:
            rows, cols = cols, rows
        weight = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cpu")
        scales = torch.empty((rows, cols // 32), dtype=torch.uint8, device="cpu")
        result = self._library.dsv41_moe_export(
            self._handle, expert, projection, weight.data_ptr(), scales.data_ptr()
        )
        if result:
            raise RuntimeError(self._library.dsv41_moe_error().decode())
        return weight, scales

    def prepare(self) -> None:
        if len(self._loaded) != self.num_experts * 3:
            raise RuntimeError("CPU expert weights are incomplete")
        if self._numa_enabled and not self._numa_logged:
            counts = self.numa_page_counts()
            if counts is not None and counts[0]:
                total, local, remote, unknown = counts
                logger.info_once(
                    "CPU expert NUMA sampled pages: local=%.1f%% remote=%.1f%% "
                    "unknown=%d samples=%d (no page migration)",
                    100 * local / total,
                    100 * remote / total,
                    unknown,
                    total,
                    scope="process",
                )
            self._numa_logged = True
        if self.config.backend != "kt" or self._kt_moe is not None:
            return
        extension = _kt_extension(self.config.library_path)
        self._pool = _kt_pool(self.config.library_path, self.config.num_threads)
        config = extension.moe.MOEConfig(
            self.num_experts, self.top_k, self.hidden_size, self.intermediate_size, 0
        )
        config.max_len = self.max_tokens
        config.quant_config.bits = 4
        config.quant_config.group_size = 32
        config.quant_config.zero_point = False
        config.swiglu_limit = self.swiglu_limit
        config.swiglu_alpha = 0.0
        config.pool = self._pool.backend_
        for projection, stem in ((0, "gate"), (1, "down"), (2, "up")):
            setattr(
                config,
                f"{stem}_projs",
                [
                    [
                        self._weights[e, projection][0].data_ptr()
                        for e in range(self.num_experts)
                    ]
                ],
            )
            setattr(
                config,
                f"{stem}_scales",
                [
                    [
                        self._weights[e, projection][1].data_ptr()
                        for e in range(self.num_experts)
                    ]
                ],
            )
        self._kt_moe = extension.moe.AVX2MXFP4_MOE(config)
        mapping = torch.arange(self.num_experts, dtype=torch.int64, device="cpu")
        self._pool.submit(self._kt_moe.load_weights_task(mapping.data_ptr()))
        self._pool.sync()
        self._weights.clear()

    def enqueue_cuda(self, tokens, hidden, ids, routes, output, stream):
        assert self._cuda_library is not None
        self.prepare()
        pointers = tuple(tensor.data_ptr() for tensor in (hidden, ids, routes, output))
        key = (tokens, *pointers)
        if key not in self._cuda_tasks:
            self._cuda_tasks[key] = self._cuda_library.dsv41_cuda_task_create(
                self._handle,
                ctypes.cast(self._library.dsv41_moe_forward, ctypes.c_void_p),
                ctypes.cast(self._library.dsv41_moe_error, ctypes.c_void_p),
                tokens,
                self.hidden_size,
                self.top_k,
                *pointers,
            )
        if not self._cuda_tasks[key]:
            del self._cuda_tasks[key]
            raise RuntimeError("Cannot create CPU MoE CUDA host callback")
        result = self._cuda_library.dsv41_cuda_task_enqueue(
            self._cuda_tasks[key], stream
        )
        if result:
            raise RuntimeError(f"Cannot enqueue CPU MoE CUDA host callback: {result}")

    def cuda_route_counts(self) -> tuple[int, int]:
        """Return cumulative GPU hits and CPU routes after stream completion."""
        if not self._cuda_tasks:
            return 0, 0
        if self._cuda_library is not None and hasattr(
            self._cuda_library, "dsv41_cuda_tasks_route_counts"
        ):
            tasks = (ctypes.c_void_p * len(self._cuda_tasks))(
                *self._cuda_tasks.values()
            )
            counts = (ctypes.c_uint64 * 2)()
            self._cuda_library.dsv41_cuda_tasks_route_counts(tasks, len(tasks), counts)
            return counts[0], counts[1]
        # Keep user-provided older CUDA bridges compatible.
        stats = self.cuda_stats()
        if stats is None:
            return 0, 0
        return stats["cached_routes"], sum(stats["expert_counts"])

    def cuda_stats(self):
        if self._cuda_library is None or not hasattr(
            self._cuda_library, "dsv41_cuda_task_stats"
        ):
            return None
        total = [0] * 388
        cached_routes = 0
        threads = []
        for task in self._cuda_tasks.values():
            if hasattr(self._cuda_library, "dsv41_cuda_task_cached_routes"):
                cached_routes += self._cuda_library.dsv41_cuda_task_cached_routes(task)
            values = (ctypes.c_uint64 * 388)()
            self._cuda_library.dsv41_cuda_task_stats(task, values)
            total = [a + b for a, b in zip(total, values)]
            if hasattr(self._cuda_library, "dsv41_cuda_task_thread_info"):
                info = (ctypes.c_uint64 * 4)()
                self._cuda_library.dsv41_cuda_task_thread_info(task, info)
                threads.append(
                    {
                        "first_tid": info[0],
                        "last_tid": info[1],
                        "switches": info[2],
                        "affinity_cpus": info[3],
                    }
                )
        return {
            "calls": total[0],
            "tokens": total[1],
            "native_seconds": total[2] / 1e9,
            "untracked_routes": total[3],
            "cached_routes": cached_routes,
            "expert_counts": total[4:],
            "callback_threads": threads,
        }

    def check_cuda_errors(self):
        for task in self._cuda_tasks.values():
            assert self._cuda_library is not None
            error = self._cuda_library.dsv41_cuda_task_error(task)
            if error:
                raise RuntimeError(error.decode())

    def forward(
        self, hidden_states: torch.Tensor, ids: torch.Tensor, routes: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("Expected [tokens, hidden_size] CPU activations")
        tokens = hidden_states.shape[0]
        if tokens > self.max_tokens:
            raise ValueError("CPU MoE batch exceeds max_tokens")
        if ids.shape != (tokens, self.top_k) or routes.shape != ids.shape:
            raise ValueError("Expected [tokens, top_k] expert IDs and weights")
        if tokens == 0:
            return torch.empty_like(hidden_states)
        self.prepare()
        routes = _cpu_contiguous(routes, torch.float32)
        if self.config.backend == "kt":
            hidden_states = _cpu_contiguous(hidden_states, torch.bfloat16)
            ids = _cpu_contiguous(ids, torch.int64)
            if ids.min().item() < 0 or ids.max().item() >= self.num_experts:
                raise ValueError("Expert ID is out of range")
            count = torch.tensor([tokens], dtype=torch.int32, device="cpu")
            output = torch.empty_like(hidden_states)
            self._pool.submit(
                self._kt_moe.forward_task(
                    count.data_ptr(),
                    self.top_k,
                    ids.data_ptr(),
                    routes.data_ptr(),
                    hidden_states.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            self._pool.sync()
        else:
            hidden_states = _cpu_contiguous(hidden_states, torch.float32)
            ids = _cpu_contiguous(ids, torch.int32)
            output = torch.empty_like(hidden_states)
            result = self._library.dsv41_moe_forward(
                self._handle,
                tokens,
                hidden_states.data_ptr(),
                ids.data_ptr(),
                routes.data_ptr(),
                output.data_ptr(),
            )
            if result:
                raise RuntimeError(self._library.dsv41_moe_error().decode())
        return output
