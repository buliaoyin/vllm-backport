# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep complete EXL3 expert decode paths with cold L2 and fixed routes."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.exl3_m32.benchmark import graph_times
from benchmarks.kernels.exl3_moe_decode.launcher import Decode
from benchmarks.kernels.exl3_prefill_int8.benchmark import load_layer
from vllm.model_executor.layers.quantization import exl3


def m32_triton_routes(x, weights, ids, ptrs, workspace, locks):
    from vllm.model_executor.layers.quantization.utils.exl3_decode import _hot_routes
    from vllm.triton_utils import triton

    rows, hidden = x.shape
    topk, slots = ids.shape[1], ids.numel()
    experts = ptrs[0].numel()
    indices = ids.long().contiguous().flatten()
    routing = weights.half().contiguous().flatten()
    counts = torch.zeros(experts + 1, dtype=torch.int64, device=x.device)
    counts.scatter_add_(0, indices, torch.ones_like(indices))
    packed_counts = torch.empty_like(counts)
    tokens = torch.empty_like(indices)
    packed_weights = torch.empty_like(routing)
    _hot_routes[(experts + 1,)](
        indices,
        routing,
        counts,
        packed_counts,
        tokens,
        packed_weights,
        slots,
        topk,
        experts,
        1,
        triton.next_power_of_2(slots),
        triton.next_power_of_2(experts),
    )
    result = torch.zeros((rows, hidden), dtype=torch.float32, device=x.device)
    torch.ops._exl3_C.moe_m32(
        x.half().contiguous(),
        result,
        packed_counts,
        tokens,
        packed_weights,
        *workspace,
        *ptrs,
        locks,
        10.0,
    )
    return result.to(x.dtype)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-sample", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path)
    parser.add_argument("--cold-library", type=Path)
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[1, 2, 4, 6, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128],
    )
    parser.add_argument("--patterns", nargs="+", default=["captured", "uniform", "hot"])
    parser.add_argument("--grids", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--replays", type=int, default=15)
    args = parser.parse_args()
    sample = torch.load(args.route_sample, weights_only=True, map_location="cuda")
    layer, method = load_layer(args.checkpoint, sample["prefix"])
    candidates = {
        f"plain_g{grid}": Decode(exl3._extension(), args.library, False, grid)
        for grid in args.grids
        if args.library is not None
    }
    if args.grouped_library:
        from benchmarks.kernels.exl3_moe_decode.grouped import GroupedDecode

        for tile in (2, 4):
            for grid in args.grids:
                candidates[f"grouped_m{tile}_g{grid}"] = GroupedDecode(
                    exl3._extension(), args.grouped_library, tile, grid
                )
    if args.cold_library:
        from benchmarks.kernels.exl3_moe_decode.cold import ColdDecode

        for threshold in (2, 3, 4, 8):
            for grid in args.grids:
                candidates[f"cold_t{threshold}_g{grid}"] = ColdDecode(
                    exl3._extension(),
                    args.cold_library,
                    method.m32_locks,
                    threshold,
                    grid,
                )
    production_scratch = None
    if args.production:
        from vllm.model_executor.layers.quantization.utils.exl3_decode import (
            moe_batched_decode,
        )

        production_scratch = torch.zeros(
            (
                1024,
                exl3._decode_workspace_stride(
                    method.hidden_size, method.intermediate_size, True
                ),
            ),
            dtype=torch.int32,
            device=method.ptrs[0].device,
        )
    result = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "sample_sha256": hashlib.sha256(args.route_sample.read_bytes()).hexdigest(),
        "scope": (
            "Complete GPU expert wrapper, allocation/compile excluded; CUDA graphs, "
            "256 MiB L2 eviction outside each interval. "
            "CUPTI is unsupported on CMP 170HX."
        ),
        "records": [],
    }
    torch.manual_seed(20260919)
    for pattern in args.patterns:
        for rows in args.rows:
            x, weights, ids = [
                sample[name][:rows].contiguous() for name in ("x", "weights", "ids")
            ]
            if x.shape[0] != rows:
                raise ValueError("Route sample is too short")
            x = x.to(torch.bfloat16)
            if pattern == "uniform":
                ids = (
                    torch.rand(rows, method.num_experts, device=x.device)
                    .topk(ids.shape[1], dim=1)
                    .indices
                )
            elif pattern == "hot":
                ids = (
                    torch.arange(ids.shape[1], device=x.device)
                    .expand(rows, -1)
                    .contiguous()
                )
            elif pattern != "captured":
                raise ValueError(pattern)
            inputs = (
                x,
                weights,
                ids,
                method.ptrs,
                method.workspace,
                method.bits,
                method.flags,
                10.0,
            )
            fused = partial(exl3._exl3_moe_fused, *inputs, method.m32_locks)
            reference = fused()
            current = partial(method.apply, layer, x, weights, ids)
            calls = {"current": current, "m32": fused}
            calls.update(
                {name: partial(impl, *inputs) for name, impl in candidates.items()}
            )
            if production_scratch is not None:
                calls["m32_triton_routes"] = partial(
                    m32_triton_routes,
                    x,
                    weights,
                    ids,
                    method.ptrs,
                    method.workspace,
                    method.m32_locks,
                )
                for residual in (False, True):
                    name = "production_residual" if residual else "production_plain"
                    calls[name] = partial(
                        moe_batched_decode,
                        x,
                        weights,
                        ids,
                        method.ptrs,
                        method.workspace,
                        method.m32_locks,
                        production_scratch,
                        10.0,
                        residual,
                    )
            times = {name: [] for name in calls}
            errors = {}
            for name, fn in calls.items():
                actual = fn()
                relative = (
                    actual.float() - reference.float()
                ).norm().item() / reference.float().norm().item()
                assert torch.isfinite(actual).all() and relative < 0.025, (
                    pattern,
                    rows,
                    name,
                    relative,
                )
                errors[name] = relative
            for iteration in range(args.rounds):
                for name in list(calls)[:: 1 if iteration % 2 == 0 else -1]:
                    times[name].extend(graph_times(calls[name], args.replays))
            for name in calls:
                result["records"].append(
                    {
                        "pattern": pattern,
                        "rows": rows,
                        "variant": name,
                        "relative_l2": errors[name],
                        "median_ms": statistics.median(times[name]),
                        "samples_ms": times[name],
                    }
                )
            args.output.write_text(json.dumps(result, indent=2))
            print(
                pattern,
                rows,
                {
                    name: round(statistics.median(values), 4)
                    for name, values in times.items()
                },
                flush=True,
            )
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
