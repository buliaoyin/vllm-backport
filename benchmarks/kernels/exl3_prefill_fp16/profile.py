# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure non-overlapping phases of temporary-weight expert wrappers."""

import argparse
import importlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.exl3_prefill_fp16.benchmark import load_layer, measure
from vllm.model_executor.layers.quantization import exl3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fp16", "int8"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="m64n128k64")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    module = importlib.import_module(
        f"benchmarks.kernels.exl3_prefill_{args.backend}.backend"
    )
    layer, method = load_layer(args.checkpoint, args.prefix)
    sample = torch.load(args.routes, weights_only=True, map_location="cuda")
    if args.rows > sample["x"].shape[0]:
        raise ValueError("The sample does not contain enough real rows")
    x, weights, ids = [
        sample[k][: args.rows].contiguous() for k in ("x", "weights", "ids")
    ]
    run = partial(method.apply, layer, x, weights, ids)
    reference = run().float()
    backend = module.Backend(args.library, args.variant)
    exl3._exl3_moe_fused = backend
    clean, actual = measure(run, args.repeats)
    error = ((actual.float() - reference).norm() / reference.norm()).item()
    if not torch.isfinite(actual).all() or error > (
        0.02 if args.backend == "int8" else 0.01
    ):
        raise RuntimeError(f"Reference relative L2: {error}")

    pool = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(64)]
    for event in pool:
        event.record()
    torch.accelerator.synchronize()
    events = []

    def timed(label, fn, *positional, **kwargs):
        index = len(events)
        start, end = pool[2 * index : 2 * index + 2]
        events.append((label, start, end))
        start.record()
        result = fn(*positional, **kwargs)
        end.record()
        return result

    original_launch = backend.launch
    backend.launch = lambda name, *a, **kw: timed(name, original_launch, name, *a, **kw)
    original_route = module.moe_align_block_size
    module.moe_align_block_size = lambda *a, **kw: timed(
        "route", original_route, *a, **kw
    )
    if args.backend == "fp16":
        original_gemm = module.invoke_fused_moe_triton_kernel
        module.invoke_fused_moe_triton_kernel = lambda *a, **kw: timed(
            "gemm", original_gemm, *a, **kw
        )
    else:

        class GridProxy:
            def __init__(self, label, kernel):
                self.label, self.kernel = label, kernel

            def __getitem__(self, grid):
                fn = self.kernel[grid]
                return lambda *a, **kw: timed(self.label, fn, *a, **kw)

        module.grouped = GridProxy("gemm", module.grouped)
        module.quantize = GridProxy("quantize", module.quantize)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    eviction = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    start, end = pool[-2:]
    records = []
    for _ in range(args.repeats):
        eviction.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        records.append(
            {
                "total_ms": start.elapsed_time(end),
                "phases": [
                    {"name": label, "ms": a.elapsed_time(b)} for label, a, b in events
                ],
            }
        )
    graph_error = ((output.float() - reference).norm() / reference.norm()).item()
    if not torch.isfinite(output).all() or graph_error > (
        0.02 if args.backend == "int8" else 0.01
    ):
        raise RuntimeError(f"Profiled graph relative L2: {graph_error}")
    phases = {
        name: statistics.median(
            sum(p["ms"] for p in r["phases"] if p["name"] == name) for r in records
        )
        for name in sorted({label for label, _, _ in events})
    }
    result = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "relative_l2": error,
        "graph_relative_l2": graph_error,
        "clean_milliseconds": clean,
        "clean_median_ms": statistics.median(clean),
        "profiled_median_ms": statistics.median(r["total_ms"] for r in records),
        "phase_medians_ms": phases,
        "records": records,
        "backend": backend.metadata(),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("records", "args")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
