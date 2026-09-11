# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure row kernels with real expert weights/routes and explicit L2 eviction."""

import hashlib
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_exl3_moe import _extension, fused_with_chunk
from benchmarks.kernels.exl3_m32.launcher import Launcher


def graph_times(fn, repeats=15):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    for _ in range(3):
        graph.replay()
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    pairs = []
    for _ in range(repeats):
        flush.zero_()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        pairs.append((start, end))
    torch.accelerator.synchronize()
    # Keep graph outputs alive through every replay and synchronization.
    assert captured is not None
    return [start.elapsed_time(end) for start, end in pairs]


@torch.inference_mode()
def run(layer, method, args, k, n):
    sample = torch.load(args.route_sample, weights_only=True, map_location="cuda")
    extension = _extension()
    native = extension.exl3_moe
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "prefix": args.prefix,
        "checkpoint": str(args.checkpoint),
        "routing": str(args.route_sample),
        "route_sha256": hashlib.sha256(args.route_sample.read_bytes()).hexdigest(),
        "library": str(args.row_kernel_library),
        "library_sha256": hashlib.sha256(
            args.row_kernel_library.read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": {"hidden": k, "intermediate": n, "experts": args.experts},
        "scope": "Full expert wrapper GPU work; one CUDA graph replay/sample; "
        "256 MiB eviction buffer written before each timed interval; 15 samples",
        "records": [],
    }
    try:
        for rows in args.prefill_rows:
            if rows > sample["x"].shape[0]:
                raise ValueError("Requested rows exceed the routing sample")
            x, weights, ids = (
                sample[name][:rows].contiguous() for name in ("x", "weights", "ids")
            )
            for chunk in args.prefill_chunks:
                workspace = [
                    t.new_empty((t.shape[0], chunk, t.shape[2]))
                    for t in method.workspace
                ]
                inputs = (
                    x,
                    weights,
                    ids,
                    method.ptrs,
                    workspace,
                    method.bits,
                    method.flags,
                    10.0,
                )

                def fn(inputs=inputs, chunk=chunk):
                    return fused_with_chunk(inputs, chunk, longest_first=True)

                extension.exl3_moe = native
                reference = fn()
                for variant in args.row_kernel_variants:
                    launcher = None
                    extension.exl3_moe = native
                    if not variant.startswith("native"):
                        launcher_class = Launcher
                        if variant.startswith("int8"):
                            from benchmarks.kernels.exl3_int8.launcher import (
                                Launcher as Int8Launcher,
                            )

                            launcher_class = Int8Launcher
                        launcher = launcher_class(
                            extension, args.row_kernel_library, variant
                        )
                        extension.exl3_moe = launcher
                    actual = fn()
                    delta = actual.float() - reference.float()
                    record = {
                        "variant": variant,
                        "rows": rows,
                        "chunk": chunk,
                        "relative_l2": float(delta.norm() / reference.float().norm()),
                        "max_abs": float(delta.abs().max()),
                        "finite": bool(torch.isfinite(actual).all()),
                        "relative_l2_limit": args.row_kernel_max_relative_error,
                    }
                    if (
                        not record["finite"]
                        or record["relative_l2"] >= args.row_kernel_max_relative_error
                    ):
                        result["records"].append(record)
                        args.output.write_text(json.dumps(result, indent=2))
                        raise AssertionError(record)
                    record["times_ms"] = graph_times(fn)
                    record["median_ms"] = statistics.median(record["times_ms"])
                    if launcher:
                        record["launch"] = {
                            key: value
                            for key, value in launcher.plans[x.device.index].items()
                            if key != "locks"
                        }
                    result["records"].append(record)
                    args.output.write_text(json.dumps(result, indent=2))
                    print(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "times_ms"
                        },
                        flush=True,
                    )
    finally:
        extension.exl3_moe = native
