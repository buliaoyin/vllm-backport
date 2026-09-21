# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare bounded INT8 expert scratch on captured routes and hot-expert stress."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.exl3_prefill_int8.benchmark import load_layer, measure
from vllm.model_executor.layers.quantization.utils import exl3_prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--routes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 4480, 6144])
    parser.add_argument("--groups", type=int, nargs="+", default=[0, 64, 48, 32])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--hot-routes", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("This sweep targets SM80")
    # The benchmark allocates each INT8 pool explicitly outside the interval.
    os.environ["VLLM_EXL3_MOE_PREFILL"] = "native"
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        Path(__file__),
        Path(exl3_prefill.__file__),
        root / "csrc/libtorch_stable/quantization/exl3/prefill.cu",
    ]
    records = []
    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "timing": (
            "Full expert wrapper CUDA Graph replay, CUDA events, "
            "256 MiB L2 eviction outside interval; CUPTI unsupported on CMP. "
            "Four warmups per variant; reverse order on alternate rounds."
        ),
        "comparison": "group=0 INT8 output; relative L2 <= 0.0002, all finite",
        "routes": [],
        "records": records,
    }
    for path in args.routes:
        sample = torch.load(path, weights_only=True, map_location="cuda")
        layer, method = load_layer(args.checkpoint, sample["prefix"])
        hidden, intermediate = method.hidden_size, method.intermediate_size
        experts, topk = method.num_experts, sample["ids"].shape[1]
        workspaces = {
            group: exl3_prefill.allocate_workspace(
                torch.device("cuda"), experts, hidden, intermediate, 6144, topk, group
            )
            for group in set([0, *args.groups])
        }
        result["routes"].append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prefix": sample["prefix"],
                "rows": sample["x"].shape[0],
            }
        )
        for rows in args.rows:
            if rows > sample["x"].shape[0]:
                raise ValueError("Captured rows cannot be repeated to extend inputs")
            x, weights, real_ids = [
                sample[key][:rows].contiguous() for key in ("x", "weights", "ids")
            ]
            patterns = {"real": real_ids}
            if args.hot_routes:
                # Unique top-k, nearly all experts empty, including tail group.
                patterns["hot"] = (
                    torch.arange(experts - topk, experts, device=x.device)
                    .expand(rows, topk)
                    .contiguous()
                    .to(real_ids.dtype)
                )
            for pattern, ids in patterns.items():
                counts = torch.bincount(ids.flatten().long(), minlength=experts)
                run = partial(exl3_prefill.moe_int8, x, weights, ids, method.ptrs)
                reference = run(workspaces[0], 10.0).float()
                torch.accelerator.synchronize()
                for round_index in range(args.rounds):
                    order = args.groups if round_index % 2 == 0 else args.groups[::-1]
                    for group in order:
                        fn = partial(run, workspaces[group], 10.0)
                        actual = fn().float()
                        error = ((actual - reference).norm() / reference.norm()).item()
                        if not torch.isfinite(actual).all() or error > 0.0002:
                            raise RuntimeError(f"group={group} relative L2 {error}")
                        times, output = measure(fn, args.repeats)
                        delta = output.float() - reference
                        graph_error = (delta.norm() / reference.norm()).item()
                        if not torch.isfinite(output).all() or graph_error > 0.0002:
                            raise RuntimeError(f"group={group} graph L2 {graph_error}")
                        record = {
                            "prefix": sample["prefix"],
                            "rows": rows,
                            "pattern": pattern,
                            "round": round_index,
                            "group": group,
                            "median_ms": statistics.median(times),
                            "milliseconds": times,
                            "relative_l2": error,
                            "graph_relative_l2": graph_error,
                            "max_abs": delta.abs().max().item(),
                            "identical_fraction": (delta == 0).float().mean().item(),
                            "workspace_bytes": workspaces[group][-1].numel(),
                            "workspace_view_bytes": [
                                t.numel() * t.element_size() for t in workspaces[group]
                            ],
                            "active_experts": (counts > 0).sum().item(),
                            "hottest_expert_rows": counts.max().item(),
                        }
                        records.append(record)
                        print(json.dumps(record), flush=True)
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
        del sample, method, layer, workspaces, run, fn, x, weights, real_ids
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
