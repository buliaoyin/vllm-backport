# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare prefill experiments on real routes, including the complete wrapper."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from benchmarks.kernels.exl3_prefill.launcher import Launcher
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod


def load_layer(checkpoint, prefix):
    config = Exl3Config({})
    config.maybe_update_config(str(checkpoint))
    k, n = config.matrices[prefix + ".0.gate_proj"].dimensions
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1),
        activation="silu",
        swiglu_limit=10.0,
    )
    layer = torch.nn.Module()
    method = Exl3MoEMethod(config, moe, prefix)
    with torch.device("cuda"):
        method.create_weights(layer, 288, k, n, torch.bfloat16)
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in list(handle.keys()):
                if not name.startswith(prefix + "."):
                    continue
                expert, projection, component = name[len(prefix) + 1 :].split(".")
                kind = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[
                    projection
                ]
                param = getattr(layer, ("w2_" if kind == "w2" else "w13_") + component)
                param.weight_loader(
                    param, handle.get_tensor(name), expert_id=int(expert), shard_id=kind
                )
    method.process_weights_after_loading(layer)
    return layer, method


def measure(fn, repeats):
    # CUPTI rejects the CMP device. Use explicit CUDA events around graph replay,
    # with the same cold-cache policy as the earlier route-based measurements.
    for _ in range(4):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    eviction = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    result = []
    for _ in range(repeats):
        eviction.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        result.append(start.elapsed_time(end))
    return result, output.clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("This sweep targets SM80")
    layer, method = load_layer(args.checkpoint, args.prefix)
    sample = torch.load(args.routes, weights_only=True, map_location="cuda")
    original = torch.ops._exl3_C.moe_m32
    if method.m32_locks is None:
        raise RuntimeError("Expected the supported production M32 reference")
    configs = json.loads(args.library.with_name("build.json").read_text())["variants"]
    variants = args.variants or list(configs)
    records = []
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "checkpoint": str(args.checkpoint),
        "prefix": args.prefix,
        "routes": str(args.routes),
        "routes_sha256": hashlib.sha256(args.routes.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "timing": (
            "CUDA events around full-wrapper CUDA Graph replay; "
            "256 MiB eviction outside interval; CUPTI rejects CMP hardware"
        ),
        "records": records,
    }
    for rows in args.rows:
        if rows > sample["x"].shape[0]:
            raise ValueError("Do not synthesize larger routes by repeating samples")
        x, weights, ids = [
            sample[k][:rows].contiguous() for k in ("x", "weights", "ids")
        ]
        if method.workspace[0].shape[1] < rows:
            raise ValueError(
                "Set VLLM_EXL3_MOE_MAX_TOKENS to the largest test row count"
            )
        run = partial(method.apply, layer, x, weights, ids)
        reference = run().float()
        torch.accelerator.synchronize()
        for variant in ["native-before", *variants, "native-after"]:
            launcher = None
            if variant.startswith("native-"):
                torch.ops._exl3_C.moe_m32 = original
            else:
                launcher = Launcher(args.library, variant)
                torch.ops._exl3_C.moe_m32 = launcher
            actual = run().float()
            torch.accelerator.synchronize()
            error = ((actual - reference).norm() / reference.norm()).item()
            if not torch.isfinite(actual).all() or error > 0.01:
                raise RuntimeError(f"{variant} relative L2 {error}")
            times, output = measure(run, args.repeats)
            graph_error = (
                (output.float() - reference).norm() / reference.norm()
            ).item()
            if not torch.isfinite(output).all() or graph_error > 0.01:
                raise RuntimeError(f"{variant} graph relative L2 {graph_error}")
            ms = statistics.median(times)
            flops = 6 * ids.numel() * method.hidden_size * method.intermediate_size
            record = {
                "rows": rows,
                "variant": variant,
                "milliseconds": times,
                "median_ms": ms,
                "relative_l2": error,
                "graph_relative_l2": graph_error,
                "useful_tflops": flops / (ms * 1e9),
                "launcher": launcher.metadata() if launcher else None,
            }
            records.append(record)
            print(rows, variant, round(ms, 4), error, flush=True)
            args.output.write_text(json.dumps(result, indent=2))
            torch.ops._exl3_C.moe_m32 = original
            del launcher
    torch.ops._exl3_C.moe_m32 = original


if __name__ == "__main__":
    main()
