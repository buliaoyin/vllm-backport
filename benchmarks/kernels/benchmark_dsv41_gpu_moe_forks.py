# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete routed MXFP4 MoE calls on real V4.1 expert weights."""

import argparse
import json
import os
import statistics
import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from cpu.benchmark_dsv41_cpu_moe import reference
from flashinfer.testing import bench_gpu_time_with_cudagraph
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types


def run_fork(runtime, permutation, hidden, w13, s13, w2, s2, ids, routes):
    tokens, hidden_size = hidden.shape
    permuted = hidden.view(tokens, hidden_size // 8, 8).index_select(2, permutation)
    topk = ids.shape[1]
    pairs = runtime.experts_tc(
        permuted.reshape_as(hidden),
        True,
        w13,
        s13,
        w2,
        s2,
        ids,
        routes,
        w2.shape[-1] * 2,
        10.0,
        tokens * topk,
        hidden.device,
        topk=topk,
    )
    return pairs.view(tokens, topk, hidden_size).sum(1).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.fork_root.resolve()))
    os.environ.setdefault("DSV41_NVCC", "/usr/local/cuda/bin/nvcc")
    from dsv41.decode import DecodeRuntime
    from dsv41.quant import tile_fp4, tile_fp4_scales

    torch.set_num_threads(1)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    weights, handles = {}, {}
    with ExitStack() as stack:
        for expert in range(args.experts):
            for projection in range(3):
                pair = []
                for suffix in ("weight", "scale"):
                    key = (
                        f"layers.{args.layer}.ffn.experts.{expert}."
                        f"w{projection + 1}.{suffix}"
                    )
                    filename = index["weight_map"][key]
                    if filename not in handles:
                        handles[filename] = stack.enter_context(
                            safe_open(args.model / filename, framework="pt")
                        )
                    pair.append(handles[filename].get_tensor(key).view(torch.uint8))
                weights[expert, projection] = tuple(pair)
        w13, w2, s13, s2 = (
            torch.stack(
                [
                    torch.cat([weights[e, p][part] for p in projections], dim=0)
                    for e in range(args.experts)
                ]
            ).to(device)
            for projections, part in (((0, 2), 0), ((1,), 0), ((0, 2), 1), ((1,), 1))
        )
        fork_weights = (
            tile_fp4(w13),
            tile_fp4_scales(s13),
            tile_fp4(w2),
            tile_fp4_scales(s2),
        )
        marlin_weights = prepare_moe_mxfp4_layer_for_marlin(
            SimpleNamespace(params_dtype=torch.bfloat16),
            w13,
            w2,
            s13,
            s2,
            None,
            None,
            inplace=True,
        )
        mw13, mw2, ms13, ms2, _, _ = marlin_weights
        workspace = marlin_make_workspace_new(device, 4)
        runtime = object.__new__(DecodeRuntime)
        permutation = torch.tensor([0, 4, 2, 6, 1, 5, 3, 7], device=device)
        payload = {
            "model": str(args.model),
            "layer": args.layer,
            "experts": args.experts,
            "gpu": torch.cuda.get_device_name(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "timing": "CUDA events, 16-call graphs, cold L2, complete routed MoE",
            "precision": {
                "marlin": "BF16 activations and output",
                "a100_fork": "BF16 input/output; FP8-rounded down-projection input",
            },
            "results": [],
        }
        for tokens in args.tokens:
            hidden = torch.randn(tokens, 5120, dtype=torch.bfloat16)
            ids = torch.stack([torch.randperm(args.experts)[:6] for _ in range(tokens)])
            ids = ids.int()
            routes = torch.rand(tokens, 6).softmax(-1) * 2.5
            expected = reference(hidden, ids, routes, weights, 10.0, device)
            hidden, ids, routes = (x.to(device) for x in (hidden, ids, routes))
            operations = {
                "marlin": partial(
                    fused_marlin_moe,
                    hidden,
                    mw13,
                    mw2,
                    None,
                    None,
                    ms13,
                    ms2,
                    routes,
                    ids,
                    scalar_types.float4_e2m1f.id,
                    global_num_experts=args.experts,
                    workspace=workspace,
                    activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
                ),
                "a100_fork": partial(
                    run_fork, runtime, permutation, hidden, *fork_weights, ids, routes
                ),
            }
            errors = {}
            for name, operation in operations.items():
                actual = operation().float().cpu()
                assert torch.isfinite(actual).all(), (name, tokens)
                errors[name] = ((actual - expected).norm() / expected.norm()).item()
                assert errors[name] < 0.035, (name, tokens, errors[name])
                for _ in range(5):
                    operation()
            print(json.dumps({"tokens": tokens, "relative_rms": errors}), flush=True)
            times = {name: [] for name in operations}
            for order in (list(operations), list(reversed(operations))):
                for name in order:
                    op = operations[name]
                    samples = bench_gpu_time_with_cudagraph(
                        op.func,
                        input_args=op.args,
                        input_kwargs=op.keywords,
                        cold_l2_cache=True,
                        num_iters_within_graph=16,
                        dry_run_iters=5,
                        repeat_iters=25,
                    )
                    times[name].append(statistics.median(samples))
            record = {
                "tokens": tokens,
                "relative_rms": errors,
                "median_ms_passes": times,
            }
            payload["results"].append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
