# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real V4.1 FP8 projections: vLLM Marlin, A100 fork, and BF16 cuBLAS."""

import argparse
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import (
    bench_gpu_time_with_cudagraph,
    bench_gpu_time_with_cupti,
)
from safetensors import safe_open

# Initialize the registry before its quantization utilities.
import vllm.model_executor.layers.fused_moe  # noqa: F401
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--timer", choices=("cupti", "cuda_graph"), default="cupti")
    args = parser.parse_args()
    os.environ.setdefault("DSV41_NVCC", "/usr/local/cuda/bin/nvcc")
    sys.path.insert(0, str(args.fork_root.resolve()))
    # Require actual CUPTI instead of silently reporting the timer's fallback.
    if args.timer == "cupti":
        from cupti import cupti  # noqa: F401
    from dsv41 import cukern
    from dsv41.w8 import W8

    torch.set_num_threads(1)
    torch.manual_seed(41)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())

    def read(key):
        with safe_open(
            args.model / index["weight_map"][key], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key).to(device)

    payload = {
        "model": str(args.model),
        "gpu": torch.cuda.get_device_name(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "fork_root": str(args.fork_root),
        "timing": f"{args.timer}, CUDA graph, cold L2; complete GPU linear call",
        "cuda_graph_iterations": 64 if args.timer == "cuda_graph" else None,
        "results": [],
    }
    for stem in (
        "layers.0.attn.wq_a",
        "layers.0.attn.wq_b",
        "layers.0.ffn.shared_experts.w2",
    ):
        raw = read(stem + ".weight").view(torch.uint8)
        scale = read(stem + ".scale").view(torch.uint8)
        n, k = raw.shape
        scale_f32 = torch.ldexp(
            torch.ones_like(scale, dtype=torch.float32), scale.int() - 127
        )
        dense = (
            raw.view(torch.float8_e4m3fn).float()
            * (scale_f32.repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k])
        )
        dense_bf16 = dense.bfloat16()
        fork_weight = W8(raw, scale)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            raw.clone().view(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            scale.repeat_interleave(32, 0)[:n].contiguous(), requires_grad=False
        )
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        prepare_mxfp8_layer_for_marlin(layer)
        for tokens in args.tokens:
            x = torch.randn(tokens, k, device=device, dtype=torch.bfloat16)
            expected = x.float() @ dense.T
            candidates = {
                "marlin": partial(
                    apply_mxfp8_marlin_linear,
                    x,
                    layer.weight,
                    layer.weight_scale,
                    layer.workspace,
                    n,
                    k,
                ),
                "a100_fork": partial(
                    cukern.fp8_gemm_tc,
                    x,
                    fork_weight.w8,
                    fork_weight.s8,
                    tiled=fork_weight.tiled,
                ),
                "bf16_cublas": partial(torch.nn.functional.linear, x, dense_bf16),
            }
            errors = {}
            for name, operation in candidates.items():
                actual = operation().float()
                error = (actual - expected).norm() / expected.norm()
                assert torch.isfinite(actual).all() and error < 0.01, (
                    stem,
                    tokens,
                    name,
                    error.item(),
                )
                errors[name] = error.item()
                for _ in range(10):
                    operation()
            torch.accelerator.synchronize()
            # Reverse the order on the second pass to expose clock/order effects.
            timings = {name: [] for name in candidates}
            for order in (list(candidates), list(reversed(candidates))):
                for name in order:
                    operation = candidates[name]
                    timer = (
                        bench_gpu_time_with_cupti
                        if args.timer == "cupti"
                        else bench_gpu_time_with_cudagraph
                    )
                    options = (
                        {"use_cuda_graph": True}
                        if args.timer == "cupti"
                        else {"num_iters_within_graph": 64}
                    )
                    samples = timer(
                        operation.func,
                        input_args=operation.args,
                        input_kwargs=operation.keywords,
                        cold_l2_cache=True,
                        dry_run_iters=5,
                        repeat_iters=50,
                        **options,
                    )
                    timings[name].append(statistics.median(samples))
            record = {
                "weight": stem,
                "shape_mnk": [tokens, n, k],
                "relative_rms": errors,
                "median_ms_passes": timings,
                "median_tflops": {
                    name: 2 * tokens * n * k / (statistics.median(ms) * 1e9)
                    for name, ms in timings.items()
                },
            }
            payload["results"].append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
