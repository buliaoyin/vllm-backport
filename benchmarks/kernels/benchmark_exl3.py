# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 adapter vs upstream LinearEXL3, same extension, cold-L2 graph timing."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph
from safetensors import safe_open

from vllm.model_executor.layers.quantization.exl3 import _exl3_linear, _extension


def prefill_ablation(weights, key, args, result):
    ext = _extension()
    k, n = weights["suh"].numel(), weights["svh"].numel()
    bits = weights["trellis"].shape[-1] // 16
    flags = ("mcg" in weights, "mul1" in weights)
    rotated_weight = torch.empty((k, n), device="cuda", dtype=torch.float16)
    ext.reconstruct_slice(rotated_weight, weights["trellis"], bits, *flags, 0)
    original_weight = torch.empty_like(rotated_weight)
    ext.reconstruct_had_slice(
        original_weight,
        weights["trellis"],
        weights["suh"],
        weights["svh"],
        bits,
        *flags,
        0,
    )

    def calculate(x, mode):
        if mode == "baseline":
            return _exl3_linear(
                x, weights["trellis"], weights["suh"], weights["svh"], *flags
            )
        inp = x.half().contiguous()
        out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float16)
        if mode == "cached_original":
            ext.hgemm(inp, original_weight, out)
        elif mode == "reconstruct_fused":
            weight = torch.empty_like(original_weight)
            ext.reconstruct_had_slice(
                weight,
                weights["trellis"],
                weights["suh"],
                weights["svh"],
                bits,
                *flags,
                0,
            )
            ext.hgemm(inp, weight, out)
        elif mode == "direct_gemm":
            scratch = torch.empty_like(inp)
            ext.exl3_gemm(
                inp,
                weights["trellis"],
                out,
                weights["suh"],
                scratch,
                weights["svh"],
                -1,
                *flags,
                0,
            )
        else:
            rotated = torch.empty_like(inp)
            ext.had_r_128(inp, rotated, weights["suh"], None, 1.0)
            if mode == "cached_rotated_torch":
                torch.mm(rotated, rotated_weight, out=out)
            else:
                ext.hgemm(rotated, rotated_weight, out)
            ext.had_r_128(out, out, None, weights["svh"], 1.0)
        return out.to(x.dtype)

    for rows in args.prefill_rows:
        x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
        reference = calculate(x, "baseline")
        for mode in (
            "baseline",
            "cached_rotated",
            "cached_rotated_torch",
            "cached_original",
            "reconstruct_fused",
            "direct_gemm",
        ):
            actual = calculate(x, mode)
            relative = (
                (actual.float() - reference.float()).norm() / reference.float().norm()
            ).item()
            assert relative < 0.01, (key, rows, mode, relative)
            for _ in range(3):
                calculate(x, mode)
            timings = bench_gpu_time_with_cudagraph(
                calculate,
                input_args=(x, mode),
                cold_l2_cache=True,
                num_iters_within_graph=1,
                dry_run_iters=3,
                repeat_iters=15,
            )
            record = {
                "matrix": key,
                "shape": [rows, k, n],
                "bits": bits,
                "mode": mode,
                "median_ms": statistics.median(timings),
                "relative_l2": relative,
                "cache_bytes": k * n * 2 if mode.startswith("cached") else 0,
            }
            result["rows"].append(record)
            print(record, flush=True)
            args.output.write_text(json.dumps(result, indent=2))


@torch.inference_mode()
def main():
    from exllamav3.modules.quant.exl3 import LinearEXL3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matrix", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-rows", type=int, nargs="+")
    args = parser.parse_args()
    torch.manual_seed(20260910)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timer": "CUDA graph",
        "cold_l2": True,
        "rows": [],
    }
    for key in args.matrix:
        weights = {}
        for shard in args.checkpoint.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cuda:0") as handle:
                for name in list(handle.keys()):
                    if name.startswith(key + "."):
                        weights[name[len(key) + 1 :]] = handle.get_tensor(name)
        if args.prefill_rows:
            prefill_ablation(weights, key, args, result)
            continue
        upstream = LinearEXL3(
            None,
            weights["suh"].numel(),
            weights["svh"].numel(),
            out_dtype=torch.float16,
            **weights,
        )

        def reference(x, upstream=upstream):
            return upstream.forward(x.half().contiguous(), {}).to(x.dtype)

        def candidate(x, weights=weights):
            return _exl3_linear(
                x,
                weights["trellis"],
                weights["suh"],
                weights["svh"],
                "mcg" in weights,
                "mul1" in weights,
            )

        for rows in [1, 8, 128, 1024]:
            x = torch.randn(
                rows, weights["suh"].numel(), device="cuda", dtype=torch.bfloat16
            )
            y = candidate(x)
            expected = reference(x)
            torch.testing.assert_close(y, expected, atol=0.02, rtol=0.02)
            error = (y.float() - expected.float()).norm() / expected.float().norm()
            assert error < 0.005
            record = {
                "matrix": key,
                "shape": [rows, weights["suh"].numel(), weights["svh"].numel()],
                "dtype": str(x.dtype),
                "bits": upstream.K,
                "relative_l2": error.item(),
            }
            # A second allocation catches wrappers that retain the first input pointer.
            replacement = x * 0.7
            torch.testing.assert_close(
                candidate(replacement), reference(replacement), atol=0.02, rtol=0.02
            )
            for label, fn in [("upstream_us", reference), ("vllm_us", candidate)]:
                for _ in range(3):
                    fn(x)
                torch.accelerator.synchronize()
                samples = bench_gpu_time_with_cudagraph(
                    fn,
                    input_args=(x,),
                    cold_l2_cache=True,
                    dry_run_iters=5,
                    repeat_iters=40,
                )
                record[label] = statistics.median(samples) * 1000
            result["rows"].append(record)
            print(record, flush=True)
            args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
