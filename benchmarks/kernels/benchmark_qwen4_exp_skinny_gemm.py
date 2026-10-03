# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune Qwen4Exp BF16 projections on RTX PRO 6000 with CUDA graph timing."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
    _qwen4_exp_low_latency_gemm,
)
from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import (
    _rowwise_fp8_projection,
    quantize_rowwise_fp8,
    rowwise_fp8_logits,
    rowwise_fp8_output,
)

# Local (N, K), TP=1. The two HC down projections are replicated under TP.
PROJECTIONS = {
    "gdn_qkvz": (16384, 2560),
    "attn_out": (2560, 6144),
    "gdn_ba": (96, 2560),
    "qsa_qkvg": (13312, 2560),
    "qsa_indexer": (640, 2560),
    "shared_gate_up": (1280, 2560),
    "shared_down": (2560, 640),
    "router": (512, 2560),
    "hc_down_inject": (336, 10240),
    "hc_final_down": (320, 10240),
    "hc_up": (10240, 320),
    "ple_kv": (12800, 2560),
    "mtp_fusion": (2560, 2560),
    "lm_head": (248320, 2560),
    "draft_lm_head": (65536, 2560),
}


def _configs(m: int, n: int, k: int) -> list[SkinnyGemmConfig]:
    candidates = [
        SkinnyGemmConfig(m, block, outputs, k_unroll=unroll, vector_width=vector)
        for block, outputs, unroll, vector in (
            (32, 4, 2, 4),
            (32, 8, 2, 4),
            (64, 2, 2, 4),
            (64, 4, 4, 4),
            (128, 2, 2, 4),
            (128, 1, 2, 4),
            (128, 4, 4, 4),
            (64, 2, 2, 8),
            (128, 2, 2, 8),
            (256, 1, 2, 4),
            (32, 4, 2, 2),
            (32, 8, 2, 2),
            (32, 2, 2, 2),
        )
        if n % outputs == 0
        and k % (block * vector) == 0
        and (m < 8 or (outputs <= 2 and vector <= 4))
    ]
    static_candidates = [
        SkinnyGemmConfig(
            m,
            config.block_size,
            config.outputs_per_block,
            vector_width=config.vector_width,
            static_k=k,
        )
        for config in candidates
        if k >= 2 * config.block_size * config.vector_width
    ]
    return candidates + static_candidates


def _bench_us(fn: Callable[[], torch.Tensor], cold: bool, repeats: int) -> float:
    samples = bench_gpu_time_with_cupti(
        fn,
        dry_run_iters=10,
        repeat_iters=repeats,
        use_cuda_graph=True,
        cold_l2_cache=cold,
    )
    return statistics.median(samples) * 1000


def _fp32_hc_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bf16: torch.Tensor,
    bf16_start: int,
) -> torch.Tensor:
    """Reproduce the former HC prefill fallback for timing comparisons."""
    m, k = x.shape
    n = weight.shape[0]
    output = x.new_empty((m, n))
    rows = max(1, 64 * 1024 * 1024 // (2 * k + 4 * m))
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        dense = weight[start:stop].bfloat16()
        lo = max(start, bf16_start)
        hi = min(stop, bf16_start + bf16.shape[0])
        if lo < hi:
            dense[lo - start : hi - start] = bf16[lo - bf16_start : hi - bf16_start]
        logits = torch.mm(x, dense.t(), out_dtype=torch.float32)
        output[:, start:stop] = logits * scales[start:stop]
    return output


def _bench_rowwise_fp8(
    x,
    weight,
    repeats: int,
    *,
    hc=False,
    dense_output=False,
    bf16_start=0,
    bf16_rows=0,
    pad_rows=0,
) -> dict:
    quantized, scales = quantize_rowwise_fp8(weight)
    bf16 = weight[bf16_start : bf16_start + bf16_rows].clone()
    quantized[bf16_start : bf16_start + bf16_rows] = 0
    scales[bf16_start : bf16_start + bf16_rows] = 1
    if pad_rows:
        quantized[-pad_rows:] = 0
        scales[-pad_rows:] = 1
        weight[-pad_rows:] = 0
    candidate = partial(
        rowwise_fp8_logits, x, quantized, scales, bf16 if hc else None, bf16_start
    )
    if dense_output:
        candidate = partial(rowwise_fp8_output, x, quantized, scales)
    previous = (
        partial(_fp32_hc_prefill, x, quantized, scales, bf16, bf16_start)
        if hc and x.shape[0] > 32
        else None
    )
    baseline = partial(_qwen4_exp_low_latency_gemm, x, weight)
    candidate()
    baseline()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = candidate()
    for sign in (-1, 1):
        x.mul_(sign)
        graph.replay()
        reference = (x.float() @ quantized.float().t()) * scales
        if dense_output and x.shape[0] > 32:
            from vllm import _custom_ops as ops

            aq, input_scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            reference = (aq.float() @ quantized.float().t()) * input_scale * scales
        reference[:, bf16_start : bf16_start + bf16_rows] = x.float() @ bf16.float().t()
        torch.testing.assert_close(actual.float(), reference, rtol=1e-2, atol=1e-1)
        if previous is not None:
            torch.testing.assert_close(
                previous().float(), reference, rtol=1e-2, atol=1e-1
            )
    base_hot = _bench_us(baseline, False, repeats)
    base_cold = _bench_us(baseline, True, repeats)
    hot = _bench_us(candidate, False, repeats)
    cold = _bench_us(candidate, True, repeats)
    base_hot = min(base_hot, _bench_us(baseline, False, repeats))
    base_cold = min(base_cold, _bench_us(baseline, True, repeats))
    minimum_bytes = (
        quantized.numel()
        + 4 * scales.numel()
        + 2 * x.numel()
        + 2 * actual.numel()
        + 2 * bf16.numel()
    )
    result = {
        "baseline": "production BF16 projection",
        "baseline_hot_us": base_hot,
        "baseline_cold_us": base_cold,
        "fp8_hot_us": hot,
        "fp8_cold_us": cold,
        "hot_speedup": base_hot / hot,
        "cold_speedup": base_cold / cold,
        "cold_gbps": minimum_bytes / (cold * 1000),
        "correctness": "FP32 quantized oracle and changed-input graph replay",
    }
    if previous is not None:
        result["previous_fp32_hot_us"] = _bench_us(previous, False, repeats)
        result["previous_fp32_cold_us"] = _bench_us(previous, True, repeats)
        result["prefill_hot_speedup"] = result["previous_fp32_hot_us"] / hot
        result["prefill_cold_speedup"] = result["previous_fp32_cold_us"] / cold
    return result


def _bench_fused_hc_up(x, weight, repeats: int, *, large_tiles=False) -> dict:
    result = _bench_rowwise_fp8(x, weight, repeats, hc=True)
    quantized, scales = quantize_rowwise_fp8(weight)
    m, k = x.shape
    n = weight.shape[0]
    reference = (x.float() @ quantized.float().t()) * scales
    candidates = []
    for bm in (64, 128) if large_tiles else (32, 64):
        for bn in (128,) if large_tiles else (32, 64):
            for bk in (64, 128) if large_tiles else (32, 64, 128):
                for stages, warps in (
                    ((2, 4), (3, 4), (2, 8), (3, 8))
                    if large_tiles
                    else ((2, 4), (3, 4))
                ):
                    config = {
                        "block_m": bm,
                        "block_n": bn,
                        "block_k": bk,
                        "num_stages": stages,
                        "num_warps": warps,
                    }

                    def candidate(bm=bm, bn=bn, bk=bk, stages=stages, warps=warps):
                        output = x.new_empty((m, n))
                        _rowwise_fp8_projection[
                            ((n + bn - 1) // bn, 1, (m + bm - 1) // bm)
                        ](
                            x,
                            quantized,
                            scales,
                            x,
                            output,
                            m,
                            n,
                            k,
                            x.stride(0),
                            x.stride(1),
                            quantized.stride(0),
                            quantized.stride(1),
                            BF16_START=0,
                            BF16_ROWS=0,
                            SPLIT_K=1,
                            BLOCK_M=bm,
                            BLOCK_N=bn,
                            BLOCK_K=bk,
                            num_warps=warps,
                            num_stages=stages,
                        )
                        return output

                    candidate()
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = candidate()
                    for sign in (-1, 1):
                        x.mul_(-1)
                        graph.replay()
                        torch.testing.assert_close(
                            actual.float(), sign * reference, rtol=1e-2, atol=1e-1
                        )
                    hot = _bench_us(candidate, False, repeats)
                    cold = _bench_us(candidate, True, repeats)
                    candidates.append(
                        {"config": config, "hot_us": hot, "cold_us": cold}
                    )
                    print(
                        f"HC up M={m} {config}: hot {hot:.2f}, cold {cold:.2f} us",
                        flush=True,
                    )
    result["fused_candidates"] = candidates
    result["best_fused"] = min(candidates, key=lambda item: item["cold_us"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--projections", choices=PROJECTIONS, nargs="+", default=list(PROJECTIONS)
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rowwise-fp8-head", action="store_true")
    parser.add_argument("--rowwise-fp8-hc", action="store_true")
    parser.add_argument("--rowwise-fp8-output", action="store_true")
    parser.add_argument("--rowwise-fp8-hc-triton", action="store_true")
    parser.add_argument(
        "--hc-prefill-tiles", choices=("small", "large"), default="small"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rowwise_fp8_hc_triton and (
        not args.rowwise_fp8_hc
        or args.projections != ["hc_up"]
        or not all(m > 32 for m in args.tokens)
    ):
        raise SystemExit("--rowwise-fp8-hc-triton requires HC up and M>32")
    if torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("This benchmark requires an SM120 GPU")
    if not shape_dynamic_skinny_gemm.is_available():
        raise SystemExit("This benchmark requires CuTe DSL")
    if not all(
        m >= 1 and (m <= 16 or args.rowwise_fp8_hc or args.rowwise_fp8_output)
        for m in args.tokens
    ):
        raise SystemExit("--tokens must be in [1, 16], or positive for HC FP8")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "timing": "FlashInfer CUPTI, CUDA graph, hot and cold L2",
        "dtype": "bfloat16",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name in args.projections:
            n, k = PROJECTIONS[name]
            weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
            for m in args.tokens:
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                if (
                    args.rowwise_fp8_head
                    or args.rowwise_fp8_hc
                    or args.rowwise_fp8_output
                ):
                    if args.rowwise_fp8_output and name != "attn_out":
                        raise SystemExit("--rowwise-fp8-output requires attn_out")
                    if args.rowwise_fp8_head and name not in (
                        "lm_head",
                        "draft_lm_head",
                    ):
                        raise SystemExit("--rowwise-fp8-head requires head projections")
                    if args.rowwise_fp8_hc and not name.startswith("hc_"):
                        raise SystemExit("--rowwise-fp8-hc requires HC projections")
                    preserved = (
                        {"bf16_start": 320, "bf16_rows": 4, "pad_rows": 12}
                        if name == "hc_down_inject"
                        else {}
                    )
                    result = (
                        _bench_fused_hc_up(
                            x,
                            weight,
                            args.repeats,
                            large_tiles=args.hc_prefill_tiles == "large",
                        )
                        if args.rowwise_fp8_hc_triton
                        else _bench_rowwise_fp8(
                            x,
                            weight,
                            args.repeats,
                            hc=args.rowwise_fp8_hc,
                            dense_output=args.rowwise_fp8_output,
                            **preserved,
                        )
                    )
                    results["cases"].append(
                        {"projection": name, "shape": [m, n, k], **result}
                    )
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(
                        f"{name} M={m}: hot {result['baseline_hot_us']:.2f} -> "
                        f"{result['fp8_hot_us']:.2f} us, cold "
                        f"{result['baseline_cold_us']:.2f} -> "
                        f"{result['fp8_cold_us']:.2f} us",
                        flush=True,
                    )
                    continue
                reference = torch.nn.functional.linear(x.float(), weight.float())
                baseline = partial(torch.nn.functional.linear, x, weight)
                baseline()
                for config in _configs(m, n, k):
                    actual = shape_dynamic_skinny_gemm(x, weight, config)
                    torch.testing.assert_close(
                        actual.float(), reference, rtol=2e-2, atol=2e-1
                    )
                base_hot = _bench_us(baseline, False, args.repeats)
                base_cold = _bench_us(baseline, True, args.repeats)
                candidates = []
                for config in _configs(m, n, k):
                    candidate = partial(shape_dynamic_skinny_gemm, x, weight, config)
                    hot = _bench_us(candidate, False, args.repeats)
                    cold = _bench_us(candidate, True, args.repeats)
                    # Theoretical lower bound: every input read and output
                    # written once. Cache reuse can exceed DRAM bandwidth.
                    min_bytes = 2 * (m * k + n * k + m * n)
                    candidates.append(
                        {
                            "config": asdict(config),
                            "hot_us": hot,
                            "cold_us": cold,
                            "cold_gbps": min_bytes / (cold * 1000),
                            "hot_speedup": base_hot / hot,
                            "cold_speedup": base_cold / cold,
                        }
                    )
                # Recheck the baseline after the sweep to expose clock drift.
                base_hot = min(base_hot, _bench_us(baseline, False, args.repeats))
                base_cold = min(base_cold, _bench_us(baseline, True, args.repeats))
                for item in candidates:
                    item["hot_speedup"] = base_hot / item["hot_us"]
                    item["cold_speedup"] = base_cold / item["cold_us"]
                winners = [
                    item
                    for item in candidates
                    if item["hot_speedup"] > 1.05 and item["cold_speedup"] > 1.05
                ]
                best = min(winners, key=lambda item: item["cold_us"], default=None)
                results["cases"].append(
                    {
                        "projection": name,
                        "shape": [m, n, k],
                        "baseline_hot_us": base_hot,
                        "baseline_cold_us": base_cold,
                        "candidates": candidates,
                        "best": best,
                    }
                )
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                if best is None:
                    print(f"{name} M={m}: retain torch linear", flush=True)
                else:
                    print(
                        f"{name} M={m}: hot {base_hot:.2f} -> "
                        f"{best['hot_us']:.2f} us, cold {base_cold:.2f} -> "
                        f"{best['cold_us']:.2f} us, {best['config']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
