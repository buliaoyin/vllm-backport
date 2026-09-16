# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure graph-safe complete decode histograms independently on each GPU."""

import argparse
import json
import statistics
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph, bench_gpu_time_with_cupti

from vllm.models.deepseek_v4_1.cache_feedback import (
    DecodeRouteFeedback,
    _record_decode_routes,
)
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timer", choices=("cupti", "event"), default="cupti")
    args = parser.parse_args()
    assert int(version("cupti-python").split(".")[0]) >= 13
    torch.accelerator.set_device_index(args.device)
    torch.cuda.set_stream(torch.cuda.current_stream(args.device))
    timer = (
        bench_gpu_time_with_cupti
        if args.timer == "cupti"
        else bench_gpu_time_with_cudagraph
    )
    result = {
        "device": args.device,
        "name": torch.cuda.get_device_name(args.device),
        "method": (
            "CUPTI graph replay, cold L2"
            if args.timer == "cupti"
            else "CUDA graph events, warm L2 (CMP does not support CUPTI)"
        ),
        "samples": 100,
        "warmups": 20,
        "cupti": version("cupti-python"),
        "triton": version("triton"),
        "rows": [],
    }
    for tokens in (1, 4, 16):
        ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)
        padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
        feedback = DecodeRouteFeedback(384, ids.device)
        feedback.set_enabled(True)
        feedback.record(ids, padding)
        expected = torch.bincount(ids.flatten().long(), minlength=384)
        observed = feedback.counts.cpu()
        torch.testing.assert_close(observed[:384], expected.cpu())
        torch.testing.assert_close(observed[384:768], (expected != 0).long().cpu())
        assert observed[-2:].tolist() == [1, tokens]
        compiled = _record_decode_routes[(1,)](
            ids,
            padding,
            feedback.enabled,
            feedback.counts,
            tokens,
            6,
            *ids.stride(),
            384,
            True,
            triton.next_power_of_2(tokens),
            8,
            512,
            num_warps=4,
        )
        ptx = args.output.with_name(f"{args.output.stem}-t{tokens}.ptx")
        ptx.write_text(compiled.asm["ptx"])
        for enabled in (False, True):
            feedback.set_enabled(enabled)
            options = {"use_cuda_graph": True} if args.timer == "cupti" else {}
            times = timer(
                partial(feedback.record, ids, padding),
                cold_l2_cache=args.timer == "cupti",
                dry_run_iters=20,
                repeat_iters=100,
                **options,
            )
            row = {
                "tokens": tokens,
                "enabled": enabled,
                "median_us": statistics.median(times) * 1000,
                "samples_ms": times,
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
                "ptx": str(ptx),
                "correctness": "exact complete route/call counts",
            }
            result["rows"].append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_ms"}),
                flush=True,
            )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
