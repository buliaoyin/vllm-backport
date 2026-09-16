# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure incremental V4.1 KV packing and save generated device code."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph

from vllm.models.deepseek_v4_1.pp_kv import _copy_packed_kv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.accelerator.set_device_index(args.device)
    torch.set_num_threads(1)
    payload = {
        "gpu": torch.cuda.get_device_name(args.device),
        "timer": "CUDA graph events, warm L2; CMP CUPTI error 42",
        "results": [],
    }
    cache = torch.randint(0, 256, (512, 64, 584), dtype=torch.uint8, device="cuda")
    for tokens in (1, 4, 128, 1024, 2048, 4096):
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
        slots = torch.where(positions % 2 == 1, positions // 2, -1)
        rows = torch.empty((tokens, 584), device="cuda", dtype=torch.uint8)
        for scatter in (False, True):

            def operation(tokens=tokens, slots=slots, rows=rows, scatter=scatter):
                return _copy_packed_kv_rows[(tokens,)](
                    cache,
                    slots,
                    rows,
                    CACHE_STRIDE=cache.stride(0),
                    PAGE_ROWS=64,
                    NUM_SLOTS=tokens,
                    SCATTER=scatter,
                    num_warps=4,
                )

            kernel = operation()
            record = {
                "tokens": tokens,
                "scatter": scatter,
                "num_regs": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
            }
            stem = args.output.with_suffix("").with_name(
                args.output.stem + f"-tokens{tokens}-scatter{int(scatter)}"
            )
            stem.with_suffix(".ptx").write_text(kernel.asm["ptx"])
            times = bench_gpu_time_with_cudagraph(
                operation,
                cold_l2_cache=False,
                num_iters_within_graph=16,
                dry_run_iters=5,
                repeat_iters=30,
            )
            record["median_us"] = statistics.median(times) * 1000
            payload["results"].append(record)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
