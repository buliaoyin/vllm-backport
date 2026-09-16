# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare allocation growth across increasing indexer context lengths.

Allocation is the subject: wall time includes wrapper allocation and FP8 conversion,
with input construction and JIT warmup excluded. CUDA graphs would hide allocation.
"""

import argparse
import gc
import json
import time

import torch

from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=524288)
    parser.add_argument("--step", type=int, default=8192)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows, heads, dim = args.rows, 64, 128
    q = torch.randn(rows, heads, dim, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(args.max_context, dim, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(args.max_context, device="cuda")
    weights = torch.randn(rows, heads, device="cuda")
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    sizes = range(args.step, args.max_context + 1, args.step)
    ends = {n: torch.full_like(ks, n) for n in sizes}

    def sequence(rounded):
        for n in sizes:
            logits = fp8_mqa_logits_triton(
                q,
                (k[:n], scales[:n]),
                weights,
                ks,
                ends[n],
                round_allocations=rounded,
            )
            del logits

    for rounded in (False, True):
        sequence(rounded)
    torch.accelerator.synchronize()
    records = []
    for rounded in (False, True, True, False):
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats()
        before = torch.accelerator.memory_allocated()
        start = time.perf_counter()
        sequence(rounded)
        torch.accelerator.synchronize()
        records.append(
            {
                "rounded": rounded,
                "seconds": time.perf_counter() - start,
                "peak_allocated_extra_mib": (
                    torch.accelerator.max_memory_allocated() - before
                )
                / 2**20,
                "peak_reserved_extra_mib": (
                    torch.accelerator.max_memory_reserved() - before
                )
                / 2**20,
            }
        )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "rows": rows,
                "heads": heads,
                "head_dim": dim,
                "max_context": args.max_context,
                "step": args.step,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
