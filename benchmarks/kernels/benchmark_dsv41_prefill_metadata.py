# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the complete DSV4.1 prefill index construction API on Ampere."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    combine_topk_swa_indices as reference,
)
from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
    DeepseekV41AmpereMLAAttention,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", default=20, type=int)
    parser.add_argument("--iterations-per-sample", default=16, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    combine_topk_swa_indices = DeepseekV41AmpereMLAAttention._combine_prefill_indices
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing": "Complete API including allocation and host synchronization; "
        "CUDA events and host wall clock. Alternating candidates, warm inputs. "
        "The Torch reference cannot be captured in a CUDA graph. "
        "CUPTI is unsupported on CMP 170HX.",
        "results": [],
        "iterations_per_sample": args.iterations_per_sample,
    }
    for tokens, ratio in [
        (128, 1),
        (512, 2),
        (1020, 2),
        (1024, 2),
        (1536, 2),
        (2048, 2),
        (1024, 0),
    ]:
        top_k = 512 if ratio else 0
        n = 33152 // ratio if ratio else 0
        m = n + 128 + 2048
        topk = torch.randint(
            0, max(n, 1), (tokens, 512), device="cuda", dtype=torch.int32
        )
        qsl = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        seqs = torch.tensor([32768], device="cuda", dtype=torch.int32)
        gather = torch.tensor(
            [tokens if ratio == 1 else tokens + 128], device="cuda", dtype=torch.int32
        )
        inputs = (topk, qsl, seqs, gather, 128, ratio, top_k, m, n)
        operations = {"torch": reference, "fused": combine_topk_swa_indices}
        expected = reference(*inputs)
        for actual, baseline in zip(combine_topk_swa_indices(*inputs), expected):
            torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        for op in operations.values():
            for _ in range(10):
                op(*inputs)
        torch.accelerator.synchronize()
        timings = {name: {"gpu_ms": [], "wall_ms": []} for name in operations}
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for repeat in range(args.repeats):
            order = list(operations) if repeat % 2 == 0 else list(reversed(operations))
            for name in order:
                start = time.perf_counter()
                events[0].record()
                for _ in range(args.iterations_per_sample):
                    result = operations[name](*inputs)
                events[1].record()
                events[1].synchronize()
                timings[name]["wall_ms"].append(
                    (time.perf_counter() - start) * 1000 / args.iterations_per_sample
                )
                timings[name]["gpu_ms"].append(
                    events[0].elapsed_time(events[1]) / args.iterations_per_sample
                )
                del result
        record = {
            "tokens": tokens,
            "ratio": ratio,
            "bit_exact": True,
            "median_ms": {
                name: {key: statistics.median(values) for key, values in times.items()}
                for name, times in timings.items()
            },
            "samples_ms": timings,
        }
        payload["results"].append(record)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in record.items() if k != "samples_ms"}),
            flush=True,
        )


if __name__ == "__main__":
    main()
