# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare request-index expansion with exact checks before timing."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.v1.attention.ops.common import fill_token_to_req_indices


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    rows = []
    cases = [
        (lengths, padding)
        for lengths in (
            [1],
            [8],
            [8] * 4,
            [8] * 64,
            [2048],
            [0, 3, 0, 5, 0],
            [1] * 1024,
        )
        for padding in (0, 3)
    ]
    for lengths, padding in cases:
        starts = torch.tensor([0] + lengths, dtype=torch.int32, device="cuda").cumsum(
            0, dtype=torch.int32
        )
        mapped = sum(lengths)
        total = mapped + padding
        buffers = [
            torch.empty(total, dtype=torch.int32, device="cuda") for _ in range(2)
        ]

        def baseline(
            starts=starts, lengths=lengths, mapped=mapped, buffers=buffers, total=total
        ):
            query_lens = starts[1:] - starts[:-1]
            result = torch.repeat_interleave(
                torch.arange(len(lengths), dtype=torch.int32, device="cuda"),
                query_lens,
                output_size=mapped,
            )
            buffers[0][:mapped].copy_(result)
            if mapped < total:
                buffers[0][mapped:].zero_()

        def fused(starts=starts, buffers=buffers, total=total):
            fill_token_to_req_indices(starts, buffers[1], total)

        funcs = [baseline, fused]
        for fn in funcs:
            for _ in range(50):
                fn()
        expected = torch.tensor(
            [i for i, length in enumerate(lengths) for _ in range(length)]
            + [0] * padding,
            dtype=torch.int32,
            device="cuda",
        )
        for output in buffers:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.accelerator.synchronize()
        graphs = []
        for fn in funcs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
        samples = [[], []]
        for repeat in range(80):
            for arm in [0, 1] if repeat % 2 == 0 else [1, 0]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                flush.zero_()
                start.record()
                graphs[arm].replay()
                end.record()
                samples[arm].append((start, end))
        torch.accelerator.synchronize()
        host_samples = [[], []]
        for repeat in range(10):
            for arm in [0, 1] if repeat % 2 == 0 else [1, 0]:
                torch.accelerator.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    funcs[arm]()
                torch.accelerator.synchronize()
                host_samples[arm].append((time.perf_counter() - start) * 1e4)
        for arm, name in enumerate(("repeat_interleave", "fused")):
            row = dict(
                arm=name,
                num_reqs=len(lengths),
                num_mapped_tokens=mapped,
                padded_tokens=padding,
                logical_bytes=4 * (len(lengths) + 1 + total),
                graph_device_us=1000
                * statistics.median(
                    start.elapsed_time(end) for start, end in samples[arm]
                ),
                eager_wall_us=statistics.median(host_samples[arm]),
                exact_match=True,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                dtype="int32",
                timing=(
                    "CUDA events on captured operation, 128 MiB L2 flush outside "
                    "interval, 80 alternating samples. Eager wall includes dispatch "
                    "and baseline temporaries, 10 alternating groups of 100 calls. "
                    "CUDA-event fallback because CUPTI rejects CMP 170HX (error 42)."
                ),
                rows=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
