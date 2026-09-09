# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep SM80 sparse-decode warps and tiles using the serving wrapper."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from benchmarks.kernels import benchmark_dsv4_sm80 as fixtures
from tests.kernels.attention.test_rocm_triton_attn_dsv4 import (
    _ref_sparse_decode_ragged,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse


class LaunchOverride:
    def __init__(self, kernel, warps):
        self.kernel = kernel
        self.warps = warps
        self.compiled = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs["num_warps"] = self.warps
            self.compiled = self.kernel[grid](*args, **kwargs)
            return self.compiled

        return launch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sparse.current_platform.is_cuda()
    assert sparse.current_platform.is_device_capability(80)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    fixtures.NUM_HEADS = 64
    device = torch.device("cuda")
    original = sparse._sparse_attn_decode_partial_kernel
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    rows = []
    cases = [
        (4, 1, 128),
        (4, 8, 128),
        (4, 8, 512),
        (4, 64, 512),
        (128, 8, 4),
        (128, 8, 256),
        (128, 64, 256),
    ]
    for ratio, queries, extra_len in cases:
        inp = fixtures._decode_inputs(
            queries, [], device, topk_len=extra_len, topk_rows=max(extra_len, 4096)
        )
        kwargs = {
            k: inp[k]
            for k in (
                "q",
                "main_cache",
                "main_indices",
                "main_indptr",
                "extra_cache",
                "extra_indices",
                "extra_indptr",
                "attn_sink",
            )
        }
        expected = _ref_sparse_decode_ragged(
            q=inp["q"],
            main_cache=inp["main_cache"],
            main_rows=inp["main_indices"].view(queries, -1).tolist(),
            scale=fixtures.SCALE,
            attn_sink=inp["attn_sink"],
            block_size=fixtures.CACHE_BLOCK_SIZE,
            extra_cache=inp["extra_cache"],
            extra_rows=inp["extra_indices"].view(queries, -1).tolist(),
            main_use_fnuz=False,
        )
        capacity_per_query = 512 if ratio == 4 else 256
        packed = inp["extra_indices"]
        capacity = torch.full(
            (queries * capacity_per_query,), -1, dtype=torch.int32, device=device
        )
        capacity[: packed.numel()].copy_(packed)
        kwargs["extra_indices"] = capacity
        arms = [(w, k) for w in (4, 8) for k in ((32, 64) if ratio == 4 else (32,))]
        graphs = []
        infos = []
        for warps, tile in arms:
            os.environ["VLLM_DSV4_UNIFORM_DECODE_BLOCK_K"] = str(int(tile == 32))
            os.environ["VLLM_DSV4_FIXED_DECODE_SPLITS"] = "0"
            override = LaunchOverride(original, warps)
            sparse._sparse_attn_decode_partial_kernel = override

            def run(kwargs=kwargs, ratio=ratio, inp=inp):
                sparse._rocm_sparse_attn_decode_ragged_triton(
                    **kwargs,
                    scale=fixtures.SCALE,
                    nope_head_dim=448,
                    rope_head_dim=64,
                    compress_ratio=ratio,
                    out=inp["out"],
                )

            for _ in range(5):
                run()
            torch.testing.assert_close(inp["out"], expected, atol=2e-2, rtol=2e-2)
            error = float((inp["out"].float() - expected.float()).abs().max())
            compiled = override.compiled
            assert compiled is not None
            infos.append(
                dict(
                    ratio=ratio,
                    queries=queries,
                    extra_len=extra_len,
                    warps=warps,
                    block_k=tile,
                    capacity_per_query=capacity_per_query,
                    max_abs_error=error,
                    n_regs=compiled.n_regs,
                    n_spills=compiled.n_spills,
                    shared_bytes=compiled.metadata.shared,
                )
            )
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graphs.append(graph)
        events = [[] for _ in arms]
        for repeat in range(80):
            order = range(len(arms)) if repeat % 2 == 0 else reversed(range(len(arms)))
            for arm in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                flush.zero_()
                start.record()
                graphs[arm].replay()
                end.record()
                events[arm].append((start, end))
        torch.accelerator.synchronize()
        for info, pairs in zip(infos, events):
            info["us"] = 1000 * statistics.median(s.elapsed_time(e) for s, e in pairs)
            info["tflops"] = (
                4 * queries * 64 * 512 * (128 + extra_len) / info["us"] * 1e-6
            )
            rows.append(info)
            print(json.dumps(info), flush=True)
        args.output.write_text(
            json.dumps(
                dict(
                    gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    heads=64,
                    dtype="bfloat16",
                    timing=(
                        "CUDA events + graph; cold 128 MiB L2; 80 alternating samples"
                    ),
                    reference="FP32 attention from fp8_ds_mla; atol=rtol=0.02",
                    rows=rows,
                ),
                indent=2,
            )
        )
    sparse._sparse_attn_decode_partial_kernel = original


if __name__ == "__main__":
    main()
