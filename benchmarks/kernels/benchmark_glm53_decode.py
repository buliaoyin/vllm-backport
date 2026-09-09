# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM decode ablations: KDA copies, NoPE layout, and PP draft scatter.

Use --timer graph on GPUs without CUPTI support. GPU timings use cold L2
and exclude setup/compilation; copy/concat operations being removed stay in
the timed wrapper. PP scatter reports wall time including host synchronization.
"""

import argparse
import functools
import json
import statistics
import time
from pathlib import Path

import torch
from flashinfer.testing import (
    bench_gpu_time_with_cudagraph,
    bench_gpu_time_with_cupti,
)

from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda
from vllm.v1.worker.gpu.pp_utils import scatter_draft_tokens


def kda(qkv, projected, gate, a_log, bias, state, indices, cu, accepted, materialize):
    tokens, heads, dim = gate.shape[1:]
    q, k, v = (x.view(1, tokens, heads, dim) for x in qkv.chunk(3, dim=-1))
    beta = projected[:, 3 * heads * dim : 3 * heads * dim + heads].unsqueeze(0)
    if materialize:
        q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    return fused_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=state,
        ssm_state_indices=indices,
        cu_seqlens=cu,
        num_accepted_tokens=accepted,
        a_log=a_log,
        g_bias=bias,
        compute_gate=True,
        sigmoid_beta=True,
        lower_bound=-5.0,
    )[0]


def nope(q, weights, output, empty_pe, token_major):
    weights = weights[:, : q.shape[-1], :]
    if token_major:
        torch.bmm(q.transpose(0, 1), weights, out=output.transpose(0, 1))
        return output
    torch.bmm(q.transpose(0, 1), weights, out=output)
    return torch.cat([output.transpose(0, 1), empty_pe], dim=-1)


def gpu_time(fn, inputs, timer):
    fn(*inputs)
    torch.accelerator.synchronize()
    method = (
        bench_gpu_time_with_cupti if timer == "cupti" else bench_gpu_time_with_cudagraph
    )
    kwargs = dict(
        input_args=inputs, cold_l2_cache=True, dry_run_iters=5, repeat_iters=40
    )
    if timer == "cupti":
        kwargs["use_cuda_graph"] = True
    return statistics.median(method(fn, **kwargs)) * 1000


def masked_scatter(dst, src, indices):
    valid = indices >= 0
    dst[indices[valid]] = src[valid]


def wall_time(fn, repeats=100):
    for _ in range(10):
        fn()
    samples = []
    for _ in range(7):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    return statistics.median(samples)


@torch.inference_mode()
def main(args):
    torch.manual_seed(42)
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        timer=args.timer,
        cold_l2=True,
        rows=[],
    )
    for requests, steps in [(1, 1), (1, 4), (16, 1), (16, 4)]:
        tokens, heads, dim = requests * steps, 64, 128
        qkv = torch.randn(tokens, 3 * heads * dim, device="cuda", dtype=torch.bfloat16)
        projected = torch.randn(
            tokens,
            3 * heads * dim + heads + 2 * dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        a_log = torch.zeros(heads, device="cuda")
        bias = torch.zeros(heads * dim, device="cuda")
        state = torch.randn(tokens + 1, heads, dim, dim, device="cuda")
        indices = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32)
        if steps > 1:
            indices = indices.view(requests, steps)
        cu = torch.arange(0, tokens + 1, steps, device="cuda", dtype=torch.int32)
        accepted = (
            torch.ones(requests, device="cuda", dtype=torch.int32)
            if steps > 1
            else None
        )
        expected_state = state.clone()
        common = (qkv, projected, gate, a_log, bias)
        expected = kda(*common, expected_state, indices, cu, accepted, True)
        actual = kda(*common, state, indices, cu, accepted, False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        times = [
            gpu_time(kda, (*common, state, indices, cu, accepted, copy), args.timer)
            for copy in (True, False)
        ]
        # Minimum state traffic: one initial read per request and one state write
        # per verification token, independent of implementation.
        state_bytes = (requests + tokens) * heads * dim * dim * 4
        result["rows"].append(
            dict(
                op="kda",
                requests=requests,
                steps=steps,
                before_us=times[0],
                after_us=times[1],
                min_state_bytes=state_bytes,
                before_GBs=state_bytes / times[0] / 1e3,
                after_GBs=state_bytes / times[1] / 1e3,
            )
        )
    for tokens in (1, 4, 16, 64):
        q = torch.randn(tokens, 64, 256, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(64, 512, 512, device="cuda", dtype=torch.bfloat16)
        empty = torch.empty(tokens, 64, 0, device="cuda", dtype=torch.bfloat16)
        before = torch.empty(64, tokens, 512, device="cuda", dtype=torch.bfloat16)
        after = torch.empty(tokens, 64, 512, device="cuda", dtype=torch.bfloat16)
        expected = nope(q, weights, before, empty, False)
        actual = nope(q, weights, after, empty, True)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        times = [
            gpu_time(nope, (q, weights, output, empty, layout), args.timer)
            for output, layout in ((before, False), (after, True))
        ]
        flops = 2 * tokens * 64 * 256 * 512
        result["rows"].append(
            dict(
                op="nope_bmm_concat",
                tokens=tokens,
                before_us=times[0],
                after_us=times[1],
                flops=flops,
                before_TFLOPs=flops / times[0] / 1e6,
                after_TFLOPs=flops / times[1] / 1e6,
            )
        )
    for requests in (1, 16, 64):
        src = torch.arange(requests * 3, device="cuda").view(requests, 3)
        indices = torch.arange(requests, device="cuda")
        indices[6::7] = -1
        dst = torch.full((requests + 8, 3), -1, device="cuda", dtype=torch.int64)
        expected = dst.clone()
        valid = indices >= 0
        expected[indices[valid]] = src[valid]
        scatter_draft_tokens(dst, src, indices)
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)

        times = [
            wall_time(functools.partial(masked_scatter, dst, src, indices)),
            wall_time(functools.partial(scatter_draft_tokens, dst, src, indices)),
        ]
        result["rows"].append(
            dict(
                op="pp_scatter_wall",
                requests=requests,
                before_us=times[0],
                after_us=times[1],
            )
        )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timer", choices=("cupti", "graph"), default="cupti")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
