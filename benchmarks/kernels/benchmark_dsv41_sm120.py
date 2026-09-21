# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM120 and portable sparse decode on the V4.1 cache format."""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import (
    bench_gpu_time_with_cudagraph,
    bench_gpu_time_with_cupti,
)

from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.utils.flashinfer import (
    flashinfer_trtllm_batch_decode_sparse_mla_dsv4,
    has_flashinfer_sparse_mla_sm120_config,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_decode_ragged_triton,
)


def pack_cache(values, block):
    cache = torch.zeros(
        (values.shape[0] // block, block, 584),
        dtype=torch.uint8,
        device=values.device,
    )
    slots = torch.arange(values.shape[0], dtype=torch.int64, device=values.device)
    quantize_and_insert_k_cache(values, cache, slots, block_size=block, use_fnuz=False)
    return cache


def unpack_cache(cache):
    block = cache.shape[1]
    slots = torch.arange(cache.shape[0] * block, device=cache.device)
    base = slots // block * cache.stride(0)
    rows = base + slots % block * 576
    scale_rows = base + block * 576 + slots % block * 8
    flat = cache.flatten()
    nope = flat[rows[:, None] + torch.arange(448, device=cache.device)]
    scales = flat[scale_rows[:, None] + torch.arange(7, device=cache.device)]
    nope = nope.view(torch.float8_e4m3fn).float() * torch.exp2(
        scales.float() - 127
    ).repeat_interleave(64, 1)
    rope = flat[rows[:, None] + 448 + torch.arange(128, device=cache.device)]
    return torch.cat((nope, rope.contiguous().view(torch.bfloat16).float()), 1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 128])
    parser.add_argument("--windows", type=int, nargs="+", default=[128, 192])
    parser.add_argument("--extra-block", type=int, choices=[64, 128], default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timer", choices=["cupti", "cuda_graph"], default="cupti")
    args = parser.parse_args()
    if args.timer == "cupti":
        from cupti import cupti  # noqa: F401

    torch.accelerator.set_device_index(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", args.device)
    assert torch.cuda.get_device_capability(device) == (12, 0)
    workspace = torch.zeros(128 * 1024**2, dtype=torch.uint8, device=device)
    main_cache = pack_cache(torch.randn(512, 512, device=device).bfloat16() * 0.2, 64)
    extra_cache = pack_cache(
        torch.randn(16384, 512, device=device).bfloat16() * 0.2, args.extra_block
    )
    main_dense, extra_dense = unpack_cache(main_cache), unpack_cache(extra_cache)
    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "torch": torch.__version__,
        "timing": f"{args.timer}, CUDA graph, warm L2, prebuilt indices",
        "results": [],
    }
    for tokens in args.tokens:
        for window in args.windows:
            q = torch.randn(tokens, 64, 512, device=device).bfloat16() * 0.2
            sink = torch.randn(64, dtype=torch.float32, device=device)
            main_ids = (
                torch.arange(window, dtype=torch.int32, device=device)[None, :]
                + torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
            )
            native_width = min(
                width
                for width in (128, 192, 256, 512, 1024)
                if width >= window and has_flashinfer_sparse_mla_sm120_config(64, width)
            )
            native_ids = torch.nn.functional.pad(
                main_ids, (0, native_width - window), value=-1
            )
            extra_ids = torch.stack(
                [torch.randperm(16384, device=device)[:512] for _ in range(tokens)]
            ).int()
            main_ptr = (
                torch.arange(tokens + 1, dtype=torch.int32, device=device) * window
            )
            extra_ptr = torch.arange(tokens + 1, dtype=torch.int32, device=device) * 512
            main_lengths = torch.full(
                (tokens,), window, dtype=torch.int32, device=device
            )
            extra_lengths = torch.full((tokens,), 512, dtype=torch.int32, device=device)
            out_portable, out_native = torch.empty_like(q), torch.empty_like(q)
            operations = {
                "portable": partial(
                    _rocm_sparse_attn_decode_ragged_triton,
                    q=q,
                    main_cache=main_cache,
                    main_indices=main_ids.flatten(),
                    main_indptr=main_ptr,
                    scale=512**-0.5,
                    attn_sink=sink,
                    nope_head_dim=448,
                    rope_head_dim=64,
                    extra_cache=extra_cache,
                    extra_indices=extra_ids.flatten(),
                    extra_indptr=extra_ptr,
                    out=out_portable,
                ),
                "sm120": partial(
                    flashinfer_trtllm_batch_decode_sparse_mla_dsv4,
                    query=q,
                    swa_kv_cache=main_cache.unsqueeze(-2),
                    compressed_kv_cache=extra_cache.unsqueeze(-2),
                    workspace_buffer=workspace,
                    sparse_indices=native_ids[:, None],
                    extra_sparse_indices=extra_ids[:, None],
                    swa_topk_lens=main_lengths,
                    extra_sparse_topk_lens=extra_lengths,
                    sinks=sink,
                    bmm1_scale=512**-0.5,
                    kv_layout="NHD",
                    out=out_native,
                ),
            }
            kv = torch.cat(
                (main_dense[main_ids.long()], extra_dense[extra_ids.long()]), 1
            )
            logits = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
            scores = torch.cat((logits, sink[None, :, None].expand(tokens, -1, 1)), -1)
            reference = torch.bmm(scores.softmax(-1)[..., :-1], kv)
            errors = {}
            max_abs = {}
            for name, operation in operations.items():
                operation()
                actual = out_portable if name == "portable" else out_native
                error = ((actual.float() - reference).norm() / reference.norm()).item()
                assert torch.isfinite(actual).all(), name
                torch.testing.assert_close(
                    actual.float(), reference, atol=2e-2, rtol=2e-2
                )
                errors[name] = error
                max_abs[name] = (actual.float() - reference).abs().max().item()
                for _ in range(5):
                    operation()
            timings = {name: [] for name in operations}
            for order in (list(operations), list(reversed(operations))):
                for name in order:
                    timer = (
                        bench_gpu_time_with_cupti
                        if args.timer == "cupti"
                        else bench_gpu_time_with_cudagraph
                    )
                    options = {"use_cuda_graph": True} if args.timer == "cupti" else {}
                    samples = timer(
                        operations[name],
                        cold_l2_cache=False,
                        dry_run_iters=5,
                        repeat_iters=30,
                        **options,
                    )
                    timings[name].append(statistics.median(samples))
            record = {
                "tokens": tokens,
                "heads": 64,
                "swa_width": window,
                "native_padded_width": native_width,
                "extra_width": 512,
                "extra_page_block": args.extra_block,
                "relative_rms": errors,
                "max_absolute_error": max_abs,
                "median_ms_by_order": timings,
                "minimum_cache_bytes": tokens * (window + 512) * 584,
            }
            payload["results"].append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
