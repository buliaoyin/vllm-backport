# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM80 DSv4 kernels at TP1; correctness precedes CUDA graph timing."""

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.kernels import benchmark_dsv4_sm80 as fixtures
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse


def time_us(fn):
    # CUPTI is unavailable in vllm-backport. Flush outside the event interval.
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(30)
    ]
    for start, end in pairs:
        flush.zero_()
        start.record()
        graph.replay()
        end.record()
    torch.accelerator.synchronize()
    return 1000 * statistics.median(start.elapsed_time(end) for start, end in pairs)


@torch.inference_mode()
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    fixtures.NUM_HEADS = 64
    fixtures.C128_MAX_MODEL_LEN = 32768
    device = torch.device("cuda")
    results = []

    def record(row):
        results.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "tp": 1,
                    "heads": 64,
                    "dtype": "bfloat16",
                    "timing": (
                        "CUDA events, CUDA graph, explicit 64 MiB L2 flush "
                        "outside timing, median of 30"
                    ),
                    "rows": results,
                },
                indent=2,
            )
        )

    for m, depth in ((128, 0), (512, 4096), (2048, 4096)):
        inp = fixtures._c128_prefill_inputs(m, depth, device)
        blocks = sparse.build_query_blocks(inp["query_start_loc_cpu"], 4, device)
        common = dict(
            q=inp["q"],
            kv=inp["kv"][:, None],
            scale=fixtures.SCALE,
            head_dim=512,
            nope_head_dim=448,
            rope_head_dim=64,
            attn_sink=inp["attn_sink"],
            output=inp["out"],
        )

        def plain(common=common, inp=inp):
            sparse.rocm_sparse_attn_prefill(
                **common, indices=inp["dense"], topk_length=inp["lens"]
            )

        def blocked(common=common, inp=inp, blocks=blocks):
            sparse.rocm_sparse_attn_prefill_blocked(
                **common,
                block_req=blocks[0],
                block_qstart=blocks[1],
                query_start_loc=inp["query_start_loc"],
                seq_lens=inp["seq_lens"],
                gather_lens=inp["gather_lens"],
                top_k=inp["top_k"],
                row_stride=inp["row_stride"],
                swa_offset=inp["swa_offset"],
                compress_ratio=128,
                window_size=128,
                block_m=4,
            )

        for name, fn in [("plain", plain), ("blocked", blocked)]:
            fn()
            error = fixtures._c128_prefill_fp32_error(inp, inp["out"])
            assert error < 0.01, (name, error)
            us = time_us(fn)
            flop = 4 * 512 * 64 * inp["rows_per_query"] * m
            record(
                dict(
                    kernel="prefill",
                    arm=name,
                    m=m,
                    depth=depth,
                    us=us,
                    tflops=flop / us * 1e-6,
                    fp32_scaled_error=error,
                )
            )

    for ratio in (4, 128):
        for batch in (1, 4, 16):
            inp = fixtures._decode_inputs(
                batch,
                [],
                device,
                topk_len=min(512, 8192 // ratio),
                topk_rows=8192 // ratio,
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

            def decode(kwargs=kwargs, ratio=ratio, inp=inp):
                return sparse._rocm_sparse_attn_decode_ragged_triton(
                    **kwargs,
                    scale=fixtures.SCALE,
                    nope_head_dim=448,
                    rope_head_dim=64,
                    compress_ratio=ratio,
                    out=inp["out"],
                )

            reference = None
            for name, uniform, splits in [
                ("baseline", 1, 0),
                ("tile", 0, 0),
                ("tile_fixed16", 0, 16),
            ]:
                os.environ["VLLM_DSV4_UNIFORM_DECODE_BLOCK_K"] = str(uniform)
                os.environ["VLLM_DSV4_FIXED_DECODE_SPLITS"] = str(splits)
                actual = decode().clone()
                if reference is None:
                    reference = actual
                torch.testing.assert_close(actual, reference, atol=5e-3, rtol=1e-2)
                error = float((actual.float() - reference.float()).abs().max())
                us = time_us(decode)
                record(
                    dict(
                        kernel="decode",
                        ratio=ratio,
                        queries=batch,
                        arm=name,
                        us=us,
                        max_abs_error=error,
                    )
                )

    for ratio, overlap in ((4, True), (128, False)):
        for count in (1, 16):
            rows = count * ratio
            width = 512 * (2 if overlap else 1)
            state = torch.randn(
                rows // 4 + 1, 4, width * 2, device=device, dtype=torch.float32
            )
            positions = torch.arange(ratio - 1, rows, ratio, device=device)
            slots = torch.arange(count, device=device)
            kv = torch.zeros(count, 1, 584, device=device, dtype=torch.uint8)
            kwargs = dict(
                state_cache=state,
                num_actual=count,
                token_to_req_indices=torch.zeros(
                    count, device=device, dtype=torch.int32
                ),
                positions=positions,
                slot_mapping=slots,
                block_table=torch.arange(
                    state.shape[0], device=device, dtype=torch.int32
                )[None],
                block_size=4,
                state_width=width,
                cos_sin_cache=torch.randn(rows, 64, device=device),
                kv_cache=kv,
                k_cache_metadata=SimpleNamespace(slot_mapping=slots),
                pdl_kwargs={},
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=ratio,
                overlap=overlap,
                use_fp4_cache=False,
                rms_norm_weight=torch.ones(512, device=device, dtype=torch.bfloat16),
                rms_norm_eps=1e-6,
                quant_block=64,
                token_stride=576,
                scale_dim=8,
            )

            def compress(kwargs=kwargs):
                compress_norm_rope_store_triton(**kwargs)

            reference = None
            for enabled in (0, 1):
                os.environ["VLLM_DSV4_SM80_COMPRESSOR_TUNING"] = str(enabled)
                compress()
                if reference is None:
                    reference = kv.clone()
                # Quantized output bytes must remain identical for this tuning.
                torch.testing.assert_close(kv, reference, atol=0, rtol=0)
                us = time_us(compress)
                minimum_bytes = rows * width * 2 * 4 + count * 584
                record(
                    dict(
                        kernel="compressor",
                        ratio=ratio,
                        queries=count,
                        tuned=enabled,
                        us=us,
                        minimum_gbps=minimum_bytes / us / 1000,
                        byte_exact=True,
                    )
                )


if __name__ == "__main__":
    run()
