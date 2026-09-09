# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GLM NoPE sparse attention with a frozen baseline kernel file."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph

from vllm.platforms import current_platform
from vllm.v1.attention.ops import triton_mla_sparse_kernel as candidate


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("glm_sparse_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main(args):
    baseline = load_baseline(args.baseline_kernel)
    torch.manual_seed(42)
    result = {
        "gpu": current_platform.get_device_name(),
        "timer": "CUDA graph, cold L2",
        "baseline_sha256": hashlib.sha256(
            args.baseline_kernel.read_bytes()
        ).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        "rows": [],
    }
    kv = torch.randn(32768, 1, 512, device="cuda", dtype=torch.bfloat16)
    warm_q = torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    warm_indices = torch.zeros(1, 1, 2176, device="cuda", dtype=torch.int32)
    # Match the backend's init-time tuning inputs, including its repeated KV row.
    for module in (baseline, candidate):
        for splits in (1, 2, 4, 8, 16, 32):
            module.triton_mla_sparse_attention(
                warm_q, kv[:64], warm_indices, 0.0625, num_kv_splits=splits
            )
    result["autotune"] = {
        name: {
            kernel: {
                str(key): str(config)
                for key, config in getattr(module, kernel).cache.items()
            }
            for kernel in ("_sparse_mla_kernel_final", "_sparse_mla_kernel_split")
        }
        for name, module in (("baseline", baseline), ("candidate", candidate))
    }
    for tokens in (1, 4, 16, 64):
        q = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.bfloat16)
        for valid in (0, 32, 256, 2048):
            indices = torch.full(
                (tokens, 1, 2176), -1, device="cuda", dtype=torch.int32
            )
            if valid:
                indices[:, :, :valid] = torch.randint(
                    0, 32768, (tokens, 1, valid), device="cuda", dtype=torch.int32
                )
                # kpool places the incomplete tail after the fixed history budget.
                indices[:, :, 2048:2051] = torch.randint(
                    0, 32768, (tokens, 1, 3), device="cuda", dtype=torch.int32
                )
            inputs = (q, kv, indices, 0.0625)
            expected = baseline.triton_mla_sparse_attention(*inputs)
            actual = candidate.triton_mla_sparse_attention(*inputs)
            torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.01)
            if not valid:
                assert torch.count_nonzero(actual).item() == 0
            else:
                chosen = indices[0, 0]
                keys = kv[chosen[chosen >= 0].long(), 0].float()
                probabilities = torch.softmax((q[0].float() @ keys.T) * 0.0625, dim=-1)
                reference = probabilities @ keys
                torch.testing.assert_close(
                    actual[0].float(), reference, rtol=0.02, atol=0.01
                )
            # QK + PV: count useful, valid-key work, excluding padding.
            flops = 4 * tokens * 64 * 512 * (valid + 3 if valid else 0)
            row = dict(
                tokens=tokens, valid_history=valid, topk=2176, useful_flops=flops
            )
            for name, module in (("baseline", baseline), ("candidate", candidate)):
                times = bench_gpu_time_with_cudagraph(
                    module.triton_mla_sparse_attention,
                    input_args=inputs,
                    dry_run_iters=3,
                    repeat_iters=15,
                    cold_l2_cache=True,
                )
                latency = statistics.median(times) * 1000
                row[name + "_us"] = latency
                row[name + "_TFLOPs"] = flops / latency / 1e6
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
