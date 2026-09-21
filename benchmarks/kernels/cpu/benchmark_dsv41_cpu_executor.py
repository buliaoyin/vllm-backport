# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare IK CPU executors on rotating real expert weights and cached routes."""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4 import (
    CPUMoEConfig,
    CPUMXFP4Experts,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path)
    parser.add_argument("--layer", type=int, default=38)
    parser.add_argument("--threads", type=int, nargs="+", default=[22])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--cached-routes", type=int, nargs="+", default=[0, 3, 4, 6])
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1234)
    config = json.loads((args.checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    experts, hidden, intermediate, topk = (
        config[key]
        for key in (
            "n_routed_experts",
            "hidden_size",
            "moe_intermediate_size",
            "num_experts_per_tok",
        )
    )
    backends = [
        CPUMXFP4Experts(
            CPUMoEConfig(
                backend="ik", library_path=str(path), num_threads=args.threads[0]
            ),
            experts,
            hidden,
            intermediate,
            topk,
            config["swiglu_limit"],
            max(args.tokens),
        )
        for path in [args.library]
        + ([args.reference_library] if args.reference_library else [])
    ]
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    with ExitStack() as stack:
        handles = {}
        for expert in range(experts):
            for projection in range(3):
                stem = f"layers.{args.layer}.ffn.experts.{expert}.w{projection + 1}"
                values = []
                for suffix in ("weight", "scale"):
                    key = f"{stem}.{suffix}"
                    shard = index["weight_map"][key]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(
                            safe_open(
                                args.checkpoint / shard, framework="pt", device="cpu"
                            )
                        )
                    values.append(handles[shard].get_tensor(key))
                for backend in backends:
                    backend.load_expert(expert, projection, *values)
    for backend in backends:
        backend.prepare()
    backend = backends[0]
    records = []
    payload = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "timing": "native CPU forward with preallocated FP32 IO; no PCIe transfer",
        "route_distribution": (
            "synthetic; tokens share the first half of routes before masking; "
            "the first cached_routes slots are then removed"
        ),
        "records": records,
    }
    try:
        for tokens in args.tokens:
            for cached in args.cached_routes:
                batches = []
                for _ in range(args.batches):
                    x = (
                        torch.randn(tokens, hidden, generator=generator)
                        .bfloat16()
                        .float()
                    )
                    ids = (
                        torch.randperm(experts, generator=generator)[:topk]
                        .int()
                        .repeat(tokens, 1)
                    )
                    for token in range(1, tokens):
                        replacement = torch.randperm(experts, generator=generator)
                        replacement = replacement[
                            ~torch.isin(replacement, ids[token, : topk // 2])
                        ]
                        ids[token, topk // 2 :] = replacement[: topk - topk // 2].int()
                    ids[:, :cached] = -2
                    routes = (
                        torch.rand(tokens, topk, generator=generator).softmax(-1) * 2.5
                    )
                    batches.append((x, ids, routes, torch.empty_like(x)))
                for batch in batches:
                    x, ids, routes, _ = batch
                    backend.set_execution_mode("graph")
                    expected = backend.forward(x, ids, routes)
                    if len(backends) > 1:
                        torch.testing.assert_close(
                            backends[1].forward(x, ids, routes),
                            expected,
                            atol=0,
                            rtol=0,
                        )
                    backend.set_execution_mode("compact")
                    torch.testing.assert_close(
                        backend.forward(x, ids, routes), expected, atol=0, rtol=0
                    )
                for threads in args.threads:
                    backend.set_num_threads(threads)
                    for mode in ("graph", "compact", "compact", "graph"):
                        backend.set_execution_mode(mode)
                        native = backend._library.dsv41_moe_forward
                        samples = []
                        for iteration in range(args.batches * 2):
                            batch = batches[iteration % args.batches]
                            pointers = [value.data_ptr() for value in batch]
                            start = time.perf_counter_ns()
                            code = native(backend._handle, tokens, *pointers)
                            elapsed = (time.perf_counter_ns() - start) / 1e6
                            assert code == 0
                            if iteration >= args.batches:
                                samples.append(elapsed)
                        record = dict(
                            tokens=tokens,
                            cached_routes=cached,
                            threads=threads,
                            mode=mode,
                            mean_ms=statistics.mean(samples),
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                        )
                        records.append(record)
                        print(
                            json.dumps(
                                {k: v for k, v in record.items() if k != "samples_ms"}
                            ),
                            flush=True,
                        )
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        for backend in backends:
            backend.close()


if __name__ == "__main__":
    main()
