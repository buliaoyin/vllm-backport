# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and full-call CPU MoE timing with native MXFP4 weights."""

import argparse
import json
import os
import platform
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4 import (
    CPUMoEConfig,
    CPUMXFP4Experts,
)


def unpack(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=weight.device,
    )
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    scale = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), scales.to(torch.int32) - 127
    )
    return values[codes] * scale.repeat_interleave(32, dim=-1)


def fp8_roundtrip(tensor, block_size=128):
    blocks = tensor.float().reshape(*tensor.shape[:-1], -1, block_size)
    scale = 2.0 ** torch.ceil(
        torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)
    )
    return (
        ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale)
        .reshape(tensor.shape)
        .bfloat16()
        .float()
    )


def reference(hidden, ids, routes, weights, limit, device="cpu", freetoken=False):
    hidden, ids, routes = (t.to(device) for t in (hidden, ids, routes))
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in ids.unique().tolist():
        token, slot = torch.where(ids == expert)
        x = hidden[token].float()
        if freetoken:
            x = fp8_roundtrip(x)
        w1, w2, w3 = (
            unpack(*(part.to(device) for part in weights[expert, p])) for p in range(3)
        )
        gate, up = x @ w1.T, x @ w3.T
        if freetoken:
            gate, up = gate.bfloat16().float(), up.bfloat16().float()
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        activated = torch.nn.functional.silu(gate) * up
        if freetoken:
            activated = fp8_roundtrip(activated.bfloat16())
        output = (activated @ w2.T) * routes[token, slot, None]
        if freetoken:
            output = output.bfloat16().float()
        result.index_add_(0, token, output)
    if freetoken:
        result = result.bfloat16().float()
    return result.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("kt", "llama", "ik", "freetoken"), required=True
    )
    parser.add_argument("--library", required=True)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 128])
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--route-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--activation-trace", type=Path)
    parser.add_argument("--cache-mode", choices=("rotated", "hot"), default="rotated")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    trace = None
    if args.activation_trace:
        from safetensors.torch import load_file

        if not args.checkpoint or args.tokens != [1]:
            parser.error("An activation trace requires --checkpoint and --tokens 1")
        trace = load_file(args.activation_trace, device="cpu")
        args.cache_mode = "hot"
    stack = ExitStack()
    shard_handles = {}
    index = None
    if args.checkpoint:
        from safetensors import safe_open

        index = json.loads(
            (args.checkpoint / "model.safetensors.index.json").read_text()
        )
        missing = [
            name
            for name in set(index["weight_map"].values())
            if not (args.checkpoint / name).is_file()
        ]
        if missing:
            parser.error(f"Checkpoint download is incomplete: {missing}")

        def read_weight(key):
            filename = index["weight_map"][key]
            if filename not in shard_handles:
                shard_handles[filename] = stack.enter_context(
                    safe_open(args.checkpoint / filename, framework="pt", device="cpu")
                )
            return shard_handles[filename].get_tensor(key)

        config_json = json.loads((args.checkpoint / "config.json").read_text())
        hf = config_json.get("text_config", config_json)
        if (args.hidden, args.intermediate, args.top_k) != (
            hf["hidden_size"],
            hf["moe_intermediate_size"],
            hf["num_experts_per_tok"],
        ):
            parser.error("Requested dimensions differ from checkpoint configuration")
        if args.experts > hf["n_routed_experts"]:
            parser.error("Requested more experts than the checkpoint contains")
    torch.manual_seed(args.seed)
    config = CPUMoEConfig(
        backend="ik" if args.backend == "freetoken" else args.backend,
        num_threads=args.threads,
        library_path=args.library,
    )
    backend_cls = CPUMXFP4Experts
    if args.backend == "freetoken":
        from freetoken_dsv41_adapter import FreeTokenCPUExperts

        backend_cls = FreeTokenCPUExperts
    backend = backend_cls(
        config,
        args.experts,
        args.hidden,
        args.intermediate,
        args.top_k,
        args.limit,
        max(args.tokens),
    )
    weights = {}
    for e in range(args.experts):
        for p in range(3):
            rows, cols = args.intermediate, args.hidden
            if p == 1:
                rows, cols = cols, rows
            if index is None:
                packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
                scales = torch.randint(119, 122, (rows, cols // 32), dtype=torch.uint8)
            else:
                stem = f"layers.{args.layer}.ffn.experts.{e}.w{p + 1}"
                packed = read_weight(stem + ".weight").view(torch.uint8)
                scales = read_weight(stem + ".scale").view(torch.uint8)
            weights[e, p] = packed, scales
            backend.load_expert(e, p, packed, scales)
    backend.prepare()
    results = []
    for tokens in args.tokens:
        batches = []
        for _ in range(args.route_batches):
            hidden = torch.randn(tokens, args.hidden, dtype=torch.bfloat16)
            ids = torch.stack(
                [torch.randperm(args.experts)[: args.top_k] for _ in range(tokens)]
            )
            routes = torch.rand(tokens, args.top_k).softmax(-1) * 2.5
            batches.append((hidden, ids, routes))
        if trace is not None:
            batches = [tuple(trace[key] for key in ("hidden", "ids", "routes"))]
        elif args.cache_mode == "hot":
            batches = [batches[0]]
        expected = reference(*batches[0], weights, args.limit, args.reference_device)
        ideal = expected
        if args.backend == "freetoken":
            expected = reference(
                *batches[0], weights, args.limit, args.reference_device, freetoken=True
            )
        actual = backend.forward(*batches[0]).float()
        error = actual - expected
        relative_rms = (error.norm() / expected.norm().clamp_min(1e-12)).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        if not torch.isfinite(actual).all() or relative_rms > 0.035 or cosine < 0.999:
            raise AssertionError(
                f"{args.backend} tokens={tokens}: relative RMS={relative_rms}, "
                f"cosine={cosine}"
            )
        warmup_iterations = max(32, len(batches))
        for iteration in range(warmup_iterations):
            backend.forward(*batches[iteration % len(batches)])
        durations = []
        for iteration in range(args.iterations):
            batch = batches[iteration % len(batches)]
            start = time.perf_counter_ns()
            backend.forward(*batch)
            durations.append((time.perf_counter_ns() - start) / 1e6)
        touched_experts = torch.cat([b[1].flatten() for b in batches]).unique().numel()
        result = dict(
            tokens=tokens,
            warmup_iterations=warmup_iterations,
            touched_experts=touched_experts,
            median_ms=statistics.median(durations),
            min_ms=min(durations),
            max_ms=max(durations),
            relative_rms=relative_rms,
            unquantized_activation_relative_rms=(
                (actual - ideal).norm() / ideal.norm()
            ).item(),
            cosine=cosine,
            max_abs_error=error.abs().max().item(),
            samples_ms=durations,
        )
        if trace is not None:
            result["model_output_relative_rms"] = (
                (actual.bfloat16().float() - trace["routed_output"].float()).norm()
                / trace["routed_output"].float().norm()
            ).item()
        results.append(result)
        print(json.dumps(result), flush=True)
    payload = dict(
        backend=args.backend,
        library=args.library,
        threads=args.threads,
        experts=args.experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        top_k=args.top_k,
        clamp=args.limit,
        seed=args.seed,
        route_batches=args.route_batches,
        cache_mode=args.cache_mode,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        reference_device=args.reference_device,
        packed_weight_bytes=args.experts
        * 3
        * args.hidden
        * args.intermediate
        // 32
        * (20 if args.backend == "kt" else 17),
        input_dtype="bfloat16",
        reference_dtype="float32",
        reference_activation="FreeToken DSV4 FP8/128"
        if args.backend == "freetoken"
        else "unquantized",
        torch=torch.__version__,
        platform=platform.platform(),
        checkpoint=str(args.checkpoint) if args.checkpoint else "synthetic",
        layer=args.layer if args.checkpoint else None,
        activation_trace=str(args.activation_trace) if args.activation_trace else None,
        results=results,
        timing="CPU full call; includes activation conversion, dispatch and reduction",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    backend.close()
    stack.close()


if __name__ == "__main__":
    main()
