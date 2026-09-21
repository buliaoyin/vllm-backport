# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnose native CPU calls versus CUDA callbacks with the same real weights."""

import argparse
import copy
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4 import CPUMoEConfig
from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--cuda-library", required=True)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--thread-sweep", type=int, nargs="+")
    parser.add_argument("--cpu-executor-sweep", choices=("graph", "compact"), nargs="+")
    parser.add_argument("--modes", choices=("direct", "callback"), nargs="+")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--gpu-cache-experts", type=int, default=0)
    parser.add_argument("--gpu-cache-device", type=int)
    parser.add_argument("--cached-routes-per-token", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(74)
    hf = json.loads((args.checkpoint / "config.json").read_text())
    hf = hf.get("text_config", hf)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )
    module = CPUExpertModule(
        CPUMoEConfig(
            backend="ik",
            library_path=args.library,
            cuda_library_path=args.cuda_library,
            num_threads=args.threads,
            gpu_cache_experts=args.gpu_cache_experts,
            gpu_cache_device=args.gpu_cache_device,
        ),
        config,
    )
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    with ExitStack() as stack:
        handles = {}

        def read(key):
            shard = index["weight_map"][key]
            if shard not in handles:
                handles[shard] = stack.enter_context(
                    safe_open(args.checkpoint / shard, framework="pt", device="cpu")
                )
            return handles[shard].get_tensor(key)

        for expert in range(hf["n_routed_experts"]):
            for projection in range(3):
                stem = f"layers.20.ffn.experts.{expert}.w{projection + 1}"
                weight, scale = read(stem + ".weight"), read(stem + ".scale")
                module.backend.load_expert(expert, projection, weight, scale)
                if module.gpu_cache is not None:
                    module.gpu_cache.load_weight(expert, projection, weight, scale)
    module.finalize()
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.exists():
        print(
            "memory",
            "; ".join(
                line
                for line in rollup.read_text().splitlines()
                if line.startswith(("Rss:", "AnonHugePages:"))
            ),
            flush=True,
        )
    # Independent IO buffers create twenty distinct host nodes without loading
    # another nineteen copies of the weight bank. This is an operator diagnostic.
    modules = [module]
    for _ in range(19):
        clone = copy.copy(module)
        clone.async_host = None
        clone.finalize()
        modules.append(clone)
    batches = []
    for _ in range(args.batches):
        hidden = torch.randn(args.tokens, hf["hidden_size"], dtype=torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(384)[:6] for _ in range(20 * args.tokens)]
        ).reshape(20, args.tokens, 6)
        if args.cached_routes_per_token:
            count = args.cached_routes_per_token
            ids = torch.stack(
                [
                    torch.cat(
                        (
                            torch.randperm(args.gpu_cache_experts)[:count],
                            torch.randperm(384 - args.gpu_cache_experts)[: 6 - count]
                            + args.gpu_cache_experts,
                        )
                    )
                    for _ in range(20 * args.tokens)
                ]
            ).reshape(20, args.tokens, 6)
        routes = torch.rand(20, args.tokens, 6).softmax(-1) * 2.5
        batches.append((hidden, ids, routes))
    device_batches = [tuple(value.cuda() for value in batch) for batch in batches]
    hidden, ids, routes = [value.clone() for value in device_batches[0]]
    context = ForwardContext({}, {}, {}, is_padding=None)
    results = []
    try:
        for sleep_cycles in (0, 500_000):
            for destination, source in zip((hidden, ids, routes), device_batches[0]):
                destination.copy_(source)
            torch.accelerator.synchronize()
            with override_forward_context(context):
                for i, clone in enumerate(modules):
                    clone(hidden, ids[i], routes[i])
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = []
                    for i, clone in enumerate(modules):
                        if sleep_cycles:
                            torch.cuda._sleep(sleep_cycles)
                        outputs.append(clone(hidden, ids[i], routes[i]))
                graph.replay()
                torch.accelerator.synchronize()
                for i, output in enumerate(outputs):
                    expected = module.backend.forward(
                        batches[0][0], batches[0][1][i], batches[0][2][i]
                    ).bfloat16()
                    if module.gpu_cache is None:
                        torch.testing.assert_close(
                            output.cpu(), expected, atol=0, rtol=0
                        )
                    else:
                        assert torch.isfinite(output).all()
                        relative = (output.cpu().float() - expected.float()).norm()
                        relative /= expected.float().norm()
                        assert relative < 0.035, relative.item()
                modes = [
                    (executor, threads, mode)
                    for executor in (args.cpu_executor_sweep or [None])
                    for threads in (args.thread_sweep or [args.threads])
                    for mode in (
                        args.modes or ["direct", "callback", "callback", "direct"]
                    )
                ]
                for executor, threads, mode in modes:
                    if executor is not None:
                        module.backend.set_execution_mode(executor)
                    if args.thread_sweep:
                        module.backend.set_num_threads(threads)
                    graph.replay()
                    torch.accelerator.synchronize()
                    times = []
                    before = module.backend.cuda_stats()
                    for batch, gpu_batch in zip(batches, device_batches):
                        if mode == "callback":
                            for destination, source in zip(
                                (hidden, ids, routes), gpu_batch
                            ):
                                destination.copy_(source)
                            torch.accelerator.synchronize()
                            start = time.perf_counter()
                            graph.replay()
                            torch.accelerator.synchronize()
                            times.append((time.perf_counter() - start) * 1000 / 20)
                        else:
                            for i in range(20):
                                if sleep_cycles:
                                    time.sleep(0.0005)
                                start = time.perf_counter()
                                module.backend.forward(
                                    batch[0], batch[1][i], batch[2][i]
                                )
                                times.append((time.perf_counter() - start) * 1000)
                    after = module.backend.cuda_stats()
                    calls = after["calls"] - before["calls"]
                    record = {
                        "mode": mode,
                        "cpu_executor": executor,
                        "threads": threads,
                        "tokens": args.tokens,
                        "gpu_cache_experts": args.gpu_cache_experts,
                        "cached_routes_per_token": args.cached_routes_per_token,
                        "sleep_cycles": sleep_cycles,
                        "wall_ms_per_call": statistics.mean(times),
                        "native_ms_per_call": (
                            (after["native_seconds"] - before["native_seconds"])
                            * 1000
                            / calls
                        )
                        if calls
                        else None,
                        "callback_threads": after["callback_threads"],
                    }
                    results.append(record)
                    print(json.dumps(record), flush=True)
                del graph
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        torch.accelerator.synchronize()
        module.backend.close()


if __name__ == "__main__":
    main()
