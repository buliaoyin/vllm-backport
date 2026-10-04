# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare exact packed weights, graph replays, and full replacement wall time."""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.models.deepseek_v4_1.expert_cache import (
    ExpertPackingWorkspace,
    GPUExpertCache,
)


def baseline_class(path):
    name = "vllm.models.deepseek_v4_1._baseline_expert_cache"
    spec = importlib.util.spec_from_file_location(
        name, path / "vllm/models/deepseek_v4_1/expert_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.GPUExpertCache


def bytes_equal(left, right):
    return all(
        torch.equal(
            a.cpu().contiguous().view(torch.uint8),
            b.cpu().contiguous().view(torch.uint8),
        )
        for a, b in zip(left, right)
    )


def capture(cache, hidden, ids, routes):
    cache.finalize(hidden.shape[0])
    with torch.cuda.stream(torch.cuda.current_stream(cache.origin)):
        for _ in range(3):
            output = cache.launch(hidden, ids, routes)
            cache.join(output)
        torch.cuda.current_stream(cache.origin).synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = cache.launch(hidden, ids, routes)
            cache.join(output)
    return graph, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("At least one measured sample is required")
    for device in {*args.devices, args.origin}:
        if not 0 <= device < torch.accelerator.device_count():
            parser.error(f"GPU {device} is not a visible CUDA ordinal")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    original = baseline_class(args.baseline)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {}
    for expert in range(16):
        for projection, name in enumerate(("w1", "w2", "w3")):
            values = []
            for suffix in ("weight", "scale"):
                key = f"layers.20.ffn.experts.{expert}.{name}.{suffix}"
                with safe_open(
                    args.model / index[key], framework="pt", device="cpu"
                ) as file:
                    values.append(file.get_tensor(key).view(torch.uint8).clone())
            weights[expert, projection] = tuple(values)
    results = []
    for device in args.devices:
        torch.accelerator.set_device_index(args.origin)
        torch.cuda.set_stream(torch.cuda.current_stream(args.origin))
        workspace = ExpertPackingWorkspace(device)
        caches = {
            "baseline": original(4, 16, device, 10.0, 6),
            "optimized": GPUExpertCache(
                4, 16, device, 10.0, 6, packing_workspace=workspace
            ),
        }
        for cache in caches.values():
            cache.weights = weights
            cache.weight_shape = tuple(weights[0, 0][0].shape)
            cache.loaded = set(weights)
            cache.select([0, 1, 2, 3])
        sibling = GPUExpertCache(4, 16, device, 10.0, 6, packing_workspace=workspace)
        sibling.weights = weights
        sibling.weight_shape = tuple(weights[0, 0][0].shape)
        sibling.loaded = set(weights)
        sibling.select([12, 13, 14, 15])
        assert sibling._raw_buffers is caches["optimized"]._raw_buffers
        hidden = torch.randn(
            8, cache.weight_shape[1] * 2, device=cache.origin, dtype=torch.bfloat16
        )
        ids = torch.arange(6, device=cache.origin).expand(8, -1).contiguous()
        routes = torch.full((8, 6), 1 / 6, device=cache.origin)
        graphs = {
            name: capture(cache, hidden, ids, routes) for name, cache in caches.items()
        }
        addresses = {
            name: [
                part.data_ptr()
                for part in (
                    *cache.packed,
                    cache.expert_map,
                    cache.membership,
                    *cache.device_io,
                )
            ]
            for name, cache in caches.items()
        }
        for index, selection in enumerate(
            ([3, 1, 0, 2], [3, 5, 0, 2], [7, 8, 0, 2], [0, 1, 2, 3])
        ):
            for name, cache in caches.items():
                cache.select(selection, preserve_slots=False)
                if name == "optimized":
                    sibling.select([6 + index, 13, 14, 15])
                assert addresses[name] == [
                    part.data_ptr()
                    for part in (
                        *cache.packed,
                        cache.expert_map,
                        cache.membership,
                        *cache.device_io,
                    )
                ]
                graphs[name][0].replay()
            torch.cuda.current_stream(args.origin).synchronize()
            assert bytes_equal(caches["baseline"].packed, caches["optimized"].packed)
            assert torch.equal(
                caches["baseline"].expert_map.cpu(),
                caches["optimized"].expert_map.cpu(),
            )
            assert torch.equal(
                caches["baseline"].membership.cpu(),
                caches["optimized"].membership.cpu(),
            )
            assert torch.equal(graphs["baseline"][1], graphs["optimized"][1])
        samples = {}
        for mode in ("repack", "host_prepacked"):
            if mode == "host_prepacked":
                prepared = {}
                for expert in range(16):
                    cache = caches["baseline"]
                    with torch.cuda.stream(cache.stream):
                        packed = cache._pack_expert(expert)
                        host = tuple(
                            torch.empty_like(part, device="cpu", pin_memory=True)
                            for part in packed
                        )
                        for destination, source in zip(host, packed):
                            destination.copy_(source, non_blocking=True)
                    cache.stream.synchronize()
                    prepared[expert] = host
                for cache in caches.values():
                    cache.host_packed = prepared
            for cache in caches.values():
                cache.select([0, 1, 2, 3], preserve_slots=False)
            mode_samples = {name: [] for name in caches}
            for repetition in range(args.samples + 4):
                for name in list(caches)[:: 1 if repetition % 2 == 0 else -1]:
                    cache = caches[name]
                    target = list(cache.selected)
                    target[0] = 8 + repetition % 8
                    torch.cuda.current_stream(args.origin).synchronize()
                    cache.stream.synchronize()
                    start = time.perf_counter()
                    cache.select(target)
                    elapsed = (time.perf_counter() - start) * 1000
                    if repetition >= 4:
                        mode_samples[name].append(elapsed)
            samples[mode] = mode_samples
        row = {
            "origin": args.origin,
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "expert_bytes": sum(
                part[0].numel() * part.element_size() for part in cache.packed
            ),
            "optimized_workspace_bytes": caches["optimized"].packing_workspace_bytes,
            "workspace_reused_by_caches": 2,
            "correctness": (
                "all packed bytes, membership/maps, captured outputs exact; "
                "graph pointers stable across reorder, one/two swaps, restore; "
                "sibling cache reuses raw workspace before graph replay"
            ),
            "samples_ms": samples,
            "median_ms": {
                mode: {
                    name: statistics.median(values) for name, values in groups.items()
                }
                for mode, groups in samples.items()
            },
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del graphs, caches, prepared, cache, sibling, workspace, hidden, ids, routes
        torch.cuda.current_stream(args.origin).synchronize()
    args.output.write_text(
        json.dumps(
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "devices": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
