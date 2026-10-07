# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure complete single-expert replacement, including host staging and copies."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm.models.deepseek_v4_1.expert_cache import GPUExpertCache


class StagedCache(GPUExpertCache):
    def _pack_expert(self, expert):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        self.stream.synchronize()
        if not hasattr(self, "staging"):
            self.staging = []
            self.raw_gpu = []
            for projections, part in (((0, 2), 0), ((1,), 0), ((0, 2), 1), ((1,), 1)):
                pieces = [self.weights[expert, p][part] for p in projections]
                shape = (1, sum(x.shape[0] for x in pieces), pieces[0].shape[1])
                host = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                self.staging.append(host)
                self.raw_gpu.append(
                    torch.empty(shape, dtype=torch.uint8, device=self.device)
                )
        for host, device, (projections, part) in zip(
            self.staging, self.raw_gpu, (((0, 2), 0), ((1,), 0), ((0, 2), 1), ((1,), 1))
        ):
            offset = 0
            for projection in projections:
                source = self.weights[expert, projection][part]
                host[0, offset : offset + source.shape[0]].copy_(source)
                offset += source.shape[0]
            device.copy_(host, non_blocking=True)
        return prepare_moe_mxfp4_layer_for_marlin(
            SimpleNamespace(params_dtype=torch.bfloat16),
            *self.raw_gpu,
            None,
            None,
            inplace=True,
        )[:4]


class PrepackedCache(StagedCache):
    def prepare_host(self):
        self.prepared = {}
        start = time.perf_counter()
        for expert in range(self.num_experts):
            with torch.cuda.stream(self.stream):
                packed = StagedCache._pack_expert(self, expert)
                host = tuple(
                    torch.empty_like(x, device="cpu", pin_memory=True) for x in packed
                )
                for destination, source in zip(host, packed):
                    destination.copy_(source, non_blocking=True)
            self.stream.synchronize()
            self.prepared[expert] = host
        self.prepare_seconds = time.perf_counter() - start

    def _pack_expert(self, expert):
        return self.prepared[expert]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {}
    for expert in range(16):
        for projection, name in enumerate(("w1", "w2", "w3")):
            pair = []
            for suffix in ("weight", "scale"):
                key = f"layers.20.ffn.experts.{expert}.{name}.{suffix}"
                with safe_open(
                    args.model / index[key], framework="pt", device="cpu"
                ) as file:
                    pair.append(file.get_tensor(key).view(torch.uint8).clone())
            weights[expert, projection] = tuple(pair)
    result = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "method": (
            "perf_counter wall time; resident checkpoint weights; staging, packing, "
            "mapping, synchronization and copies included; first allocations warmed "
            "separately; alternating methods"
        ),
        "devices": [],
    }
    for device in args.devices:
        torch.accelerator.set_device_index(3)
        torch.cuda.set_stream(torch.cuda.current_stream(3))
        caches = {
            "baseline": GPUExpertCache(8, 16, device, 10.0, 6),
            "staged": StagedCache(8, 16, device, 10.0, 6),
            "prepacked": PrepackedCache(8, 16, device, 10.0, 6),
        }
        for cache in caches.values():
            cache.weights = weights
            if isinstance(cache, PrepackedCache):
                cache.prepare_host()
            cache.select(list(range(8)))
        for expert in (0, 7, 9):
            packed = []
            for cache in caches.values():
                with torch.cuda.stream(cache.stream):
                    value = tuple(x.cpu().clone() for x in cache._pack_expert(expert))
                cache.stream.synchronize()
                packed.append(value)
            assert all(
                torch.equal(a, b)
                for variant in packed[1:]
                for a, b in zip(packed[0], variant)
            )
        samples = {name: [] for name in caches}
        payload_bytes = sum(
            x[0].numel() * x.element_size() for x in caches["baseline"].packed
        )
        for repetition in range(args.samples):
            target = 8 + repetition % 8
            for name in list(caches)[:: 1 if repetition % 2 == 0 else -1]:
                cache = caches[name]
                selection = list(cache.selected)
                selection[0] = target
                torch.accelerator.synchronize(device)
                start = time.perf_counter()
                cache.select(selection)
                samples[name].append((time.perf_counter() - start) * 1000)
        row = {
            "origin_device": 3,
            "device": device,
            "name": torch.cuda.get_device_name(device),
            "expert_packed_bytes": payload_bytes,
            "correctness": "all four packed tensors exactly equal",
            "prepack_setup_seconds": caches["prepacked"].prepare_seconds,
            "prepacked_host_bytes": 16 * payload_bytes,
            "samples_ms": samples,
            "median_ms": {k: statistics.median(v) for k, v in samples.items()},
        }
        result["devices"].append(row)
        print(json.dumps(row), flush=True)
        del caches, packed
        torch.accelerator.synchronize(device)
        torch.accelerator.empty_cache()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
