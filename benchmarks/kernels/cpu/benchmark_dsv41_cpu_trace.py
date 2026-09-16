# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay recorded CPU expert inputs with identical weights and cached routes."""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4 import (
    CPUMoEConfig,
    CPUMXFP4Experts,
)


def read_trace(path):
    data = path.read_bytes()
    if data[:8] != b"DSV41TR1":
        raise ValueError(f"Invalid CPU trace: {path}")
    count = struct.unpack_from("<I", data, 8)[0]
    offset = 12
    frames = []
    for _ in range(count):
        tokens, hidden, topk = struct.unpack_from("<III", data, offset)
        offset += 12
        arrays = []
        for dtype, shape in (
            (np.float32, (tokens, hidden)),
            (np.int32, (tokens, topk)),
            (np.float32, (tokens, topk)),
        ):
            size = shape[0] * shape[1]
            array = np.frombuffer(data, dtype=dtype, count=size, offset=offset).copy()
            arrays.append(torch.from_numpy(array.reshape(shape)))
            offset += size * 4
        frames.append((*arrays, torch.empty_like(arrays[0])))
    if offset != len(data):
        raise ValueError(f"Unexpected trailing trace bytes: {path}")
    return frames


def invoke(backend, batch):
    pointers = [value.data_ptr() for value in batch]
    status = backend._library.dsv41_moe_forward(
        backend._handle, batch[0].shape[0], *pointers
    )
    if status:
        raise RuntimeError(backend._library.dsv41_moe_error().decode())


def load_weights(checkpoint, index, layer, experts, backends):
    with ExitStack() as stack:
        handles = {}
        for expert in range(experts):
            for projection in range(3):
                stem = f"layers.{layer}.ffn.experts.{expert}.w{projection + 1}"
                tensors = []
                for suffix in ("weight", "scale"):
                    key = f"{stem}.{suffix}"
                    shard = index["weight_map"][key]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(
                            safe_open(checkpoint / shard, framework="pt", device="cpu")
                        )
                    tensors.append(handles[shard].get_tensor(key))
                for backend in backends:
                    backend.load_expert(expert, projection, *tensors)
    for backend in backends:
        backend.prepare()
        backend.set_execution_mode("compact")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--layer", type=int)
    selection.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--threads", type=int, nargs="+", default=[22])
    parser.add_argument("--schedules", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    layers = args.layers or [args.layer]
    frames, trace_paths = [], []
    for path in args.traces:
        paths = (
            [
                path / f"language_model.model.layers.{layer}.ffn.experts.bin"
                for layer in layers
            ]
            if args.layers
            else [path]
        )
        trace_paths.extend(paths)
        batches = [read_trace(p) for p in paths]
        if len({len(batch) for batch in batches}) != 1:
            raise ValueError(f"Layer trace lengths differ: {path}")
        count = len(batches[0])
        if args.max_frames:
            count = min(count, args.max_frames)
        for step in range(count):
            if len({batch[step][0].shape[0] for batch in batches}) != 1:
                raise ValueError(f"Layer token counts differ at {path}:{step}")
            frames.extend((layer, batch[step]) for layer, batch in zip(layers, batches))
    if not frames:
        raise ValueError("No CPU trace frames")
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
    assert all(
        x.shape[1] == hidden and ids.shape[1] == topk for _, (x, ids, _, _) in frames
    )
    paths = [args.library] + (
        [args.reference_library] if args.reference_library else []
    )

    def create(path):
        return CPUMXFP4Experts(
            CPUMoEConfig(
                backend="ik", library_path=str(path), num_threads=args.threads[0]
            ),
            experts,
            hidden,
            intermediate,
            topk,
            config["swiglu_limit"],
            16,
        )

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    backends = {}

    def set_schedule(backend, value):
        setter = getattr(backend._library, "dsv41_moe_set_schedule", None)
        if setter:
            setter.argtypes = [ctypes.c_void_p, ctypes.c_int]
            setter.restype = ctypes.c_int
            if setter(backend._handle, value):
                raise RuntimeError(backend._library.dsv41_moe_error().decode())
        elif value:
            raise ValueError("Library does not support schedule selection")

    def schedule(value):
        for backend in backends.values():
            set_schedule(backend, value)

    counts = []
    hist = [0] * 16
    for _, (_, ids, _, _) in frames:
        _, sizes = torch.unique(ids[ids >= 0], return_counts=True)
        counts.append(sizes.numel())
        for size in sizes.tolist():
            hist[size - 1] += 1
    records = []
    payload = {
        "arguments": {
            key: [str(x) for x in value]
            if key == "traces"
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "library_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "trace_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in trace_paths
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "frames": len(frames),
        "layers_in_execution_order": layers,
        "layer_per_frame": [layer for layer, _ in frames],
        "group_size_histogram": hist,
        "timing": "native CPU forward; preallocated FP32 IO; real routes; no PCIe",
        "records": records,
    }
    try:
        expected = [None] * len(frames)
        for layer in layers:
            backend = create(args.library)
            backends[layer] = backend
            reference = (
                create(args.reference_library) if args.reference_library else backend
            )
            load_weights(
                args.checkpoint,
                index,
                layer,
                experts,
                [backend, reference] if reference is not backend else [backend],
            )
            set_schedule(reference, 0)
            for i, (frame_layer, frame) in enumerate(frames):
                if frame_layer == layer:
                    invoke(reference, frame)
                    expected[i] = frame[3].clone()
            if reference is not backend:
                reference.close()
            print(json.dumps({"loaded_layer": layer}), flush=True)
        for threads in args.threads:
            for backend in backends.values():
                backend.set_num_threads(threads)
            for variant in args.schedules:
                schedule(variant)
                for (layer, frame), output in zip(frames, expected):
                    invoke(backends[layer], frame)
                    torch.testing.assert_close(frame[3], output, atol=0, rtol=0)
            for repetition in range(args.rounds):
                order = args.schedules if repetition % 2 == 0 else args.schedules[::-1]
                for variant in order:
                    schedule(variant)
                    for layer, frame in frames:
                        invoke(backends[layer], frame)
                    samples = []
                    for layer, frame in frames:
                        backend = backends[layer]
                        native = backend._library.dsv41_moe_forward
                        pointers = [x.data_ptr() for x in frame]
                        tokens = frame[0].shape[0]
                        start = time.perf_counter_ns()
                        status = native(backend._handle, tokens, *pointers)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                        assert status == 0
                    elapsed_ms = sum(samples)
                    record = {
                        "threads": threads,
                        "schedule": variant,
                        "round": repetition,
                        "total_ms": elapsed_ms,
                        "mean_ms": elapsed_ms / len(frames),
                        "layer_mean_ms": {
                            str(layer): sum(
                                elapsed
                                for elapsed, (frame_layer, _) in zip(samples, frames)
                                if frame_layer == layer
                            )
                            / sum(frame_layer == layer for frame_layer, _ in frames)
                            for layer in layers
                        },
                        "logical_group_GBs": sum(counts)
                        * 3
                        * hidden
                        * intermediate
                        * (17 / 32)
                        / elapsed_ms
                        / 1e6,
                        "samples_ms": samples,
                    }
                    records.append(record)
                    args.output.write_text(json.dumps(payload, indent=2) + "\n")
                    print(
                        json.dumps(
                            {k: v for k, v in record.items() if k != "samples_ms"}
                        ),
                        flush=True,
                    )
        payload["all_outputs_bit_exact"] = True
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        for backend in backends.values():
            backend.close()


if __name__ == "__main__":
    main()
