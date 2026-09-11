# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded prefill-only caches of reconstructed non-expert projection weights."""

from functools import wraps

import torch

from vllm.model_executor.layers.quantization import exl3
from vllm.utils.torch_utils import direct_register_custom_op

STATE = {"budget_bytes": 0, "devices": {}, "hits": 0, "fallbacks": 0}


def reconstruct(weights):
    cached = torch.empty(
        (weights["suh"].numel(), weights["svh"].numel()),
        device=weights["suh"].device,
        dtype=torch.float16,
    )
    exl3._extension().reconstruct_had_slice(
        cached,
        weights["trellis"],
        weights["suh"],
        weights["svh"],
        weights["trellis"].shape[-1] // 16,
        "mcg" in weights,
        "mul1" in weights,
        0,
    )
    return cached


def linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
    cached: torch.Tensor,
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    if rows < 1024:
        STATE["fallbacks"] += 1
        return exl3._exl3_linear(x, trellis, suh, svh, mcg, mul1)
    STATE["hits"] += 1
    inp = x.reshape(rows, x.shape[-1]).to(torch.float16).contiguous()
    out = torch.empty((rows, cached.shape[1]), dtype=torch.float16, device=x.device)
    exl3._extension().hgemm(inp, cached, out)
    return out.reshape(x.shape[:-1] + (cached.shape[1],)).to(x.dtype)


def fake(x, trellis, suh, svh, mcg, mul1, cached):
    return x.new_empty(x.shape[:-1] + (cached.shape[1],))


direct_register_custom_op("exl3_cached_prefill_linear", linear, fake_impl=fake)


def install(budget_gib):
    STATE["budget_bytes"] = int(budget_gib * 1024**3)
    original_load = exl3.Exl3LinearMethod.process_weights_after_loading
    original_apply = exl3.Exl3LinearMethod.apply

    @wraps(original_load)
    def load(method, layer):
        result = original_load(method, layer)
        method.prefill_weights = {}
        for index, (part, weights) in enumerate(zip(method.parts, method.weights)):
            if not part.quantized or ".layers." not in part.name:
                continue
            device = weights["trellis"].device
            if torch.cuda.get_device_capability(device) != (8, 0):
                continue
            k, n = part.dimensions
            if n > 32768:
                continue
            state = STATE["devices"].setdefault(str(device), {"bytes": 0, "parts": []})
            size = k * n * 2
            if state["bytes"] + size > STATE["budget_bytes"]:
                continue
            method.prefill_weights[index] = reconstruct(weights)
            state["bytes"] += size
            state["parts"].append(part.name)
        return result

    @wraps(original_apply)
    def apply(method, layer, x, bias=None):
        caches = getattr(method, "prefill_weights", {})
        if not caches:
            return original_apply(method, layer, x, bias)
        outputs = []
        for index, (part, weights) in enumerate(zip(method.parts, method.weights)):
            if part.quantized:
                args = (
                    x,
                    weights["trellis"],
                    weights["suh"],
                    weights["svh"],
                    "mcg" in weights,
                    "mul1" in weights,
                )
                if index in caches:
                    y = torch.ops.vllm.exl3_cached_prefill_linear(*args, caches[index])
                else:
                    y = torch.ops.vllm.exl3_linear(*args)
            else:
                y = torch.nn.functional.linear(x, weights["weight"].to(x.dtype))
            outputs.append(y)
        result = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
        return result if bias is None else result + bias

    exl3.Exl3LinearMethod.process_weights_after_loading = load
    exl3.Exl3LinearMethod.apply = apply
