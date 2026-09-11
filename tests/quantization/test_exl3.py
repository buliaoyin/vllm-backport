# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 loading preserves independently rotated matrices and expert identities."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3LinearMethod,
    Exl3Matrix,
    Exl3MoEMethod,
    Exl3Tensor,
    _exl3_linear,
    _extension,
)
from vllm.model_executor.models.utils import WeightsMapper


def _weights(k=128, n=256, bits=4, device="cpu"):
    return {
        "trellis": torch.randint(
            -32768,
            32767,
            (k // 16, n // 16, bits * 16),
            dtype=torch.int16,
            device=device,
        ),
        "suh": torch.randn(k, device=device, dtype=torch.float16) * 0.2,
        "svh": torch.randn(n, device=device, dtype=torch.float16) * 0.2,
        "mul1": torch.tensor(1, dtype=torch.int32, device=device),
    }


def _spec(name, weights):
    return Exl3Matrix(
        name, {key: Exl3Tensor(tuple(t.shape), t.dtype) for key, t in weights.items()}
    )


def test_header_metadata_preserves_per_projection_bits(tmp_path):
    """Average bpw cannot determine allocation for mixed-bit fused projections."""
    a, b = _weights(bits=3), _weights(bits=6)
    save_file(
        {
            **{f"hf.gate_proj.{k}": v for k, v in a.items()},
            **{f"hf.up_proj.{k}": v for k, v in b.items()},
        },
        tmp_path / "model.safetensors",
    )
    config = Exl3Config({"bits": 4.0})
    config.maybe_update_config(str(tmp_path))
    config.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"hf.": "model."}))
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    parts = config.resolve("model.gate_up_proj")
    method, layer = Exl3LinearMethod(parts), nn.Module()
    method.create_weights(layer, 128, [256, 256], 128, 512, torch.bfloat16)
    for shard, weights in enumerate((a, b)):
        for key, tensor in weights.items():
            param = getattr(layer, key)
            param.weight_loader(param, tensor, shard)
    for i, weights in enumerate((a, b)):
        for key, tensor in weights.items():
            offset, size, shape = method.views[i, key]
            torch.testing.assert_close(
                getattr(layer, key)[offset : offset + size].view(shape), tensor
            )
    assert layer.trellis.numel() == a["trellis"].numel() + b["trellis"].numel()


def test_fused_qkv_and_dense_gates_keep_tuple_shard(tmp_path, monkeypatch):
    """A QKV Hadamard must span all three heads; BF16 gate shards stay separate."""
    qkv = _weights(n=384)
    gate = {"weight": torch.randn(32, 128, dtype=torch.float16)}
    method = Exl3LinearMethod([_spec("qkv", qkv), _spec("gate", gate)])
    layer = nn.Module()
    method.create_weights(layer, 128, [128, 128, 128, 32], 128, 416, torch.bfloat16)
    for shard, weights in (((0, 1, 2), qkv), (3, gate)):
        for key, value in weights.items():
            param = getattr(layer, key)
            param.weight_loader(param, value, shard)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.exl3._extension", lambda: None
    )
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(method.weights[0]["trellis"], qkv["trellis"])
    torch.testing.assert_close(method.weights[1]["weight"], gate["weight"].bfloat16())


@pytest.mark.parametrize("include_unpacked", [False, True])
def test_packed_sign_loading_preserves_explicit_scales(include_unpacked, monkeypatch):
    """Legacy signs decode correctly; explicit scales take precedence if present."""
    weights = _weights()
    weights["su"] = torch.tensor([-1] + [0] * 7, dtype=torch.int16)
    weights["sv"] = torch.full((16,), -32768, dtype=torch.int16)
    expected_u, expected_v = weights["suh"], weights["svh"]
    if not include_unpacked:
        del weights["suh"], weights["svh"]
        expected_u = torch.tensor([-1] * 16 + [1] * 112, dtype=torch.float16)
        expected_v = torch.tensor(([1] * 15 + [-1]) * 16, dtype=torch.float16)
    method, layer = Exl3LinearMethod([_spec("proj", weights)]), nn.Module()
    method.create_weights(layer, 128, [256], 128, 256, torch.float16)
    for key, tensor in weights.items():
        param = getattr(layer, key)
        param.weight_loader(param, tensor)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.exl3._extension", lambda: None
    )
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(method.weights[0]["suh"], expected_u)
    torch.testing.assert_close(method.weights[0]["svh"], expected_v)


def test_missing_rotations_fail_before_inference():
    weights = _weights()
    method, layer = Exl3LinearMethod([_spec("proj", weights)]), nn.Module()
    method.create_weights(layer, 128, [256], 128, 256, torch.float16)
    for key in ("trellis", "mul1"):
        param = getattr(layer, key)
        param.weight_loader(param, weights[key])
    with pytest.raises(ValueError, match="Incomplete EXL3"):
        method.process_weights_after_loading(layer)


def _reference(x, weights):
    """Explicit normalized Sylvester rotations, independent of EXL3 GEMM."""
    ext = _extension()
    k, n = weights["suh"].numel(), weights["svh"].numel()
    matrix = torch.empty(k, n, dtype=torch.float16, device=x.device)
    ext.reconstruct(
        matrix, weights["trellis"], weights["trellis"].shape[-1] // 16, False, True
    )
    had = torch.ones(1, 1, device=x.device)
    for _ in range(7):
        had = torch.cat((torch.cat((had, had), 1), torch.cat((had, -had), 1)), 0)
    had = (had / 128**0.5).half()
    rotated = ((x.half() * weights["suh"]).reshape(-1, k // 128, 128) @ had).reshape(
        -1, k
    )
    result = rotated @ matrix
    return (
        (result.reshape(-1, n // 128, 128) @ had).reshape(-1, n) * weights["svh"]
    ).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("rows,int8_mode", [(1, 0), (1, 2), (145, 0), (1024, 0)])
def test_linear_decode_and_prefill_match_rotated_reference(
    bits, rows, int8_mode, monkeypatch
):
    monkeypatch.setenv("EXL3_INT8_GEMV", str(int8_mode))
    pytest.importorskip("exllamav3_ext")
    torch.manual_seed(1)
    weights = _weights(bits=bits, device="cuda")
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    actual = _exl3_linear(
        x, weights["trellis"], weights["suh"], weights["svh"], False, True
    )
    expected = _reference(x, weights)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    # Plain INT8 GEMV intentionally rounds activations; retain the FP16 bound.
    assert relative < (0.015 if int8_mode == 2 else 0.006)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,capacity",
    [(1, 128), (3, 128), (8, 128), (129, 128), (129, 512), (512, 512), (513, 512)],
)
def test_moe_routing_and_chunk_boundaries(rows, capacity, monkeypatch):
    """A hot expert exceeding workspace capacity must survive chunking and replay."""
    monkeypatch.setenv("VLLM_EXL3_MOE_MAX_TOKENS", str(capacity))
    pytest.importorskip("exllamav3_ext")
    torch.manual_seed(3)
    config = Exl3Config({})
    weights = {}
    num_experts = 32 if rows == 513 else 3
    for expert in range(num_experts):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            weights[expert, kind] = _weights(k=128, n=128, device="cuda")
            name = f"experts.{expert}.{kind}"
            config.matrices[name] = _spec(name, weights[expert, kind])
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1),
        activation="silu",
        swiglu_limit=10.0,
    )
    method, layer = Exl3MoEMethod(config, moe, "experts"), nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, num_experts, 128, 128, torch.float16)
    for expert in range(num_experts):
        for shard, kind in (
            ("w1", "gate_proj"),
            ("w3", "up_proj"),
            ("w2", "down_proj"),
        ):
            for key, tensor in weights[expert, kind].items():
                param = getattr(layer, ("w2_" if shard == "w2" else "w13_") + key)
                param.weight_loader(param, tensor, shard_id=shard, expert_id=expert)
    method.process_weights_after_loading(layer)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    ids = torch.stack(
        (
            torch.full((rows,), num_experts - 1, device="cuda", dtype=torch.long),
            torch.arange(rows, device="cuda") % 2,
        ),
        -1,
    )
    routing = torch.full((rows, 2), 0.5, device="cuda")
    actual = method.apply(layer, x, routing, ids)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(num_experts):
        gate = _reference(x, weights[expert, "gate_proj"]).clamp(max=10)
        up = _reference(x, weights[expert, "up_proj"]).clamp(-10, 10)
        activated = (torch.nn.functional.silu(gate.float()) * up).half()
        result = _reference(activated, weights[expert, "down_proj"])
        expected += result.float() * ((ids == expert).float() * routing).sum(
            -1, keepdim=True
        )
    relative = (actual.float() - expected).norm() / expected.norm()
    assert relative < 0.01
    if rows in (3, 513):
        for _ in range(3):
            method.apply(layer, x, routing, ids)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, x, routing, ids)
        x.mul_(2)
        ids[:, 1].copy_(1 - ids[:, 1])
        graph.replay()
        torch.testing.assert_close(captured, method.apply(layer, x, routing, ids))


def test_shared_experts_serialize_device_wide_exl3_workspace(monkeypatch):
    """Shared and routed EXL3 GEMMs must never compete for device-wide locks."""
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )

    def unexpected_stream():
        raise AssertionError("A single-stream backend must not allocate an aux stream")

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.shared_experts.aux_stream",
        unexpected_stream,
    )
    moe = SimpleNamespace()
    method = Exl3MoEMethod(Exl3Config({}), moe, "experts")
    shared = SharedExperts(
        nn.Identity(),
        moe,
        False,
        lambda: False,
        disable_overlap=not method.supports_multi_stream,
    )
    x = torch.ones(2, 128)
    shared(x, SharedExpertsOrder.NO_OVERLAP)
    torch.testing.assert_close(shared.output, x)
