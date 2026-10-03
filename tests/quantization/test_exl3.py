# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 loading preserves independently rotated matrices and expert identities."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.config import CUDAGraphMode
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


def test_ngram_sidecar_metadata_is_separate_from_linear_matrices(tmp_path):
    """Unindexed packed embedding rows must not be validated as GEMM trellises."""
    weights = {f"head.{k}": v for k, v in _weights().items()}
    save_file(weights, tmp_path / "model.safetensors")
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    table = {
        f"{prefix}.trellis": torch.zeros((4, 61), dtype=torch.int16),
        f"{prefix}.head_bias": torch.zeros((2, 160), dtype=torch.float16),
        f"{prefix}.head_offsets": torch.tensor([0, 2]),
        f"{prefix}.head_vocab_sizes": torch.tensor([2, 2]),
        f"{prefix}.layer_multipliers": torch.tensor([3, 5, 7]),
    }
    save_file(table, tmp_path / "ngram_embedding.safetensors")
    vision_prefix = "model.visual.blocks.0.mlp.linear_fc1"
    save_file(
        {f"{vision_prefix}.{k}": v for k, v in _weights(bits=6).items()},
        tmp_path / "vision_k6.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    config = Exl3Config({})
    config.maybe_update_config(str(tmp_path))
    config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "language_model.model."}
        )
    )
    mapped = prefix.replace("model.language_model.", "language_model.model.")
    assert set(config.matrices) == {"head", vision_prefix}
    assert config.vision_safetensors_files == ("vision_k6.safetensors",)
    assert config.ngram_tables[mapped]["trellis"] == Exl3Tensor((4, 61), torch.int16)


def test_routed_experts_loads_individual_exl3_trellises():
    """A 3D trellis is one expert's matrix, not a fused expert batch."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    config = Exl3Config({})
    tensors = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            values = _weights(k=128, n=128)
            tensors[expert, projection] = values
            name = f"experts.{expert}.{projection}"
            config.matrices[name] = _spec(name, values)
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1), activation="silu"
    )
    layer = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.layer_name = "experts"
    layer.quant_method = Exl3MoEMethod(config, moe, "experts")
    layer.quant_method.create_weights(layer, 2, 128, 128, torch.bfloat16)
    mapping = [
        (f"experts.{base}_", f"experts.{expert}.{projection}.", expert, shard)
        for expert in range(2)
        for base, projection, shard in (
            ("w13", "gate_proj", "w1"),
            ("w13", "up_proj", "w3"),
            ("w2", "down_proj", "w2"),
        )
    ]
    layer.get_expert_mapping = lambda **kwargs: mapping
    weights = (
        (f"{expert}.{projection}.{component}", tensor)
        for (expert, projection), values in tensors.items()
        for component, tensor in values.items()
    )
    loaded = set(layer.load_weights(weights))
    assert "w2_trellis" in loaded
    assert len(layer.quant_method.loaded) == 24


def test_qwen4_mtp_resolves_offset_layers_without_duplicating_scratch(monkeypatch):
    """The HC drafter uses layer 48's names while preserving target storage."""
    from vllm.models.qwen4_exp.nvidia import mtp

    target, draft = Exl3Config({}), Exl3Config({})
    layer = "mtp.layers.0.mlp.experts.0.gate_proj"
    draft.matrices[layer] = _spec(layer, _weights())
    target.workspaces["shared"] = [torch.empty(1)]
    draft_model = SimpleNamespace(
        hf_text_config=SimpleNamespace(mtp_num_hidden_layers=1)
    )

    @dataclass
    class Config:
        model_config: object
        quant_config: Exl3Config
        speculative_config: object

    config = Config(
        model_config=object(),
        quant_config=target,
        speculative_config=SimpleNamespace(draft_model_config=draft_model),
    )
    monkeypatch.setattr(mtp, "get_draft_quant_config", lambda _: draft)
    result = mtp._make_draft_vllm_config(config, 48)
    assert result.quant_config.resolve(layer.replace("layers.0.", "layers.48."))
    assert result.quant_config.workspaces is target.workspaces
    assert result.model_config is draft_model
    assert config.quant_config is target
    assert not target.matrices


def test_draft_loading_keeps_target_and_mtp_metadata_independent(tmp_path, monkeypatch):
    """A target mapper must not drop the drafter's quantized head or MTP weights."""
    from vllm.config import CompilationMode, LoadConfig
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    weights = _weights()
    save_file(
        {
            f"{name}.{k}": v.clone()
            for name in ("lm_head", "mtp.fc")
            for k, v in weights.items()
        },
        tmp_path / "model.safetensors",
    )
    target = Exl3Config({})
    target.maybe_update_config(str(tmp_path))
    target.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={"lm_head.": "language_model.lm_head.", "mtp.": None}
        )
    )
    draft_config = SimpleNamespace(quantization="exl3", model=str(tmp_path))
    spec = SimpleNamespace(
        draft_model_config=draft_config,
        moe_backend=None,
        kv_cache_dtype=None,
        attention_backend=None,
        enforce_eager=False,
    )

    @dataclass
    class Config:
        quant_config: Exl3Config
        speculative_config: object
        load_config: object
        compilation_config: object

    config = Config(
        quant_config=target,
        speculative_config=spec,
        load_config=LoadConfig(),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    class Loaded(Exception):
        pass

    def fresh_quant(model_config, load_config):
        draft = Exl3Config({})
        draft.maybe_update_config(model_config.model)
        return draft

    def load(*, vllm_config, model_config):
        quant = vllm_config.quant_config
        configure_quant_config(quant, Qwen3_5MTP)
        assert quant.resolve("lm_head")[0].quantized
        assert quant.resolve("mtp.fc")[0].quantized
        assert quant.workspaces is target.workspaces
        assert quant.prefill_workspaces is target.prefill_workspaces
        assert quant.decode_workspaces is target.decode_workspaces
        assert quant.m32_locks is target.m32_locks
        assert quant.prefill_disabled is target.prefill_disabled
        raise Loaded

    monkeypatch.setattr(utils.VllmConfig, "get_quantization_config", fresh_quant)
    monkeypatch.setattr(utils, "get_model", load)
    with pytest.raises(Loaded):
        utils.load_eagle_model(nn.Module(), config)
    assert config.quant_config is target
    assert set(target.matrices) == {"language_model.lm_head"}


@pytest.mark.parametrize("indexed_mtp", [False, True])
@pytest.mark.parametrize("load_mtp", [False, True])
def test_mtp_sidecar_loads_once_without_loading_unindexed_files(
    tmp_path, indexed_mtp, load_mtp
):
    """MTP metadata and weights survive a target-only index; other files do not."""
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    target = {f"model.layers.0.proj.{k}": v for k, v in _weights().items()}
    mtp = {f"model.layers.1.eh_proj.{k}": v for k, v in _weights().items()}
    save_file(target, tmp_path / "model.safetensors")
    save_file(mtp, tmp_path / "mtp.safetensors")
    save_file({"unused.weight": torch.ones(4, 4)}, tmp_path / "unused.safetensors")
    index = dict.fromkeys(target, "model.safetensors")
    if indexed_mtp:
        index.update(dict.fromkeys(mtp, "mtp.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )

    config = Exl3Config({})
    config.maybe_update_config(str(tmp_path))
    assert set(config.matrices) == {"model.layers.0.proj", "model.layers.1.eh_proj"}
    assert config.resolve("model.layers.1.eh_proj")[0].quantized

    model = nn.Module()
    if load_mtp:
        model.extra_safetensors_files = ("mtp.safetensors",)
    model_config = SimpleNamespace(
        model=str(tmp_path), revision=None, quantization="exl3"
    )
    loader = DefaultModelLoader(LoadConfig(use_tqdm_on_load=False))
    weights = list(loader.get_all_weights(model_config, model))
    expected = target | mtp if indexed_mtp or load_mtp else target
    assert len(weights) == len(expected)
    assert {name for name, _ in weights} == set(expected)
    for name, value in weights:
        torch.testing.assert_close(value, expected[name])


@pytest.mark.parametrize("model_type", ["glm5next", "qwen4_exp"])
@pytest.mark.parametrize("layout", ["separate", "separate_with_fallback", "fused"])
def test_vision_loads_exl3_qkv_with_bias(layout, model_type):
    """Visual QKV keeps each rotation and bias, including with obsolete weights."""
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from vllm.models.glm5next.nvidia.multimodal import Glm5NextVisionTransformer
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpVisionTransformer

    cls = (
        Glm5NextVisionTransformer
        if model_type == "glm5next"
        else Qwen4ExpVisionTransformer
    )
    qkv_name = "qkv_proj" if model_type == "glm5next" else "qkv"

    projections = (
        {"qkv": _weights(n=384, bits=4)}
        if layout == "fused"
        else {p: _weights(n=128, bits=b) for p, b in zip("qkv", (3, 4, 6))}
    )
    config = Exl3Config({})
    config.packed_modules_mapping = {qkv_name: ["q_proj", "k_proj", "v_proj"]}
    for proj, weights in projections.items():
        projection_name = qkv_name if proj == "qkv" else f"{proj}_proj"
        name = f"visual.blocks.0.attn.{projection_name}"
        config.matrices[name] = _spec(name, weights)
    if layout == "separate_with_fallback":
        name = "visual.blocks.0.attn.qkv"
        config.matrices[name] = _spec(name, {"weight": torch.zeros(384, 128)})
    qkv = QKVParallelLinear(
        128,
        64,
        2,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix=f"visual.blocks.0.attn.{qkv_name}",
        disable_tp=True,
    )
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].attn = nn.Module()
    model.blocks[0].attn.qkv = qkv
    tensors = {}
    biases = []
    for i, (proj, weights) in enumerate(projections.items()):
        prefix = "blocks.0.attn.qkv" if proj == "qkv" else f"blocks.0.attn.{proj}_proj"
        tensors.update({f"{prefix}.{k}": v for k, v in weights.items()})
        bias = torch.full((384 if proj == "qkv" else 128,), i + 1, dtype=torch.bfloat16)
        tensors[f"{prefix}.bias"] = bias
        biases.append(bias)
    if layout == "separate_with_fallback":
        tensors["blocks.0.attn.qkv.weight"] = torch.zeros(384, 128)
        tensors["blocks.0.attn.qkv.bias"] = torch.zeros(384)
    loaded = model.load_weights(tensors.items())
    assert loaded == set(dict(model.named_parameters()))
    torch.testing.assert_close(qkv.bias, torch.cat(biases))
    for i, weights in enumerate(projections.values()):
        for key, tensor in weights.items():
            offset, size, shape = qkv.quant_method.views[i, key]
            actual = getattr(qkv, key)[offset : offset + size].view(shape)
            torch.testing.assert_close(actual, tensor)


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
@pytest.mark.parametrize(
    "rows,int8_mode", [(1, 0), (1, 1), (1, 2), (2, 2), (145, 0), (1024, 0)]
)
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

    method = Exl3LinearMethod([_spec("proj", weights)])
    method.weights = [weights]
    for bias_dtype in (torch.float16, torch.float32):
        bias = torch.randn(actual.shape[-1], device=x.device, dtype=bias_dtype)
        original_bias = bias.clone()
        torch.testing.assert_close(method.apply(None, x, bias), actual + bias)
        torch.testing.assert_close(bias, original_bias)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,capacity",
    [(1, 128), (3, 128), (8, 128), (129, 128), (129, 512), (512, 512), (513, 512)],
)
def test_moe_routing_and_chunk_boundaries(rows, capacity, monkeypatch):
    """A hot expert exceeding workspace capacity must survive chunking and replay."""
    _check_moe_routing(rows, capacity, 128, monkeypatch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [3, 129, 513])
def test_native_moe_skips_graph_padding_routes(rows, monkeypatch):
    """Padding must not count as an expert or survive graph replay as a route."""
    _check_moe_routing(
        rows,
        128,
        128,
        monkeypatch,
        intermediate_dim=640,
        topk=10,
        num_experts=32,
        check_graph=True,
        padded=True,
    )


def _check_moe_routing(
    rows,
    capacity,
    hidden_dim,
    monkeypatch,
    relative_limit=0.01,
    dtype=torch.float16,
    m_tile=16,
    decode="native",
    intermediate_dim=None,
    topk=2,
    num_experts=None,
    cold_routes=False,
    check_graph=False,
    padded=False,
):
    if capacity is None:
        monkeypatch.delenv("VLLM_EXL3_MOE_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("VLLM_EXL3_MOE_MAX_TOKENS", str(capacity))
    if m_tile is None:
        monkeypatch.delenv("VLLM_EXL3_MOE_M_TILE", raising=False)
    else:
        monkeypatch.setenv("VLLM_EXL3_MOE_M_TILE", str(m_tile))
    if decode is None:
        monkeypatch.delenv("VLLM_EXL3_MOE_DECODE", raising=False)
    else:
        monkeypatch.setenv("VLLM_EXL3_MOE_DECODE", decode)
    intermediate_dim = intermediate_dim or hidden_dim
    pytest.importorskip("exllamav3_ext")
    torch.manual_seed(3)
    config = Exl3Config({})
    weights = {}
    if num_experts is None:
        num_experts = 32 if rows == 513 else max(3, topk)
    for expert in range(num_experts):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            k, n = (
                (intermediate_dim, hidden_dim)
                if kind == "down_proj"
                else (hidden_dim, intermediate_dim)
            )
            weights[expert, kind] = _weights(k=k, n=n, device="cuda")
            name = f"experts.{expert}.{kind}"
            config.matrices[name] = _spec(name, weights[expert, kind])
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1),
        activation="silu",
        swiglu_limit=10.0,
        experts_per_token=topk,
    )
    method, layer = Exl3MoEMethod(config, moe, "experts"), nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, num_experts, hidden_dim, intermediate_dim, dtype)
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
    x = torch.randn(rows, hidden_dim, device="cuda", dtype=dtype)
    ids = torch.stack(
        (
            torch.full((rows,), num_experts - 1, device="cuda", dtype=torch.long),
            torch.arange(rows, device="cuda") % (num_experts - 1 if cold_routes else 2),
        ),
        -1,
    )
    if topk != 2:
        ids = (
            torch.arange(topk, device="cuda")[None, :]
            + torch.arange(rows, device="cuda")[:, None]
        ) % num_experts
    if padded:
        ids[::2] = -1
    routing = torch.full((rows, topk), 1 / topk, device="cuda")
    if dtype == torch.bfloat16 and rows == 3:
        x[0].zero_()
        routing[1].zero_()
        routing[2] = torch.tensor(
            [0.2] + [0.8 / (topk - 1)] * (topk - 1), device="cuda"
        )
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
    assert relative < relative_limit
    if padded:
        torch.testing.assert_close(actual[::2], torch.zeros_like(actual[::2]))
    if check_graph or rows in (3, 513, 4097):
        for _ in range(3):
            method.apply(layer, x, routing, ids)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, x, routing, ids)
        x.mul_(2)
        if padded:
            ids.copy_(
                (
                    torch.arange(topk, device="cuda")[None, :]
                    + torch.arange(rows, device="cuda")[:, None]
                    + 1
                )
                % num_experts
            )
            ids[1::3] = -1
        else:
            ids.copy_((ids + 1) % num_experts)
        if dtype == torch.bfloat16:
            routing.copy_(routing.flip(1))
        graph.replay()
        # Exercise the same scratch at another batch size before replaying.
        if method.decode_workspace is not None:
            method.apply(layer, x[:1], routing[:1], ids[:1])
            graph.replay()
        torch.testing.assert_close(captured, method.apply(layer, x, routing, ids))
    return method


@pytest.mark.parametrize("rows", [9, 33, 513, 2049])
def test_default_m32_routes_tails_and_workspace_overflow(rows, monkeypatch):
    """Default M32 must compute routes and chunk overflow without upstream MoE."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Default M32 requires SM80")
    pytest.importorskip("vllm._exl3_C")

    def reject_upstream(*args):
        pytest.fail("Supported SM80 experts should use the default M32 kernel")

    monkeypatch.setattr(_extension(), "exl3_moe", reject_upstream)
    _check_moe_routing(rows, None, 256, monkeypatch, m_tile=None)


@pytest.mark.parametrize(
    "rows,topk", [(1, 2), (2, 2), (3, 2), (4, 2), (8, 2), (3, 8), (8, 8)]
)
@pytest.mark.parametrize("policy", [None, "native", "plain", "residual"])
def test_expert_decode_defaults_overrides_and_shared_scratch(
    rows, topk, policy, monkeypatch
):
    """Guard default dispatch, disable/override and gate/down scratch reuse."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((8, 0), (12, 0)):
        pytest.skip("Native expert INT8 requires SM80 or SM120")
    pytest.importorskip("vllm._exl3_C")
    from vllm.model_executor.layers.quantization import exl3

    def unexpected(*args, **kwargs):
        pytest.fail("Unexpected expert decode fallback or opt-in")

    if policy == "native":
        monkeypatch.setattr(exl3, "_exl3_moe_decode_int8", unexpected)
    else:
        monkeypatch.setattr(exl3, "_exl3_moe_decode", unexpected)
    residual = policy == "residual" or (policy is None and capability == (12, 0))
    method = _check_moe_routing(
        rows,
        2048,
        256,
        monkeypatch,
        relative_limit=0.01 if residual else 0.02,
        dtype=torch.bfloat16,
        decode=policy,
        intermediate_dim=512,
        topk=topk,
    )
    assert method.decode_mode == (0 if policy == "native" else 2 if residual else 1)


@pytest.mark.parametrize("dtype,hidden", [(torch.float16, 256), (torch.bfloat16, 128)])
def test_expert_decode_default_falls_back_for_unsupported_inputs(
    dtype, hidden, monkeypatch
):
    """The default retains the upstream path for unsupported dtype or alignment."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.model_executor.layers.quantization import exl3

    def unexpected(*args):
        pytest.fail("Unsupported input should use upstream expert decode")

    monkeypatch.setattr(exl3, "_exl3_moe_decode_int8", unexpected)
    method = _check_moe_routing(3, 128, hidden, monkeypatch, dtype=dtype, decode=None)
    assert method.decode_mode == 0


@pytest.mark.parametrize("rows", [9, 16, 17, 31, 32, 33, 63, 64, 65, 129, 513, 1025])
def test_experimental_m32_tails_and_replay(rows, monkeypatch):
    """Check short row tiles, hot experts and replay against rotated dense weights."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_ROWS_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional SM80 row kernel build")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Row kernel is SM80-only")
    from benchmarks.kernels.exl3_m32.launcher import Launcher

    extension = _extension()
    monkeypatch.setattr(extension, "exl3_moe", Launcher(extension, library))
    _check_moe_routing(rows, 1024, 256, monkeypatch)


@pytest.mark.parametrize("rows", [9, 16, 17, 31, 32, 33, 65, 513, 1025])
@pytest.mark.parametrize("variant", ["int8", "int8_residual"])
def test_experimental_int8_tails_and_replay(rows, variant, monkeypatch):
    """Bound added activation/codebook error against independently rotated weights."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_INT8_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional SM80 INT8 kernel build")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("INT8 experiment is SM80-only")
    from benchmarks.kernels.exl3_int8.launcher import Launcher

    extension = _extension()
    monkeypatch.setattr(extension, "exl3_moe", Launcher(extension, library, variant))
    _check_moe_routing(rows, 1024, 256, monkeypatch, relative_limit=0.02)


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
        is_multistream_safe=lambda: True,
        disable_overlap=not method.supports_multi_stream,
    )
    x = torch.ones(2, 128)
    shared(x, SharedExpertsOrder.NO_OVERLAP)
    torch.testing.assert_close(shared.output, x)


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("residual", [False, True])
def test_experimental_expert_int8_decode(rows, residual, monkeypatch):
    """Check expert identity and graph replay against independently rotated weights."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_DECODE_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional expert INT8 decode build")
    from benchmarks.kernels.exl3_moe_decode.launcher import Decode
    from vllm.model_executor.layers.quantization import exl3

    monkeypatch.setattr(
        exl3, "_exl3_moe_decode", Decode(_extension(), library, residual)
    )
    _check_moe_routing(
        rows,
        1024,
        256,
        monkeypatch,
        relative_limit=0.01 if residual else 0.02,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("rows", [9, 33, 513])
@pytest.mark.parametrize("variant", ["nobar_m32_k32_n256", "half_m32_k32_n256"])
def test_experimental_moe_pipeline(rows, variant, monkeypatch):
    """Bound pipeline arithmetic and synchronize reused tiles on graph replay."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_PIPELINE_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional pipeline ablation build")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Pipeline experiment is SM80-only")
    from benchmarks.kernels.exl3_m32.launcher import Launcher

    class Pipeline(Launcher):
        variants = ("nobar_m32_k32_n256", "half_m32_k32_n256")

    extension = _extension()
    monkeypatch.setattr(extension, "exl3_moe", Pipeline(extension, library, variant))
    _check_moe_routing(rows, 1024, 256, monkeypatch, relative_limit=0.02)


@pytest.mark.parametrize("rows", [9, 33, 65, 513])
@pytest.mark.parametrize(
    "variant",
    [
        "control",
        "compact",
        "loadfirst",
        "loadfirst_s4",
        "singlefrag",
        "k16_resident",
        "k16_singlefrag",
        "k16_n128",
        "m64_k16_n128",
        "lookup",
        "lookup_k16",
        "wide_m32",
        "wide_m64",
        "wide_m64_single",
        "fixed_m32",
        "fixed_k16",
        "fixed_wide_m32",
        "fixed_wide_m64",
        "cached_m32_k64",
        "cached_m32_k128",
        "cached_m64_k64",
        "cached_m64_k128",
        "cached_m32_n256_k64",
        "cached_m32_n256_k128",
        "cached_m64_n256_k64",
        "cached_m64_n256_k128",
        "adaptive64",
        "adaptive96",
        "adaptive128",
        "adaptive192",
    ],
)
def test_experimental_prefill_residency_and_codebook(rows, variant, monkeypatch):
    """Guard exact lookup, changed K reduction and resident-group scratch reuse."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_PREFILL_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional prefill experiment library")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Prefill residency experiments require SM80")
    pytest.importorskip("vllm._exl3_C")
    from benchmarks.kernels.exl3_prefill.launcher import Launcher

    monkeypatch.setattr(torch.ops._exl3_C, "moe_m32", Launcher(library, variant))
    hidden, intermediate = (4096, 2048) if variant.startswith("fixed_") else (256, 512)
    _check_moe_routing(
        rows, 1024, hidden, monkeypatch, m_tile=None, intermediate_dim=intermediate
    )


@pytest.mark.parametrize("rows", [9, 1024, 2048])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_prefill_projection_cache_preserves_quantized_outputs(rows, bits):
    """Cached rotated weights must match reconstruction and survive graph replay."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("exllamav3_ext")
    from benchmarks.kernels.exl3_prefill.linear_cache import linear, reconstruct

    torch.manual_seed(3)
    w = _weights(k=256, n=512, bits=bits, device="cuda")
    cached = reconstruct(w)
    x = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
    args = (w["trellis"], w["suh"], w["svh"], False, True)
    torch.testing.assert_close(linear(x, *args, cached), _exl3_linear(x, *args))
    for _ in range(3):
        linear(x, *args, cached)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = linear(x, *args, cached)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(actual, _exl3_linear(x, *args))


@pytest.mark.parametrize("rows", [9, 65, 513, 2049])
def test_experimental_batched_fp16_prefill(rows, monkeypatch):
    """Batched reconstruction and grouped GEMM must preserve routes and rotations."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_FP16_PREFILL_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional batched FP16 prefill library")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("The prototype is built for SM80")
    from benchmarks.kernels.exl3_prefill_fp16.backend import Backend
    from vllm.model_executor.layers.quantization import exl3

    monkeypatch.setattr(exl3, "_exl3_moe_fused", Backend(library))
    _check_moe_routing(rows, 1024, 256, monkeypatch, m_tile=None, intermediate_dim=512)


@pytest.mark.parametrize("rows", [9, 65, 513, 2049])
@pytest.mark.parametrize("hidden", [256, 768])
def test_experimental_batched_int8_prefill(rows, hidden, monkeypatch):
    """Bound expanded INT8 weight/activation error against rotated references."""
    import os

    library = os.environ.get("VLLM_EXL3_TEST_BATCHED_INT8_LIBRARY")
    if not library or not torch.cuda.is_available():
        pytest.skip("Requires the optional batched INT8 prefill library")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("The prototype is built for SM80")
    from benchmarks.kernels.exl3_prefill_int8.backend import Backend
    from vllm.model_executor.layers.quantization import exl3

    monkeypatch.setattr(exl3, "_exl3_moe_fused", Backend(library))
    _check_moe_routing(
        rows,
        1024,
        hidden,
        monkeypatch,
        m_tile=None,
        intermediate_dim=512,
        relative_limit=0.02,
    )


@pytest.mark.parametrize(
    "group,expected_group,expected_mib",
    [(None, 48, 960.375), (32, 32, 960.375), (64, 64, 1088.375), (0, 288, 2880.375)],
)
def test_int8_workspace_glm_memory_budget_and_group_override(
    group, expected_group, expected_mib, monkeypatch
):
    """The tuned default fits the memory budget without ignoring overrides."""
    from vllm import envs
    from vllm.model_executor.layers.quantization.utils.exl3_prefill import (
        allocate_workspace,
    )

    if group is None:
        monkeypatch.delenv("VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP", raising=False)
    else:
        monkeypatch.setenv("VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP", str(group))
    workspace = allocate_workspace(
        "meta", 288, 4096, 2048, 6144, 8, envs.VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP
    )
    assert workspace[0].numel() == expected_group * 4096 * 2048
    assert workspace[-1].numel() == int(expected_mib * 2**20)
    # Other capacities retain the prior group width unless explicitly overridden.
    other = allocate_workspace("meta", 288, 4096, 2048, 4096, 8, None)
    assert other[0].numel() == 64 * 4096 * 2048


@pytest.mark.parametrize("hidden,intermediate", [(256, 512), (512, 256), (2048, 256)])
def test_int8_workspace_preserves_live_values_across_projection_phases(
    hidden, intermediate
):
    """Aliasing must preserve live GEMM operands for both expansion directions."""
    from vllm.model_executor.layers.quantization.utils.exl3_prefill import (
        allocate_workspace,
    )

    weight, stage, gate, up, q, scales, sums, gather, arena = allocate_workspace(
        "cpu", 7, hidden, intermediate, 513, 2, 1
    )
    old_bytes = (
        weight.numel()
        + 2 * 513 * 2 * (hidden + 2 * intermediate)
        + q.numel()
        + 8 * 513 * 2
    )
    assert arena.numel() <= old_bytes
    gather.fill_(1)
    q.fill_(2)
    scales.fill_(3)
    sums.fill_(4)
    assert torch.all(gather == 1)
    gate.fill_(5)
    weight.fill_(6)
    assert torch.all(q == 2)
    gather.fill_(7)
    assert torch.all(gate == 5)
    q.fill_(8)
    assert torch.all(gather == 7)
    up.fill_(9)
    assert torch.all(gate == 5) and torch.all(q == 8)
    gate.fill_(10)
    assert torch.all(up == 9)
    q.fill_(11)
    assert torch.all(gate == 10)
    weight.fill_(12)
    stage.fill_(13)
    assert torch.all(q == 11) and torch.all(weight == 12)
    assert torch.all(scales == 3) and torch.all(sums == 4)


@pytest.mark.parametrize("int8_capacity,num_experts", [(128, 7), (512, 3)])
@pytest.mark.parametrize("native_capacity", [32, 2048])
def test_native_int8_workspace_switching_and_graph_replay(
    int8_capacity, num_experts, native_capacity, monkeypatch
):
    """Shared/fallback pools survive native tails and replay after being overwritten."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Native EXL3 INT8 prefill targets SM80")
    from vllm.model_executor.layers.quantization import exl3
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "int8")
    monkeypatch.setenv("VLLM_EXL3_MOE_INT8_MIN_TOKENS", "64")
    monkeypatch.delenv("VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP", raising=False)
    monkeypatch.setattr(exl3_prefill, "INT8_MAX_ROWS", int8_capacity)
    original = Exl3MoEMethod.apply
    checked = False

    def check(method, layer, x, routing, ids):
        nonlocal checked
        if not checked:
            checked = True
            arena = method.prefill_workspace[-1]
            shared = all(
                t.untyped_storage().data_ptr() == arena.data_ptr()
                for t in method.workspace
            )
            assert shared == (native_capacity == 32)
            native = [torch.empty_like(t) for t in method.workspace]
            prefill = [torch.empty_like(t) for t in method.prefill_workspace]
            for _ in range(3):
                original(method, layer, x, routing, ids)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = original(method, layer, x, routing, ids)
            for rows in (33, int8_capacity, 1, 7):
                original(method, layer, x[:rows], routing[:rows], ids[:rows])
                arena.fill_(-1)
                graph.replay()
                expected = exl3._exl3_moe(
                    x,
                    routing,
                    ids,
                    method.ptrs,
                    native,
                    prefill,
                    method.bits,
                    method.flags,
                    10.0,
                    method.m32_locks,
                    method.decode_workspace,
                    method.decode_mode,
                    64,
                )
                relative = (captured.float() - expected.float()).norm()
                relative /= expected.float().norm()
                assert torch.isfinite(captured).all() and relative < 2e-4
        return original(method, layer, x, routing, ids)

    monkeypatch.setattr(Exl3MoEMethod, "apply", check)
    _check_moe_routing(
        int8_capacity + 1,
        native_capacity,
        512,
        monkeypatch,
        m_tile=32,
        intermediate_dim=256,
        num_experts=num_experts,
        dtype=torch.bfloat16,
        relative_limit=0.02,
    )
    assert checked


@pytest.mark.parametrize("rows", [4095, 4096, 4097, 6145])
@pytest.mark.parametrize("hidden", [256, 768])
def test_batched_int8_prefill_routing_and_capacity(rows, hidden, monkeypatch):
    """Production IMMA matches independent rotations across scratch reuse/tails."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Native EXL3 INT8 prefill targets SM80")
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "int8")
    calls = []
    original = exl3_prefill.moe_int8

    def observed(x, *args):
        calls.append(x.shape[0])
        return original(x, *args)

    monkeypatch.setattr(exl3_prefill, "moe_int8", observed)
    method = _check_moe_routing(
        rows,
        2048,
        hidden,
        monkeypatch,
        relative_limit=0.02,
        dtype=torch.bfloat16,
        m_tile=32,
        intermediate_dim=512,
    )
    if rows >= 4096:
        assert calls and calls[0] == min(rows, 6144)
    else:
        assert not calls
    assert len(method.config.prefill_workspaces) == 1


@pytest.mark.parametrize("group_size", [1, 4, 16])
def test_int8_prefill_grouped_weights_preserve_routes_and_replay(
    group_size, monkeypatch
):
    """Bound scratch without changing hot/empty experts, tail groups or replay."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("INT8 prefill targets SM80")
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "int8")
    monkeypatch.setenv("VLLM_EXL3_MOE_INT8_MIN_TOKENS", "9")
    monkeypatch.setenv("VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP", str(group_size))
    original = exl3_prefill.moe_int8
    compared = False

    def compare_once(x, routing, ids, ptrs, workspace, limit):
        nonlocal compared
        actual = original(x, routing, ids, ptrs, workspace, limit)
        if not compared:
            full = exl3_prefill.allocate_workspace(x.device, 7, 256, 512, 513, 2)
            expected = original(x, routing, ids, ptrs, full, limit)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            compared = True
        return actual

    monkeypatch.setattr(exl3_prefill, "moe_int8", compare_once)
    method = _check_moe_routing(
        513,
        2048,
        256,
        monkeypatch,
        m_tile=32,
        intermediate_dim=512,
        num_experts=7,
        dtype=torch.bfloat16,
        relative_limit=0.02,
    )
    assert compared
    assert method.prefill_workspace[0].numel() == min(group_size, 7) * 256 * 512


def test_int8_prefill_large_topk_uses_native(monkeypatch):
    """Unsupported expert fan-out must fall back before allocating an INT8 pool."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("INT8 prefill targets SM80")
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "int8")
    monkeypatch.setenv("VLLM_EXL3_MOE_INT8_MIN_TOKENS", "9")

    def unexpected_allocation(*args):
        raise AssertionError("Unsupported top-k must use native prefill")

    monkeypatch.setattr(exl3_prefill, "allocate_workspace", unexpected_allocation)
    _check_moe_routing(64, 128, 256, monkeypatch, topk=16, m_tile=32)


@pytest.mark.parametrize("policy", ["auto", "int8"])
def test_int8_prefill_workspace_oom_fallback(policy, monkeypatch):
    """Auto retains correct native output; an explicit INT8 request reports OOM."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("INT8 prefill targets SM80")
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", policy)
    allocate = exl3_prefill.allocate_workspace
    empty = torch.empty
    native = []

    def record_empty(*args, **kwargs):
        result = empty(*args, **kwargs)
        if (
            result.is_cuda
            and result.dtype == torch.float16
            and result.ndim == 3
            and result.shape[1:] == (2048, 256)
        ):
            native.append(result)
        return result

    def unavailable(device, *args):
        if device == "meta":
            return allocate(device, *args)
        # This native pool is larger than INT8's; it must be reserved first.
        assert len(native) == 4
        raise torch.OutOfMemoryError("test workspace reservation failure")

    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setattr(exl3_prefill, "allocate_workspace", unavailable)
    if policy == "int8":
        with pytest.raises(torch.OutOfMemoryError, match="workspace reservation"):
            _check_moe_routing(9, 2048, 256, monkeypatch, m_tile=32)
    else:
        method = _check_moe_routing(9, 2048, 256, monkeypatch, m_tile=32)
        assert not method.prefill_workspace
        assert method.ptrs[0].device in method.config.prefill_disabled


def test_int8_prefill_full_graph_uses_native_arithmetic(monkeypatch):
    """Padding a decode graph must not enable request-dependent INT8 prefill."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("INT8 prefill targets SM80")
    from vllm import forward_context
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "auto")
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )

    def unexpected(*args):
        pytest.fail("FULL graph must retain native arithmetic")

    monkeypatch.setattr(exl3_prefill, "moe_int8", unexpected)
    _check_moe_routing(4096, 2048, 256, monkeypatch, m_tile=32)


def test_int8_prefill_uses_callers_stream(monkeypatch):
    """Reconstruction, Hadamard and IMMA must share the caller's stream."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("INT8 prefill targets SM80")
    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "int8")
    with torch.cuda.stream(torch.cuda.Stream()):
        _check_moe_routing(
            4096,
            2048,
            256,
            monkeypatch,
            m_tile=32,
            dtype=torch.bfloat16,
            relative_limit=0.02,
        )
    torch.accelerator.synchronize()


def test_int8_prefill_uses_tensor_device(monkeypatch):
    """Triton must launch on the input device and restore the caller's device."""
    if torch.accelerator.device_count() < 2 or torch.cuda.get_device_capability(0) != (
        8,
        0,
    ):
        pytest.skip("Requires SM80 device 0 and a second CUDA device")
    from vllm.model_executor.layers.quantization.utils import exl3_prefill

    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "int8")
    original = exl3_prefill.moe_int8

    def from_other_device(*args):
        with torch.accelerator.device_index(1):
            result = original(*args)
            assert torch.accelerator.current_device_index() == 1
            return result

    monkeypatch.setattr(exl3_prefill, "moe_int8", from_other_device)
    with torch.accelerator.device_index(0):
        _check_moe_routing(
            4096,
            2048,
            256,
            monkeypatch,
            m_tile=32,
            dtype=torch.bfloat16,
            relative_limit=0.02,
        )


@pytest.mark.parametrize(
    "rows,hidden,intermediate,topk",
    [(rows, 256, 512, 2) for rows in (9, 12, 16, 32, 96, 128)]
    + [(32, 4096, 2048, 2), (96, 4096, 2048, 2), (12, 4096, 2048, 3)],
)
@pytest.mark.parametrize("residual", [False, True])
def test_batched_decode_preserves_sparse_hot_experts_and_replay(
    rows, hidden, intermediate, topk, residual, monkeypatch
):
    """Mixed expert paths preserve rotations, sparse routing and scratch reuse."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Batched expert decode requires SM80")
    from vllm.model_executor.layers.quantization.utils.exl3_decode import (
        moe_batched_decode,
    )

    monkeypatch.setenv("VLLM_EXL3_MOE_BATCHED_DECODE", "0")
    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL", "native")

    def apply(method, layer, x, weights, ids):
        if method.decode_workspace.shape[0] == 64:
            method.decode_workspace = torch.zeros(
                (1024, method.decode_workspace.shape[1]),
                dtype=torch.int32,
                device=x.device,
            )
        if rows == 12:
            x = x.t().contiguous().t()
        current_device = 1 if torch.accelerator.device_count() > 1 else 0
        with torch.accelerator.device_index(current_device):
            return moe_batched_decode(
                x,
                weights,
                ids,
                method.ptrs,
                method.workspace,
                method.m32_locks,
                method.decode_workspace,
                10.0,
                residual,
            )

    monkeypatch.setattr(Exl3MoEMethod, "apply", apply)
    _check_moe_routing(
        rows,
        256,
        hidden,
        monkeypatch,
        relative_limit=0.02,
        dtype=torch.bfloat16,
        m_tile=32,
        decode="residual" if residual else "plain",
        intermediate_dim=intermediate,
        topk=topk,
        num_experts=rows + 1,
        cold_routes=True,
        check_graph=True,
    )


@pytest.mark.parametrize(
    "metadata,mode,capacity,batched",
    [
        (SimpleNamespace(num_prefills=0, num_decodes=16), CUDAGraphMode.NONE, 16, True),
        (SimpleNamespace(num_prefills=0, num_decodes=16), CUDAGraphMode.NONE, 8, False),
        (SimpleNamespace(num_prefills=0, num_decodes=16), CUDAGraphMode.FULL, 16, True),
        (
            SimpleNamespace(num_prefills=0, num_decodes=16),
            CUDAGraphMode.PIECEWISE,
            16,
            False,
        ),
        (
            SimpleNamespace(num_prefills=0, num_decodes=0, num_spec_decodes=8),
            CUDAGraphMode.FULL,
            16,
            True,
        ),
        (
            SimpleNamespace(num_prefills=1, num_decodes=15),
            CUDAGraphMode.NONE,
            16,
            False,
        ),
        (SimpleNamespace(num_prefills=1, num_decodes=0), CUDAGraphMode.NONE, 16, False),
        (None, CUDAGraphMode.NONE, 16, False),
    ],
)
def test_batched_decode_does_not_quantize_native_prefill(
    metadata, mode, capacity, batched, monkeypatch
):
    """Short or mixed prefill cannot inherit the new decode arithmetic."""
    from vllm import forward_context
    from vllm.model_executor.layers.quantization import exl3
    from vllm.model_executor.layers.quantization.utils import exl3_decode

    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            attn_metadata={"attention": metadata}, cudagraph_runtime_mode=mode
        ),
    )
    x = torch.zeros((16, 256), dtype=torch.bfloat16)
    expected = torch.ones_like(x)
    native = torch.full_like(x, 2)
    monkeypatch.setattr(exl3_decode, "moe_batched_decode", lambda *args: expected)
    monkeypatch.setattr(exl3, "_exl3_moe_fused", lambda *args: native)
    result = exl3._exl3_moe(
        x,
        torch.ones((16, 2)),
        torch.zeros((16, 2), dtype=torch.long),
        [],
        [torch.empty((1, capacity, 256))],
        [],
        [4, 4, 4],
        [False, True] * 3,
        10.0,
        m32_locks=torch.zeros(1, dtype=torch.int32),
        decode_workspace=torch.zeros((1024, 1), dtype=torch.int32),
        decode_mode=1,
        batched_decode_mode=1,
    )
    torch.testing.assert_close(result, expected if batched else native)
