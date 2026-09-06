# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia import model as glm5_model
from vllm.sequence import IntermediateTensors


class _DeferredMhcLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_state: tuple[torch.Tensor | None, ...] | None = None
        self.post_state: tuple[torch.Tensor, ...] | None = None

    def forward(self, positions, hidden_states, residual, post, comb):
        self.forward_state = (residual, post, comb)
        return (
            hidden_states + 1,
            hidden_states + 2,
            hidden_states + 3,
            hidden_states + 4,
        )

    def mhc_post_op(self, hidden_states, residual, post, comb):
        self.post_state = (hidden_states, residual, post, comb)
        return hidden_states + residual + post + comb


def _make_stage(layer: nn.Module) -> glm5_model.Glm5NextModel:
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model._active_layers = nn.ModuleList([layer])
    model.is_sequence_parallel = False
    return model


def test_glm5next_pp_intermediate_tensor_shape():
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=8,
        mhc=True,
        mhc_num_residual_streams=4,
    )

    intermediate = model.make_empty_intermediate_tensors(
        batch_size=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert set(intermediate.tensors) == {"hidden_states"}
    assert intermediate["hidden_states"].shape == (3, 4, 8)
    assert intermediate["hidden_states"].dtype == torch.bfloat16


def test_glm5next_pp_materializes_deferred_mhc_state(monkeypatch):
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    layer = _DeferredMhcLayer()
    model = _make_stage(layer)
    hidden_states = torch.ones(2, 3)

    output = model(
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=None,
        inputs_embeds=hidden_states,
    )

    assert isinstance(output, IntermediateTensors)
    assert set(output.tensors) == {"hidden_states"}
    torch.testing.assert_close(output["hidden_states"], hidden_states * 4 + 10)
    assert layer.post_state is not None


def test_glm5next_pp_receiving_stage_starts_new_mhc_fusion(monkeypatch):
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    layer = _DeferredMhcLayer()
    model = _make_stage(layer)
    hidden_states = torch.ones(2, 4, 3)

    model(
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=IntermediateTensors({"hidden_states": hidden_states}),
    )

    assert layer.forward_state == (None, None, None)


@pytest.mark.parametrize(
    ("checkpoint_prefix", "parameter_prefix"),
    [
        ("attn_hc.", "hc_attn_"),
        ("ffn_hc.", "hc_ffn_"),
        ("hc_attn_", "hc_attn_"),
        ("hc_ffn_", "hc_ffn_"),
    ],
)
def test_glm5next_pp_loads_mhc_checkpoint_names(checkpoint_prefix, parameter_prefix):
    """Load both checkpoint formats while skipping another PP stage's weights."""
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(is_moe=False, mla_nope=False)
    model.quant_config = None
    layer = nn.Module()
    layer.register_parameter(parameter_prefix + "base", nn.Parameter(torch.zeros(2)))
    model.layers = nn.ModuleList([layer, glm5_model.PPMissingLayer()])
    weight = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    loaded = model.load_weights(
        [(f"layers.{i}.{checkpoint_prefix}base", weight) for i in range(2)]
    )

    assert loaded == {f"layers.0.{parameter_prefix}base"}
    torch.testing.assert_close(
        getattr(layer, parameter_prefix + "base"), weight.float()
    )


def test_glm5next_loads_fused_kda_convolution_and_forget_gate():
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(is_moe=False, mla_nope=False)
    model.quant_config = None
    layer = nn.Module()
    layer.self_attn = nn.Module()
    for proj in ("q", "k", "v"):
        setattr(
            layer.self_attn, f"{proj}_conv1d", nn.Conv1d(2, 2, 4, groups=2, bias=False)
        )
    layer.self_attn.dt_bias = nn.Parameter(torch.zeros(2))
    model.layers = nn.ModuleList([layer])
    fused = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4)
    bias = torch.tensor([1.0, 2.0])

    model.load_weights(
        [
            ("layers.0.self_attn.conv1d.weight", fused),
            ("layers.0.self_attn.forget_gate.dt_bias", bias),
        ]
    )

    for proj, expected in zip(("q", "k", "v"), fused.chunk(3)):
        torch.testing.assert_close(
            getattr(layer.self_attn, f"{proj}_conv1d").weight, expected
        )
    torch.testing.assert_close(layer.self_attn.dt_bias, bias)


@pytest.mark.parametrize("symmetric", [False, True])
def test_glm5next_loads_packed_attention_weights_in_any_order(symmetric):
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
    )

    quant_args = QuantizationArgs(
        num_bits=4, strategy="group", group_size=16, symmetric=symmetric
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=quant_args)
    scales = torch.full((8, 2), 0.5)
    zero_point = torch.zeros((8, 2), dtype=torch.int8)
    if not symmetric:
        zero_point[:, 0] = -1
        zero_point[:, 1] = 1
    integers = (torch.arange(256).reshape(8, 32) % 8 - 4).float()
    expected = (integers - zero_point.repeat_interleave(16, dim=1)) * 0.5
    packed = PackedQuantizationCompressor.compress(
        {"weight": expected, "weight_scale": scales, "weight_zero_point": zero_point},
        scheme,
    )
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(is_moe=False, mla_nope=False)
    model.quant_config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={"Linear": {"weights": quant_args}},
        ignore=[],
        quant_format="pack-quantized",
    )
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(32, 8, bias=False)
    model.layers = nn.ModuleList([layer])

    loaded = model.load_weights(
        [
            (f"layers.0.self_attn.o_proj.{name}", value)
            for name, value in reversed(list(packed.items()))
        ]
    )

    assert loaded == {"layers.0.self_attn.o_proj.weight"}
    torch.testing.assert_close(layer.self_attn.o_proj.weight, expected)


def test_glm5next_awq_preserves_unquantized_fused_mlp():
    from compressed_tensors.quantization import QuantizationArgs

    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config

    config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={"Linear": {"weights": QuantizationArgs(num_bits=4)}},
        ignore=[
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
        ],
        quant_format="pack-quantized",
    )
    configure_quant_config(config, glm5_model.Glm5NextForConditionalGeneration)
    configure_quant_config(config, glm5_model.Glm5NextForCausalLM)
    layer = nn.Linear(32, 64, bias=False)

    assert (
        config.get_scheme_dict(layer, "language_model.model.layers.0.mlp.gate_up_proj")
        is None
    )
    assert (
        config.get_scheme_dict(layer, "language_model.model.layers.1.mlp.gate_up_proj")
        is not None
    )


@pytest.mark.parametrize("load_target_first", [False, True])
def test_glm5next_awq_preserves_unquantized_mtp_layers(load_target_first):
    from compressed_tensors.quantization import QuantizationArgs

    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP

    config = compressed_tensors.CompressedTensorsConfig(
        target_scheme_map={"Linear": {"weights": QuantizationArgs(num_bits=4)}},
        ignore=[
            f"model.language_model.layers.45.mlp.{expert}.{proj}_proj"
            for expert in ("experts.0", "shared_experts")
            for proj in ("gate", "up")
        ],
        quant_format="pack-quantized",
    )
    if load_target_first:
        configure_quant_config(config, glm5_model.Glm5NextForConditionalGeneration)
        configure_quant_config(config, glm5_model.Glm5NextForCausalLM)
    configure_quant_config(config, Glm5NextMTP)
    layer = nn.Linear(32, 64, bias=False)

    for expert in ("experts.0", "shared_experts"):
        prefix = f"model.layers.45.mlp.{expert}.gate_up_proj"
        assert config.get_scheme_dict(layer, prefix) is None
    assert (
        config.get_scheme_dict(layer, "model.layers.44.mlp.experts.0.gate_up_proj")
        is not None
    )


@pytest.mark.parametrize("checkpoint_prefix", ["model.", "model.language_model."])
@pytest.mark.parametrize("include_embedding", [True, False])
def test_glm5next_mtp_requires_embedding_when_pp_cannot_share(
    checkpoint_prefix, include_embedding, monkeypatch
):
    """The last PP rank must load the embedding from the target checkpoint."""
    from vllm.models.glm5next.nvidia import mtp as glm5_mtp
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP

    monkeypatch.setattr(glm5_mtp, "get_pp_group", lambda: SimpleNamespace(world_size=4))
    model = Glm5NextMTP.__new__(Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_routed_experts=0,
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        mla_nope=False,
    )
    model.quant_config = None
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 45
    model.model.num_mtp_layers = 1
    model.model.embed_tokens = nn.Embedding(4, 2)
    model.model.embed_tokens.weight.data.zero_()
    layer = nn.Module()
    layer.eh_proj = nn.Linear(4, 2, bias=False)
    model.model.layers = nn.ModuleDict({"45": layer})
    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    weights = [(checkpoint_prefix + "layers.45.eh_proj.weight", torch.ones(2, 4))]
    if not include_embedding:
        with pytest.raises(ValueError, match="requires embedding weights"):
            model.load_weights(weights)
        return

    weights.insert(0, (checkpoint_prefix + "embed_tokens.weight", weight))
    loaded = model.load_weights(weights)

    torch.testing.assert_close(
        model.model.embed_tokens(torch.tensor([1, 3])), weight[[1, 3]]
    )
    assert "model.embed_tokens.weight" in loaded
