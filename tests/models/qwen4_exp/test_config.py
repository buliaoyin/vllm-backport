# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.platforms import current_platform
from vllm.v1.spec_decode.dynamic.adaptive import supports_adaptive_mtp
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from ...utils import spawn_new_process_for_each_test


@pytest.mark.parametrize("online", [False, True])
def test_qwen4_exp_qsa_excluded_projection_retains_online_quantization(online):
    """QSA QKV may quantize BF16 weights without applying checkpoint FP4."""
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.models.qwen4_exp.nvidia.model import without_modelopt_fp4

    quant_config = ModelOptNvFp4Config(exclude_modules=[])
    overlay = (
        OnlineQuantizationConfig(
            QuantizationConfigArgs(targets={"*.self_attn.qkv_proj": "mxfp8"})
        )
        if online
        else None
    )
    quant_config.online_quantization_config = overlay

    assert without_modelopt_fp4(quant_config) is overlay


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 0,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("layers", [0, 1, 2])
def test_adaptive_qwen4_exp_mtp_requires_one_resolved_layer(wrapped, layers):
    """Only the validated single-layer HC drafter may use adaptive budgets."""
    text = _text_config(mtp_num_hidden_layers=layers)
    config = Qwen4ExpConfig(text_config=text) if wrapped else text
    config.architectures = [
        "Qwen4ExpForConditionalGeneration" if wrapped else "Qwen4ExpForCausalLM"
    ]
    draft = SpeculativeConfig.hf_config_override(config)
    spec = SimpleNamespace(
        method="mtp", draft_model_config=SimpleNamespace(hf_config=draft)
    )
    assert supports_adaptive_mtp(spec) == (layers == 1)


def test_qwen4_exp_mtp_returns_sample_and_multi_streams() -> None:
    from vllm.models.qwen4_exp.nvidia.mtp import (
        Qwen4ExpMultiTokenPredictor,
    )

    model = object.__new__(Qwen4ExpMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.layers = [
        lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["hidden_states"],
            torch.zeros(kwargs["hidden_states"].shape[0], 2),
        ),
    ]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    with patch(
        "vllm.models.qwen4_exp.nvidia.mtp.get_pp_group",
        return_value=pp_group,
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(2),
            intermediate_tensors={"hidden_states": multi_hidden},
        )

    torch.testing.assert_close(
        sample_hidden,
        multi_hidden.unflatten(-1, (2, 4)).mean(dim=-2),
    )
    assert returned_multi_hidden is multi_hidden


def _make_token_map_draft(device: str):
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP

    config = VllmConfig()
    draft = object.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(draft)
    draft.vllm_config = config
    draft.config = SimpleNamespace(vocab_size=8)
    draft.model = torch.nn.Module()
    with set_current_vllm_config(config), torch.device(device):
        target_head = ParallelLMHead(
            8, 4, params_dtype=torch.bfloat16, padding_size=1, disable_tp=True
        )
        target_head.weight.data.copy_(torch.arange(32).view(8, 4))
        draft.lm_head = target_head
        draft.logits_processor = LogitsProcessor(8)
    draft.register_buffer("draft_id_to_target_id", None, persistent=False)
    draft.register_buffer("mtp_token_map", None, persistent=False)
    return draft, target_head


@pytest.mark.parametrize(
    "unsupported", ["capability", "dtype", "head_dtype", "tp", "pp", "tied", "lora"]
)
@pytest.mark.parametrize("feature", ["head", "hc", "output"])
def test_qwen4_exp_rowwise_fp8_head_rejects_unsupported_config(
    monkeypatch, unsupported, feature
) -> None:
    """Weight-only FP8 must fail before loading outside its supported scope."""
    from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import (
        rowwise_fp8_hc_enabled,
        rowwise_fp8_head_enabled,
        rowwise_fp8_output_enabled,
    )

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: True
    )
    text = SimpleNamespace(tie_word_embeddings=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**{f"sm120_rowwise_fp8_{feature}": True}),
            hf_text_config=text,
            dtype=torch.bfloat16,
            head_dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        lora_config=None,
    )
    enabled = {
        "head": rowwise_fp8_head_enabled,
        "hc": rowwise_fp8_hc_enabled,
        "output": rowwise_fp8_output_enabled,
    }[feature]
    assert enabled(config)
    if unsupported == "capability":
        monkeypatch.setattr(
            current_platform, "is_device_capability", lambda capability: False
        )
    elif unsupported in ("dtype", "head_dtype"):
        setattr(config.model_config, unsupported, torch.float32)
    elif unsupported in ("tp", "pp"):
        name = (
            "tensor_parallel_size" if unsupported == "tp" else "pipeline_parallel_size"
        )
        setattr(config.parallel_config, name, 2)
    elif unsupported == "tied":
        text.tie_word_embeddings = True
    else:
        config.lora_config = object()
    if feature != "head" and unsupported in ("head_dtype", "tied"):
        assert enabled(config)
    else:
        with pytest.raises(ValueError, match=f"sm120_rowwise_fp8_{feature} requires"):
            enabled(config)


def test_qwen4_exp_rowwise_fp8_head_reload_refreshes_values_and_scales() -> None:
    """A BF16 checkpoint reload must refresh both FP8 tensors in-place."""
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import (
        install_rowwise_fp8_head,
        quantize_rowwise_fp8,
    )

    draft, head = _make_token_map_draft("cpu")
    install_rowwise_fp8_head(head)
    record_metadata_for_reloading(head)
    head.quant_method.process_weights_after_loading(head)
    weight_ptr = head.weight.data_ptr()
    scale_ptr = head.weight_scale.data_ptr()
    replacement = torch.arange(32, dtype=torch.bfloat16).view(8, 4).neg() / 3
    expected_weight, expected_scale = quantize_rowwise_fp8(replacement)
    with set_current_vllm_config(draft.vllm_config):
        initialize_layerwise_reload(head)
        head.weight.weight_loader(head.weight, replacement)
        finalize_layerwise_reload(head, None)
    assert head.weight.data_ptr() == weight_ptr
    assert head.weight_scale.data_ptr() == scale_ptr
    torch.testing.assert_close(head.weight.float(), expected_weight.float())
    torch.testing.assert_close(head.weight_scale, expected_scale)


def test_qwen4_exp_rowwise_fp8_head_rejects_direct_resident_reload() -> None:
    """Plain BF16 copying cannot leave an FP8 head with stale row scales."""
    from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import install_rowwise_fp8_head

    _, head = _make_token_map_draft("cpu")
    install_rowwise_fp8_head(head)
    head.quant_method.process_weights_after_loading(head)
    values, scales = head.weight.float().clone(), head.weight_scale.clone()
    with pytest.raises(ValueError, match="layerwise weight reload"):
        head.weight.weight_loader(head.weight, torch.zeros(8, 4, dtype=torch.bfloat16))
    torch.testing.assert_close(head.weight.float(), values)
    torch.testing.assert_close(head.weight_scale, scales)


def test_qwen4_exp_rowwise_fp8_hc_reload_preserves_bf16_injection(monkeypatch) -> None:
    """Merged HC reloads quantize mixing rows and retain original injection."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.models.qwen4_exp.nvidia.ops import rowwise_fp8

    draft, layer = _make_token_map_draft("cpu")
    layer.quant_method = UnquantizedLinearMethod()
    layer.weight.data.copy_(torch.randn_like(layer.weight))
    original = layer.weight.clone()
    rowwise_fp8.install_rowwise_fp8_hc(layer, bf16_start=4, bf16_rows=2, pad_rows=2)
    record_metadata_for_reloading(layer)
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.equal(layer.bf16_weight, original[4:6])
    assert layer.weight.float()[4:].count_nonzero() == 0
    pointers = tuple(
        value.data_ptr()
        for value in (layer.weight, layer.weight_scale, layer.bf16_weight)
    )
    replacement = original.neg() / 3
    with set_current_vllm_config(draft.vllm_config):
        initialize_layerwise_reload(layer)
        layer.weight.weight_loader(layer.weight, replacement)
        finalize_layerwise_reload(layer, None)
    assert pointers == tuple(
        value.data_ptr()
        for value in (layer.weight, layer.weight_scale, layer.bf16_weight)
    )
    assert torch.equal(layer.bf16_weight, replacement[4:6])
    monkeypatch.setattr(rowwise_fp8, "_CHUNK_BYTES", 16)
    hidden = torch.ones(33, 4, dtype=torch.bfloat16)
    actual = layer.quant_method.apply(layer, hidden)
    expected = (hidden.float() @ layer.weight.float().t()) * layer.weight_scale
    expected[:, 4:6] = hidden.float() @ replacement[4:6].float().t()
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    assert actual[:, 6:].count_nonzero() == 0


def test_qwen4_exp_rowwise_fp8_output_reload_keeps_registered_scales() -> None:
    """Output projection reloads update resident FP8 values and row scales."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import install_rowwise_fp8_output

    draft, layer = _make_token_map_draft("cpu")
    layer.weight = torch.nn.Parameter(
        torch.ones(2560, 6144, dtype=torch.bfloat16), requires_grad=False
    )
    layer.weight.weight_loader = default_weight_loader
    layer.quant_method = UnquantizedLinearMethod()
    install_rowwise_fp8_output(layer)
    record_metadata_for_reloading(layer)
    layer.quant_method.process_weights_after_loading(layer)
    weight_ptr, scale_ptr = layer.weight.data_ptr(), layer.weight_scale.data_ptr()
    original_scale = layer.weight_scale.clone()
    replacement = torch.full((2560, 6144), 3, dtype=torch.bfloat16)
    replacement[:, 0] = 0
    with set_current_vllm_config(draft.vllm_config):
        initialize_layerwise_reload(layer)
        layer.weight.weight_loader(layer.weight, replacement)
        finalize_layerwise_reload(layer, None)
    assert (layer.weight.data_ptr(), layer.weight_scale.data_ptr()) == (
        weight_ptr,
        scale_ptr,
    )
    torch.testing.assert_close(layer.weight_scale, original_scale * 3)
    assert layer.weight.float()[:, 0].count_nonzero() == 0
    assert torch.all(layer.weight.float()[:, 1:] == 448)
    with pytest.raises(ValueError, match="unquantized BF16"):
        install_rowwise_fp8_output(layer)
    with pytest.raises(ValueError, match="layerwise weight reload"):
        layer.weight.weight_loader(layer.weight, replacement)


def test_qwen4_exp_mtp_token_map_selects_matching_fp8_scales(tmp_path) -> None:
    """Reduced FP8 heads retain target logits and map draft IDs correctly."""
    from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import install_rowwise_fp8_head

    draft, target_head = _make_token_map_draft("cpu")
    install_rowwise_fp8_head(target_head)
    target_head.quant_method.process_weights_after_loading(target_head)
    hidden = torch.tensor([[1] * 4, [-1] * 4], dtype=torch.bfloat16)
    expected_target = draft.compute_logits(hidden).clone()
    target_processor = draft.logits_processor
    target_weight = target_head.weight
    target_scale = target_head.weight_scale
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor([6, 2, 4]), path)
    draft.configure_mtp_token_map(str(path))
    assert target_head.weight is target_weight
    assert target_head.weight_scale is target_scale
    torch.testing.assert_close(target_processor(target_head, hidden), expected_target)
    torch.testing.assert_close(draft.lm_head.weight_scale, target_scale[[2, 4, 6]])
    expected = torch.full_like(expected_target, -torch.inf)
    expected[:, [2, 4, 6]] = expected_target[:, [2, 4, 6]]
    torch.testing.assert_close(draft.compute_logits(hidden), expected)
    torch.testing.assert_close(draft.get_top_tokens(hidden), expected.argmax(-1))
    with pytest.raises(ValueError, match="cannot be changed"):
        draft.configure_mtp_token_map(str(path))


@pytest.mark.parametrize("pp_size", [1, 2])
def test_qwen4_exp_mtp_token_map_survives_target_head_sharing(
    monkeypatch, tmp_path, pp_size
) -> None:
    """Reduced drafts return target IDs while target logits remain unchanged."""
    from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp_speculator

    draft, target_head = _make_token_map_draft("cpu")
    draft.vllm_config.parallel_config.pipeline_parallel_size = pp_size
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.mtp.get_pp_group",
        lambda: SimpleNamespace(world_size=pp_size, is_last_rank=True),
    )
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor([6, 2, 4]), path)
    hidden = torch.tensor([[1] * 4, [-1] * 4], dtype=torch.bfloat16)
    target_logits_processor = draft.logits_processor
    expected_target = draft.compute_logits(hidden).clone()

    def load_shared_model(*args):
        draft.lm_head = target_head
        return draft

    monkeypatch.setattr(mtp_speculator, "load_eagle_model", load_shared_model)
    runner = object.__new__(mtp_speculator.MTPSpeculator)
    runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            mtp_token_map=str(path),
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
    )
    assert runner.load_draft_model(torch.nn.Module(), set()) is draft
    assert draft.lm_head is not target_head
    assert draft.lm_head.weight.shape == (3, 4)
    torch.testing.assert_close(
        target_logits_processor(target_head, hidden), expected_target
    )
    expected = torch.full_like(expected_target, -torch.inf)
    expected[:, [2, 4, 6]] = expected_target[:, [2, 4, 6]]
    torch.testing.assert_close(draft.compute_logits(hidden), expected)
    torch.testing.assert_close(draft.get_top_tokens(hidden), expected.argmax(-1))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("token_ids", [[6, 2, 4], [6, 0, 2, 4]])
def test_qwen4_exp_mtp_tp2_token_map_balances_rows_and_preserves_global_ids(
    monkeypatch, tmp_path, rank, token_ids
) -> None:
    """Repartition hot rows across TP ranks without changing target logits."""
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding
    from vllm.models.qwen4_exp.nvidia import mtp

    draft, _ = _make_token_map_draft("cpu")
    draft.config.vocab_size = 7
    draft.vllm_config.parallel_config.tensor_parallel_size = 2
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: 2
    )
    full_weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    full_weight[-1].zero_()
    hidden = torch.tensor([[1] * 4, [-1] * 4, [0] * 4], dtype=torch.bfloat16)
    with set_current_vllm_config(draft.vllm_config):
        target_head = ParallelLMHead(7, 4, params_dtype=torch.bfloat16, padding_size=2)
        target_head.weight.data.copy_(full_weight[rank * 4 : (rank + 1) * 4])
        draft.lm_head = target_head
        draft.logits_processor = LogitsProcessor(7)
    target_processor = draft.logits_processor
    original_weight = target_head.weight.clone()
    original_pointer = target_head.weight.data_ptr()
    gathered_weights = []

    def gather_weights(weight, dim):
        assert dim == 0 and weight is target_head.weight
        gathered_weights.append(weight)
        return full_weight.clone()

    monkeypatch.setattr(mtp, "tensor_model_parallel_all_gather", gather_weights)
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor(token_ids), path)
    draft.configure_mtp_token_map(str(path))
    assert len(gathered_weights) == 1
    assert target_head.weight.data_ptr() == original_pointer
    torch.testing.assert_close(target_head.weight, original_weight)
    sorted_ids = torch.tensor(sorted(token_ids))
    padded_rows = draft.lm_head.num_embeddings_padded
    reduced_weight = torch.zeros(padded_rows, 4, dtype=torch.bfloat16)
    reduced_weight[: len(token_ids)] = full_weight[sorted_ids]
    rows_per_rank = padded_rows // 2
    torch.testing.assert_close(
        draft.lm_head.weight,
        reduced_weight[rank * rows_per_rank : (rank + 1) * rows_per_rank],
    )
    target_logits = hidden @ full_weight.t()
    reduced_logits = hidden @ reduced_weight.t()
    masked_logits = reduced_logits.clone()
    masked_logits[:, len(token_ids) :] = -torch.inf
    pairs = []
    for peer in range(2):
        values, indices = masked_logits[
            :, peer * rows_per_rank : (peer + 1) * rows_per_rank
        ].max(-1)
        pairs.append(
            torch.stack((values.float(), (indices + peer * rows_per_rank).float()), -1)
        )

    def gather_logits(logits, dim=-1):
        assert dim == -1
        if logits.dtype == torch.float32:
            torch.testing.assert_close(logits, pairs[rank])
            return torch.cat(pairs, dim=-1)
        expected = target_logits if logits.shape[-1] == 4 else reduced_logits
        width = logits.shape[-1]
        torch.testing.assert_close(
            logits, expected[:, rank * width : (rank + 1) * width]
        )
        return expected

    monkeypatch.setattr(
        logits_processor, "tensor_model_parallel_all_gather", gather_logits
    )
    monkeypatch.setattr(logits_processor, "tensor_model_parallel_gather", gather_logits)
    torch.testing.assert_close(
        target_processor(target_head, hidden), target_logits[:, :7]
    )
    expected = torch.full((hidden.shape[0], 7), -torch.inf, dtype=torch.bfloat16)
    expected[:, sorted_ids] = reduced_logits[:, : len(token_ids)]
    torch.testing.assert_close(draft.compute_logits(hidden), expected)
    torch.testing.assert_close(draft.get_top_tokens(hidden), expected.argmax(-1))


@pytest.mark.parametrize(
    "token_ids",
    [[], [[2, 4]], [2.0, 4.0], [2, 2], [-1, 2], [2, 8]],
)
def test_qwen4_exp_mtp_rejects_invalid_token_map(tmp_path, token_ids) -> None:
    """Invalid hot vocabularies must fail before replacing the shared head."""
    draft, target_head = _make_token_map_draft("cpu")
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor(token_ids), path)
    with pytest.raises(ValueError, match="mtp_token_map"):
        draft.configure_mtp_token_map(str(path))
    assert draft.lm_head is target_head


@pytest.mark.parametrize(
    "unsupported",
    [
        "tp",
        "head_tp",
        "head_padding",
        "pp_non_last",
        "pp_size",
        "head_vocab",
        "added_vocab",
        "tp2_fp8",
        "fp32",
    ],
)
def test_qwen4_exp_mtp_token_map_rejects_unsupported_head(
    monkeypatch, tmp_path, unsupported
) -> None:
    """A token map cannot reinterpret sharded or non-BF16 head weights."""
    draft, target_head = _make_token_map_draft("cpu")
    if unsupported == "tp":
        draft.vllm_config.parallel_config.tensor_parallel_size = 2
    elif unsupported == "head_tp":
        target_head.tp_size = 2
    elif unsupported == "head_padding":
        draft.vllm_config.parallel_config.tensor_parallel_size = 2
        target_head.tp_size = 2
    elif unsupported == "pp_non_last":
        draft.vllm_config.parallel_config.pipeline_parallel_size = 2
        monkeypatch.setattr(
            "vllm.models.qwen4_exp.nvidia.mtp.get_pp_group",
            lambda: SimpleNamespace(world_size=2, is_last_rank=False),
        )
    elif unsupported == "pp_size":
        draft.vllm_config.parallel_config.pipeline_parallel_size = 3
    elif unsupported == "head_vocab":
        target_head.org_vocab_size = 4
    elif unsupported == "added_vocab":
        target_head.num_embeddings = 9
    elif unsupported == "tp2_fp8":
        from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import (
            install_rowwise_fp8_head,
        )

        draft.vllm_config.parallel_config.tensor_parallel_size = 2
        target_head.tp_size = 2
        install_rowwise_fp8_head(target_head)
        target_head.quant_method.process_weights_after_loading(target_head)
    else:
        target_head.weight.data = target_head.weight.float()
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor([2, 4, 6]), path)
    with pytest.raises(ValueError, match="mtp_token_map requires"):
        draft.configure_mtp_token_map(str(path))
    assert draft.lm_head is target_head


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="CUDA graph requires an available CUDA device",
)
@pytest.mark.parametrize("rowwise_fp8", [False, True])
@torch.inference_mode()
def test_qwen4_exp_mtp_reduced_head_cuda_graph_remaps_fresh_tokens(
    tmp_path, rowwise_fp8
) -> None:
    """Captured draft sampling retains the reduced head and returns target IDs."""
    draft, target_head = _make_token_map_draft("cuda")
    if rowwise_fp8:
        from vllm.models.qwen4_exp.nvidia.ops.rowwise_fp8 import (
            install_rowwise_fp8_head,
        )

        install_rowwise_fp8_head(target_head)
        target_head.quant_method.process_weights_after_loading(target_head)
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor([2, 4, 6]), path)
    draft.configure_mtp_token_map(str(path))
    hidden = torch.ones(2, 4, dtype=torch.bfloat16, device="cuda")
    draft.get_top_tokens(hidden)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens = draft.get_top_tokens(hidden)
        logits = draft.compute_logits(hidden)
    for sign in (-1, 1):
        hidden.fill_(sign)
        graph.replay()
        reference = hidden.float() @ draft.lm_head.weight.float().t()
        if rowwise_fp8:
            reference *= draft.lm_head.weight_scale
        expected = draft.mtp_token_map[reference.argmax(-1)]
        torch.testing.assert_close(tokens, expected)
        torch.testing.assert_close(logits.argmax(-1), expected)
        assert torch.isneginf(logits[:, [0, 1, 3, 5, 7]]).all()


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
def test_qwen4_exp_mtp_remaps_mixed_precision_layer_indices(backend: str) -> None:
    mtp_module = import_module(f"vllm.models.qwen4_exp.{backend}.mtp")

    quantized_layers = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.0.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }

    assert mtp_module._remap_quantized_layers(quantized_layers, 48) == {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.48.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


@pytest.mark.parametrize("ple_layer_ids", [[1], []])
def test_qwen4_exp_accepts_pipeline_parallel_with_ple(ple_layer_ids) -> None:
    """PLE consumes the raw IDs now available on every pipeline rank."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(ple_layer_ids=ple_layer_ids),
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2, enable_dbo=False, ubatch_size=1
        ),
        speculative_config=None,
    )
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 1
    input_batch.idx_mapping = torch.tensor([0])
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)
        # Dummy runs establish the addresses used during CUDA graph capture.
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)

    expected_query_start_loc = torch.full((9,), 4, dtype=torch.int32)
    expected_query_start_loc[:4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    torch.testing.assert_close(second["query_start_loc"], expected_query_start_loc)
    torch.testing.assert_close(
        second["ngram_context"], torch.full((8, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr
