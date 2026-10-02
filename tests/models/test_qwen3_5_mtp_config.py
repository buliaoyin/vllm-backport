# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 MTP speculative decoding config overrides."""

from types import SimpleNamespace
from typing import Any

import pytest
from transformers import AutoConfig, PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm.v1.spec_decode.dynamic.adaptive import supports_adaptive_mtp

_CHECKPOINTS = {
    "qwen3_5": "Qwen/Qwen3.8-27B",
    "qwen3_5_moe": "Qwen/Qwen3.6-35B-A3B",
}


def _mtp_config(model_type: str) -> PretrainedConfig:
    """Create a top-level MTP configuration with mtp_num_hidden_layers."""
    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "architectures": ["SomeArch"],
        "mtp_num_hidden_layers": 1,
    }
    return PretrainedConfig(**kwargs)


def _multimodal_wrapper_mtp_config(
    model_type: str, mtp_layers: int = 1
) -> PretrainedConfig:
    """Download a multimodal wrapper config via AutoConfig.from_pretrained
    and configure mtp_num_hidden_layers in text_config.

    Uses real-world Hugging Face Hub checkpoints:
    - Dense (qwen3_5):     Qwen/Qwen3.8-27B
    - MoE   (qwen3_5_moe): Qwen/Qwen3.6-35B-A3B
    """
    repo = _CHECKPOINTS[model_type]
    config: PretrainedConfig = AutoConfig.from_pretrained(repo)
    text_config = config.get_text_config()
    text_config.mtp_num_hidden_layers = mtp_layers
    return config


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        # Text-only config variants must map to the same MTP architectures.
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(
    model_type: str, expected_arch: str
) -> None:
    """Verify that text-only config variants map to the expected MTP architectures."""
    cfg = SpeculativeConfig.hf_config_override(_mtp_config(model_type))
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("layers", [0, 1, 2])
def test_adaptive_mtp_requires_one_resolved_qwen_layer(wrapped, layers):
    """The scheduler accepts a single MTP layer in text and wrapper configs."""
    if wrapped:
        cfg = Qwen3_5Config(
            text_config={"mtp_num_hidden_layers": layers},
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
    else:
        cfg = _mtp_config("qwen3_5_text")
        cfg.mtp_num_hidden_layers = layers
    draft = SpeculativeConfig.hf_config_override(cfg)
    spec = SimpleNamespace(
        method="mtp", draft_model_config=SimpleNamespace(hf_config=draft)
    )
    assert supports_adaptive_mtp(spec) == (layers == 1)


def test_adaptive_mtp_excludes_other_models_using_qwen_draft_type():
    """A shared model_type must not enable an unvalidated MTP architecture."""
    draft = SpeculativeConfig.hf_config_override(_mtp_config("qwen3_5"))
    draft.architectures = ["InternS2MobiusMTP"]
    spec = SimpleNamespace(
        method="mtp", draft_model_config=SimpleNamespace(hf_config=draft)
    )
    assert not supports_adaptive_mtp(spec)


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_extracts_n_predict_from_multimodal_wrapper(
    model_type: str, expected_arch: str
) -> None:
    """Verify that multimodal wrapper checkpoints with mtp_num_hidden_layers
    in text_config resolve n_predict and architecture correctly."""
    cfg = SpeculativeConfig.hf_config_override(
        _multimodal_wrapper_mtp_config(model_type, mtp_layers=2)
    )
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 2


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_top_level_precedence_over_nested_text_config(
    model_type: str, expected_arch: str
) -> None:
    """Verify that an explicit top-level mtp_num_hidden_layers takes precedence
    over a nested text_config value."""
    cfg = _multimodal_wrapper_mtp_config(model_type, mtp_layers=2)
    cfg.mtp_num_hidden_layers = 3
    overridden = SpeculativeConfig.hf_config_override(cfg)
    assert overridden.model_type == "qwen3_5_mtp"
    assert overridden.architectures == [expected_arch]
    assert overridden.n_predict == 3


@pytest.mark.parametrize(
    "model_id,expected_arch",
    [
        ("Qwen/Qwen3.8-27B", "Qwen3_5MTP"),
        ("Qwen/Qwen3.6-35B-A3B", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_downloads_real_hf_hub_configs(
    model_id: str, expected_arch: str
) -> None:
    """Verify that unmodified real-world checkpoints downloaded via
    AutoConfig.from_pretrained resolve n_predict=1 from text_config."""
    hf_config = AutoConfig.from_pretrained(model_id)
    cfg = SpeculativeConfig.hf_config_override(hf_config)
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1
