# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import (
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger

logger = init_logger("test_expert_parallel")


def test_get_kv_connector_cache_layout_without_kv_connector():
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout is None


def test_get_kv_connector_cache_layout_with_lmcache_connector():
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"), kv_transfer_config=kv_transfer_config
    )
    with set_current_vllm_config(vllm_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout is None


def test_get_kv_connector_cache_layout_with_nixl_connector():
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    model_config = ModelConfig()
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        model_config=model_config,
        kv_transfer_config=kv_transfer_config,
    )
    with set_current_vllm_config(vllm_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout == "LBHNC"


def test_get_kv_connector_cache_layout_with_multi_connector():
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "ExampleConnector", "kv_role": "kv_both"},
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
            ]
        },
    )
    model_config = ModelConfig()
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        model_config=model_config,
        kv_transfer_config=kv_transfer_config,
    )
    with set_current_vllm_config(vllm_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout == "LBHNC"


@pytest.mark.parametrize("requested", [None, "BLNHC"])
def test_pipeline_stages_resolve_a_layout_supported_by_every_backend(
    monkeypatch, requested
):
    """SWA-only PP stages must not prevent sharing a compressed-KV layout."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    if requested:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", requested)
    config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(pipeline_parallel_size=4),
    )
    supported = [
        ["LBNHC", "LBHNC", "BLNHC", "BLHNC", "BHLNC", "LHBNC"],
        *[["BLHNC", "BLNHC"]] * 3,
    ]
    layout = resolve_kv_cache_layout(config, supported)
    assert layout.name == (requested or "BLHNC")
    assert config.cache_config.kv_cache_layout == layout.name


@pytest.mark.parametrize(
    ("supported", "requested"),
    [([["LBNHC"], ["BLHNC"]], None), ([["LBNHC", "BLHNC"], ["BLHNC"]], "LBNHC")],
)
def test_pipeline_layout_rejects_incompatible_worker_or_explicit_choice(
    monkeypatch, supported, requested
):
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    if requested:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", requested)
    config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(pipeline_parallel_size=2),
    )
    with pytest.raises(ValueError, match="supported set"):
        resolve_kv_cache_layout(config, supported)
    assert config.cache_config.kv_cache_layout is None


def test_tensor_parallel_workers_still_require_identical_layout_preferences():
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig("cpu"))
    with pytest.raises(AssertionError, match="Workers disagree"):
        resolve_kv_cache_layout(config, [["BLHNC", "BLNHC"], ["BLHNC"]])
