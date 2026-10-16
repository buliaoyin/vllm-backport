# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CPU MoE numerics, loading, and GPU graph-boundary correctness."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4 import (
    CPUMoEConfig,
    CPUMXFP4Experts,
)


def test_static_cache_selects_capacity_and_device_per_layer():
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v4_1.cpu_moe import cpu_moe_config

    settings = {
        "backend": "ik",
        "cuda_library_path": "unused.so",
        "gpu_cache_experts": 8,
        "gpu_cache_device": 1,
        "gpu_cache_selections": {"20": [0, 2], "21": [1, 3, 5]},
        "gpu_cache_devices": {"20": 0, "21": 2},
    }
    config = SimpleNamespace(
        additional_config={"cpu_moe": settings},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=40)),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    first, second = (cpu_moe_config(config, layer) for layer in (20, 21))
    assert (first.gpu_cache_experts, first.gpu_cache_device) == (2, 0)
    assert (second.gpu_cache_experts, second.gpu_cache_device) == (3, 2)
    assert first.gpu_cache_static_experts == (0, 2)
    assert second.gpu_cache_static_experts == (1, 3, 5)
    assert settings["gpu_cache_experts"] == 8
    assert settings["gpu_cache_device"] == 1
    assert cpu_moe_config(config, 19) is None
    settings["gpu_cache_prefill"] = True
    settings["gpu_cache_dynamic"] = {"20": {"mutable_experts": [2]}}
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    assert cpu_moe_config(config, 20).gpu_cache_dynamic == {"mutable_experts": [2]}
    assert cpu_moe_config(config, 21).gpu_cache_dynamic is None
    config.scheduler_config.max_num_seqs = 2
    assert cpu_moe_config(config, 20).gpu_cache_dynamic == {"mutable_experts": [2]}
    config.scheduler_config.max_num_seqs = 1
    with pytest.raises(ValueError, match="Missing static expert selection"):
        cpu_moe_config(config, 22)


@pytest.mark.parametrize("cache_env", ["VLLM_CACHE_ROOT", "XDG_CACHE_HOME"])
@pytest.mark.parametrize("explicit_path", [False, True])
def test_expert_cache_profile_resolves_path_and_restores_after_restart(
    tmp_path, monkeypatch, cache_env, explicit_path
):
    from vllm.models.deepseek_v4_1.cache_profile import (
        ExpertCacheProfile,
        model_fingerprint,
    )

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    root = tmp_path / "xdg" / "vllm"
    if cache_env == "VLLM_CACHE_ROOT":
        root = tmp_path / "cache"
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(root))
    config = SimpleNamespace(
        model="test-checkpoint",
        revision=None,
        hf_config=SimpleNamespace(to_dict=lambda: {"n_routed_experts": 4}),
    )
    expected = root / "expert_profiles" / "deepseek_v41"
    expected /= f"{model_fingerprint(config)}.json"
    settings = {}
    if explicit_path:
        expected = tmp_path / "custom.json"
        settings["expert_profile"] = str(expected)
    profile = ExpertCacheProfile.from_config(settings, config, 4, [20])
    assert profile.path == expected
    state = {"history": [0.0, 1.0, 0.0, 0.0], "requests": 1, "mass": 1}
    profile.save([(20, SimpleNamespace(selected=[1], learning_state=lambda: state))])
    restarted = ExpertCacheProfile.from_config(settings, config, 4, [20])
    assert restarted.load() == {"20": {"selected": [1], **state}}
    config.revision = "different-checkpoint"
    different = ExpertCacheProfile.from_config(settings, config, 4, [20])
    if not explicit_path:
        assert different.path != profile.path
    assert different.load() == {}


def test_expert_cache_profile_disable_skips_checkpoint_access():
    from vllm.models.deepseek_v4_1.cache_profile import ExpertCacheProfile

    assert (
        ExpertCacheProfile.from_config({"expert_profile": False}, None, 4, [20]) is None
    )


def test_expert_cache_profile_restores_demand_and_adjusts_resident_capacity(tmp_path):
    from vllm.models.deepseek_v4_1.cache_profile import ExpertCacheProfile

    profile = ExpertCacheProfile(tmp_path / "profile.json", "checkpoint", 4, [20])
    state = {"history": [0.0, 0.5, 0.25, 1.0], "requests": 2, "mass": 1.95}
    cache = SimpleNamespace(
        selected=[1, 3],
        learning_state=lambda: {**state, "pending": [900], "decode_totals": [800]},
    )
    assert profile.load() == {}
    profile.save([(20, cache)])
    entries = profile.load()
    assert entries == {"20": {"selected": [1, 3], **state}}
    assert profile.selection(entries["20"], 2) == [1, 3]
    assert profile.selection(entries["20"], 1) == [3]
    assert profile.selection(entries["20"], 3) == [1, 3, 2]
    assert list(tmp_path.iterdir()) == [profile.path]


def test_expert_cache_profile_limits_writes_and_preserves_previous_on_failure(
    tmp_path, monkeypatch
):
    import json

    from vllm.models.deepseek_v4_1.cache_profile import ExpertCacheProfile

    profile = ExpertCacheProfile(tmp_path / "profile.json", "checkpoint", 4, [20], 2)
    cache = SimpleNamespace(
        selected=[0],
        learning_state=lambda: {"history": [1.0, 0, 0, 0], "requests": 1, "mass": 1},
    )
    profile.save([(20, cache)])
    original = profile.path.read_bytes()
    cache.selected = [1]
    profile.save([(20, cache)])
    assert profile.path.read_bytes() == original
    profile.save([(20, cache)])
    assert json.loads(profile.path.read_text())["layers"]["20"]["selected"] == [1]
    previous = profile.path.read_bytes()

    def failed_replace(*args):
        raise OSError("disk full")

    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.cache_profile.os.replace", failed_replace
    )
    profile.save([(20, cache)])
    profile.save([(20, cache)])
    assert profile.path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [profile.path]

    def failed_cleanup(*args, **kwargs):
        raise PermissionError("temporary directory permissions changed")

    monkeypatch.setattr(Path, "unlink", failed_cleanup)
    profile.save([(20, cache)])
    profile.save([(20, cache)])
    assert profile.path.read_bytes() == previous


@pytest.mark.parametrize("corruption", ["fingerprint", "duplicate", "nan", "large"])
def test_expert_cache_profile_ignores_stale_or_corrupt_demand(tmp_path, corruption):
    import json

    from vllm.models.deepseek_v4_1.cache_profile import (
        MAX_PROFILE_BYTES,
        ExpertCacheProfile,
    )

    profile = ExpertCacheProfile(tmp_path / "profile.json", "checkpoint", 4, [20])
    entry = {"selected": [0, 1], "history": [1, 0, 0, 0], "requests": 1, "mass": 1}
    saved = {"version": 1, "fingerprint": "checkpoint", "layers": {"20": entry}}
    profile.path.write_text(json.dumps(saved))
    assert profile.load() == {"20": entry}
    if corruption == "fingerprint":
        saved["fingerprint"] = "other-checkpoint"
    elif corruption == "duplicate":
        entry["selected"] = [0, 0]
    elif corruption == "nan":
        entry["history"] = [float("nan"), 0, 0, 0]
    if corruption == "large":
        profile.path.write_bytes(b" " * (MAX_PROFILE_BYTES + 1))
    else:
        profile.path.write_text(json.dumps(saved))
    assert profile.load() == {}
    invalid_data = profile.path.read_bytes()
    profile.save([])
    assert profile.path.read_bytes() == invalid_data


def test_expert_cache_profile_fingerprint_tracks_checkpoint_revision_and_files(
    tmp_path,
):
    from vllm.models.deepseek_v4_1.cache_profile import model_fingerprint

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"original")
    config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        hf_config=SimpleNamespace(to_dict=lambda: {"n_routed_experts": 4}),
    )
    initial = model_fingerprint(config)
    assert model_fingerprint(config) == initial
    config.hf_config._commit_hash = "another-resolved-revision"
    assert model_fingerprint(config) != initial
    config.hf_config._commit_hash = None
    shard.write_bytes(b"replaced-checkpoint")
    assert model_fingerprint(config) != initial


def test_expert_cache_profile_preserves_inactive_layer_history(tmp_path):
    """A layer with no resident slots retains its saved learning state."""
    from vllm.models.deepseek_v4_1.cache_profile import ExpertCacheProfile

    profile = ExpertCacheProfile(tmp_path / "profile.json", "checkpoint", 33, [20, 21])
    caches = [
        SimpleNamespace(
            selected=list(range(16)),
            learning_state=lambda history=history: {
                "history": history,
                "requests": 1,
                "mass": 1,
            },
        )
        for history in ([1.0] * 32 + [0.0], [0.0] * 33)
    ]
    profile.save(zip((20, 21), caches))
    profile = ExpertCacheProfile(profile.path, "checkpoint", 33, [20, 21], interval=1)
    initial = profile.load()
    caches[0].selected = list(range(32))
    profile.save([(20, caches[0])])
    restored = profile.load()
    assert restored["20"]["selected"] == list(range(32))
    assert restored["21"] == initial["21"]


@pytest.mark.parametrize(
    "rank,requests,mode,valid",
    [
        (4, 1, "FULL_DECODE_ONLY", True),
        (6, 1, "FULL_DECODE_ONLY", False),
        (4, 2, "FULL_DECODE_ONLY", True),
        (4, 8, "FULL_DECODE_ONLY", True),
        (4, 32, "FULL_DECODE_ONLY", False),
        (4, 1, "FULL", False),
        (4, 8, "NONE", False),
    ],
)
def test_route_tail_skip_requires_model_topk_and_decode_graph(
    rank, requests, mode, valid
):
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v4_1.cpu_moe import cpu_moe_config

    config = SimpleNamespace(
        additional_config={
            "cpu_moe": {
                "backend": "ik",
                "cuda_library_path": "unused.so",
                "route_tail_skip": rank,
            }
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=40, num_experts_per_tok=6)
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode[mode],
            max_cudagraph_capture_size=max(16, requests * 8),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=requests),
        uniform_decode_query_len=8,
    )
    if valid:
        assert cpu_moe_config(config, 20).route_tail_skip == rank
    else:
        with pytest.raises(ValueError, match="route_tail_skip"):
            cpu_moe_config(config, 20)


@pytest.mark.parametrize("draft_tokens", [3, 5, 7])
def test_cpu_callbacks_accept_reachable_decode_graphs(draft_tokens):
    """Unused capture candidates must not reject a supported 16-request batch."""
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v4_1.cpu_moe import (
        cpu_moe_config,
        max_callback_tokens,
    )

    config = SimpleNamespace(
        additional_config={
            "cpu_moe": {"backend": "ik", "cuda_library_path": "unused.so"}
        },
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=40)),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=32 * (draft_tokens + 1),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        uniform_decode_query_len=draft_tokens + 1,
    )
    assert cpu_moe_config(config, 20) is not None
    assert max_callback_tokens(config) == 16 * (draft_tokens + 1)
    config.scheduler_config.max_num_seqs = 64
    config.compilation_config.max_cudagraph_capture_size = 256
    with pytest.raises(ValueError, match="CPU MoE needs"):
        cpu_moe_config(config, 20)


def test_tail_cache_admission_preserves_pinned_experts_and_pays_for_transfers():
    """Deduplicate routes, ignore padding, and reject unprofitable swaps."""
    import numpy as np

    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy

    policy = TailCachePolicy(
        mutable_experts=(1, 2),
        group_size=4,
        future_tokens=64,
        cpu_call_ms=1,
        transfer_ms=2,
        max_swaps=2,
        budget_ms=2,
        min_tokens=4,
    )
    ids = np.array([[3, 3, 4]] * 4 + [[3, 3, 2]] * 4 + [[-1, -1, -1]] * 8)
    selected, saving, counts = policy.plan(ids, [0, 1, 2], {0}, 5)
    assert counts == [0, 0, 1, 2, 1]
    assert selected == [0, 3, 2]
    assert saving == 16
    assert policy.plan(ids, [0, 3, 2], {0}, 5)[0] == [0, 3, 2]
    assert replace(policy, transfer_ms=20, budget_ms=40).plan(ids, [0, 1, 2], {0}, 5)[
        0
    ] == [0, 1, 2]
    assert policy.plan(np.full((8, 3), -1), [0, 1, 2], {0}, 5)[0] == [0, 1, 2]
    assert policy.plan(ids[:3], [0, 1, 2], {0}, 5)[0] == [0, 1, 2]
    assert replace(policy, mutable_experts=(0, 1, 2)).plan(ids, [0, 1, 2], set(), 5)[
        0
    ] == [3, 1, 2]


def test_decode_history_admits_experts_absent_from_tail_within_copy_budget():
    import numpy as np

    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy

    policy = TailCachePolicy(
        mutable_experts=(0, 1),
        future_tokens=64,
        cpu_call_ms=1,
        transfer_ms=1,
        feedback_weight=1,
        feedback_budget_ms=4,
    )
    ids = np.array([[0, 2]] * 128)
    assert policy.plan(ids, [0, 1], set(), 4)[0] == [0, 2]
    selected, _, _ = policy.plan(
        ids,
        [0, 1],
        set(),
        4,
        history=np.array([0, 0.01, 0.05, 0.8]),
        costs=np.array([1, 1, 10, 1]),
        feedback=True,
    )
    assert selected == [3, 1]


def test_decode_history_normalizes_early_requests_and_ignores_empty_samples():
    import numpy as np

    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy

    policy = TailCachePolicy((0,), feedback_decay=0.95, feedback_debias=True)
    history, mass = policy.update_history(None, 0, [10, 0], 10)
    history, mass = policy.update_history(history, mass, [0, 20], 20)
    np.testing.assert_allclose(history, np.array([0.95, 1.0]) / 1.95)
    assert mass == 1.95
    unchanged, same_mass = policy.update_history(history, mass, [0, 0], 0)
    assert unchanged is history and same_mass == mass


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_dynamic_cache_rejects_invalid_host_memory_limits(invalid):
    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy

    with pytest.raises(ValueError, match="Host cache byte limit"):
        TailCachePolicy((0,), host_cache_bytes=invalid)


@pytest.mark.parametrize("horizon, expected", [(1, [0, 1]), (8, [0, 2])])
def test_decode_history_amortizes_reusable_experts_across_requests(horizon, expected):
    import numpy as np

    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy

    policy = TailCachePolicy(
        (1,),
        group_size=4,
        cpu_call_ms=0.3,
        feedback_weight=1.0,
        expected_tokens_per_step=2.5,
        feedback_horizon_requests=horizon,
    )
    selected, _, _ = policy.plan(
        np.zeros((128, 1), dtype=np.int64),
        [0, 1],
        {0},
        4,
        history=np.array([0.9, 0, 0.1, 0]),
        costs=np.full(4, 3.2),
        feedback=True,
    )
    assert selected == expected


@pytest.mark.parametrize("tokens", [1, 4, 16, 64, 128])
@pytest.mark.parametrize("num_experts", [8, 384])
@pytest.mark.parametrize("device", [0, 1])
def test_decode_feedback_preserves_routes_and_counts_graph_replays(
    tokens, num_experts, device
):
    from vllm.models.deepseek_v4_1.cache_feedback import DecodeRouteFeedback

    if torch.accelerator.device_count() <= device:
        pytest.skip("The requested CUDA device is unavailable")
    torch.accelerator.set_device_index(device)
    generator = torch.Generator(device="cpu").manual_seed(79)
    original = torch.randint(-2, num_experts + 1, (tokens, 6), generator=generator)
    original[0, :3] = torch.tensor([num_experts - 1, num_experts - 1, 0])
    ids = original.to("cuda")
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    feedback = DecodeRouteFeedback(num_experts, ids.device)
    feedback.set_enabled(True)
    feedback.record(ids, padding)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=ids.device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=stream):
        feedback.record(ids, padding)
    feedback.counts.fill_(2**32)
    expected = feedback.counts.cpu()
    addresses = (feedback.counts.data_ptr(), feedback.enabled.data_ptr())
    for enabled, valid_rows in [(False, tokens), (True, tokens), (True, 0), (True, 1)]:
        padding.copy_(torch.arange(tokens, device="cuda") >= valid_rows)
        feedback.set_enabled(enabled)
        graph.replay()
        if enabled:
            valid = original[:valid_rows]
            in_range = (valid >= 0) & (valid < num_experts)
            counts = torch.bincount(valid[in_range], minlength=num_experts)
            expected[:num_experts] += counts
            expected[num_experts : 2 * num_experts] += counts != 0
            actual = int(in_range.any(dim=1).sum())
            expected[-2] += actual > 0
            expected[-1] += actual
    torch.accelerator.synchronize()
    torch.testing.assert_close(feedback.counts.cpu(), expected)
    torch.testing.assert_close(ids.cpu(), original)
    assert (feedback.counts.data_ptr(), feedback.enabled.data_ptr()) == addresses


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_route_diagnostics_preserve_ids_and_count_reuse_without_padding(device):
    """Cache selection needs full routes, unique expert calls, and phase separation."""
    from benchmarks.kernels.cpu.dsv41_prefill_diagnostics import ExpertRouteHistogram

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    histogram = ExpertRouteHistogram(8, device)
    ids = torch.tensor([[0, 3, 7], [3, 3, -1], [7, -2, 8], [2, 6, 6]], device=device)
    original = ids.clone()
    padding = torch.tensor([False, False, True, True], device=device)
    histogram.record(ids, padding)
    histogram.record(ids, padding)
    replay_ids = torch.tensor([1, 5, 5], device=device).expand(128, -1)
    replay_padding = torch.zeros(128, dtype=torch.bool, device=device)
    replay_padding[0] = True
    histogram.record(replay_ids, replay_padding, actual_tokens=17)
    result = histogram.snapshot()
    assert result["decode"] == {
        "route_counts": [2, 0, 0, 6, 0, 0, 0, 2],
        "expert_calls": [2, 0, 0, 2, 0, 0, 0, 2],
        "tokens": 4,
        "calls": 2,
    }
    assert result["prefill"] == {
        "route_counts": [0, 16, 0, 0, 0, 32, 0, 0],
        "expert_calls": [0, 1, 0, 0, 0, 1, 0, 0],
        "tokens": 16,
        "calls": 1,
    }
    torch.testing.assert_close(ids, original)
    assert torch.equal(replay_ids[0], torch.tensor([1, 5, 5], device=device))


@pytest.fixture(params=["kt", "llama", "ik"])
def backend_config(request):
    name = request.param
    path = os.environ.get(f"DSV41_TEST_{name.upper()}_LIBRARY")
    if not path or not Path(path).is_file():
        pytest.skip(f"Build the optional {name} CPU backend and set its test library")
    return CPUMoEConfig(backend=name, num_threads=4, library_path=path)


def load_weights(backend, gpu_cache=None, *, seed=13):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    reference_weights = {}
    for e in range(backend.num_experts):
        for p in range(3):
            rows, cols = backend.intermediate_size, backend.hidden_size
            if p == 1:
                rows, cols = cols, rows
            packed = torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
            )
            exponents = torch.randint(
                120, 124, (rows, cols // 32), dtype=torch.uint8, generator=generator
            )
            backend.load_expert(e, p, packed, exponents)
            if gpu_cache is not None:
                gpu_cache.load_weight(e, p, packed, exponents)
            codes = torch.empty((rows, cols), dtype=torch.long)
            codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
            scales = (2.0 ** (exponents.int() - 127)).repeat_interleave(32, -1)
            reference_weights[e, p] = values[codes] * scales
    backend.prepare()
    return reference_weights


@pytest.mark.parametrize("numa_partitions", [0, 4], indirect=True)
@pytest.mark.parametrize("hidden,intermediate", [(256, 128), (5120, 2304)])
def test_native_export_preserves_bytes_after_checkpoint_is_released(
    backend_config, hidden, intermediate, numa_partitions
):
    """GPU refills must need only resident weights, including exact scale bytes."""
    if backend_config.backend != "ik":
        pytest.skip("Resident export uses the IK R8 layout")
    backend = CPUMXFP4Experts(
        backend_config, 2, hidden, intermediate, 1, swiglu_limit=10.0, max_tokens=1
    )
    try:
        if not backend.supports_export:
            pytest.skip("Native library lacks resident export")
        generator = torch.Generator().manual_seed(82)
        for expert in range(2):
            for projection in range(3):
                rows, cols = intermediate, hidden
                if projection == 1:
                    rows, cols = cols, rows
                weight = torch.randint(
                    256, (rows, cols // 2), dtype=torch.uint8, generator=generator
                )
                scales = torch.randint(
                    256, (rows, cols // 32), dtype=torch.uint8, generator=generator
                )
                expected = weight.clone(), scales.clone()
                backend.load_expert(expert, projection, weight, scales)
                weight.zero_()
                scales.zero_()
                restored = backend.export_expert(expert, projection)
                for actual, reference_bytes in zip(restored, expected):
                    assert torch.equal(actual, reference_bytes)
                    actual.zero_()
                for actual, reference_bytes in zip(
                    backend.export_expert(expert, projection), expected
                ):
                    assert torch.equal(actual, reference_bytes)
        if backend.supports_r8_staging:
            staged = torch.empty(
                3 * hidden * intermediate * 17 // 32, dtype=torch.uint8
            )
            for expert in range(2):
                staged.fill_(0xAA)
                backend.stage_expert_r8(expert, staged)
                expected_r8 = staged.clone()
                staged.fill_(0x55)
                backend.stage_expert_r8(expert, staged)
                torch.testing.assert_close(staged, expected_r8, atol=0, rtol=0)
            with pytest.raises(ValueError, match="unloaded"):
                backend.stage_expert_r8(2, staged)
            with pytest.raises(ValueError, match="byte buffer"):
                backend.stage_expert_r8(0, staged[:-1])
        with pytest.raises(ValueError, match="unloaded"):
            backend.export_expert(2, 0)
    finally:
        backend.close()


@pytest.mark.parametrize("cache_device", [0, 1])
@pytest.mark.parametrize("hidden,intermediate", [(64, 64), (128, 192), (5120, 2304)])
def test_r8_refill_preserves_marlin_bytes_across_shared_staging_streams(
    backend_config, hidden, intermediate, cache_device
):
    """Two streams reusing a bounded pool must retain every expert's packed bytes."""
    if (
        backend_config.backend != "ik"
        or not torch.cuda.is_available()
        or torch.accelerator.device_count() <= cache_device
    ):
        pytest.skip("Native R8 refill requires IK and the requested CUDA device")
    from vllm.models.deepseek_v4_1.expert_cache import GPUExpertCache
    from vllm.models.deepseek_v4_1.r8_refill import R8Refill

    backend = CPUMXFP4Experts(
        backend_config, 4, hidden, intermediate, 2, swiglu_limit=10.0, max_tokens=4
    )
    try:
        if not backend.supports_r8_staging:
            pytest.skip("Build the native R8 staging interface")
        generator = torch.Generator().manual_seed(93)
        for expert in range(4):
            for projection in range(3):
                rows, cols = intermediate, hidden
                if projection == 1:
                    rows, cols = cols, rows
                weight = torch.randint(
                    256, (rows, cols // 2), dtype=torch.uint8, generator=generator
                )
                scale = torch.randint(
                    256, (rows, cols // 32), dtype=torch.uint8, generator=generator
                )
                backend.load_expert(expert, projection, weight, scale)
        cache = GPUExpertCache(
            2, 4, cache_device, 10.0, 2, weight_source=backend.export_expert
        )
        origin = torch.accelerator.current_device_index()
        refills = [R8Refill(backend, cache.device) for _ in range(2)]
        assert refills[0].workspace is refills[1].workspace
        assert torch.accelerator.current_device_index() == origin
        streams = [cache.stream, torch.cuda.Stream(device=cache.device)]
        results = []
        for step in range(8):
            with torch.cuda.stream(streams[step % 2]):
                results.append(
                    (
                        refills[step % 2].pack(step % 4),
                        cache._pack_source_expert(step % 4),
                    )
                )
        torch.accelerator.synchronize(cache.device.index)
        for actual, expected in results:
            for a, b in zip(actual, expected):
                torch.testing.assert_close(
                    a.view(torch.uint8), b.view(torch.uint8), atol=0, rtol=0
                )
    finally:
        backend.close()


def reference(x, ids, routes, weights, limit):
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot, e in enumerate(ids[token].tolist()):
            gate = weights[e, 0] @ x[token].float()
            up = weights[e, 2] @ x[token].float()
            if limit > 0:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            activation = gate.sigmoid() * gate * up
            output[token] += routes[token, slot] * (weights[e, 1] @ activation)
    return output


@pytest.mark.parametrize("limit", [0.0, 0.125])
def test_native_moe_preserves_routing_and_clipped_swiglu(backend_config, limit):
    backend = CPUMXFP4Experts(backend_config, 4, 128, 96, 3, limit, 32)
    weights = load_weights(backend)
    generator = torch.Generator().manual_seed(17)
    try:
        for tokens in (1, 7, 1):
            x = torch.randn(tokens, 128, generator=generator).bfloat16()
            ids = torch.stack(
                [torch.randperm(4, generator=generator)[:3] for _ in range(tokens)]
            )
            routes = torch.rand(tokens, 3, generator=generator) * 1.7
            expected = reference(x, ids, routes, weights, limit)
            actual = backend.forward(x, ids, routes).float()
            assert torch.isfinite(actual).all()
            relative_rms = (actual - expected).norm() / expected.norm()
            assert relative_rms < 0.035, relative_rms.item()
            if backend_config.backend != "kt" and hasattr(
                backend._library, "dsv41_moe_set_threads"
            ):
                for threads in (2, 1, 4):
                    backend.set_num_threads(threads)
                    retuned = backend.forward(x, ids, routes).float()
                    torch.testing.assert_close(retuned, actual, atol=1e-5, rtol=1e-5)
                with pytest.raises(ValueError, match="positive"):
                    backend.set_num_threads(0)
    finally:
        backend.close()


@pytest.mark.parametrize("limit", [0.0, 0.125])
@pytest.mark.parametrize("intermediate", [96, 128, 160, 224])
def test_ik_compact_matches_graph_with_cached_routes(
    backend_config, limit, intermediate
):
    # Guard route compaction, quantization tails and the large-batch fallback.
    if backend_config.backend != "ik":
        pytest.skip("The compact executor uses IQK kernels")
    backend = CPUMXFP4Experts(backend_config, 4, 256, intermediate, 3, limit, 129)
    load_weights(backend)
    generator = torch.Generator().manual_seed(93)
    try:
        for tokens in (1, 4, 16, 17, 32, 64, 65, 128, 129, 1):
            x = torch.randn(tokens, 256, generator=generator).bfloat16()
            ids = torch.stack(
                [torch.randperm(4, generator=generator)[:3] for _ in range(tokens)]
            )
            routes = torch.rand(tokens, 3, generator=generator)
            for cached in (0, 1, 2, 3):
                selected = ids.clone()
                selected[:, :cached] = -2
                backend.set_execution_mode("graph")
                expected = backend.forward(x, selected, routes)
                backend.set_execution_mode("compact")
                for schedule in (-1, 0, 2, 7, 8, 15, 16, 24, 31):
                    backend.set_schedule(schedule)
                    for threads in (1, 4):
                        backend.set_num_threads(threads)
                        actual = backend.forward(x, selected, routes)
                        if backend._numa_enabled and tokens > 128:
                            # NUMA chunks can select GEMV for the final token.
                            torch.testing.assert_close(
                                actual, expected, atol=1e-7, rtol=1e-5
                            )
                        else:
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                if tokens > 128:
                    small = backend.forward(x[:1], selected[:1], routes[:1])
                    backend.set_execution_mode("graph")
                    expected_small = backend.forward(x[:1], selected[:1], routes[:1])
                    torch.testing.assert_close(small, expected_small, atol=0, rtol=0)
        with pytest.raises(ValueError, match="graph or compact"):
            backend.set_execution_mode("unknown")
    finally:
        backend.close()


@pytest.mark.parametrize("packed", [0x77, 0xFF])
def test_ik_avx2_preserves_extreme_quantized_dot_products(backend_config, packed):
    if backend_config.backend != "ik":
        pytest.skip("The AVX2 compact executor uses IK")
    backend = CPUMXFP4Experts(backend_config, 1, 128, 128, 1, 0.125, 16)
    try:
        weight = torch.full((128, 64), packed, dtype=torch.uint8)
        scales = torch.full((128, 4), 127, dtype=torch.uint8)
        for projection in range(3):
            backend.load_expert(0, projection, weight, scales)
        backend.prepare()
        ids = torch.zeros((1, 1), dtype=torch.int64)
        routes = torch.ones((1, 1))
        for sign in (1.0, -1.0):
            x = torch.full((1, 128), sign, dtype=torch.bfloat16)
            backend.set_execution_mode("graph")
            expected = backend.forward(x, ids, routes)
            backend.set_execution_mode("compact")
            for schedule in (-1, 8, 15, 24, 31):
                backend.set_schedule(schedule)
                actual = backend.forward(x, ids, routes)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        backend.close()


def test_ik_profile_counts_group_reuse_and_preserves_inputs(backend_config, tmp_path):
    if backend_config.backend != "ik":
        pytest.skip("CPU diagnostics use the compact IQK executor")
    backend = CPUMXFP4Experts(backend_config, 4, 256, 160, 3, 10.0, 4)
    load_weights(backend)
    x = torch.randn(4, 256).bfloat16()
    ids = torch.tensor([[0, 1, -2], [0, 2, -2], [0, 2, 3], [0, -2, -2]])
    routes = torch.rand(4, 3)
    try:
        expected = backend.forward(x, ids, routes)
        backend.set_profile(1)
        torch.testing.assert_close(
            backend.forward(x, ids, routes), expected, atol=0, rtol=0
        )
        cached = backend.forward(x, torch.full_like(ids, -2), routes)
        assert torch.count_nonzero(cached) == 0
        trace_path = tmp_path / "trace.bin"
        stats = backend.profile_stats(trace_path)
        assert (stats["calls"], stats["cpu_routes"], stats["expert_groups"]) == (
            2,
            8,
            4,
        )
        assert stats["expert_group_size_histogram"] == [2, 1, 0, 1] + [0] * 12
        assert stats["all_cached_calls"] == 1 and stats["sampled_calls"] == 1
        data = bytearray(trace_path.read_bytes())
        assert data[:8] == b"DSV41TR1" and int.from_bytes(data[8:12], "little") == 2
        recorded = torch.frombuffer(
            data, dtype=torch.float32, count=x.numel(), offset=24
        )
        torch.testing.assert_close(recorded.reshape(x.shape), x.float(), atol=0, rtol=0)
        backend.set_profile(0)
        assert backend.profile_stats(trace_path)["calls"] == 0
    finally:
        backend.close()


def test_native_moe_rejects_unloaded_weights(backend_config):
    backend = CPUMXFP4Experts(backend_config, 4, 128, 96, 3, 10.0)
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            backend.prepare()
    finally:
        backend.close()


def test_cpu_experts_skip_padding_and_replay_new_inputs(backend_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is needed for the hybrid graph boundary test")
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") != "1":
        pytest.skip("Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 before importing vLLM")
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

    torch.cuda.init()
    before = torch.accelerator.memory_allocated()
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                n_routed_experts=4,
                hidden_size=128,
                moe_intermediate_size=96,
                num_experts_per_tok=3,
                swiglu_limit=0.125,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=128),
    )
    with torch.device("cuda"):
        module = CPUExpertModule(backend_config, config)
    assert not list(module.parameters())
    assert torch.accelerator.memory_allocated() == before
    load_weights(module.backend)
    metadata = SimpleNamespace(num_actual_tokens=1)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"attention": metadata}, slot_mapping={}
    )
    native_forward = module.backend.forward
    calls = []

    def checked_forward(hidden, ids, routes):
        # Reject the old padded call before IK's native NaN assertion aborts.
        expected_tokens = metadata.num_actual_tokens
        if context.is_padding is not None:
            expected_tokens = (~context.is_padding[:expected_tokens]).sum().item()
        assert hidden.shape[0] == expected_tokens
        calls.append(hidden.shape[0])
        return native_forward(hidden, ids, routes)

    module.backend.forward = checked_forward
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 3]] * 16, device="cuda")
    routes = torch.tensor([[0.2, 0.6, 0.3]] * 16, device="cuda")
    valid_x, valid_ids, valid_routes = x.clone(), ids.clone(), routes.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with override_forward_context(context), torch.cuda.stream(stream):
            x[1:].fill_(float("nan"))
            ids[1:].fill_(-1)
            routes[1:].fill_(float("nan"))
            with torch.inference_mode():
                module(x, ids, routes)
            capture = BreakableCUDAGraphCapture()
            with capture:
                output = module(x, ids, routes) * 2
            for count in (1, 7, 0, 2):
                metadata.num_actual_tokens = count
                x.fill_(float("nan"))
                ids.fill_(-1)
                routes.fill_(float("nan"))
                x[:count].copy_(valid_x[:count] + count * 0.1)
                ids[:count].copy_(valid_ids[:count])
                routes[:count].copy_(valid_routes[:count] * 0.9)
                previous_calls = len(calls)
                capture.replay()
                assert len(calls) == previous_calls + bool(count)
                expected = torch.zeros_like(x)
                if count:
                    expected[:count].copy_(
                        native_forward(
                            x[:count].cpu(), ids[:count].cpu(), routes[:count].cpu()
                        ).to(dtype=x.dtype)
                    )
                torch.accelerator.synchronize()
                torch.testing.assert_close(output, expected * 2, atol=0, rtol=0)
                assert module.last_tokens == count
        # Full-context profiling marks the entire physical batch as padding.
        metadata.num_actual_tokens = x.shape[0]
        context.is_padding = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        x.fill_(float("nan"))
        previous_calls = len(calls)
        with override_forward_context(context):
            result = module(x, ids, routes)
        torch.accelerator.synchronize()
        assert len(calls) == previous_calls
        assert torch.count_nonzero(result) == 0
        # Sparse valid rows must be compacted for the CPU and scattered back.
        valid = torch.tensor([0, 7, 15], device=x.device)
        context.is_padding[valid] = False
        x[valid] = valid_x[valid]
        ids[valid] = valid_ids[valid]
        routes[valid] = valid_routes[valid]
        with override_forward_context(context):
            result = module(x, ids, routes)
        expected = torch.zeros_like(x)
        expected[valid] = native_forward(
            x[valid].cpu(), ids[valid].cpu(), routes[valid].cpu()
        ).to(device=x.device, dtype=x.dtype)
        torch.accelerator.synchronize()
        assert calls[-1] == 3
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        assert capture._num_eager_breaks == 1
    finally:
        torch.accelerator.synchronize()
        module.backend.close()


@pytest.mark.parametrize("numa_partitions", [0, 4], indirect=True)
def test_native_cpu_callback_replays_without_python_and_masks_padding(
    backend_config, numa_partitions
):
    cuda_library = os.environ.get("DSV41_TEST_CUDA_LIBRARY")
    if backend_config.backend == "kt" or not cuda_library:
        pytest.skip("Build the optional CUDA host-callback library for llama/ik")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for host-callback graph replay")
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                n_routed_experts=4,
                hidden_size=128,
                moe_intermediate_size=96,
                num_experts_per_tok=3,
                swiglu_limit=0.125,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=128),
    )
    module = CPUExpertModule(
        replace(backend_config, cuda_library_path=cuda_library), config
    )
    load_weights(module.backend)
    with torch.inference_mode():
        module.finalize()
    for buffer in module.async_host:
        buffer.zero_()
    hidden = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 3]] * 128, device="cuda")
    routes = torch.full((128, 3), 0.5, device="cuda")
    padding = torch.zeros(128, device="cuda", dtype=torch.bool)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={}, is_padding=padding
    )
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with override_forward_context(context), torch.cuda.stream(stream):
            module(hidden, ids, routes)
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                output = module(hidden, ids, routes)
            for valid in (128, 96, 64, 17, 1, 0, 3):
                before = module.backend.cuda_stats()
                hidden.normal_()
                hidden[valid:].fill_(float("nan"))
                padding.fill_(True)
                padding[:valid].fill_(False)
                graph.replay()
                torch.accelerator.synchronize()
                module.backend.check_cuda_errors()
                after = module.backend.cuda_stats()
                if before is not None:
                    assert after["tokens"] - before["tokens"] == valid
                    assert sum(after["expert_counts"]) == after["tokens"] * 3
                    assert after["native_seconds"] > 0
                assert torch.isfinite(output).all()
                assert torch.count_nonzero(output[valid:]) == 0
                if valid:
                    # Graph replay must preserve the direct native arithmetic.
                    expected = module.backend.forward(
                        hidden[:valid].cpu(),
                        ids[:valid].cpu(),
                        routes[:valid].cpu(),
                    ).to(output.dtype)
                    torch.testing.assert_close(
                        output[:valid].cpu(), expected, atol=0, rtol=0
                    )
            ids[0, 1] = 99
            graph.replay()
            torch.accelerator.synchronize()
            with pytest.raises(RuntimeError, match="invalid or unloaded"):
                module.backend.check_cuda_errors()
    finally:
        del graph
        module.backend.close()


@pytest.mark.parametrize("initial_rank", [0, 2])
def test_route_tail_skip_preserves_batch_heads_and_gpu_hits_in_graph_replay(
    backend_config,
    initial_rank,
):
    """Heads in another request protect all routes; padding must not protect them."""
    bridge = os.environ.get("DSV41_TEST_CUDA_LIBRARY")
    if backend_config.backend != "ik" or not bridge or not torch.cuda.is_available():
        pytest.skip("An IK backend and CUDA bridge are required")
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                n_routed_experts=8,
                hidden_size=128,
                moe_intermediate_size=128,
                num_experts_per_tok=4,
                swiglu_limit=0.125,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128, max_num_seqs=8),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=16),
    )
    module = CPUExpertModule(
        replace(
            backend_config,
            cuda_library_path=bridge,
            gpu_cache_experts=2,
            gpu_cache_device=0,
            gpu_cache_static_experts=(3, 7),
            route_tail_skip=initial_rank,
        ),
        config,
    )
    load_weights(module.backend, module.gpu_cache)
    with torch.inference_mode():
        module.finalize()
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 2, 3]] * 4, device="cuda")
    routes = torch.tensor(
        [
            [0.6, 0.05, 0.2, 0.03],
            [0.06, 0.02, 0.7, 0.01],
            [0.02, 0.01, 0.7, 0.2],
            [0.01, 0.9, 0.02, 0.01],
        ],
        device="cuda",
    )
    padding = torch.tensor([False, False, False, True], device="cuda")
    hidden[3].fill_(float("nan"))
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        is_padding=padding,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with override_forward_context(context), torch.cuda.stream(stream):
            module(hidden, ids, routes)
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                output = module(hidden, ids, routes)
            for rank, valid in ((0, 3), (2, 3), (2, 4), (2, 1), (2, 0), (3, 3), (0, 3)):
                torch.accelerator.synchronize()
                hidden.normal_()
                hidden[valid:].fill_(float("nan"))
                padding.fill_(True)
                padding[:valid].fill_(False)
                module.backend.set_route_tail_skip(rank)
                before = module.backend.cuda_stats()
                graph.replay()
                torch.accelerator.synchronize()
                module.backend.check_cuda_errors()
                after = module.backend.cuda_stats()
                skipped = valid if rank == 2 and valid < 4 else 0
                assert after["cached_routes"] - before["cached_routes"] == valid
                assert (
                    after["tail_skipped_routes"] - before["tail_skipped_routes"]
                    == skipped
                )
                assert after["tail_skipped_experts"] - before[
                    "tail_skipped_experts"
                ] == bool(skipped)
                assert after["tail_candidate_routes"] - before[
                    "tail_candidate_routes"
                ] == (3 * valid if rank else 0)
                cpu_routes = sum(after["expert_counts"]) - sum(before["expert_counts"])
                assert cpu_routes + skipped + valid == valid * 4
                assert torch.count_nonzero(output[valid:]) == 0
                if not valid:
                    continue
                expected_ids = ids[:valid].cpu().clone()
                expected_ids[:, 3] = -2
                if skipped:
                    expected_ids[:, 1] = -2
                cpu = module.backend.forward(
                    hidden[:valid].cpu(), expected_ids, routes[:valid].cpu()
                ).to(hidden.dtype)
                gpu = module.gpu_cache.launch(
                    hidden[:valid], ids[:valid], routes[:valid]
                )
                module.gpu_cache.join(gpu)
                torch.testing.assert_close(
                    output[:valid].cpu(), cpu.add_(gpu.cpu()), atol=0, rtol=0
                )
            module.backend.set_route_tail_skip(2)
            context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            before = module.backend.cuda_stats()
            unfiltered = module(hidden, ids, routes)
            torch.accelerator.synchronize()
            after = module.backend.cuda_stats()
            assert after["tail_skipped_routes"] == before["tail_skipped_routes"]
            assert after["tail_candidate_routes"] == before["tail_candidate_routes"]
            torch.testing.assert_close(
                unfiltered[:3].cpu(), output[:3].cpu(), atol=0, rtol=0
            )
            context.cudagraph_runtime_mode = CUDAGraphMode.FULL
            module.async_host[1][0, 1] = 99
            module.backend.enqueue_cuda(
                4, *module.async_host, stream.cuda_stream, allow_tail_skip=True
            )
            torch.accelerator.synchronize()
            with pytest.raises(RuntimeError, match="invalid or unloaded"):
                module.backend.check_cuda_errors()
    finally:
        del graph
        module.backend.close()


@pytest.mark.parametrize("decode_tokens", [8, 64, 96])
@pytest.mark.parametrize("origin_device,cache_device", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("hidden_size", [128, 5120])
@pytest.mark.parametrize(
    "prepack_resident,host_capacity_experts",
    [(True, None), (False, None), (False, 0), (False, 1), (False, 2)],
)
def test_gpu_cache_preserves_mixed_routes_across_prefill_and_graph_replay(
    backend_config,
    decode_tokens,
    origin_device,
    cache_device,
    hidden_size,
    prepack_resident,
    host_capacity_experts,
):
    if backend_config.backend != "ik":
        pytest.skip("GPU expert caching currently uses IK")
    bridge = os.environ.get("DSV41_TEST_CUDA_LIBRARY")
    if not bridge or torch.accelerator.device_count() <= max(
        origin_device, cache_device
    ):
        pytest.skip("A CUDA bridge and the requested cache GPU are required")
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )
    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule
    from vllm.utils.torch_utils import current_stream

    origin = torch.device("cuda", origin_device)
    torch.accelerator.set_device_index(origin_device)
    torch.cuda.set_stream(torch.cuda.current_stream(origin_device))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                n_routed_experts=8,
                hidden_size=hidden_size,
                moe_intermediate_size=128,
                num_experts_per_tok=3,
                swiglu_limit=0.125,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max(16, decode_tokens)
        ),
    )
    dynamic_settings = {
        "prepack_resident": prepack_resident,
        "feedback_weight": 0.9 if not prepack_resident else 0.0,
        "host_lru_experts": 1 if not prepack_resident else 0,
        "mutable_experts": [1, 2],
        "cpu_call_ms": 1.0,
        "transfer_ms": 0.01,
        "min_tokens": 64,
    }
    if host_capacity_experts is not None:
        expert_bytes = 3 * hidden_size * 128 * 17 // 32
        dynamic_settings["host_cache_bytes"] = (
            host_capacity_experts * expert_bytes + expert_bytes // 2
        )
    module = CPUExpertModule(
        replace(
            backend_config,
            cuda_library_path=bridge,
            gpu_cache_static_experts=(0, 1, 2),
            gpu_cache_dynamic=dynamic_settings,
            gpu_cache_experts=3,
            gpu_cache_device=cache_device,
            gpu_cache_prefill=True,
        ),
        config,
    )
    weights = load_weights(module.backend, module.gpu_cache)
    module.finalize()
    with torch.inference_mode():
        module.gpu_cache.device_io = module.gpu_cache._allocate_io(decode_tokens)
    module.gpu_cache.warmup()
    assert current_stream().device == origin
    shared_layer = torch.nn.Linear(
        hidden_size, 128, bias=False, dtype=torch.bfloat16, device=origin
    ).requires_grad_(False)
    shared = SharedExperts(
        shared_layer,
        SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                enable_eplb=False,
                use_fi_nvl_two_sided_kernels=False,
                dp_size=1,
                tp_size=1,
            )
        ),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
        is_multistream_safe=lambda: True,
    )
    generator = torch.Generator(device=origin).manual_seed(71)
    hidden = torch.randn(
        decode_tokens,
        hidden_size,
        device=origin,
        dtype=torch.bfloat16,
        generator=generator,
    )
    # Keep gate variance comparable when increasing the dot-product dimension.
    hidden.mul_((128 / hidden_size) ** 0.5)
    original = hidden.cpu()
    ids = torch.zeros(decode_tokens, 3, device=origin, dtype=torch.int64)
    routes = torch.ones(decode_tokens, 3, device=origin) * 0.7
    padding = torch.zeros(decode_tokens, device=origin, dtype=torch.bool)
    context = ForwardContext({}, {}, {}, is_padding=padding)

    def forward():
        result = module(hidden, ids, routes)
        assert current_stream() == torch.cuda.current_stream(origin_device)
        assert not shared.maybe_forward_async(result)
        shared.maybe_sync_shared_experts_stream(result)
        shared(result, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
        return result, shared.output

    try:
        with override_forward_context(context):
            stream = torch.cuda.Stream(device=origin)
            stream.wait_stream(torch.cuda.current_stream(origin_device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            stream.synchronize()
            pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output, shared_output = forward()
            consumer = torch.cuda.CUDAGraph()
            with torch.cuda.graph(consumer, pool=pool, stream=stream):
                consumed = shared_output * 1.5
            torch.accelerator.synchronize(origin_device)
            torch.accelerator.synchronize(cache_device)
            torch.accelerator.empty_cache()
            module.gpu_cache.select_static([0, 1, 2])
            cache = module.gpu_cache
            if module.backend.supports_export:
                assert not cache.weights
            addresses = [part.data_ptr() for part in cache.packed]
            cache.prepare_dynamic(dynamic_settings)
            expected_host = list(range(1 if prepack_resident else 3, 8))
            if host_capacity_experts is not None:
                expected_host = expected_host[: max(0, host_capacity_experts - 1)]
                allocated = sum(
                    cache.dynamic_stats[key] for key in ("host_bytes", "lru_host_bytes")
                )
                assert allocated <= dynamic_settings["host_cache_bytes"]
                assert allocated == host_capacity_experts * expert_bytes
            assert set(cache.host_packed) == set(expected_host)
            cache.adapt(torch.tensor([[0, 3, 4]] * 128, device=origin))
            assert cache.selected == [0, 3, 4]
            host_addresses = {
                e: tuple(p.data_ptr() for p in parts)
                for e, parts in cache.host_packed.items()
            }
            module.finalize()
            assert cache.selected == [0, 3, 4]
            assert host_addresses == {
                e: tuple(p.data_ptr() for p in parts)
                for e, parts in cache.host_packed.items()
            }
            with pytest.raises(ValueError, match="Cannot change"):
                cache.prepare_dynamic({**dynamic_settings, "max_swaps": 1})
            for expert in (3, 4):
                if expert not in cache.host_packed:
                    continue
                prepared = cache.host_packed.pop(expert)
                with torch.cuda.stream(cache.stream):
                    expected_packed = cache._pack_expert(expert)
                    cache.stream.synchronize()
                for actual, expected in zip(prepared, expected_packed):
                    assert torch.equal(actual, expected.cpu())
                cache.host_packed[expert] = prepared
            cache.select([0, 1, 2], preserve_slots=False)
            assert [part.data_ptr() for part in cache.packed] == addresses
            if not prepack_resident and host_capacity_experts is None:
                lru_addresses = [part.data_ptr() for part in cache.host_lru]
                cache.select([0, 3, 2], preserve_slots=False)
                cache.select([0, 3, 4], preserve_slots=False)
                hits = cache.dynamic_stats["lru_hits"]
                cache.select([0, 3, 2], preserve_slots=False)
                assert cache.dynamic_stats["lru_hits"] == hits + 1
                assert list(cache.host_lru_slots) == [2]
                assert [part.data_ptr() for part in cache.host_lru] == lru_addresses
                cache.select([0, 3, 4], preserve_slots=False)
                hits = cache.dynamic_stats["lru_hits"]
                repacks = cache.dynamic_stats["repacked_experts"]
                cache.select([1, 3, 2], preserve_slots=False)
                assert cache.dynamic_stats["lru_hits"] == hits + 1
                assert cache.dynamic_stats["repacked_experts"] == repacks + 1
                cache.select([0, 1, 2], preserve_slots=False)
            if host_capacity_experts and cache.host_lru is not None:
                hits = cache.dynamic_stats["lru_hits"]
                cache.select([2, 0, 1], preserve_slots=False)
                assert cache.dynamic_stats["lru_hits"] == hits + 1
                assert [part.data_ptr() for part in cache.packed] == addresses
                cache.select([0, 1, 2], preserve_slots=False)
            # Allocate and reuse a larger replay buffer after decode capture.
            for valid, selected in (
                (128, [0, 3, 4]),
                (17, [0, 1, 2]),
                (127, [5, 6, 7]),
                (0, [0, 3, 4]),
            ):
                replay = original.repeat((127 + decode_tokens) // decode_tokens, 1)
                replay = replay[:128].to(origin)
                replay[valid:].fill_(float("nan"))
                replay_ids = torch.tensor(selected, device=origin).expand(128, -1)
                replay_routes = torch.full((128, 3), 0.7, device=origin)
                replay_context = ForwardContext(
                    {},
                    {"replay": SimpleNamespace(num_actual_tokens=valid)},
                    {},
                    is_padding=torch.arange(128, device=origin) >= valid,
                )
                with override_forward_context(replay_context):
                    replay_output = module(replay, replay_ids, replay_routes)
                torch.accelerator.synchronize(origin_device)
                torch.accelerator.synchronize(cache_device)
                expected = torch.zeros_like(replay, device="cpu").float()
                if valid:
                    expected[:valid] = reference(
                        replay[:valid].cpu(),
                        replay_ids[:valid].cpu(),
                        replay_routes[:valid].cpu(),
                        weights,
                        0.125,
                    )
                assert torch.isfinite(replay_output).all()
                error = (replay_output.float().cpu() - expected).norm()
                assert error / expected.norm().clamp_min(1e-8) < 0.035
                assert torch.count_nonzero(replay_output[valid:]) == 0
                assert 0 in cache.selected
                assert [part.data_ptr() for part in cache.packed] == addresses
            assert cache.dynamic_stats["observations"] == 3
            assert cache.dynamic_stats["swaps"] == 6
            cache.dynamic_enabled = False
            for enabled, cached in (
                (True, [0, 1, 2]),
                (False, [0, 1, 2]),
                (True, [1, 2, 3]),
                (True, [3, 4, 5]),
                (True, [5, 4, 3]),
            ):
                module.gpu_cache.set_enabled(enabled)
                module.gpu_cache.select_static(cached)
                module.gpu_cache.calibrate(torch.full((128, 3), 7))
                assert set(module.gpu_cache.selected) == set(cached)
                assert current_stream() == torch.cuda.current_stream(origin_device)
                cold = [e for e in range(8) if e not in cached][:3]
                for valid, selected in (
                    (decode_tokens, cached),
                    (decode_tokens, cold),
                    (6, cached),
                    (3, cold),
                    (1, [cached[0], *cold[:2]]),
                    (0, cold),
                ):
                    hidden.copy_(original)
                    hidden[valid:].fill_(float("nan"))
                    ids.copy_(
                        torch.tensor(selected, device=origin).expand(decode_tokens, -1)
                    )
                    padding.copy_(torch.arange(decode_tokens, device=origin) >= valid)
                    graph.replay()
                    consumer.replay()
                    torch.accelerator.synchronize(origin_device)
                    torch.accelerator.synchronize(cache_device)
                    module.backend.check_cuda_errors()
                    expected = torch.zeros_like(original).float()
                    if valid:
                        expected[:valid] = reference(
                            original[:valid],
                            ids[:valid].cpu(),
                            routes[:valid].cpu(),
                            weights,
                            0.125,
                        )
                    assert torch.isfinite(output).all()
                    error = (output.float().cpu() - expected).norm()
                    assert error / expected.norm().clamp_min(1e-8) < 0.035
                    assert torch.count_nonzero(output[valid:]) == 0
                    torch.testing.assert_close(shared_output, shared_layer(output))
                    torch.testing.assert_close(consumed, shared_output * 1.5)
    finally:
        torch.accelerator.synchronize(origin_device)
        torch.accelerator.synchronize(cache_device)
        module.backend.close()


@pytest.mark.parametrize(
    "capacities",
    [
        [600, 600, 160],
        [1700, 1500, 1500, 2000],
        [345, 64, 402],
        [327, 46, 384],
        [15000, 15000, 15000, 15000],
        [0, 8, 0],
        [0, 0, 0],
    ],
)
def test_automatic_cache_respects_each_devices_remaining_budget(capacities):
    from vllm.models.deepseek_v4_1.hybrid import plan_expert_cache

    plan = plan_expert_cache(capacities, local_device=len(capacities) - 1)
    assert len(plan) == 20
    for device, budget in enumerate(capacities):
        assert sum(count for owner, count in plan if owner == device) <= budget
    assert all(0 <= count < 384 for _, count in plan)
    assert all(count % 8 == 0 for _, count in plan)
    if sum(capacities) > 20 * 8:
        assert all(count > 0 for _, count in plan)


def test_automatic_cache_fills_aligned_spare_budget_on_assigned_device():
    """The three-card startup must use PP1's spare 24 slots within its budget."""
    from vllm.models.deepseek_v4_1.hybrid import plan_expert_cache

    capacities = [345, 64, 402]
    plan = plan_expert_cache(capacities, local_device=2)
    assert [sum(n for d, n in plan if d == device) for device in range(3)] == [
        344,
        64,
        400,
    ]
    assert plan[18] == (1, 64)
    assert [device for device, _ in plan] == [2] * 10 + [0] * 8 + [1, 0]
    assert capacities == [345, 64, 402]


@pytest.mark.parametrize(
    "capacities,layers,expected",
    [
        ([1960, 1736, 2104, 64], 20, [1960, 1736, 2104, 64]),
        ([16, 16, 19], 20, [16, 16, 16]),
        ([24, 16], 1, [24, 0]),
    ],
)
def test_automatic_cache_uses_small_device_only_when_total_capacity_grows(
    capacities, layers, expected
):
    """An unused GPU can hold a partial layer without losing more slots elsewhere."""
    from vllm.models.deepseek_v4_1.hybrid import plan_expert_cache

    plan = plan_expert_cache(
        capacities, num_layers=layers, local_device=len(capacities) - 1
    )
    assert [
        sum(n for d, n in plan if d == device) for device in range(len(capacities))
    ] == (expected)
    assert all(count % 8 == 0 and 0 <= count < 384 for _, count in plan)


def test_measured_expert_budget_can_expand_beyond_previous_total():
    """808 slots are a previous result, not a limit on measured free space."""
    from vllm.models.deepseek_v4_1.hybrid import expand_expert_cache, plan_expert_cache

    before = plan_expert_cache([345, 64, 402], local_device=2)
    after = expand_expert_cache(before, [345, 112, 402])
    assert sum(count for _, count in before) == 808
    assert sum(count for _, count in after) == 856
    assert after[18] == (1, 112)
    assert [d for d, _ in before] == [d for d, _ in after]
    assert before[18] == (1, 64)


def test_calibrated_expert_budget_accounts_for_existing_weights_and_peak():
    from vllm.models.deepseek_v4_1.hybrid import GiB, calibrated_expert_capacities

    memory = [
        (5 * GiB // 2, 16 * GiB, GiB // 4),
        (3 * GiB, 16 * GiB, GiB // 2),
        (GiB // 2, 16 * GiB, GiB // 2),
    ]
    assert calibrated_expert_capacities(
        memory, [(0, 8), (1, 4), (2, 8)], GiB // 8, 15 / 16
    ) == [18, 16, 0]


@pytest.mark.parametrize(
    "minimum_free, expected", [(149, 21), (150, 32), (249, 32), (250, 32), (330, 40)]
)
def test_cache_retains_slots_within_forty_percent_reserve_shortfall(
    minimum_free, expected
):
    """Retain at the 60% boundary; expansion still needs the full reserve."""
    from vllm.models.deepseek_v4_1.hybrid import calibrated_expert_capacities

    memory = [(minimum_free + 40, 1000, 40)]
    assert calibrated_expert_capacities(memory, [(0, 32)], 10, 0.75) == [expected]


@pytest.mark.parametrize("additional_peak", [0, 400 * 1024**2])
def test_recorded_peak_shortfall_retains_856_slots_until_tolerance_is_exceeded(
    additional_peak,
):
    """Replay the measured 614/578 MiB margins that previously removed 16 slots."""
    from vllm.models.deepseek_v4_1.cache_budget import (
        cached_expert_capacities,
        fit_cached_plan,
    )

    plan = (
        [(2, 48)] * 3
        + [(2, 40)] * 5
        + [(0, 40)] * 5
        + [(0, 48), (0, 48), (1, 48), (0, 48), (1, 48), (0, 40), (1, 32)]
    )
    expert_bytes = 18800640
    slots = [sum(n for d, n in plan if d == i) for i in range(3)]
    growth = [1064304640, 871366656, 2128805888]
    capture_free = [1778515968, 1524760576, 2572353536]
    saved = {
        "plan": plan,
        "growth": growth,
        "headroom": [1134559232 + additional_peak, 918552576, 1656750080],
        "capture_free": capture_free,
    }
    memory = [
        (free + retained + count * expert_bytes, 68063395840, 0)
        for free, retained, count in zip(capture_free, growth, slots)
    ]
    capacities = cached_expert_capacities(memory, saved, expert_bytes, 0.99)
    fitted = fit_cached_plan(plan, capacities, 384)
    assert fitted is not None
    counts = [sum(n for d, n in fitted if d == i) for i in range(3)]
    assert counts[1] == 128
    if additional_peak:
        assert counts[0] < 384
    else:
        assert counts[0] == 384
        assert sum(counts) >= 856


def test_graph_calibration_accepts_small_shortfall_without_864_856_rebuilds(
    monkeypatch,
):
    """The user's logged probe and second capture converge in two captures."""
    from vllm.models.deepseek_v4_1 import hybrid_runtime as runtime

    captures, saved = [], []
    worker = SimpleNamespace(
        _hybrid_graph_calibration=True,
        _hybrid_cached_budget=None,
        _hybrid_validated_headrooms=None,
        _hybrid_cache_plan=(
            [(2, 16)] * 3
            + [(2, 24)] * 6
            + [(0, 24)] * 7
            + [(1, 24), (0, 24), (1, 24), (2, 16)]
        ),
        _hybrid_calibration_started=runtime.time.monotonic(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=5120, moe_intermediate_size=2304, n_routed_experts=384
                )
            ),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.99),
        ),
    )

    def capture():
        captures.append(list(worker._hybrid_cache_plan))
        return 123

    worker.model_runner = SimpleNamespace(capture_model=capture)
    monkeypatch.setattr(
        runtime, "get_pp_group", lambda: SimpleNamespace(world_size=3, rank_in_group=2)
    )
    phases = iter(
        [
            [(5.065, 0.889), (2.865, 0.734), (4.657, 1.602)],
            [(1.516, 0.912), (1.420, 0.756), (2.290, 1.584)],
        ]
    )

    def profile(*args, **kwargs):
        return [
            (int(free * runtime.GiB), 68063395840, int(peak * runtime.GiB))
            for free, peak in next(phases)
        ]

    monkeypatch.setattr(runtime, "_profile_hybrid_runtime_memory", profile)
    monkeypatch.setattr(runtime, "_hybrid_memory", lambda _: [(0, 68063395840, 0)] * 3)
    monkeypatch.setattr(runtime, "_clear_hybrid_graphs_and_cache", lambda _: None)
    monkeypatch.setattr(
        runtime,
        "_allocate_hybrid_cache",
        lambda w, p: setattr(w, "_hybrid_cache_plan", p),
    )
    monkeypatch.setattr(
        runtime,
        "_save_hybrid_budget",
        lambda w, _: saved.append(list(w._hybrid_cache_plan)),
    )
    monkeypatch.setattr(runtime, "_log_hybrid_cache_plan", lambda _: None)
    assert runtime.capture_hybrid_graphs(worker) == 123
    assert len(captures) == 2
    assert [sum(n for d, n in saved[0] if d == i) for i in range(3)] == [392, 128, 344]


@pytest.mark.parametrize("restored_free", [500, 5])
def test_graph_calibration_rechecks_restored_layout_after_final_expansion(
    monkeypatch, restored_free
):
    """A failed last expansion restores a safe layout, never bypassing its check."""
    from vllm.models.deepseek_v4_1 import hybrid_runtime as runtime

    captures = []
    worker = SimpleNamespace(
        _hybrid_graph_calibration=True,
        _hybrid_cached_budget=None,
        _hybrid_validated_headrooms=None,
        _hybrid_cache_plan=[(0, 8)],
        _hybrid_calibration_started=runtime.time.monotonic(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=32, moe_intermediate_size=32, n_routed_experts=384
                )
            ),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.99),
        ),
    )

    def capture():
        captures.append(list(worker._hybrid_cache_plan))
        return 123

    worker.model_runner = SimpleNamespace(capture_model=capture)
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(runtime, "get_pp_group", lambda: group)
    monkeypatch.setattr(runtime, "_hybrid_memory", lambda _: [(500, 1000, 20)])
    profiles = iter([(500, 1000, 20)] * 3 + [(5, 1000, 20), (restored_free, 1000, 20)])
    monkeypatch.setattr(
        runtime, "_profile_hybrid_runtime_memory", lambda *a, **kw: [next(profiles)]
    )
    monkeypatch.setattr(runtime, "plan_expert_cache", lambda *a: [(0, 16)])
    expansions = iter([[(0, 24)], [(0, 32)], [(0, 32)]])
    monkeypatch.setattr(runtime, "expand_expert_cache", lambda *a: next(expansions))
    monkeypatch.setattr(runtime, "_clear_hybrid_graphs_and_cache", lambda _: None)
    monkeypatch.setattr(
        runtime,
        "_allocate_hybrid_cache",
        lambda w, plan: setattr(w, "_hybrid_cache_plan", plan),
    )
    saved = []
    monkeypatch.setattr(
        runtime,
        "_save_hybrid_budget",
        lambda w, _: saved.append(list(w._hybrid_cache_plan)),
    )
    monkeypatch.setattr(runtime, "_log_hybrid_cache_plan", lambda _: None)

    if restored_free == 500:
        assert runtime.capture_hybrid_graphs(worker) == 123
        assert saved == [[(0, 24)]]
    else:
        with pytest.raises(RuntimeError, match="could not preserve runtime headroom"):
            runtime.capture_hybrid_graphs(worker)
        assert not saved
    assert captures == [[(0, 8)], [(0, 16)], [(0, 24)], [(0, 32)], [(0, 24)]]


def test_hybrid_memory_includes_remote_allocator_headroom():
    """PP2's allocations on PP0 and PP1 contribute to those GPUs' peaks."""
    from vllm.models.deepseek_v4_1.hybrid_runtime import _aggregate_hybrid_memory

    samples = [
        {0: (100, 1000, 10, 11, 12, 16, 5)},
        {1: (200, 1000, 20, 22, 22, 25, 4)},
        {
            0: (100, 1000, 5, 7, 8, 11, 0),
            1: (200, 1000, 6, 9, 9, 12, 0),
            2: (300, 1000, 30, 34, 35, 41, 7),
        },
    ]
    assert _aggregate_hybrid_memory(samples) == [
        (100, 1000, 7),
        (200, 1000, 7),
        (300, 1000, 7),
    ]


@pytest.fixture
def dsv41_startup_budget(tmp_path, monkeypatch):
    from vllm.models.deepseek_v4_1.cache_budget import CacheBudgetProfile

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    profile = CacheBudgetProfile("model-config-hardware", [1000, 1000], 2, 384)
    payload = {
        "version": 2,
        "fingerprint": profile.fingerprint,
        "plan": [[0, 8], [1, 16]],
        "growth": [50, 70],
        "headroom": [100, 200],
        "prefill_headroom": [80, 160],
        "capture_free": [900, 850],
        "workspaces": [[32, 0], [64, 16]],
    }
    return profile, payload


def test_cache_budget_restores_plan_and_ignores_corrupt_memory(dsv41_startup_budget):
    import json

    profile, payload = dsv41_startup_budget
    assert profile.load() is None
    profile.save(payload)
    assert profile.load() == payload
    for invalid in (
        dict(payload, headroom=[-1, 200]),
        dict(payload, prefill_headroom=[101, 160]),
        dict(payload, capture_free=[900, 1001]),
    ):
        profile.path.write_text(json.dumps(invalid))
        assert profile.load() is None
    profile.save(payload)
    profile.fingerprint = "different-configuration"
    assert profile.load() is None


def test_serving_peak_is_saved_for_next_startup_and_never_lowered(dsv41_startup_budget):
    profile, payload = dsv41_startup_budget
    profile.save(payload)
    updated = profile.record_runtime_peak(payload, [700, 800])
    assert updated["headroom"] == [200, 200]
    assert profile.load() == updated
    assert profile.record_runtime_peak(updated, [800, 850]) == updated


def test_reused_cache_budget_adapts_slots_without_moving_io_owners():
    from vllm.models.deepseek_v4_1.cache_budget import fit_cached_plan
    from vllm.models.deepseek_v4_1.hybrid import plan_expert_cache

    plan = plan_expert_cache([345, 64, 402], local_device=2)
    increased = fit_cached_plan(plan, [360, 96, 408], 384)
    decreased = fit_cached_plan(plan, [320, 56, 384], 384)
    assert sum(count for _, count in increased) == 864
    for result, capacities in (
        (increased, [360, 96, 408]),
        (decreased, [320, 56, 384]),
    ):
        assert [device for device, _ in result] == [device for device, _ in plan]
        assert all(count % 8 == 0 and count >= 8 for _, count in result)
        for device, capacity in enumerate(capacities):
            assert sum(count for owner, count in result if owner == device) <= capacity
    assert fit_cached_plan(plan, [0, 64, 402], 384) is None


@pytest.fixture
def dsv41_budget_config():
    from vllm.config import CacheConfig, CompilationConfig, CUDAGraphMode, VllmConfig

    class Config(SimpleNamespace):
        compute_hash = VllmConfig.compute_hash

    return Config(
        **dict.fromkeys(
            (
                "parallel_config",
                "device_config",
                "load_config",
                "offload_config",
                "attention_config",
                "engram_config",
                "lora_config",
                "speculative_config",
                "structured_outputs_config",
                "profiler_config",
                "quant_config",
                "kernel_config",
                "kv_transfer_config",
                "ec_transfer_config",
            )
        ),
        model_config=SimpleNamespace(
            model="test-checkpoint",
            revision=None,
            max_model_len=524288,
            hf_config=SimpleNamespace(to_dict=lambda: {"n_routed_experts": 384}),
            multimodal_config=None,
            compute_hash=lambda: "model-graph",
        ),
        observability_config=SimpleNamespace(compute_hash=lambda: "metrics"),
        cache_config=CacheConfig(
            gpu_memory_utilization=0.985,
            enable_prefix_caching=True,
            kv_offloading_size=32,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            compute_hash=lambda: "scheduler-graph",
        ),
        uniform_decode_query_len=8,
        num_lookahead_tokens=7,
        additional_config={
            "deepseek_v41_hybrid": {
                "pipeline_layers": [7, 8, 25],
                "expert_profile": "first.json",
                "cpu_threads": [24, 24],
                "route_tail_skip": 0,
                "host_cache_gib": 0,
                "engram_storage": "ssd",
                "engram_cache_gib": 1,
                "r8_refill": True,
                "overlap_decode": True,
                "adaptive_cache": {"min_experts": 8, "interval": 128},
            },
            "cpu_moe": {"backend": "ik", "num_threads": 24, "route_tail_skip": 0},
            "cpu_phase_threads": [24, 24],
            "ced_prefill": True,
        },
    )


def test_budget_identity_ignores_history_path_but_invalidates_memory_changes(
    dsv41_budget_config,
):
    from vllm.models.deepseek_v4_1.cache_budget import budget_fingerprint

    config = dsv41_budget_config
    memory = [(1000, 2000, 10)]
    first = budget_fingerprint(config, memory, [(20, [128], 2)])
    assert config.cache_config.kv_offloading_size == 32
    assert budget_fingerprint(config, [(500, 2000, 100)], [(20, [128], 2)]) == first
    assert budget_fingerprint(config, [(1000, 4000, 10)], [(20, [128], 2)]) != first
    config.additional_config["deepseek_v41_hybrid"]["expert_profile"] = "second.json"
    assert budget_fingerprint(config, memory, [(20, [128], 2)]) == first
    assert budget_fingerprint(config, memory, [(40, [256], 2)]) != first
    config.scheduler_config.max_num_seqs = 4
    assert budget_fingerprint(config, memory, [(20, [128], 2)]) != first


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("deepseek_v41_hybrid.route_tail_skip", 4),
        ("cpu_moe.route_tail_skip", 4),
        ("deepseek_v41_hybrid.cpu_threads", [16, 20]),
        ("cpu_moe.num_threads", 16),
        ("cpu_phase_threads", [16, 20]),
        ("deepseek_v41_hybrid.host_cache_gib", 12),
        ("deepseek_v41_hybrid.engram_cache_gib", 4),
        ("deepseek_v41_hybrid.expert_profile", False),
        ("deepseek_v41_hybrid.expert_profile_interval", 1),
        ("deepseek_v41_hybrid.adaptive_cache.interval", 64),
        ("deepseek_v41_hybrid.adaptive_cache.max_swaps", 32),
        ("deepseek_v41_hybrid.adaptive_cache.online", False),
        ("deepseek_v41_hybrid.adaptive_cache.safety_factor", 2.0),
    ],
)
def test_budget_identity_reuses_peaks_for_host_policies(
    dsv41_budget_config, path, value
):
    import copy

    from vllm.models.deepseek_v4_1.cache_budget import budget_fingerprint

    config = dsv41_budget_config
    memory, kv_sizes = [(1000, 2000, 10)], [(20, [128], 2)]
    first = budget_fingerprint(config, memory, kv_sizes)
    target = config.additional_config
    *parents, field = path.split(".")
    for parent in parents:
        target = target[parent]
    target[field] = value
    original = copy.deepcopy(config.additional_config)
    assert budget_fingerprint(config, memory, kv_sizes) == first
    assert config.additional_config == original


@pytest.mark.parametrize(
    ("section", "field", "value", "invalidates"),
    [
        ("cache_config", "gpu_memory_utilization", 0.99, False),
        ("cache_config", "kv_offloading_size", 64, False),
        ("cache_config", "kv_offloading_size", None, True),
        ("cache_config", "kv_offloading_backend", "lmcache", True),
        ("cache_config", "enable_prefix_caching", False, True),
        ("compilation_config", "cudagraph_capture_sizes", [1, 2, 4], True),
        ("scheduler_config", "max_num_batched_tokens", 4096, True),
        ("model_config", "max_model_len", 131072, True),
        (None, "uniform_decode_query_len", 6, True),
        (None, "num_lookahead_tokens", 5, True),
        ("deepseek_v41_hybrid", "pipeline_layers", [6, 9, 25], True),
        ("deepseek_v41_hybrid", "engram_storage", "ram", True),
        ("deepseek_v41_hybrid", "r8_refill", False, True),
        ("deepseek_v41_hybrid", "overlap_decode", False, True),
        ("deepseek_v41_hybrid", "adaptive_cache", False, True),
        ("deepseek_v41_hybrid", "adaptive_cache", {"min_experts": 16}, True),
    ],
)
def test_budget_identity_separates_memory_shape_from_budget_targets(
    dsv41_budget_config, section, field, value, invalidates
):
    from vllm.models.deepseek_v4_1.cache_budget import budget_fingerprint

    config = dsv41_budget_config
    memory, kv_sizes = [(1000, 2000, 10)], [(20, [128], 2)]
    first = budget_fingerprint(config, memory, kv_sizes)
    if section == "deepseek_v41_hybrid":
        config.additional_config[section][field] = value
    else:
        setattr(getattr(config, section) if section else config, field, value)
    assert (budget_fingerprint(config, memory, kv_sizes) != first) == invalidates


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize(
    ("requests", "max_batched_tokens"),
    [(8, 2048), (16, 2048), (16, 1792), (16, 128)],
)
def test_serving_memory_profile_uses_prefill_then_decode_and_cleans_requests(
    monkeypatch,
    sliding,
    requests,
    max_batched_tokens,
):
    from vllm.models.deepseek_v4_1 import hybrid_runtime as runtime
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    group = SimpleNamespace(world_size=1, cpu_group=None, barrier=lambda: None)
    monkeypatch.setattr(runtime, "get_pp_group", lambda: group)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda result, value, group: result.__setitem__(0, value),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_batched_tokens),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sliding_window=128), get_vocab_size=lambda: 64
        ),
        num_lookahead_tokens=7,
    )
    steps: list[SchedulerOutput] = []
    connector_states: list[bool] = []
    adaptive = object()
    spec = (
        SlidingWindowSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            sliding_window=128,
        )
        if sliding
        else SimpleNamespace(block_size=128)
    )
    runner = SimpleNamespace(
        vllm_config=config,
        max_model_len=8192,
        decode_query_len=8,
        num_speculative_steps=7,
        adaptive_verification=adaptive,
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_cache_config=SimpleNamespace(
            num_blocks=5 * requests,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
        ),
        kv_connector=SimpleNamespace(set_disabled=connector_states.append),
    )

    computed: dict[str, int] = {}

    def execute(output):
        assert output.total_num_scheduled_tokens <= max_batched_tokens
        assert output.total_num_scheduled_tokens == sum(
            output.num_scheduled_tokens.values()
        )
        for req in output.scheduled_new_reqs:
            assert req.req_id not in computed
            computed[req.req_id] = req.num_computed_tokens
        cached = output.scheduled_cached_reqs
        assert [computed[req_id] for req_id in cached.req_ids] == (
            cached.num_computed_tokens
        )
        for req_id, count in output.num_scheduled_tokens.items():
            computed[req_id] += count
        for req_id in output.finished_req_ids:
            computed.pop(req_id)
        steps.append(output)

    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        execute_model=execute,
        sample_tokens=lambda grammar: None,
    )
    runtime._profile_hybrid_serving_step(worker, 8 * requests, requests, decode=True)
    *prefills, decode, cleanup = steps
    new_reqs = [req for step in prefills for req in step.scheduled_new_reqs]
    assert len({req.req_id for req in new_reqs}) == len(new_reqs) == requests
    assert all(req.sampling_params.prompt_logprobs is None for req in new_reqs)
    assert all(req.prompt_len - req.num_computed_tokens == 256 for req in new_reqs)
    for req in new_reqs:
        assert (
            sum(step.num_scheduled_tokens.get(req.req_id, 0) for step in prefills)
            == 256
        )
    block_ids = {
        block
        for req in new_reqs
        for group_blocks in req.block_ids
        for block in group_blocks
    }
    assert max(block_ids) < runner.kv_cache_config.num_blocks
    assert all(req.num_computed_tokens % 128 == 0 for req in new_reqs)
    if sliding:
        assert 0 in block_ids
        assert all(req.num_computed_tokens > 4096 for req in new_reqs)
    else:
        assert 0 not in block_ids
    assert decode.total_num_scheduled_tokens == 8 * requests
    assert set(decode.scheduled_cached_reqs.req_ids) == {req.req_id for req in new_reqs}
    assert decode.scheduled_cached_reqs.num_computed_tokens == [
        req.prompt_len for req in new_reqs
    ]
    assert all(
        len(drafts) == 7 for drafts in decode.scheduled_spec_decode_tokens.values()
    )
    assert cleanup.finished_req_ids == set(decode.num_scheduled_tokens)
    assert not computed
    assert connector_states == [True, False]
    assert runner.adaptive_verification is adaptive


def test_profile_kv_initializes_shared_backing_before_synthetic_prefix_reads():
    from vllm.models.deepseek_v4_1.hybrid_runtime import _zero_hybrid_profile_kv

    storage = torch.full((512,), 255, dtype=torch.uint8)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_caches=[storage[:128], storage[128:].view(torch.int16)]
        )
    )
    _zero_hybrid_profile_kv(worker)
    assert torch.count_nonzero(storage) == 0


@pytest.mark.parametrize(
    "connections,external", [(None, False), ("8", False), (None, True)]
)
def test_owned_mps_respects_and_restores_work_queue_configuration(
    monkeypatch, tmp_path, connections, external
):
    from vllm.models.deepseek_v4_1 import hybrid_runtime as runtime

    for key in (
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_MAX_CONNECTIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    if connections is not None:
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", connections)
    if external:
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/external/mps")
    for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(runtime, "hybrid_settings", lambda _: {})
    monkeypatch.setattr(runtime, "check_host_headroom", lambda _: None)
    monkeypatch.setattr(runtime, "physical_cpus", lambda: [0])
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/test/mps-control")
    monkeypatch.setattr(runtime.tempfile, "tempdir", str(tmp_path))
    launched = []

    def run(*args, **kwargs):
        launched.append(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    hf = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=32,
        num_hidden_layers=40,
        n_routed_experts=8,
        engram_num_embeddings=[1],
        engram_head_dim=32,
    )
    resources = runtime.HybridExecutorResources(
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf),
            additional_config={"cpu_phase_threads": [1, 1]},
            parallel_config=SimpleNamespace(pipeline_parallel_size=4),
        )
    )
    resources.close()
    assert launched == ([] if external else [connections or "16", connections])
    assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == connections
    assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == (
        "/external/mps" if external else None
    )


@pytest.mark.parametrize("failure", ["timeout", "exit_code", "missing_command"])
def test_owned_mps_quit_failure_still_reclaims_only_its_own_processes(
    monkeypatch, tmp_path, failure
):
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1 import hybrid_runtime as runtime

    resources = runtime.HybridMPS()
    resources.path = tmp_path / "owned"
    resources.path.mkdir()
    owned: list[Mock] = []
    unrelated: list[Mock] = []
    for name, pipe, uid, group in (
        ("nvidia-cuda-mps-control", resources.path / "pipe", os.getuid(), owned),
        ("nvidia-cuda-mps-server", resources.path / "pipe", os.getuid(), owned),
        ("nvidia-cuda-mps-server", tmp_path / "external", os.getuid(), unrelated),
        ("python", resources.path / "pipe", os.getuid(), unrelated),
        ("nvidia-cuda-mps-server", resources.path / "pipe", os.getuid() + 1, unrelated),
    ):
        proc = Mock()
        proc.info = {"name": name, "uids": SimpleNamespace(real=uid)}
        proc.environ.return_value = {"CUDA_MPS_PIPE_DIRECTORY": str(pipe)}
        group.append(proc)
    monkeypatch.setattr(runtime.psutil, "process_iter", lambda _: owned + unrelated)
    monkeypatch.setattr(
        runtime.psutil, "wait_procs", Mock(side_effect=[([], owned), (owned, [])])
    )
    error = {
        "timeout": runtime.subprocess.TimeoutExpired("mps-control", 50),
        "exit_code": runtime.subprocess.CalledProcessError(1, "mps-control"),
        "missing_command": FileNotFoundError("mps-control"),
    }[failure]
    monkeypatch.setattr(runtime.subprocess, "run", Mock(side_effect=error))

    resources.close()
    resources.close()

    assert resources.path is None
    assert not (tmp_path / "owned").exists()
    for proc in owned:
        proc.kill.assert_called_once_with()
    for proc in unrelated:
        proc.kill.assert_not_called()


def test_owned_mps_reports_unreaped_processes_and_keeps_cleanup_directory(
    monkeypatch, tmp_path
):
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1 import hybrid_runtime as runtime

    resources = runtime.HybridMPS()
    resources.path = tmp_path
    proc = Mock(pid=123, **{"status.return_value": runtime.psutil.STATUS_DISK_SLEEP})
    monkeypatch.setattr(resources, "_processes", lambda: [proc])
    monkeypatch.setattr(runtime.subprocess, "run", Mock())
    monkeypatch.setattr(
        runtime.psutil, "wait_procs", lambda *args, **kwargs: ([], [proc])
    )

    with pytest.raises(RuntimeError, match="still alive.*123"):
        resources.close()

    assert resources.path == tmp_path
    assert tmp_path.exists()


def test_registered_lru_dma_does_not_cross_registration_boundaries(monkeypatch):
    """GPU expert copies must fit one registered span, including remote GPUs."""
    from vllm.models.deepseek_v4_1 import host_memory

    monkeypatch.setattr(host_memory, "LOAD_CHUNK_BYTES", 4 * 1024**2)
    row_bytes = 3 * 1024**2
    with torch.accelerator.device_index(0):
        host, owner = host_memory.empty_registered((4, row_bytes), torch.uint8)
    for device in range(min(2, torch.accelerator.device_count())):
        with torch.accelerator.device_index(device):
            source = torch.full(
                (row_bytes,), 17 + device, dtype=torch.uint8, device="cuda"
            )
            for row in host:
                row.copy_(source, non_blocking=True)
            torch.accelerator.synchronize()
            assert host.min().item() == host.max().item() == 17 + device


def test_expert_cache_stats_exclude_warmup_and_do_not_recount(monkeypatch):
    """Combine callback and eager routes, including layers with no GPU slots."""
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.metrics.stats import ExpertCacheStats

    def module(hits, misses, cache=None):
        result = CPUExpertModule.__new__(CPUExpertModule)
        torch.nn.Module.__init__(result)
        result.backend = SimpleNamespace(
            route_tail_skip=0,
            cuda_route_stats=Mock(return_value=(hits, misses, 10, 2, 30)),
        )
        result.gpu_cache = cache
        result._eager_gpu_hits = result._eager_cpu_misses = 0
        result._last_cache_stats = ExpertCacheStats()
        return result

    cache = SimpleNamespace(
        reload_updates=1,
        reloaded_experts=8,
        reload_seconds=1.0,
        dynamic_stats={"lru_hits": 2, "repacked_experts": 8},
    )
    cached = module(100, 20, cache)
    uncached = module(0, 120)
    state = object.__new__(DeepseekV41ModelState)
    state.cpu_expert_modules = [cached, uncached]
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    state.reset_expert_cache_stats()
    stream.synchronize.assert_called_once()
    cached.backend.cuda_route_stats.return_value = (107, 21, 13, 3, 40)
    uncached.backend.cuda_route_stats.return_value = (0, 128, 12, 4, 50)
    cached._eager_gpu_hits, cached._eager_cpu_misses = 2, 3
    cache.reload_updates += 1
    cache.reloaded_experts += 4
    cache.reload_seconds += 0.5
    cache.dynamic_stats["lru_hits"] += 3
    cache.dynamic_stats["repacked_experts"] += 1
    assert state.take_expert_cache_stats() == ExpertCacheStats(
        gpu_hits=9,
        cpu_misses=12,
        tail_skipped_routes=5,
        tail_skipped_experts=3,
        tail_candidate_routes=30,
        updates=1,
        experts_reloaded=4,
        reload_seconds=0.5,
        host_lru_hits=3,
        repacked_experts=1,
    )
    assert state.take_expert_cache_stats() == ExpertCacheStats()
    stream.synchronize.assert_called_once()  # Collection adds no synchronization.
    cached.gpu_cache = None
    state.reset_expert_cache_stats()
    assert state.take_expert_cache_stats() is None
    uncached.backend.route_tail_skip = 4
    state.reset_expert_cache_stats()
    assert state.take_expert_cache_stats() == ExpertCacheStats()
    assert stream.synchronize.call_count == 2


@pytest.mark.parametrize("bridge_api", ["stats", "counts", "legacy"])
def test_cuda_route_counts_aggregate_all_graph_shapes(bridge_api):
    """Old bridges retain route counts; new bridges include batch skip counts."""
    from unittest.mock import Mock

    backend = object.__new__(CPUMXFP4Experts)
    backend._cuda_tasks = {"one": 11, "four": 22}
    backend._cuda_library = SimpleNamespace()
    if bridge_api != "legacy":

        def collect(tasks, count, output):
            assert list(tasks) == [11, 22]
            assert count == 2
            output[0], output[1] = 17, 3

        backend._cuda_library.dsv41_cuda_tasks_route_counts = collect
    else:
        backend.cuda_stats = Mock(
            return_value={
                "cached_routes": 17,
                "expert_counts": [1, 0, 2],
            }
        )
    if bridge_api == "stats":

        def collect_stats(tasks, count, output):
            assert list(tasks) == [11, 22]
            assert count == 2
            output[:] = (17, 3, 5, 2, 8)

        backend._cuda_library.dsv41_cuda_tasks_route_stats = collect_stats
    assert backend.cuda_route_counts() == (17, 3)
    assert backend.cuda_route_stats() == (
        (17, 3, 5, 2, 8) if bridge_api == "stats" else (17, 3, 0, 0, 0)
    )
    backend._cuda_tasks.clear()
    assert backend.cuda_route_counts() == (0, 0)
    assert backend.cuda_route_stats() == (0, 0, 0, 0, 0)


def test_eager_expert_cache_counts_exclude_padding(monkeypatch):
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

    module = SimpleNamespace(
        host_tokens=3,
        host_hidden=torch.empty(3, 2),
        host_ids=torch.empty(3, 2, dtype=torch.int64),
        host_routes=torch.empty(3, 2),
        host_output=torch.empty(3, 2),
        backend=SimpleNamespace(forward=lambda x, *_: torch.zeros_like(x)),
        gpu_cache=None,
        _eager_gpu_hits=0,
        _eager_cpu_misses=0,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    hidden = torch.ones(3, 2)
    ids = torch.tensor([[-2, 7], [9, -2], [-2, 7]])
    routes = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.0, 0.0]])
    CPUExpertModule._forward_cpu(
        module, hidden, ids, routes, torch.empty_like(hidden), tokens_override=3
    )
    assert (module._eager_gpu_hits, module._eager_cpu_misses) == (2, 2)


def test_decode_cache_policy_uses_observed_groups_and_bounded_transfer_budget():
    import numpy as np

    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy

    policy = TailCachePolicy(
        (1, 2),
        cpu_call_ms=1,
        decode_interval=4,
        decode_budget_ms=8,
        decode_max_swaps=2,
        decode_horizon_steps=256,
    )
    calls = np.array([4, 0, 0, 4, 3])
    # A cold admission uses 6.4 ms, so this cycle cannot admit both candidates.
    selected, _, _ = policy.plan_decode(calls, 4, [0, 1, 2], {0}, costs=np.full(5, 6.4))
    assert selected == [0, 3, 2]
    assert policy.plan_decode(
        calls, 4, [0, 1, 2], {0}, costs=np.full(5, 6.4), remaining_steps=2
    )[0] == [0, 1, 2]
    assert policy.plan_decode(calls, 3, [0, 1, 2], {0})[0] == [0, 1, 2]
    assert policy.plan_decode(calls, 4, [0, 1, 2], {0}, costs=np.ones(5))[0] == [
        0,
        3,
        4,
    ]


def test_adaptive_capacity_preserves_floor_pins_and_transfer_budget():
    import numpy as np

    from vllm.models.deepseek_v4_1.cache_capacity import CapacityPolicy
    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy
    from vllm.models.deepseek_v4_1.expert_cache import GPUExpertCache

    donor, target = (object.__new__(GPUExpertCache) for _ in range(2))
    for cache, selected in ((donor, [0, 1, 2]), (target, [0, 1])):
        cache.num_experts, cache.selected = 5, selected
        cache.dynamic_policy = TailCachePolicy(
            tuple(selected), cpu_call_ms=1, decode_budget_ms=8
        )
        cache.host_packed, cache.host_lru_slots = {}, {}
    caches = [donor, target]
    scores = {
        donor: np.array([0.8, 0.01, 0.03, 0, 0]),
        target: np.array([0.6, 0.7, 0.9, 0.8, 0.65]),
    }
    protected = {donor: {0}, target: set()}
    policy = CapacityPolicy(min_experts=1)
    diagnostics: dict[str, int] = {}
    plan = policy.plan(caches, scores, 128, protected, diagnostics=diagnostics)
    assert [move[:4] for move in plan] == [(donor, 1, target, 2)]
    assert diagnostics["accepted"] == 1
    assert diagnostics["transfer_budget"] == 2
    assert diagnostics["candidates"] == 3
    assert donor.selected == [0, 1, 2]  # Planning cannot change live graph maps.
    diagnostics = {}
    assert policy.plan(caches, scores, 1, protected, diagnostics=diagnostics) == []
    assert diagnostics == {"candidates": 3, "gross_payback": 3}
    target.dynamic_policy = replace(target.dynamic_policy, decode_budget_ms=20)
    plan = policy.plan(caches, scores, 128, protected)
    assert [move[:4] for move in plan] == [(donor, 1, target, 2), (donor, 2, target, 3)]
    assert (
        policy.plan(caches, scores, 128, protected, cpu_costs={donor: 100, target: 1})
        == []
    )
    assert replace(policy, resize=False).plan(caches, scores, 128, protected) == []


@pytest.mark.parametrize("cache_device", [0, 1])
def test_adaptive_capacity_moves_shared_slots_without_recapturing_mixed_decode(
    backend_config, cache_device
):
    """Physical slots beyond a layer's expert IDs remain valid after a transfer."""
    bridge = os.environ.get("DSV41_TEST_CUDA_LIBRARY")
    if backend_config.backend != "ik" or not bridge:
        pytest.skip("Requires the IK CUDA callback bridge")
    if torch.accelerator.device_count() <= cache_device:
        pytest.skip("Requires the requested cache GPU")
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v4_1.cache_capacity import (
        CapacityPolicy,
        SharedExpertPool,
    )
    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

    torch.accelerator.set_device_index(0)
    torch.cuda.set_stream(torch.cuda.current_stream(0))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                n_routed_experts=8,
                hidden_size=128,
                moe_intermediate_size=128,
                num_experts_per_tok=3,
                swiglu_limit=0.125,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=16),
    )
    pool = SharedExpertPool(12, CapacityPolicy(min_experts=1))
    modules, weights = [], []
    graph = None
    try:
        for layer in range(2):
            module = CPUExpertModule(
                replace(
                    backend_config,
                    cuda_library_path=bridge,
                    gpu_cache_experts=6,
                    gpu_cache_prefill=True,
                    gpu_cache_device=cache_device,
                    gpu_cache_static_experts=tuple(range(6)),
                    gpu_cache_dynamic={
                        "mutable_experts": tuple(range(6)),
                        "feedback_weight": 0.9,
                        "host_cache_bytes": 0,
                        "prepack_resident": False,
                    },
                ),
                config,
            )
            modules.append(module)
            pool.register(module.gpu_cache, 20 + layer, list(range(6)))
            weights.append(
                load_weights(module.backend, module.gpu_cache, seed=13 + layer)
            )
            module.finalize()
            module.gpu_cache.warmup()
        caches = [module.gpu_cache for module in modules]
        assert max(caches[1].slot_ids) >= caches[1].num_experts
        addresses = [part.data_ptr() for part in pool.packed]
        map_addresses = [cache.expert_map.data_ptr() for cache in caches]
        hidden = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor([[5, 6, 7]] * 8, device="cuda")
        routes = torch.full((8, 3), 0.7, device="cuda")
        context = ForwardContext({}, {}, {})
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with override_forward_context(context):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    for module in modules:
                        module(hidden, ids, routes)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [module(hidden, ids, routes) for module in modules]
            for phase in range(4):
                if phase == 1:
                    pool.move(caches[0], 5, caches[1], 6)
                    assert [cache.capacity for cache in caches] == [5, 7]
                elif phase == 2:
                    caches[1].select([1, 2, 3, 4, 5, 6, 7])
                elif phase == 3:
                    pool.restore_initial()
                    assert [cache.capacity for cache in caches] == [6, 6]
                graph.replay()
                torch.accelerator.synchronize(0)
                torch.accelerator.synchronize(cache_device)
                for i, module in enumerate(modules):
                    module.backend.check_cuda_errors()
                    expected = reference(
                        hidden.cpu(), ids.cpu(), routes.cpu(), weights[i], 0.125
                    )
                    error = (outputs[i].float().cpu() - expected).norm()
                    assert error / expected.norm() < 0.035
                    assert (
                        module.gpu_cache.membership.sum().item()
                        == module.gpu_cache.capacity
                    )
                assert sum(cache.capacity for cache in caches) == pool.capacity
                assert [part.data_ptr() for part in pool.packed] == addresses
                assert [
                    cache.expert_map.data_ptr() for cache in caches
                ] == map_addresses
                assert all(cache.packed is pool.packed for cache in caches)
    finally:
        torch.accelerator.synchronize(0)
        torch.accelerator.synchronize(cache_device)
        del graph
        for module in modules:
            module.backend.close()


@pytest.fixture
def decode_refresh_cache(monkeypatch):
    from collections import OrderedDict
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.cache_feedback import DecodeRouteFeedback
    from vllm.models.deepseek_v4_1.cache_policy import TailCachePolicy
    from vllm.models.deepseek_v4_1.expert_cache import GPUExpertCache

    cache = object.__new__(GPUExpertCache)
    cache.num_experts, cache.capacity = 4, 2
    cache.device = cache.origin = torch.device("cpu")
    cache.stream = Mock()
    cache.selected = [0, 1]
    cache.packed = (torch.zeros(2, 2),)
    cache.expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    cache.membership = torch.tensor([True, True, False, False])
    cache.host_packed, cache.host_lru_slots = {}, OrderedDict()
    cache.dynamic_policy = TailCachePolicy(
        (1,),
        feedback_weight=0.9,
        decode_interval=4,
        decode_max_swaps=1,
        cpu_call_ms=1,
        repack_transfer_ms=4,
        feedback_debias=True,
    )
    cache.dynamic_pinned = {0}
    cache.enabled = cache.dynamic_enabled = cache.feedback_enabled = True
    cache.decode_feedback = DecodeRouteFeedback(4, cache.origin)
    cache.reloaded_experts = cache.reload_updates = 0
    cache.reload_seconds = 0.0
    cache.feedback_history = None
    cache.feedback_mass, cache.feedback_requests = 0.0, 0
    cache.dynamic_stats = dict.fromkeys(
        (
            "observations",
            "decode_observations",
            "observe_seconds",
            "updates",
            "update_seconds",
            "swaps",
            "estimated_saved_ms",
            "feedback_requests",
        ),
        0,
    )
    cache._pack_expert = lambda e: (torch.full((1, 2), float(e)),)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cache.begin_request(1000)
    return cache


@pytest.mark.parametrize("history", [False, True])
def test_adaptive_capacity_counts_hits_before_and_after_layer_transfer(
    decode_refresh_cache, monkeypatch, history
):
    import copy

    from vllm.models.deepseek_v4_1.cache_capacity import (
        CapacityPolicy,
        SharedExpertPool,
    )
    from vllm.models.deepseek_v4_1.cache_feedback import DecodeRouteFeedback

    donor = decode_refresh_cache
    target = copy.copy(donor)
    target.decode_feedback = DecodeRouteFeedback(4, target.origin)
    target.dynamic_stats = dict(donor.dynamic_stats)
    target.expert_map = donor.expert_map.clone()
    target.membership = donor.membership.clone()
    pool = SharedExpertPool(
        4,
        CapacityPolicy(interval=4, min_experts=1, history=history, online=not history),
    )
    monkeypatch.setattr(pool, "_wait", lambda cache: None)
    for i, cache in enumerate((donor, target)):
        pool.register(cache, 20 + i, [0, 1])
        cache.selected = []
        cache.packed = None
        pool.select(cache, [0, 1])
        cache.begin_request(100)
    pool.begin_request()
    donor_window = torch.tensor([4, 0, 0, 0, 4, 0, 0, 0, 4, 4])
    target_window = torch.tensor([4, 4, 4, 0, 4, 4, 4, 0, 4, 4])
    donor.decode_feedback.counts.add_(donor_window)
    target.decode_feedback.counts.add_(target_window)
    for cache, window in ((donor, donor_window), (target, target_window)):
        cache.feedback_history = window[4:8].numpy() / 4
    pool.refresh(4, 96, history=not history)
    assert pool.stats["checks"] == 0
    pool.refresh(3, 96)
    assert pool.stats["swaps"] == 0
    pool.refresh(4, 96, history=history)
    phase = "history" if history else "decode"
    assert pool.stats[f"{phase}_swaps"] == 1
    assert pool.stats[f"{phase}_decisions"]["accepted"] == 1
    assert donor.selected == [0]
    assert target.selected == [0, 1, 2]
    assert pool.stats["cross_layer_swaps"] == 1
    donor.decode_feedback.counts.add_(donor_window)
    target.decode_feedback.counts.add_(target_window)
    assert donor.finish_request() == (8, 8, 8)
    assert target.finish_request() == (20, 24, 8)
    pool.restore_initial()
    assert [donor.capacity, target.capacity] == [2, 2]
    assert donor.selected == target.selected == [0, 1]
    assert pool.stats["swaps"] == 0


def test_capacity_hit_accounting_does_not_shorten_local_admission_window(
    decode_refresh_cache,
):
    from unittest.mock import Mock

    cache = decode_refresh_cache
    pool = Mock()
    cache.capacity_pool = pool
    window = torch.tensor([2, 0, 2, 0, 2, 0, 2, 0, 2, 2])
    cache.decode_feedback.counts.add_(window)
    cache._track_decode_snapshot(cache.decode_feedback.snapshot())
    cache.decode_feedback.counts.add_(window)
    cache.refresh_decode(4)
    pool.select.assert_called_once_with(cache, [0, 2], True)
    saved = cache.learning_state()
    cache.restore_learning_state(saved)
    assert cache.learning_state() == saved
    assert cache.finish_request() == (4, 8, 4)


def test_decode_refresh_follows_changing_hot_experts_with_stable_cache_storage(
    decode_refresh_cache,
):
    """Refresh mid-request and count each window against its actual selection."""
    cache = decode_refresh_cache
    addresses = [part.data_ptr() for part in cache.packed]
    # Two tokens can share an expert: routes and per-group calls are distinct.
    window = torch.tensor([4, 0, 8, 0, 4, 0, 4, 0, 4, 8])
    cache.decode_feedback.counts.add_(window)
    cache.refresh_decode(3)
    assert cache.selected == [0, 1]
    cache.refresh_decode(4)
    assert cache.selected == [0, 2]
    assert cache.dynamic_stats["swaps"] == 1
    torch.testing.assert_close(cache.decode_feedback.counts, window)
    cache.refresh_decode(4)
    assert cache.dynamic_stats["decode_observations"] == 1
    cache.decode_feedback.counts.add_(window)
    cache.refresh_decode(8)
    assert cache.dynamic_stats["swaps"] == 1  # Stable demand must not churn.
    drift = torch.tensor([4, 0, 0, 8, 4, 0, 0, 4, 4, 8])
    cache.decode_feedback.counts.add_(drift)
    cache.refresh_decode(12)
    assert cache.selected == [0, 3]
    assert cache.dynamic_stats["swaps"] == 2
    assert [part.data_ptr() for part in cache.packed] == addresses
    cache.decode_feedback.counts.add_(drift)
    cache.restore_learning_state(cache.learning_state())
    assert cache.finish_request() == (32, 48, 16)
    assert cache.feedback_requests == 1
    assert cache.feedback_history.tolist() == [1.0, 0.0, 0.5, 0.5]
    cache.begin_request(10)
    assert cache.finish_request() == (0, 0, 0)


@pytest.mark.parametrize("flag", ["dynamic_enabled", "enabled", "feedback_enabled"])
def test_decode_refresh_does_not_update_during_warmup_or_when_disabled(
    decode_refresh_cache,
    flag,
):
    cache = decode_refresh_cache
    setattr(cache, flag, False)
    cache.refresh_decode(100)
    assert cache.dynamic_stats["decode_observations"] == 0
    assert cache._last_decode_check == 0


def test_decode_refresh_visits_one_layer_per_step_and_limits_nearly_finished_request():
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    state = object.__new__(DeepseekV41ModelState)
    state.cpu_async_modules = [SimpleNamespace(gpu_cache=Mock()) for _ in range(20)]
    state.hybrid_active = "request"
    state.hybrid_requests = {"request": 1000}
    state.hybrid_capacity_pools = (Mock(),)
    for step in range(1, 81):
        state.hybrid_steps = state.hybrid_request_steps = step
        state.hybrid_generated = step * 3
        state._refresh_decode_cache()
        assert (
            sum(m.gpu_cache.refresh_decode.call_count for m in state.cpu_async_modules)
            == step
        )
    assert all(
        m.gpu_cache.refresh_decode.call_count == 4 for m in state.cpu_async_modules
    )
    state.hybrid_requests["request"] = state.hybrid_generated + 2
    state._refresh_decode_cache()
    assert state.cpu_async_modules[-1].gpu_cache.refresh_decode.call_args.kwargs == {
        "remaining_steps": pytest.approx(1 / 3)
    }
    state.hybrid_capacity_pools[0].refresh.assert_called_with(80, pytest.approx(1 / 3))
    state.cpu_async_modules[-1].gpu_cache = None
    state._refresh_decode_cache()
    assert state.hybrid_capacity_pools[0].refresh.call_count == 82


@pytest.fixture
def numa_partitions(request, monkeypatch):
    """Exercise node ownership even on single-node CI, without fake OS nodes."""
    from vllm.model_executor.layers.fused_moe.experts import cpu_mxfp4 as adapter
    from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4_numa import (
        NumaNode,
        cpu_numa_nodes,
    )

    count = request.param
    if not count:
        return 0
    available = cpu_numa_nodes()
    if available[0].node_id < 0 or len(available[0].cpus) < count * 2:
        pytest.skip("Need enough cores on one real node to emulate row partitions")
    cpus = available[0].cpus[: count * 2]
    # Unequal core counts exercise proportional row ownership as well.
    sizes = [1, 3] + [2] * (count - 2)
    nodes, first = [], 0
    for size in sizes:
        nodes.append(NumaNode(available[0].node_id, cpus[first : first + size]))
        first += size
    monkeypatch.setattr(adapter, "cpu_numa_nodes", lambda: tuple(nodes))
    return count * 2


@pytest.mark.parametrize("numa_partitions", [0, 2, 4, 8], indirect=True)
@pytest.mark.parametrize(
    "hidden,intermediate", [(512, 1056), (5120, 1056), (5120, 2304)]
)
def test_native_numa_rows_survive_thread_switches_and_large_batches(
    backend_config, numa_partitions, hidden, intermediate
):
    """Node-local scheduling must preserve graph numerics, cached routes and tails."""
    if backend_config.backend != "ik":
        pytest.skip("NUMA row sharding uses IK")
    backend = CPUMXFP4Experts(backend_config, 4, hidden, intermediate, 3, 0.125, 257)
    original_affinity = os.sched_getaffinity(0)
    try:
        load_weights(backend)
        assert os.sched_getaffinity(0) == original_affinity
        pages = backend.numa_page_counts()
        if pages is not None:
            assert pages[0] > 0
            assert pages[0] == sum(pages[1:])
            # MPOL_LOCAL permits fallback, and malloc can reuse resident pages.
            assert pages[3] == 0
            per_expert = [backend.numa_page_counts(e) for e in range(4)]
            assert tuple(map(sum, zip(*per_expert))) == pages
        for invalid_expert in (-1, 4):
            with pytest.raises(ValueError, match="Invalid expert index"):
                backend.numa_page_counts(invalid_expert)
        with pytest.raises(RuntimeError, match="before loading"):
            backend._configure_numa()
        generator = torch.Generator().manual_seed(93)
        for tokens in (1, 4, 8, 96, 128, 257):
            x = torch.randn(tokens, hidden, generator=generator).bfloat16()
            ids = torch.stack(
                [torch.randperm(4, generator=generator)[:3] for _ in range(tokens)]
            )
            ids[::2, 0] = -2
            routes = torch.rand(tokens, 3, generator=generator)
            backend.set_execution_mode("graph")
            expected = backend.forward(x, ids, routes)
            backend.set_execution_mode("compact")
            partitioned = None
            for threads in dict.fromkeys((1, 3, numa_partitions or 4, 4)):
                backend.set_num_threads(threads)
                actual = backend.forward(x, ids, routes)
                # Chunking can select GEMV for the last token instead of GEMM.
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
                if partitioned is None:
                    partitioned = actual.clone()
                else:
                    torch.testing.assert_close(actual, partitioned, atol=0, rtol=0)
                assert os.sched_getaffinity(0) == original_affinity
    finally:
        backend.close()


@pytest.mark.parametrize("numa_partitions", [0, 4], indirect=True)
@pytest.mark.parametrize("reuse_workers", [False, True])
def test_native_numa_callback_workers_inherit_all_configured_nodes(
    backend_config, numa_partitions, reuse_workers, monkeypatch
):
    """GPU-local callers must not strand OpenMP workers on one node at barriers."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vllm.model_executor.layers.fused_moe.experts import cpu_mxfp4 as adapter
    from vllm.model_executor.layers.fused_moe.experts.cpu_mxfp4_numa import NumaNode

    nodes = adapter.cpu_numa_nodes()
    if backend_config.backend != "ik" or len(nodes) < 2:
        pytest.skip("Requires IK and multiple NUMA row partitions")
    backend = CPUMXFP4Experts(backend_config, 4, 128, 96, 3, 0.125, 4)
    cpus = {cpu for node in nodes for cpu in node.cpus}
    local = {nodes[-1].cpus[-1]}
    unsharded = None
    try:
        load_weights(backend)
        if reuse_workers:
            with monkeypatch.context() as patch:
                patch.setattr(
                    adapter,
                    "cpu_numa_nodes",
                    lambda: (NumaNode(nodes[0].node_id, tuple(sorted(cpus))),),
                )
                unsharded = CPUMXFP4Experts(backend_config, 4, 128, 96, 3, 0.125, 4)
            load_weights(unsharded)
        x = torch.randn(4, 128)
        ids = torch.tensor([[0, 1, 3]] * 4)
        routes = torch.full((4, 3), 0.5)
        expected = backend.forward(x, ids, routes)
        previous = {int(p.name) for p in Path("/proc/self/task").iterdir()}

        def callback():
            os.sched_setaffinity(0, local)
            if unsharded is not None:
                unsharded.forward(x, ids, routes)
            result = backend.forward(x, ids, routes)
            assert os.sched_getaffinity(0) == local
            return threading.get_native_id(), result

        with ThreadPoolExecutor(max_workers=1) as executor:
            caller, actual = executor.submit(callback).result()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            created = {int(p.name) for p in Path("/proc/self/task").iterdir()}
            workers = created - previous - {caller}
            assert len(workers) == backend_config.num_threads - 1
            assert all(os.sched_getaffinity(tid) == cpus for tid in workers)
    finally:
        if unsharded is not None:
            unsharded.close()
        backend.close()


def test_registered_host_mapping_interleaves_before_first_touch(monkeypatch):
    """Pinning/copying must follow policy setup without allocating extra storage."""
    import ctypes
    import mmap

    from vllm.models.deepseek_v4_1 import host_memory

    calls = []

    def mbind(address, length, mode, mask, maxnode, flags):
        calls.append(
            (length.value, mode.value, tuple(mask), maxnode.value, flags.value)
        )
        return 0

    monkeypatch.setattr(host_memory, "host_memory_nodes", lambda: (1, 3, 67))
    monkeypatch.setattr(
        host_memory, "get_libnuma", lambda: SimpleNamespace(mbind=mbind)
    )
    with mmap.mmap(-1, mmap.PAGESIZE) as owner:
        host_memory.interleave_host_mapping(owner)
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        expected = [0] * ((67 + bits) // bits)
        for node in (1, 3, 67):
            expected[node // bits] |= 1 << (node % bits)
        assert calls == [
            (mmap.PAGESIZE, 3, tuple(expected), len(expected) * bits + 1, 0)
        ]
        assert owner[:8] == bytes(8)


def test_registered_host_mapping_keeps_highest_node_in_kernel_policy(monkeypatch):
    """Linux subtracts one from maxnode; a mask's last node must survive that."""
    import ctypes
    import mmap

    from vllm.models.deepseek_v4_1 import host_memory

    nodes = host_memory.host_memory_nodes()
    libnuma = host_memory.get_libnuma()
    if not nodes or libnuma is None:
        pytest.skip("NUMA policy APIs unavailable")
    node = max(nodes)
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * max(16, (node + bits) // bits))()
    mask[node // bits] = 1 << (node % bits)
    with mmap.mmap(
        -1, mmap.PAGESIZE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    ) as probe:
        address = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(probe)))
        rc = libnuma.mbind(
            address,
            ctypes.c_ulong(len(probe)),
            3,
            mask,
            ctypes.c_ulong(len(mask) * bits + 1),
            0,
        )
        if rc:
            pytest.skip("Kernel denies mbind for this test process")
    # A duplicate real node exercises policy setup on single-node test hosts.
    monkeypatch.setattr(host_memory, "host_memory_nodes", lambda: (node, node))
    with mmap.mmap(
        -1, mmap.PAGESIZE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    ) as owner:
        host_memory.interleave_host_mapping(owner)
        mode = ctypes.c_int()
        actual = type(mask)()
        address = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(owner)))
        rc = libnuma.get_mempolicy(
            ctypes.byref(mode),
            actual,
            ctypes.c_ulong(len(actual) * bits + 1),
            address,
            2,
        )
        assert rc == 0 and mode.value == 3
        assert tuple(actual) == tuple(mask)


def test_checkpoint_reclaim_preserves_private_data_and_excludes_other_storage(
    tmp_path, monkeypatch
):
    """Reclaim only full, consumed checkpoint pages after the native copy."""
    import ctypes
    import mmap

    from vllm.models.deepseek_v4_1 import host_memory

    if not Path("/proc/self/maps").exists():
        pytest.skip("Checkpoint page reclaim uses Linux mappings")
    page = mmap.PAGESIZE
    path = tmp_path / "experts.safetensors"
    path.write_bytes(bytes(5 * page + 300))
    calls = []
    native = ctypes.CDLL(None, use_errno=True).madvise
    native.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    native.restype = ctypes.c_int

    def pageout(address, length, advice):
        calls.append((address, length, advice))
        return native(address, length, advice)

    monkeypatch.setattr(
        host_memory.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(madvise=pageout)
    )
    with path.open("rb") as file:
        owner = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
    source = torch.frombuffer(owner, dtype=torch.uint8)
    try:
        source[2 * page] = 113
        reclaimer = host_memory.CheckpointPageReclaimer()
        reclaimer.add(source[3:-7], torch.zeros(4 * page), source[::2])
        assert not calls
        reclaimer.flush()
        assert calls == [(source.data_ptr() + page, 4 * page, 21)]
        assert source[2 * page].item() == 113
        assert source.sum().item() == 113
        assert not reclaimer.weights and reclaimer.pending_bytes == 0
        reclaimer.flush()
        assert len(calls) == 1
        monkeypatch.setattr(host_memory, "LOAD_CHUNK_BYTES", page)
        reclaimer.add(source[3:-7])
        reclaimer.add(source[3:-7])
        reclaimer.flush()
        assert len(calls) == 3
        assert reclaimer.executor is None and reclaimer.pending is None
    finally:
        del source
        owner.close()
