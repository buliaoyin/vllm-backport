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
    with pytest.raises(ValueError, match="single request"):
        cpu_moe_config(config, 20)
    config.scheduler_config.max_num_seqs = 1
    with pytest.raises(ValueError, match="Missing static expert selection"):
        cpu_moe_config(config, 22)


@pytest.mark.parametrize("draft_tokens", [3, 5])
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


def load_weights(backend, gpu_cache=None):
    generator = torch.Generator(device="cpu").manual_seed(13)
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


@pytest.mark.parametrize("hidden,intermediate", [(256, 128), (5120, 2304)])
def test_native_export_preserves_bytes_after_checkpoint_is_released(
    backend_config, hidden, intermediate
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
        with pytest.raises(ValueError, match="unloaded"):
            backend.export_expert(2, 0)
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
@pytest.mark.parametrize("intermediate", [128, 160])
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
        for tokens in (1, 4, 16, 17, 32, 64, 65, 129, 1):
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


def test_native_cpu_callback_replays_without_python_and_masks_padding(backend_config):
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
    module.finalize()
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
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
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
        assert shared.maybe_forward_async(result)
        shared.wait()
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
    "capacities", [[600, 600, 160], [1700, 1500, 1500, 2000], [0, 8, 0]]
)
def test_automatic_cache_respects_each_devices_remaining_budget(capacities):
    from vllm.models.deepseek_v4_1.hybrid import plan_expert_cache

    plan = plan_expert_cache(capacities, local_device=len(capacities) - 1)
    assert len(plan) == 20
    for device, budget in enumerate(capacities):
        assert sum(count for owner, count in plan if owner == device) <= budget
    assert all(0 <= count < 384 for _, count in plan)
    if sum(capacities) > 20 * 8:
        assert all(count > 0 for _, count in plan)


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
            cuda_route_counts=Mock(return_value=(hits, misses))
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
    cached.backend.cuda_route_counts.return_value = (107, 21)
    uncached.backend.cuda_route_counts.return_value = (0, 128)
    cached._eager_gpu_hits, cached._eager_cpu_misses = 2, 3
    cache.reload_updates += 1
    cache.reloaded_experts += 4
    cache.reload_seconds += 0.5
    cache.dynamic_stats["lru_hits"] += 3
    cache.dynamic_stats["repacked_experts"] += 1
    assert state.take_expert_cache_stats() == ExpertCacheStats(
        gpu_hits=9,
        cpu_misses=12,
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


@pytest.mark.parametrize("batched", [False, True])
def test_cuda_route_counts_aggregate_all_graph_shapes(batched):
    """Both bridge versions count actual routes across all captured shapes."""
    from unittest.mock import Mock

    backend = object.__new__(CPUMXFP4Experts)
    backend._cuda_tasks = {"one": 11, "four": 22}
    backend._cuda_library = SimpleNamespace()
    if batched:

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
    assert backend.cuda_route_counts() == (17, 3)
    backend._cuda_tasks.clear()
    assert backend.cuda_route_counts() == (0, 0)


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
