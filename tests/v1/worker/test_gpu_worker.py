# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu.pp_utils import (
    PPRecvBufferGuard,
    PPRecvBufferState,
)
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


def test_pp_recv_buffer_guard_allows_consecutive_dspark_batches():
    guard = PPRecvBufferGuard()

    for expected_generation, stream in enumerate((object(), object()), start=1):
        generation = guard.begin_receive(stream)
        assert generation == expected_generation
        assert guard.state is PPRecvBufferState.RECEIVING

        guard.finish_receive(generation, stream)
        assert guard.state is PPRecvBufferState.MODEL

        guard.begin_dspark(stream)
        assert guard.state is PPRecvBufferState.DSPARK

        guard.release(stream)
        assert guard.state is PPRecvBufferState.FREE


def test_pp_recv_buffer_guard_tracks_stream_per_generation():
    guard = PPRecvBufferGuard()
    warmup_stream = object()
    other_stream = object()

    generation = guard.begin_receive(warmup_stream)
    guard.finish_receive(generation, warmup_stream)

    with pytest.raises(RuntimeError, match="owned by stream"):
        guard.begin_dspark(other_stream)

    guard.begin_dspark(warmup_stream)
    guard.release(warmup_stream)

    generation = guard.begin_receive(other_stream)
    guard.finish_receive(generation, other_stream)
    guard.release(other_stream)


def test_pp_recv_buffer_guard_rejects_overlapping_receive():
    guard = PPRecvBufferGuard()
    generation = guard.begin_receive()

    with pytest.raises(RuntimeError, match="receiving"):
        guard.begin_receive()
    with pytest.raises(RuntimeError, match="Stale PP receive completion"):
        guard.finish_receive(generation + 1)

    guard.finish_receive(generation)
    guard.release()


def test_async_intermediate_tensors_marks_complete_after_comm():
    events: list[str] = []

    class DummyWork:
        def wait(self) -> None:
            events.append("wait")

    intermediate_tensors = AsyncIntermediateTensors(
        {"hidden_states": torch.zeros(1)},
        comm_handles=[DummyWork()],  # type: ignore[list-item]
        comm_postprocess=[lambda: events.append("postprocess")],
        comm_complete_callback=lambda: events.append("complete"),
    )

    assert not events
    assert intermediate_tensors.tensors["hidden_states"].item() == 0
    assert events == ["wait", "postprocess", "complete"]

    _ = intermediate_tensors.tensors
    assert events == ["wait", "postprocess", "complete"]
