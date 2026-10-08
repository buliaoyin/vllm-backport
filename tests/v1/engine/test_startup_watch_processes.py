# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import signal
import weakref
from contextlib import nullcontext
from multiprocessing import connection, get_context
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest
import zmq

import vllm.platforms as platforms
from vllm.v1.engine import core as core_module
from vllm.v1.engine import utils as engine_utils
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    CoreEngineProcManager,
    EngineZmqAddresses,
    wait_for_engine_startup,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("is_rocm", "request_timeout", "manager_timeout", "process_timeout"),
    [
        (True, 0, 0, 15.0),
        (True, 0, 7, 7),
        (True, 0, None, None),
        (False, 0, 0, 0),
        (True, 7, 0, 0),
    ],
)
def test_engine_core_process_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    request_timeout: float | None,
    manager_timeout: float | None,
    process_timeout: float | None,
):
    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = request_timeout
    manager._process_cleanup_timeout = 0
    manager._hybrid_mps = None
    manager._shutdown_lock = Lock()
    manager.manager_stopped = Event()
    manager.processes = [object()]
    detach_results = iter((object(), None))
    manager._finalizer = SimpleNamespace(detach=lambda: next(detach_results))

    shutdown_calls = []
    monkeypatch.setattr(
        engine_utils,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: is_rocm),
    )
    monkeypatch.setattr(
        engine_utils,
        "shutdown",
        lambda processes, timeout: shutdown_calls.append((processes, timeout)),
    )

    manager.shutdown(timeout=manager_timeout)
    manager.shutdown(timeout=manager_timeout)

    assert manager.manager_stopped.is_set()
    assert shutdown_calls == [(manager.processes, process_timeout)]


@pytest.mark.parametrize(
    (
        "is_rocm",
        "shutdown_state",
        "has_work",
        "shutdown_timeout",
        "exit_code",
        "expected_calls",
    ),
    [
        (
            True,
            EngineShutdownState.SHUTTING_DOWN,
            False,
            0,
            None,
            ["shutdown", "freeze"],
        ),
        (False, EngineShutdownState.SHUTTING_DOWN, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.RUNNING, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, True, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 7, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 0, 1, ["shutdown"]),
    ],
)
def test_freeze_gc_after_clean_rocm_engine_core_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    shutdown_state: EngineShutdownState,
    has_work: bool,
    shutdown_timeout: int,
    exit_code: int | None,
    expected_calls: list[str],
):
    calls: list[str] = []
    vllm_config = SimpleNamespace(shutdown_timeout=shutdown_timeout)
    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: has_work,
        vllm_config=vllm_config,
    )

    def run_busy_loop():
        proc.shutdown_state = shutdown_state
        raise SystemExit(exit_code)

    proc.run_busy_loop = run_busy_loop
    proc.shutdown = lambda: calls.append("shutdown")
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    vllm_config.parallel_config = parallel_config

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: is_rocm)
    )
    monkeypatch.setattr(core_module.gc, "freeze", lambda: calls.append("freeze"))

    with pytest.raises(SystemExit):
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert calls == expected_calls


def _run_cleanup_with_signal(completed, followup_signal):
    monkeypatch = pytest.MonkeyPatch()
    config = SimpleNamespace(
        shutdown_timeout=0,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            numa_bind=False,
            reconfigure_for_independent_dp_rank=lambda: None,
        ),
    )
    proc = SimpleNamespace(
        vllm_config=config,
        has_work=lambda: False,
        shutdown_state=EngineShutdownState.SHUTTING_DOWN,
    )

    def run_busy_loop():
        raise SystemExit(0)

    def shutdown():
        os.kill(os.getpid(), followup_signal)
        completed.write_text("finished")

    proc.run_busy_loop = run_busy_loop
    proc.shutdown = shutdown
    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )
    EngineCoreProc.run_engine_core(vllm_config=config)


@pytest.mark.parametrize("followup_signal", [signal.SIGTERM, signal.SIGINT])
def test_engine_cleanup_survives_followup_shutdown_signal(tmp_path, followup_signal):
    """Ctrl+C and the API's subsequent terminate must allow cleanup to finish."""
    completed = tmp_path / "cleaned"
    child = get_context("spawn").Process(
        target=_run_cleanup_with_signal, args=(completed, followup_signal)
    )
    child.start()
    try:
        child.join(timeout=30)
        assert child.exitcode == 0
        assert completed.read_text() == "finished"
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def _wait_past_shutdown_deadline(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        signal.pause()


@pytest.mark.parametrize("finalize", [False, True])
def test_parent_reclaims_mps_after_engine_exceeds_shutdown_deadline(finalize):
    """MPS cleanup must run after the real engine process is force-killed."""
    ctx = get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(target=_wait_past_shutdown_deadline, args=(ready,))
    child.start()
    calls = []

    def close_mps():
        child.join(timeout=5)
        assert child.exitcode == -signal.SIGKILL
        calls.append("closed")

    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = 0
    manager._process_cleanup_timeout = 0.1
    manager._hybrid_mps = SimpleNamespace(close=close_mps)
    manager._shutdown_lock = Lock()
    manager.manager_stopped = Event()
    manager.processes = [child]
    manager._finalizer = weakref.finalize(
        manager, manager._shutdown, manager.processes, manager._hybrid_mps, timeout=0.1
    )
    try:
        assert ready.wait(timeout=30)
        if finalize:
            manager._finalizer()
        else:
            manager.shutdown(timeout=0)
        manager.shutdown(timeout=0)
        assert calls == ["closed"]
    finally:
        if child.is_alive():
            child.kill()
        child.join(timeout=5)


def test_engine_start_failure_closes_parent_owned_mps(monkeypatch):
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1 import hybrid_runtime

    mps = Mock()
    monkeypatch.setattr(hybrid_runtime, "HybridMPS", lambda: mps)
    proc = Mock(exitcode=None, **{"is_alive.return_value": False})
    proc.name = "EngineCore"
    proc.start.side_effect = RuntimeError("spawn failed")
    monkeypatch.setattr(
        engine_utils,
        "get_mp_context",
        lambda: SimpleNamespace(Process=lambda **_: proc),
    )
    monkeypatch.setattr(
        engine_utils.numa_utils,
        "configure_subprocess",
        lambda *args, **kwargs: nullcontext(),
    )
    config = SimpleNamespace(
        shutdown_timeout=0,
        additional_config={"deepseek_v41_hybrid": {}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1, assigned_physical_gpu_ids=None
        ),
    )
    with pytest.raises(RuntimeError, match="spawn failed"):
        CoreEngineProcManager(1, 0, 0, config, True, "unused", object, False)
    mps.close.assert_called_once_with()
    mps.restore_environment.assert_called_once_with()


def test_shutdown_waits_for_mps_cleanup_started_by_another_thread():
    entered, release, completed = Event(), Event(), Event()

    def close_mps():
        entered.set()
        assert release.wait(timeout=10)

    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = manager._process_cleanup_timeout = 0
    manager._shutdown_lock = Lock()
    manager._hybrid_mps = SimpleNamespace(close=close_mps)
    manager.manager_stopped = Event()
    manager.processes = []
    manager._finalizer = weakref.finalize(
        manager, manager._shutdown, manager.processes, manager._hybrid_mps
    )
    monitor = Thread(target=manager.shutdown)

    def api_shutdown():
        manager.shutdown()
        completed.set()

    api = Thread(target=api_shutdown)
    monitor.start()
    try:
        assert entered.wait(timeout=5)
        api.start()
        assert not completed.wait(timeout=0.1)
    finally:
        release.set()
        monitor.join(timeout=5)
        if api.ident is not None:
            api.join(timeout=5)
    assert completed.is_set()


class _FinishedProcess:
    name = "RustFrontend"

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @property
    def exitcode(self):
        return 1


def test_wait_for_engine_startup_reports_watched_process_exit():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [_FinishedProcess(recv)]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,  # type: ignore[arg-type]
                coordinated_dp=False,
                cache_config=None,  # type: ignore[arg-type]
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)
