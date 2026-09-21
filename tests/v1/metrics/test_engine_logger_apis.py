# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger


@pytest.fixture
def log_stats_enabled_engine_args():
    """
    Shared fixture providing common AsyncEngineArgs configuration
    used across multiple tests.
    """
    return AsyncEngineArgs(
        model="distilbert/distilgpt2",
        dtype="half",
        disable_log_stats=False,
        enforce_eager=True,
    )


@pytest.mark.asyncio
async def test_async_llm_replace_default_loggers(log_stats_enabled_engine_args):
    """
    RayPrometheusStatLogger should replace the default PrometheusStatLogger
    """

    engine = AsyncLLM.from_engine_args(
        log_stats_enabled_engine_args, stat_loggers=[RayPrometheusStatLogger]
    )
    assert isinstance(engine.logger_manager.stat_loggers[0], RayPrometheusStatLogger)
    engine.shutdown()


@pytest.mark.asyncio
async def test_async_llm_add_to_default_loggers(log_stats_enabled_engine_args):
    """
    It's still possible to use custom stat loggers exclusively by passing
    disable_log_stats=True in addition to a list of custom stat loggers.
    """
    # Create engine_args with disable_log_stats=True for this test
    disabled_log_engine_args = copy.deepcopy(log_stats_enabled_engine_args)
    disabled_log_engine_args.disable_log_stats = True

    # Disable default loggers; pass custom stat logger to the constructor
    engine = AsyncLLM.from_engine_args(
        disabled_log_engine_args, stat_loggers=[DummyStatLogger]
    )

    assert len(engine.logger_manager.stat_loggers) == 2
    assert len(engine.logger_manager.stat_loggers[0].per_engine_stat_loggers) == 1
    assert isinstance(
        engine.logger_manager.stat_loggers[0].per_engine_stat_loggers[0],
        DummyStatLogger,
    )

    # log_stats is still True, since custom stat loggers are used
    assert engine.log_stats

    engine.shutdown()


@pytest.mark.parametrize("aggregate", [False, True])
def test_expert_cache_logging_is_weighted_per_interval_and_follows_spec(
    aggregate, monkeypatch
):
    """Route-weighted deltas must survive IPC, aggregate, and reset each log."""
    from types import SimpleNamespace

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger, LoggingStatLogger
    from vllm.v1.metrics.stats import ExpertCacheStats, SchedulerStats
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
    from vllm.v1.spec_decode.metrics import SpecDecodingStats

    config = SimpleNamespace(
        model_config=None,
        kv_transfer_config=None,
        observability_config=SimpleNamespace(
            cudagraph_metrics=False, enable_mfu_metrics=False
        ),
    )
    stat_logger = (
        AggregatedLoggingStatLogger(config, [0, 1])
        if aggregate
        else LoggingStatLogger(config)
    )
    messages = []
    emit = lambda fmt, *args: messages.append(fmt % args)
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.info", emit)
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.debug", emit)
    first = ExpertCacheStats(
        gpu_hits=90,
        cpu_misses=10,
        decode_checks=2,
        updates=2,
        experts_reloaded=4,
        reload_seconds=0.025,
        host_lru_hits=3,
        repacked_experts=1,
    )
    stats = SchedulerStats(
        expert_cache_stats=first,
        spec_decoding_stats=SpecDecodingStats(
            num_spec_tokens=3,
            num_drafts=1,
            num_draft_tokens=3,
            num_accepted_tokens=2,
            num_accepted_tokens_per_pos=[1, 1, 0],
        ),
    )
    output = EngineCoreOutputs(scheduler_stats=stats)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(output))
    stat_logger.record(decoded.scheduler_stats, None, engine_idx=0)
    stat_logger.record(
        SchedulerStats(expert_cache_stats=ExpertCacheStats(gpu_hits=1, cpu_misses=9)),
        None,
        engine_idx=1 if aggregate else 0,
    )
    stat_logger.log()
    assert len(messages) == 3
    assert "Avg prompt throughput" in messages[0]
    assert "SpecDecoding metrics" in messages[1]
    assert "GPU hit rate: 82.73% (91/110 routes), CPU routes: 19" in messages[2]
    assert (
        "Expert replacements: 4, Cache updates: 2, Reload time: 25.0 ms" in messages[2]
    )
    assert "Host LRU hits: 3, Repacked experts: 1, Decode checks: 2" in messages[2]
    assert first.gpu_hits == 90  # Recording must not mutate the worker's payload.
    stat_logger.record(
        SchedulerStats(expert_cache_stats=ExpertCacheStats(gpu_hits=1)), None
    )
    messages.clear()
    stat_logger.log()
    assert "GPU hit rate: 100.00% (1/1 routes), CPU routes: 0" in messages[-1]
    assert "Expert replacements: 0, Cache updates: 0" in messages[-1]
    messages.clear()
    stat_logger.log()
    assert len(messages) == 1  # No stale expert statistics on an idle interval.


def test_expert_cache_logging_without_routes_is_not_a_measured_zero(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.metrics.loggers import LoggingStatLogger
    from vllm.v1.metrics.stats import ExpertCacheStats, SchedulerStats

    config = SimpleNamespace(
        model_config=None,
        kv_transfer_config=None,
        observability_config=SimpleNamespace(
            cudagraph_metrics=False, enable_mfu_metrics=False
        ),
    )
    stat_logger = LoggingStatLogger(config)
    messages = []
    emit = lambda fmt, *args: messages.append(fmt % args)
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.info", emit)
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.debug", emit)
    stat_logger.record(SchedulerStats(), None)
    stat_logger.log()
    assert len(messages) == 1  # Models without an expert cache keep the old log.
    stat_logger.record(SchedulerStats(expert_cache_stats=ExpertCacheStats()), None)
    stat_logger.log()
    assert "GPU hit rate: N/A (0/0 routes)" in messages[-1]
