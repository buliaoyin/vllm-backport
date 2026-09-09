# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import Mock, call

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


def test_batch_pop_updates_grammar_before_ingesting_bound_drafts():
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = Mock()
    engine_core.scheduler.update_from_output.return_value = {}
    engine_core._process_aborts_queue = Mock()
    engine_core.capture_iteration_details = lambda *_: nullcontext(None)
    engine_core.log_error_detail = lambda *_: nullcontext()
    engine_core._attach_iteration_details = Mock()

    draft_token_ids = DraftTokenIds(
        req_ids=["request"],
        draft_token_ids=[[1, 2, 3]],
        producer_step_id=7,
    )
    model_output = ModelRunnerOutput(
        req_ids=["request"],
        req_id_to_index={"request": 0},
        sampled_token_ids=[[4]],
        draft_token_ids=draft_token_ids,
    )
    model_future: Future[ModelRunnerOutput] = Future()
    model_future.set_result(model_output)
    execute_future: Future[None] = Future()
    execute_future.set_result(None)
    scheduler_output = SchedulerOutput.make_empty()
    batch_queue = deque([(model_future, scheduler_output, execute_future)])

    engine_core._pop_and_process_batch(batch_queue)

    assert engine_core.scheduler.mock_calls[:2] == [
        call.update_from_output(scheduler_output, model_output),
        call.update_draft_token_ids(draft_token_ids),
    ]
