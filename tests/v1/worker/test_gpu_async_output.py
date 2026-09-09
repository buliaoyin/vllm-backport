# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.async_utils import AsyncOutput

pytestmark = pytest.mark.cpu_test


def _make_async_output_stub() -> AsyncOutput:
    output = object.__new__(AsyncOutput)
    output.model_runner_output = ModelRunnerOutput(
        req_ids=["plain", "structured"],
        req_id_to_index={"plain": 0, "structured": 1},
    )
    output.copy_event = Mock()
    output._main_stream = Mock()
    output._copy_stream = Mock()
    output._draft_copy_event = None
    output._draft_token_ids = None
    output._draft_req_ids = []
    output._draft_req_indices = []
    output._draft_producer_step_id = None
    output.sampled_token_ids = np.array([[1], [2]])
    output.num_sampled_tokens_np = np.array([1, 1])
    output.sampling_mask_tensors = None
    output.routed_experts_cpu = None
    output.num_nans = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}
    output._has_fault = None
    return output


def test_async_outputs_keep_independent_draft_snapshots(
    monkeypatch: pytest.MonkeyPatch,
):
    draft_events = [Mock(), Mock()]
    monkeypatch.setattr(
        async_utils.torch.cuda,
        "Event",
        Mock(side_effect=draft_events),
    )
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(
        async_utils,
        "async_copy_to_np",
        lambda tensor: tensor.array.copy(),
    )

    output_a = _make_async_output_stub()
    output_b = _make_async_output_stub()
    tensor_a = SimpleNamespace(
        array=np.array([[11, 12], [21, 22]]),
        record_stream=Mock(),
    )
    tensor_b = SimpleNamespace(
        array=np.array([[31, 32], [41, 42]]),
        record_stream=Mock(),
    )

    output_a.set_draft_token_ids(
        req_ids=["plain", "structured"],
        draft_token_ids=tensor_a,
        structured_output_request_ids=["structured"],
        producer_step_id=7,
    )
    output_b.set_draft_token_ids(
        req_ids=["plain", "structured"],
        draft_token_ids=tensor_b,
        structured_output_request_ids=["structured"],
        producer_step_id=8,
    )

    result_a = output_a.get_output()
    result_b = output_b.get_output()

    assert result_a.draft_token_ids is not None
    assert result_a.draft_token_ids.producer_step_id == 7
    assert result_a.draft_token_ids.req_ids == ["structured"]
    assert result_a.draft_token_ids.draft_token_ids == [[21, 22]]
    assert result_b.draft_token_ids is not None
    assert result_b.draft_token_ids.producer_step_id == 8
    assert result_b.draft_token_ids.req_ids == ["structured"]
    assert result_b.draft_token_ids.draft_token_ids == [[41, 42]]
    assert output_a._draft_copy_event is draft_events[0]
    assert output_b._draft_copy_event is draft_events[1]
    draft_events[0].synchronize.assert_called_once_with()
    draft_events[1].synchronize.assert_called_once_with()


def test_sync_output_preserves_drafts_for_plain_requests(monkeypatch):
    monkeypatch.setattr(async_utils.torch.cuda, "Event", Mock(return_value=Mock()))
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(
        async_utils, "async_copy_to_np", lambda tensor: tensor.array.copy()
    )
    output = _make_async_output_stub()
    output.set_draft_token_ids(
        req_ids=["plain", "structured"],
        draft_token_ids=SimpleNamespace(
            array=np.array([[3, 4], [5, 6]]), record_stream=Mock()
        ),
        structured_output_request_ids=["plain", "structured"],
        producer_step_id=None,
    )
    drafts = output.get_output().draft_token_ids
    assert drafts is not None
    assert drafts.req_ids == ["plain", "structured"]
    assert drafts.draft_token_ids == [[3, 4], [5, 6]]
    assert drafts.producer_step_id is None
