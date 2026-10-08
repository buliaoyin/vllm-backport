# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from benchmarks.kernels.cpu.benchmark_dsv41_serving import request


@pytest.mark.parametrize("first_count", [1, 3])
def test_decode_counts_tokens_after_first_streamed_group(first_count):
    """Exclude the first group and count tokens even without visible text."""
    events = [
        {
            "choices": [{"text": "start", "token_ids": [11] * first_count}],
            "usage": {"completion_tokens": first_count},
        },
        {"choices": [], "usage": {"completion_tokens": 5}},
        {"choices": [{"text": "end"}], "usage": {"completion_tokens": 8}},
        {"choices": [], "usage": {"completion_tokens": 8, "prompt_tokens": 32}},
    ]
    body = "\n".join("data: " + json.dumps(event) for event in events)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    with (
        httpx.Client(transport=transport, base_url="http://test") as client,
        patch(
            "benchmarks.kernels.cpu.benchmark_dsv41_serving.time",
            SimpleNamespace(
                perf_counter=Mock(side_effect=[10, 11, 12, 13, 14]),
                time=lambda: 0,
            ),
        ),
    ):
        result = request(
            client, "/v1/completions", {"ignore_eos": True, "max_tokens": 8}, "test"
        )
    assert result["decode_tps"] == (8 - first_count) / 2
    assert result["token_events"] == [[11, first_count], [12, 5], [13, 8]]
    assert result["decode_counting"] == "cumulative_output_tokens"
    assert result["text"] == "startend"
    assert result["token_ids"] == [11] * first_count
