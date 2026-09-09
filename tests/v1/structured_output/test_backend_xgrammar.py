# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import xgrammar as xgr

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

pytestmark = pytest.mark.cpu_test


class _FakeMatcher:
    def __init__(self, stop_token: int, *, terminated: bool = False):
        self.stop_token = stop_token
        self.terminated = terminated
        self.accepted_tokens: list[int] = []
        self.accept_calls: list[int] = []
        self.rollback_calls: list[int] = []

    def accept_token(self, token: int) -> bool:
        self.accept_calls.append(token)
        if self.terminated:
            return False
        self.accepted_tokens.append(token)
        self.terminated = token == self.stop_token
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def rollback(self, num_tokens: int) -> None:
        self.rollback_calls.append(num_tokens)
        del self.accepted_tokens[-num_tokens:]
        self.terminated = self.stop_token in self.accepted_tokens


def _make_grammar(matcher: _FakeMatcher) -> XgrammarGrammar:
    grammar = XgrammarGrammar(
        vocab_size=4,
        matcher=cast(xgr.GrammarMatcher, matcher),
        ctx=cast(xgr.CompiledGrammar, None),
    )
    grammar._is_terminated = matcher.is_terminated()
    return grammar


def test_validate_tokens_stops_after_stop_token_and_rolls_back():
    matcher = _FakeMatcher(stop_token=2)
    grammar = _make_grammar(matcher)

    assert grammar.validate_tokens([1, 2, 3]) == [1, 2]
    assert matcher.accept_calls == [1, 2]
    assert matcher.rollback_calls == [2]
    assert matcher.accepted_tokens == []
    assert not matcher.is_terminated()


def test_validate_tokens_does_not_probe_terminated_matcher():
    matcher = _FakeMatcher(stop_token=2, terminated=True)
    grammar = _make_grammar(matcher)

    assert grammar.validate_tokens([3]) == []
    assert matcher.accept_calls == []
    assert matcher.rollback_calls == []
