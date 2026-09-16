# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-limited expert admission from prompt tails and previous decode routes."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TailCachePolicy:
    mutable_experts: tuple[int, ...]
    group_size: int = 1
    future_tokens: int = 255
    cpu_call_ms: float = 0.25
    transfer_ms: float = 1.0
    max_swaps: int = 4
    budget_ms: float = 6.0
    min_tokens: int = 64
    safety_factor: float = 2.0
    prepack_resident: bool = True
    feedback_weight: float = 0.0
    feedback_decay: float = 0.8
    feedback_debias: bool = False
    feedback_horizon_requests: float = 1.0
    feedback_max_swaps: int = 16
    feedback_budget_ms: float = 24.0
    expected_tokens_per_step: float = 1.0
    repack_transfer_ms: float = 6.4
    host_lru_experts: int = 0
    host_cache_bytes: int | None = None
    decode_interval: int = 64
    decode_horizon_steps: int = 256
    decode_max_swaps: int = 2
    decode_budget_ms: float = 8.0

    def __post_init__(self):
        if type(self.feedback_debias) is not bool:
            raise ValueError("feedback_debias must be a boolean")
        if type(self.prepack_resident) is not bool:
            raise ValueError("prepack_resident must be a boolean")
        if not 0 <= self.feedback_weight <= 1 or not 0 <= self.feedback_decay < 1:
            raise ValueError("Feedback weights must be probabilities")
        if type(self.host_lru_experts) is not int or self.host_lru_experts < 0:
            raise ValueError("Host LRU capacity must be a nonnegative integer")
        if self.host_cache_bytes is not None and (
            type(self.host_cache_bytes) is not int or self.host_cache_bytes < 0
        ):
            raise ValueError("Host cache byte limit must be a nonnegative integer")
        integers = (
            self.feedback_max_swaps,
            self.group_size,
            self.future_tokens,
            self.max_swaps,
            self.min_tokens,
            self.decode_interval,
            self.decode_horizon_steps,
            self.decode_max_swaps,
        )
        costs = (
            self.cpu_call_ms,
            self.transfer_ms,
            self.budget_ms,
            self.safety_factor,
            self.feedback_budget_ms,
            self.expected_tokens_per_step,
            self.repack_transfer_ms,
            self.feedback_horizon_requests,
            self.decode_budget_ms,
        )
        if any(type(n) is not int or n <= 0 for n in integers):
            raise ValueError("Dynamic cache counts must be positive integers")
        if any(not math.isfinite(n) or n <= 0 for n in costs):
            raise ValueError("Dynamic cache costs must be finite and positive")
        if self.safety_factor < 1 or len(set(self.mutable_experts)) != len(
            self.mutable_experts
        ):
            raise ValueError(
                "Dynamic cache needs unique slots and a safety factor >= 1"
            )

    def update_history(self, history, mass, calls, steps):
        if not steps:
            return history, mass
        frequency = np.asarray(calls, dtype=np.float64) / steps
        mass = self.feedback_decay * mass + 1
        blend = 1 / mass if self.feedback_debias else 1 - self.feedback_decay
        history = (
            frequency if history is None else (1 - blend) * history + blend * frequency
        )
        return history, mass

    def plan(
        self,
        ids,
        selected,
        pinned,
        num_experts,
        *,
        history=None,
        costs=None,
        feedback=False,
    ):
        """Return a slot-preserving selection and estimated CPU savings in ms.

        Count an expert once per verification group, even if several tokens
        route to it. The budget bounds estimated transfer time, not wall time.
        """
        if ids.ndim != 2:
            raise ValueError("Dynamic cache routes must have shape [tokens, top_k]")
        valid_rows = np.any((ids >= 0) & (ids < num_experts), axis=1)
        ids = ids[valid_rows]
        counts = np.zeros(num_experts, dtype=np.int64)
        if len(ids) < self.min_tokens:
            return list(selected), 0.0, counts.tolist()
        groups = np.broadcast_to(
            np.arange(len(ids))[:, None] // self.group_size, ids.shape
        )
        valid = (ids >= 0) & (ids < num_experts)
        unique = np.unique((groups * num_experts + ids)[valid])
        counts = np.bincount(unique % num_experts, minlength=num_experts)
        groups_count = math.ceil(len(ids) / self.group_size)
        scores = counts.astype(np.float64) / groups_count
        if feedback and history is not None:
            scores = (
                1 - self.feedback_weight
            ) * scores + self.feedback_weight * history
        future_steps = self.future_tokens / (
            self.expected_tokens_per_step if feedback else self.group_size
        )
        if feedback:
            future_steps *= self.feedback_horizon_requests
        result, saving = self._plan_scores(
            scores,
            selected,
            pinned,
            future_steps,
            costs,
            self.feedback_max_swaps if feedback else self.max_swaps,
            self.feedback_budget_ms if feedback else self.budget_ms,
        )
        return result, saving, counts.tolist()

    def plan_decode(
        self, calls, steps, selected, pinned, *, costs=None, remaining_steps=None
    ):
        """Use recent verification groups, with a bounded within-request horizon."""
        calls = np.asarray(calls, dtype=np.int64)
        if steps < self.decode_interval:
            return list(selected), 0.0, calls.tolist()
        future_steps = self.decode_horizon_steps
        if remaining_steps is not None:
            future_steps = min(future_steps, max(0.0, remaining_steps))
        result, saving = self._plan_scores(
            calls.astype(np.float64) / steps,
            selected,
            pinned,
            future_steps,
            costs,
            min(self.decode_max_swaps, self.feedback_max_swaps),
            min(self.decode_budget_ms, self.feedback_budget_ms),
        )
        return result, saving, calls.tolist()

    def _plan_scores(
        self, scores, selected, pinned, future_steps, costs, limit, budget
    ):
        num_experts = len(scores)
        if costs is None:
            costs = np.full(num_experts, self.transfer_ms)
        candidates = sorted(
            set(range(num_experts)) - set(selected), key=lambda e: (-scores[e], e)
        )
        victims = sorted(set(selected) - pinned, key=lambda e: (scores[e], e))
        result, saving, swaps = list(selected), 0.0, 0
        for incoming in candidates:
            if swaps >= min(limit, len(victims)):
                break
            outgoing = victims[swaps]
            benefit = float(scores[incoming] - scores[outgoing]) * (
                future_steps * self.cpu_call_ms
            )
            cost = float(costs[incoming])
            if cost > budget or benefit <= self.safety_factor * cost:
                continue
            result[result.index(outgoing)] = incoming
            saving += benefit
            budget -= cost
            swaps += 1
        return result, saving
