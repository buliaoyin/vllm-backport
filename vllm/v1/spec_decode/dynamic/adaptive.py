# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side draft budgets for CPU hybrid pipeline execution."""

import math
import statistics
from collections import Counter, deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def uses_scheduler_adaptive_verification(config: "VllmConfig") -> bool:
    spec = getattr(config, "speculative_config", None)
    extra = getattr(config, "additional_config", None)
    return bool(
        spec is not None
        and getattr(spec, "enable_adaptive_verification", False)
        and isinstance(extra, dict)
        and isinstance(extra.get("deepseek_v41_hybrid"), dict)
    )


def verification_lengths(max_drafts: int) -> tuple[int, ...]:
    return tuple(range(1, max_drafts + 1))


@dataclass
class Acceptance:
    conditional: list[float]

    def observe(self, verified: int, accepted: int) -> None:
        # Only the accepted prefix and first rejection were actually observed.
        for index in range(min(verified, accepted + 1)):
            self.conditional[index] += 0.15 * (
                float(index < accepted) - self.conditional[index]
            )

    def expected(self, drafts: int) -> float:
        probability = 1.0
        result = 1.0
        for value in self.conditional[:drafts]:
            probability *= value
            result += probability
        return result


@dataclass
class BudgetCosts:
    samples: dict[int, deque[float]] = field(default_factory=dict)
    decisions: int = 0
    selected: int = 3


class AdaptiveDraftBudget:
    """Choose a uniform K from observed acceptance and end-to-end step costs.

    The scheduler owns the decision, so PP stages receive exact CPU boundaries.
    The selected length applies to the next proposal; existing drafts retain
    their producer's length through verification.
    """

    def __init__(self, max_drafts: int):
        self.max_drafts = max_drafts
        self.lengths = verification_lengths(max_drafts)
        self.default = min(3, max_drafts)
        self.requests: dict[str, Acceptance] = {}
        self.costs: dict[tuple[int, int, bool], BudgetCosts] = {}
        self.request_costs: dict[str, dict[tuple[int, int, bool], BudgetCosts]] = {}
        self.pending: dict[
            int, tuple[float, tuple[int, int, bool], dict[str, int]]
        ] = {}
        self.counts: Counter[int] = Counter()
        self.confidences: dict[str, tuple[int, list[float]]] = {}

    def observe_confidences(self, req_id: str, step: int, values: list[float]) -> None:
        if len(values) != self.max_drafts:
            return
        # Short proposals leave a NaN suffix in the fixed-size output buffer.
        length = next((i for i, x in enumerate(values) if math.isnan(x)), len(values))
        if not length or any(not math.isnan(x) for x in values[length:]):
            return
        values = values[:length]
        if any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
            return
        previous = self.confidences.get(req_id)
        if previous is None or step > previous[0]:
            self.confidences[req_id] = (step, list(values))

    @staticmethod
    def key(num_reqs: int, context: int, mixed: bool) -> tuple[int, int, bool]:
        return num_reqs, max(0, context // 32768), mixed

    def _costs(
        self, req_ids: Collection[str], key: tuple[int, int, bool]
    ) -> BudgetCosts:
        table = self.costs
        if len(req_ids) == 1:
            # Expert hit rates and routing costs differ between tasks.
            table = self.request_costs.setdefault(next(iter(req_ids)), {})
        return table.setdefault(key, BudgetCosts(selected=self.default))

    def choose(self, req_ids: list[str], context: int, mixed: bool) -> int:
        if not req_ids:
            return self.default
        for req_id in req_ids:
            if req_id not in self.requests:
                self.requests[req_id] = Acceptance([0.75] * self.max_drafts)
        key = self.key(len(req_ids), context, mixed)
        costs = self._costs(req_ids, key)
        costs.decisions += 1
        # Measure each shape on real work. Medians discard first-use outliers.
        for drafts in (self.default, *self.lengths):
            if len(costs.samples.get(drafts, ())) < 4:
                return drafts
        # Refresh censored positions and costs even after settling on a short K.
        if costs.decisions % 32 == 0:
            return self.lengths[(costs.decisions // 32 - 1) % len(self.lengths)]
        scores = {
            drafts: sum(self.requests[r].expected(drafts) for r in req_ids)
            / statistics.median(costs.samples[drafts])
            for drafts in self.lengths
        }
        best = max(scores, key=scores.__getitem__)
        if scores[best] > scores[costs.selected] * 1.03:
            costs.selected = best
        return costs.selected

    def scheduled(
        self,
        step: int,
        started: float,
        context: int,
        mixed: bool,
        drafts: dict[str, int],
    ) -> None:
        if drafts:
            self.pending[step] = (
                started,
                self.key(len(drafts), context, mixed),
                drafts,
            )

    def complete(self, step: int, now: float, sampled: dict[str, int]) -> None:
        pending = self.pending.pop(step, None)
        if pending is None:
            return
        started, key, drafts = pending
        for req_id, verified in drafts.items():
            count = sampled.get(req_id, 0)
            if count <= 0 or req_id not in self.requests:
                continue
            self.requests[req_id].observe(verified, min(count - 1, verified))
            self.counts[verified] += 1
        lengths = set(drafts.values())
        elapsed = now - started
        if (
            len(lengths) != 1
            or not all(r in self.requests and sampled.get(r, 0) > 0 for r in drafts)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            return
        drafts_count = lengths.pop()
        costs = self._costs(drafts, key)
        costs.samples.setdefault(drafts_count, deque(maxlen=12)).append(elapsed)

    def retain_requests(self, req_ids: Collection[str]) -> None:
        for req_id in self.requests.keys() - req_ids:
            del self.requests[req_id]
            self.request_costs.pop(req_id, None)
        for req_id in self.confidences.keys() - req_ids:
            del self.confidences[req_id]


@dataclass
class ConfidenceCosts:
    req_ids: frozenset[str]
    selected: int
    completed: int = 0
    changed: int = 0


class ConfidenceDraftBudget(AdaptiveDraftBudget):
    """Calibrate confidence against verified prefixes; measure batched PP costs."""

    def __init__(self, max_drafts: int):
        super().__init__(max_drafts)
        self.shapes: dict[
            tuple[str | None, tuple[int, int, bool]], ConfidenceCosts
        ] = {}
        # Single-request timing varies with routing as well as K. Keep a stable
        # cost prior instead of feeding those fluctuations back into decoding.
        self.relative_costs = {k: 1 + 0.08 * (k - self.default) for k in self.lengths}
        self.smoothed_confidences: dict[str, list[float]] = {}
        self.confidence_bias: dict[str, list[float]] = {}
        self.last_observations: dict[str, tuple[int, int, int]] = {}

    def observe_confidences(self, req_id: str, step: int, values: list[float]) -> None:
        before = self.confidences.get(req_id)
        super().observe_confidences(req_id, step, values)
        if self.confidences.get(req_id) == before:
            return
        observed = self.last_observations.get(req_id)
        if observed is not None and observed[0] == step and observed[2] >= 0:
            _, verified, accepted = observed
            bias = self.confidence_bias.setdefault(req_id, [0.0] * self.max_drafts)
            for index in range(min(verified, accepted + 1)):
                error = float(index < accepted) - values[index]
                bias[index] = max(-0.2, min(0.2, 0.98 * bias[index] + 0.02 * error))
        values = self.confidences[req_id][1]
        previous = self.smoothed_confidences.get(req_id)
        if previous is None:
            previous = [float("nan")] * self.max_drafts
            self.smoothed_confidences[req_id] = previous
        for index, value in enumerate(values):
            previous[index] = (
                value
                if math.isnan(previous[index])
                else 0.85 * previous[index] + 0.15 * value
            )

    def _shape(
        self, req_ids: Collection[str], key: tuple[int, int, bool]
    ) -> ConfidenceCosts:
        identity = next(iter(req_ids)) if len(req_ids) == 1 else None
        cohort = frozenset(req_ids)
        state = self.shapes.get((identity, key))
        if state is None or state.req_ids != cohort:
            state = ConfidenceCosts(
                req_ids=cohort,
                selected=self.default,
            )
            self.shapes[(identity, key)] = state
        return state

    def _expected(self, req_id: str, drafts: int) -> float:
        empirical = self.requests[req_id].expected(drafts)
        if req_id not in self.confidences:
            return empirical
        probability, predicted = 1.0, 1.0
        bias = self.confidence_bias.get(req_id, [0.0] * self.max_drafts)
        for index, value in enumerate(self.smoothed_confidences[req_id][:drafts]):
            if math.isnan(value):
                value = self.requests[req_id].conditional[index]
            probability *= max(0.0, min(1.0, value + bias[index]))
            predicted += probability
        return 0.35 * predicted + 0.65 * empirical

    def choose(self, req_ids: list[str], context: int, mixed: bool) -> int:
        if not req_ids:
            return self.default
        if len(req_ids) > 1 or mixed:
            return self._choose_batch(req_ids, context, mixed)
        measured_budget = super().choose(req_ids, context, mixed)
        key = self.key(len(req_ids), context, mixed)
        state = self._shape(req_ids, key)
        if state.completed == 0:
            state.selected = self.default
        has_prediction = all(r in self.confidences for r in req_ids)
        if not has_prediction and state.completed >= 4:
            return measured_budget
        if not has_prediction or (
            state.completed >= 3 and state.completed - state.changed < 3
        ):
            return state.selected
        scores = {
            k: sum(self._expected(r, k) for r in req_ids) / self.relative_costs[k]
            for k in self.lengths
        }
        best = max(scores, key=scores.__getitem__)
        if scores[best] > 1.02 * scores[state.selected]:
            state.selected = best
            state.changed = state.completed
        return state.selected

    def _choose_batch(self, req_ids: list[str], context: int, mixed: bool) -> int:
        for req_id in req_ids:
            self.requests.setdefault(req_id, Acceptance([0.75] * self.max_drafts))
        costs = self._costs(req_ids, self.key(len(req_ids), context, mixed))
        costs.decisions += 1
        # Start with two measured anchors, then measure promising budgets.
        for drafts in (self.default, 1):
            if len(costs.samples.get(drafts, ())) < 3:
                return drafts
        measured = {
            k: statistics.median(samples)
            for k, samples in costs.samples.items()
            if len(samples) >= 3
        }
        baseline = measured[self.default]
        slope = max(
            0.01,
            min(
                0.30,
                (baseline - measured[1]) / (max(1, self.default - 1) * baseline),
            ),
        )
        scores = {
            k: sum(self._expected(req_id, k) for req_id in req_ids)
            / measured.get(k, baseline * max(0.25, 1 + slope * (k - self.default)))
            for k in self.lengths
        }
        best = max(scores, key=scores.__getitem__)
        if best not in measured:
            return best
        if costs.decisions % 32 == 0:
            return self.lengths[(costs.decisions // 32 - 1) % len(self.lengths)]
        if scores[best] > 1.03 * scores[costs.selected]:
            costs.selected = best
        return costs.selected

    def complete(self, step: int, now: float, sampled: dict[str, int]) -> None:
        entry = self.pending.get(step)
        self.last_observations = (
            {}
            if entry is None
            else {
                req_id: (step, verified, sampled.get(req_id, 0) - 1)
                for req_id, verified in entry[2].items()
            }
        )
        super().complete(step, now, sampled)
        if entry is None or len(entry[2]) > 1 or entry[1][2]:
            return
        _, key, drafts = entry
        lengths = set(drafts.values())
        if len(lengths) != 1 or not all(
            r in self.requests and sampled.get(r, 0) > 0 for r in drafts
        ):
            return
        state = self._shape(drafts, key)
        state.completed += 1

    def retain_requests(self, req_ids: Collection[str]) -> None:
        super().retain_requests(req_ids)
        for req_id in self.smoothed_confidences.keys() - req_ids:
            del self.smoothed_confidences[req_id]
            self.confidence_bias.pop(req_id, None)
        for req_id in self.last_observations.keys() - req_ids:
            del self.last_observations[req_id]
        for key in list(self.shapes):
            if key[0] is not None and key[0] not in req_ids:
                del self.shapes[key]
