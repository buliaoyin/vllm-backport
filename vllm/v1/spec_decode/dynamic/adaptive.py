# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side draft budgets for pipeline execution."""

import math
import statistics
from collections import Counter, deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig


def supports_adaptive_mtp(spec: "SpeculativeConfig") -> bool:
    if getattr(spec, "method", None) != "mtp":
        return False
    draft = getattr(spec, "draft_model_config", None)
    hf_config = getattr(draft, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    if model_type == "glm5_next_mtp":
        return getattr(hf_config, "num_nextn_predict_layers", 1) == 1
    if model_type == "qwen3_5_mtp":
        architectures = getattr(hf_config, "architectures", None) or []
        return (
            architectures in (["Qwen3_5MTP"], ["Qwen3_5MoeMTP"])
            and getattr(hf_config, "n_predict", None) == 1
        )
    if model_type == "qwen4_exp_mtp":
        return (
            getattr(hf_config, "architectures", None) == ["Qwen4ExpMTP"]
            and getattr(hf_config, "n_predict", None) == 1
        )
    return False


def uses_scheduler_adaptive_verification(config: "VllmConfig") -> bool:
    spec = getattr(config, "speculative_config", None)
    extra = getattr(config, "additional_config", None)
    return bool(
        spec is not None
        and getattr(spec, "enable_adaptive_verification", False)
        and (
            supports_adaptive_mtp(spec)
            or (
                isinstance(extra, dict)
                and isinstance(extra.get("deepseek_v41_hybrid"), dict)
            )
        )
    )


def verification_lengths(max_drafts: int) -> tuple[int, ...]:
    return tuple(range(1, max_drafts + 1))


def prefers_default_mtp_budget(config: "VllmConfig") -> bool:
    model_config = getattr(config, "model_config", None)
    quantization = getattr(model_config, "quantization", None)
    pp_size = getattr(
        getattr(config, "parallel_config", None), "pipeline_parallel_size", 1
    )
    return bool(
        uses_scheduler_adaptive_verification(config)
        and getattr(getattr(model_config, "hf_text_config", None), "model_type", None)
        == "qwen4_exp_text"
        and (quantization == "exl3" or (quantization == "modelopt_fp4" and pp_size > 1))
    )


def uses_mtp_confidence_forecast(config: "VllmConfig") -> bool:
    extra = getattr(config, "additional_config", None)
    model = getattr(config, "model_config", None)
    parallel = getattr(config, "parallel_config", None)
    spec = getattr(config, "speculative_config", None)
    return bool(
        isinstance(extra, dict)
        and extra.get("mtp_confidence_forecast") is True
        and prefers_default_mtp_budget(config)
        and getattr(model, "quantization", None) == "modelopt_fp4"
        and getattr(parallel, "pipeline_parallel_size", 1) == 2
        and getattr(parallel, "tensor_parallel_size", 1) == 1
        and getattr(spec, "draft_sample_method", None) == "greedy"
        and getattr(spec, "use_local_argmax_reduction", False)
    )


@dataclass
class Acceptance:
    conditional: list[float]
    weight: float = 0.15
    observations: int = 0
    verified_prefixes: list[deque[int]] | None = None
    prefix_updated: list[int] | None = None

    def observe(self, verified: int, accepted: int) -> None:
        self.observations += 1
        if self.verified_prefixes is not None:
            if self.prefix_updated is None:
                self.prefix_updated = [0] * len(self.verified_prefixes)
            for index in range(min(verified, len(self.verified_prefixes))):
                prefixes = self.verified_prefixes[index]
                if self.observations - self.prefix_updated[index] > (
                    prefixes.maxlen or 32
                ):
                    prefixes.clear()
                prefixes.append(min(index + 1, accepted))
                self.prefix_updated[index] = self.observations
        # Only the accepted prefix and first rejection were actually observed.
        for index in range(min(verified, accepted + 1)):
            self.conditional[index] += self.weight * (
                float(index < accepted) - self.conditional[index]
            )

    def expected(self, drafts: int) -> float:
        if self.verified_prefixes is not None and drafts:
            observed = self.verified_prefixes[drafts - 1]
            if len(observed) >= 4:
                return 1 + sum(observed) / len(observed)
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
    changed: int = -8
    probing: int | None = None
    probe_samples_remaining: int = 0
    refresh_remaining: int = 0


class AdaptiveDraftBudget:
    """Choose a uniform K from observed acceptance and end-to-end step costs.

    The scheduler owns the decision, so PP stages receive exact CPU boundaries.
    The selected length applies to the next proposal; existing drafts retain
    their producer's length through verification.
    """

    acceptance_weight = 0.15

    def __init__(self, max_drafts: int):
        self.max_drafts = max_drafts
        self.lengths = verification_lengths(max_drafts)
        self.default = min(3, max_drafts)
        self.requests: dict[str, Acceptance] = {}
        self.costs: dict[tuple[int, int, bool], BudgetCosts] = {}
        self.request_costs: dict[str, dict[tuple[int, int, bool], BudgetCosts]] = {}
        self.pending: dict[
            int, tuple[float, tuple[int, int, bool], dict[str, int], int | None]
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
                self.requests[req_id] = Acceptance(
                    [0.75] * self.max_drafts, self.acceptance_weight
                )
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
        return self._select(scores, costs)

    def _select(self, scores: dict[int, float], costs: BudgetCosts) -> int:
        best = max(scores, key=scores.__getitem__)
        if scores[best] > scores[costs.selected] * 1.03:
            costs.selected = best
        return costs.selected

    def _expected(self, req_id: str, drafts: int) -> float:
        return self.requests[req_id].expected(drafts)

    def _batch_scores(
        self, req_ids: list[str], costs: BudgetCosts
    ) -> tuple[dict[int, float], dict[int, float]]:
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
        return scores, measured

    def scheduled(
        self,
        step: int,
        started: float,
        context: int,
        mixed: bool,
        drafts: dict[str, int],
        *,
        proposal_drafts: int | None = None,
    ) -> None:
        if drafts:
            self.pending[step] = (
                started,
                self.key(len(drafts), context, mixed),
                drafts,
                proposal_drafts,
            )

    def complete(self, step: int, now: float, sampled: dict[str, int]) -> None:
        pending = self.pending.pop(step, None)
        if pending is None:
            return
        started, key, drafts, proposal_drafts = pending
        for req_id, verified in drafts.items():
            count = sampled.get(req_id, 0)
            if count <= 0 or req_id not in self.requests:
                continue
            self.requests[req_id].observe(verified, min(count - 1, verified))
            self.counts[verified] += 1
        lengths = set(drafts.values())
        # A transition verifies the old K while generating the new K.
        if proposal_drafts is not None and lengths != {proposal_drafts}:
            return
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
        if costs.probing == drafts_count and costs.probe_samples_remaining:
            costs.probe_samples_remaining -= 1

    def retain_requests(self, req_ids: Collection[str]) -> None:
        for req_id in self.requests.keys() - req_ids:
            del self.requests[req_id]
            self.request_costs.pop(req_id, None)
        for req_id in self.confidences.keys() - req_ids:
            del self.confidences[req_id]


@dataclass
class MTPConfidenceRank:
    observations: deque[tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=128)
    )
    buckets: dict[int, list[int]] = field(default_factory=dict)
    losses: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))

    def estimate(self, bucket: int) -> float | None:
        count, successes = self.buckets.get(bucket, (0, 0))
        return (successes + 1) / (count + 2) if count >= 8 else None

    def observe(
        self, bucket: int, accepted: int, baseline: float, predicted: float | None
    ) -> None:
        if len(self.observations) == self.observations.maxlen:
            old_bucket, old_accepted = self.observations.popleft()
            old = self.buckets[old_bucket]
            old[0] -= 1
            old[1] -= old_accepted
            if not old[0]:
                del self.buckets[old_bucket]
        self.observations.append((bucket, accepted))
        counts = self.buckets.setdefault(bucket, [0, 0])
        counts[0] += 1
        counts[1] += accepted
        if predicted is not None:
            self.losses.append(
                ((predicted - accepted) ** 2, (baseline - accepted) ** 2)
            )

    def weight(self) -> float:
        if len(self.losses) < 64 or sum(n >= 8 for n, _ in self.buckets.values()) < 2:
            return 0.0
        predicted = sum(x for x, _ in self.losses)
        baseline = sum(x for _, x in self.losses)
        differences = [b - p for p, b in self.losses]
        mean = sum(differences) / len(differences)
        variance = max(
            0.0, sum(d * d for d in differences) / len(differences) - mean * mean
        )
        if mean <= 2 * math.sqrt(variance / len(differences)):
            return 0.0
        return (
            min(0.75, 1 - predicted / baseline) if predicted < baseline * 0.7 else 0.0
        )


class MTPConfidenceForecast:
    """Predict future prefixes from a previous block's calibrated feature.

    FR head max-softmax is only a feature. Training labels come from the
    later verification of the proposal that used that feature, never from
    the same block that supplied it. Only conditional opportunities and the
    first rejection contribute labels.
    """

    def __init__(self, max_drafts: int):
        self.max_drafts = max_drafts
        self.ranks: dict[str, list[MTPConfidenceRank]] = {}
        self.latest: dict[str, tuple[int, int]] = {}
        self.producers: dict[
            str,
            tuple[int, int, int, int, int, tuple[float, ...], tuple[float | None, ...]]
            | None,
        ] = {}
        self.pending: dict[
            int,
            dict[
                str,
                tuple[int, int, int, int, tuple[float, ...], tuple[float | None, ...]],
            ],
        ] = {}
        self.epochs: Counter[str] = Counter()
        self.verification_epochs: dict[int, dict[str, int]] = {}
        self.chosen: dict[
            str, tuple[int, int, tuple[float, ...], tuple[float | None, ...]]
        ] = {}
        self.predictions: dict[str, tuple[float, tuple[float, ...]]] = {}
        self.observed_steps: dict[str, int] = {}
        self.active = False
        self.training_pairs: Counter[str] = Counter()
        self.forecast_decisions: Counter[str] = Counter()
        self.scored: set[str] = set()
        self.latest_pair: dict[str, tuple[int, int, int]] = {}

    @staticmethod
    def bucket(value: float) -> int:
        value = min(1 - 1e-6, max(1e-6, value))
        return math.floor(math.log2(value / (1 - value)))

    def prepare(self, requests: dict[str, Acceptance], req_ids: list[str]) -> None:
        self.chosen.clear()
        self.predictions.clear()
        self.scored.clear()
        for r in req_ids:
            latest = self.latest.get(r)
            if (
                latest is None
                or latest[0] != self.observed_steps.get(r)
                or r not in requests
            ):
                continue
            _, bucket = latest
            baseline = tuple(requests[r].conditional)
            ranks = self.ranks.setdefault(
                r, [MTPConfidenceRank() for _ in range(self.max_drafts)]
            )
            predicted = tuple(rank.estimate(bucket) for rank in ranks)
            self.chosen[r] = (latest[0], bucket, baseline, predicted)
            weight = ranks[0].weight()
            if weight and predicted[0] is not None:
                self.predictions[r] = (
                    weight,
                    tuple(
                        (
                            p
                            if i == 0
                            else baseline[i] + ranks[i].weight() * (p - baseline[i])
                        )
                        if p is not None
                        else baseline[i]
                        for i, p in enumerate(predicted)
                    ),
                )

    def prediction(self, req_id: str) -> tuple[float, tuple[float, ...]] | None:
        prediction = self.predictions.get(req_id) if self.active else None
        if prediction is not None and req_id not in self.scored:
            self.scored.add(req_id)
            self.forecast_decisions[req_id] += 1
        return prediction

    def expected(self, req_id: str, drafts: int, empirical: float) -> float:
        prediction = self.prediction(req_id)
        if prediction is None:
            return empirical
        weight, conditional = prediction
        probability, result = 1.0, 1.0
        for p in conditional[:drafts]:
            probability *= p
            result += probability
        return (1 - weight) * empirical + weight * result

    def prefix(self, req_id: str, position: int, empirical: float) -> float:
        prediction = self.prediction(req_id)
        if prediction is None:
            return empirical
        weight, conditional = prediction
        return (1 - weight) * empirical + weight * math.prod(conditional[:position])

    def scheduled(
        self, step: int, mixed: bool, drafts: dict[str, int], proposal: int | None
    ) -> None:
        pending = {}
        if drafts:
            self.verification_epochs[step] = {r: self.epochs[r] for r in drafts}
        for r, verified in drafts.items():
            previous = self.producers.get(r)
            if not mixed and previous is not None and previous[1] == verified:
                producer, _, epoch, feature_step, bucket, baseline, predicted = previous
                pending[r] = (
                    producer,
                    epoch,
                    feature_step,
                    bucket,
                    baseline,
                    predicted,
                )
            chosen = self.chosen.get(r)
            if mixed or proposal is None or chosen is None:
                self.producers[r] = None
            else:
                feature_step, bucket, baseline, predicted = chosen
                self.producers[r] = (
                    step,
                    proposal,
                    self.epochs[r],
                    feature_step,
                    bucket,
                    baseline,
                    predicted,
                )
        if pending:
            self.pending[step] = pending

    def invalidate(self, req_id: str) -> None:
        self.epochs[req_id] += 1
        for table in (
            self.producers,
            self.latest,
            self.observed_steps,
            self.chosen,
            self.predictions,
        ):
            table.pop(req_id, None)
        self.scored.discard(req_id)

    def complete(
        self, step: int, drafts: dict[str, int], sampled: dict[str, int]
    ) -> None:
        pending = self.pending.pop(step, {})
        epochs = self.verification_epochs.pop(step, {})
        for r, verified in drafts.items():
            if epochs.get(r) != self.epochs[r]:
                continue
            count = sampled.get(r, 0)
            if count <= 0:
                self.invalidate(r)
                continue
            self.observed_steps[r] = step
            record = pending.get(r)
            if record is None or record[1] != self.epochs[r]:
                continue
            producer, _, feature_step, bucket, baseline, predicted = record
            self.training_pairs[r] += 1
            self.latest_pair[r] = (feature_step, producer, step)
            ranks = self.ranks[r]
            accepted = min(verified, count - 1)
            for i in range(min(verified, accepted + 1)):
                ranks[i].observe(bucket, int(i < accepted), baseline[i], predicted[i])

    def observe_confidences(
        self, req_id: str, step: int, confidence: tuple[int, list[float]] | None
    ) -> None:
        if self.observed_steps.get(req_id) != step:
            return
        if confidence is None or confidence[0] != step:
            self.latest.pop(req_id, None)
        else:
            self.latest[req_id] = (step, self.bucket(confidence[1][0]))

    def retain_requests(self, req_ids: Collection[str]) -> None:
        retained = set(req_ids)
        for table in (
            self.ranks,
            self.latest,
            self.producers,
            self.epochs,
            self.chosen,
            self.predictions,
            self.observed_steps,
            self.training_pairs,
            self.forecast_decisions,
            self.latest_pair,
        ):
            for r in table.keys() - retained:
                del table[r]
        self.scored.intersection_update(retained)
        for step, pending_records in list(self.pending.items()):
            for r in pending_records.keys() - retained:
                del pending_records[r]
            if not pending_records:
                del self.pending[step]
        for step, epoch_records in list(self.verification_epochs.items()):
            for r in epoch_records.keys() - retained:
                del epoch_records[r]
            if not epoch_records:
                del self.verification_epochs[step]


class MTPDraftBudget(AdaptiveDraftBudget):
    """Reuse measured shapes while learning acceptance separately per request."""

    acceptance_weight = 0.05

    def __init__(
        self,
        max_drafts: int,
        *,
        prefer_default: bool = False,
        use_confidence: bool = False,
    ):
        super().__init__(max_drafts)
        self._num_reqs = 1
        self.prefer_default = prefer_default
        self._forecast = (
            MTPConfidenceForecast(max_drafts)
            if prefer_default and use_confidence
            else None
        )
        self._forecast_owner = True
        self._extension = MTPDraftExtension(self) if max_drafts > 3 else None
        if self._extension is not None:
            self._extension.core._forecast = self._forecast
            self._extension.core._forecast_owner = False

    def invalidate_confidence(self, req_id: str) -> None:
        if self._forecast is not None:
            self._forecast.invalidate(req_id)

    def observe_confidences(self, req_id: str, step: int, values: list[float]) -> None:
        super().observe_confidences(req_id, step, values)
        if self._forecast is not None:
            self._forecast.observe_confidences(
                req_id, step, self.confidences.get(req_id)
            )

    def _expected(self, req_id: str, drafts: int) -> float:
        empirical = self.requests[req_id].expected(drafts)
        if self._forecast is None:
            return empirical
        return self._forecast.expected(req_id, drafts, empirical)

    def choose(self, req_ids: list[str], context: int, mixed: bool) -> int:
        forecast = self._forecast if self._forecast_owner else None
        if forecast is None:
            return self._choose(req_ids, context, mixed)
        requests = self.requests
        forecast.prepare(requests, req_ids)
        forecast.active = not mixed
        try:
            return self._choose(req_ids, context, mixed)
        finally:
            forecast.active = False

    def _choose(self, req_ids: list[str], context: int, mixed: bool) -> int:
        self._num_reqs = len(req_ids)
        if self._extension is not None:
            return self._extension.choose(req_ids, context, mixed)
        if self.prefer_default:
            return self._choose_from_default(req_ids, context, mixed)
        if self._num_reqs <= 1:
            return super().choose(req_ids, context, mixed)
        for req_id in req_ids:
            self.requests.setdefault(
                req_id, Acceptance([0.75] * self.max_drafts, self.acceptance_weight)
            )
        costs = self._costs(req_ids, self.key(self._num_reqs, context, mixed))
        if not costs.decisions:
            costs.selected = 1
        costs.decisions += 1
        # Measure short budgets first; larger ones need evidence of a gain.
        for drafts in self.lengths[: self.default]:
            if len(costs.samples.get(drafts, ())) < 3:
                return drafts
        if costs.probing is not None:
            if (
                len(costs.samples.get(costs.probing, ())) < 3
                or costs.probe_samples_remaining
            ):
                return costs.probing
            costs.probing = None
        scores, measured = self._batch_scores(req_ids, costs)
        best = max(scores, key=scores.__getitem__)
        if best not in measured:
            costs.probing = best
            return best
        if costs.decisions % 32 == 0:
            drafts = self.lengths[(costs.decisions // 32 - 1) % len(self.lengths)]
            # A one-step probe cannot time MTP: it still verifies the old K.
            costs.probing = drafts
            costs.probe_samples_remaining = 1
            return drafts
        return self._select(scores, costs)

    def _choose_from_default(
        self, req_ids: list[str], context: int, mixed: bool
    ) -> int:
        if not req_ids:
            return self.default
        for req_id in req_ids:
            if req_id not in self.requests:
                self.requests[req_id] = Acceptance(
                    [0.75] * self.max_drafts,
                    self.acceptance_weight,
                    verified_prefixes=[deque(maxlen=32) for _ in self.lengths],
                )
        costs = self._costs(req_ids, self.key(len(req_ids), context, mixed))
        costs.decisions += 1
        if (
            mixed
            or len(costs.samples.get(self.default, ())) < 3
            or min(self.requests[r].observations for r in req_ids) < 32
        ):
            return self.default
        if (
            costs.selected != self.default
            and costs.decisions % 96 == 0
            and sum(self.requests[r].conditional[0] for r in req_ids)
            >= 0.5 * len(req_ids)
        ):
            costs.refresh_remaining = 4
        if costs.refresh_remaining:
            costs.refresh_remaining -= 1
            return self.default
        measured = {
            k: statistics.median(v) for k, v in costs.samples.items() if len(v) >= 3
        }
        baseline = measured[self.default]
        scores = {
            k: sum(self._expected(r, k) for r in req_ids)
            / measured.get(k, baseline * (1 - 0.10 * (self.default - k)))
            for k in self.lengths
        }
        if costs.probing is not None:
            if costs.probing not in measured or costs.probe_samples_remaining:
                return costs.probing
            costs.probing = None
        best = max(scores, key=scores.__getitem__)
        if best not in measured:
            if scores[best] > scores[self.default] * 1.08:
                costs.probing = best
                costs.probe_samples_remaining = 3
                return best
            return self.default
        return self._select(scores, costs)

    def scheduled(
        self,
        step: int,
        started: float,
        context: int,
        mixed: bool,
        drafts: dict[str, int],
        *,
        proposal_drafts: int | None = None,
    ) -> None:
        super().scheduled(
            step, started, context, mixed, drafts, proposal_drafts=proposal_drafts
        )
        if self._forecast is not None and self._forecast_owner:
            self._forecast.scheduled(step, mixed, drafts, proposal_drafts)
        if self._extension is not None:
            self._extension.scheduled(
                step, started, context, mixed, drafts, proposal_drafts=proposal_drafts
            )

    def retain_requests(self, req_ids: Collection[str]) -> None:
        super().retain_requests(req_ids)
        if self._forecast is not None and self._forecast_owner:
            self._forecast.retain_requests(req_ids)
        if self._extension is not None:
            self._extension.retain_requests(req_ids)

    def extension_stats(
        self, req_ids: list[str], context: int, mixed: bool
    ) -> dict[str, object]:
        if self._extension is None or not req_ids:
            return {}
        extension = self._extension
        key = self.key(len(req_ids), context, mixed)
        identity = req_ids[0] if len(req_ids) == 1 else frozenset(req_ids)
        state = extension.extensions.get((identity, key))
        if state is None:
            return {}
        prefix_acceptance = {}
        for k in range(3, self.max_drafts + 1):
            p = extension.probability(req_ids, k)
            prefix_acceptance[k] = round(p, 3) if p is not None else None
        costs = self._costs(req_ids, key)
        forecast = self._forecast
        confidence_stats = (
            {}
            if forecast is None
            else {
                "training_pairs": sum(forecast.training_pairs[r] for r in req_ids),
                "weighted_decisions": sum(
                    forecast.forecast_decisions[r] for r in req_ids
                ),
                "ready_requests": len(set(req_ids) & forecast.predictions.keys()),
                "first_rank_weight": {
                    r: round(forecast.predictions[r][0], 3)
                    for r in req_ids
                    if r in forecast.predictions
                },
                "last_pair_steps": {
                    r: forecast.latest_pair[r]
                    for r in req_ids
                    if r in forecast.latest_pair
                },
                "conditional_samples": {
                    r: [len(rank.observations) for rank in forecast.ranks[r]]
                    for r in req_ids
                    if r in forecast.ranks
                },
            }
        )
        return {
            "confidence_forecast": confidence_stats,
            "prefix_acceptance": prefix_acceptance,
            "step_ms": {
                k: round(statistics.median(v) * 1000, 2)
                for k, v in sorted(
                    (state.samples if self.prefer_default else costs.samples).items()
                )
                if len(v) >= 3
            },
            "trial_k": state.probing,
            "next_trial_in": max(0, state.next_probe - state.decisions),
            "excess_ms_per_step": round(state.debt / extension.amortization * 1000, 2),
        }

    def _costs(
        self, req_ids: Collection[str], key: tuple[int, int, bool]
    ) -> BudgetCosts:
        if len(req_ids) != 1:
            return super()._costs(req_ids, key)
        samples = self.costs.setdefault(key, BudgetCosts()).samples
        table = self.request_costs.setdefault(next(iter(req_ids)), {})
        return table.setdefault(
            key, BudgetCosts(samples=samples, selected=self.default)
        )

    def _select(self, scores: dict[int, float], costs: BudgetCosts) -> int:
        best = max(scores, key=scores.__getitem__)
        if self.prefer_default:
            if scores[best] <= scores[self.default] * 1.05:
                best = self.default
            margin = 1.0 if best == self.default else 1.08
            hold = 2 if best == self.default else 4
        elif self._num_reqs > 1:
            margin = 1.03 if best > costs.selected else 1.01
            hold = 2
        else:
            # Avoid hiding later acceptance after a single-request rejection burst.
            margin = 1.005 if best > costs.selected else 1.05
            hold = 8
        if (
            best != costs.selected
            and (
                (self.prefer_default and best == self.default)
                or scores[best] > scores[costs.selected] * margin
            )
            and costs.decisions - costs.changed >= hold
        ):
            costs.selected = best
            costs.changed = costs.decisions
        return costs.selected

    def complete(self, step: int, now: float, sampled: dict[str, int]) -> None:
        pending = self.pending.get(step)
        if pending is not None and self._forecast is not None and self._forecast_owner:
            self._forecast.complete(step, pending[2], sampled)
        if pending is not None:
            # Future proposals may already have a different batch size.
            weight = 0.15 if len(pending[2]) > 1 else self.acceptance_weight
            for req_id in pending[2]:
                if req_id in self.requests:
                    self.requests[req_id].weight = weight
        if self._extension is not None:
            self._extension.observe_completion(pending, now, sampled)
        super().complete(step, now, sampled)
        if self._extension is not None:
            self._extension.core.complete(step, now, sampled)


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
        scores, measured = self._batch_scores(req_ids, costs)
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
        _, key, drafts, _ = entry
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


@dataclass
class MTPExtensionCosts:
    """Bound trial work and amortize its recent excess cost before retrying."""

    cohort: frozenset[str] = field(default_factory=frozenset)
    decisions: int = 0
    selected: int = 0
    changed: int = -2
    probing: int | None = None
    attempts: int = 0
    stable: int = 0
    next_probe: int = 0
    interval: int = 128
    debt: float = 0.0
    prefix_floor: float = 1.0
    incumbent_updated: int = 0
    incumbent_refreshed: int = 0
    refresh_remaining: int = 0
    samples: dict[int, deque[float]] = field(default_factory=dict)


class MTPDraftExtension:
    """Use the three-draft policy until longer prefixes justify bounded trials."""

    prefix_threshold = 0.5
    prefix_samples = 4
    amortization = 128
    max_observation_age = 1024

    def __init__(self, parent: MTPDraftBudget) -> None:
        self.parent = parent
        self.core = MTPDraftBudget(3, prefer_default=parent.prefer_default)
        self.observed: dict[str, dict[int, deque[tuple[int, int]]]] = {}
        self.clocks: dict[str, int] = {}
        self.extensions: dict[
            tuple[str | frozenset[str], tuple[int, int, bool]], MTPExtensionCosts
        ] = {}

    def state(
        self, req_ids: Collection[str], key: tuple[int, int, bool]
    ) -> MTPExtensionCosts:
        identity = next(iter(req_ids)) if len(req_ids) == 1 else frozenset(req_ids)
        return self.extensions.setdefault((identity, key), MTPExtensionCosts())

    def probability(
        self, req_ids: list[str], position: int, *, predict: bool = True
    ) -> float | None:
        values = []
        for req_id in req_ids:
            observed = self.observed.get(req_id, {}).get(position, ())
            if position > 3 and isinstance(observed, deque):
                clock = self.clocks.get(req_id, 0)
                while observed and clock - observed[0][0] > self.max_observation_age:
                    observed.popleft()
            if len(observed) < (self.prefix_samples if position <= 3 else 3):
                return None
            successes = sum(accepted >= position for _, accepted in observed)
            observations = len(observed)
            if position == 3:
                prefix = math.prod(self.core.requests[req_id].conditional[:3])
                if self.parent.prefer_default:
                    prefixes = self.core.requests[req_id].verified_prefixes
                    if prefixes is None or len(prefixes[2]) < self.prefix_samples:
                        return None
                    successes = sum(accepted >= position for accepted in prefixes[2])
                    observations = len(prefixes[2])
                    prefix = self.core.requests[req_id].conditional[0]
                values.append(min(successes / (observations + 1), prefix))
            else:
                preceding = self.probability([req_id], position - 1, predict=False)
                if preceding is None:
                    return None
                # A stale marginal rate must not overvalue a changing prefix.
                opportunities = sum(
                    accepted >= position - 1 for _, accepted in observed
                )
                if self.parent.prefer_default and opportunities < 3:
                    return None
                values.append(preceding * successes / max(1, opportunities))
        forecast = self.parent._forecast if predict else None
        if forecast is not None:
            values = [forecast.prefix(r, position, p) for r, p in zip(req_ids, values)]
        return sum(values) / len(values)

    def yields(self, req_ids: list[str], k: int) -> float | None:
        value = sum(self.core._expected(r, min(k, 3)) for r in req_ids)
        for i in range(4, k + 1):
            p = self.probability(req_ids, i)
            if p is None:
                return None
            value += len(req_ids) * p
        return value

    def _probe_score(
        self,
        req_ids: list[str],
        drafts: int,
        measured: dict[int, float],
        costs: BudgetCosts,
        debt: float,
        *,
        samples: dict[int, deque[float]] | None = None,
    ) -> float:
        previous = self.yields(req_ids, drafts - 1)
        prefix = self.probability(req_ids, drafts - 1)
        if previous is None or prefix is None:
            return 0.0
        increment = 0.03 * measured[3]
        if not self.parent.prefer_default:
            increment = max(measured[3] - measured.get(2, measured[3] * 0.9), increment)
        table = costs.samples if samples is None else samples
        anchor = next(
            (k for k in range(drafts, 3, -1) if len(table.get(k, ())) >= 3),
            3,
        )
        elapsed = statistics.median(table[anchor]) if anchor > 3 else measured[3]
        elapsed += (drafts - anchor) * increment + debt / self.amortization
        return (previous + len(req_ids) * prefix) / elapsed

    def choose(self, req_ids: list[str], context: int, mixed: bool) -> int:
        if not req_ids:
            return self.parent.default
        for r in req_ids:
            self.clocks[r] = self.clocks.get(r, 0) + 1
            self.parent.requests.setdefault(
                r,
                Acceptance(
                    [0.75] * self.parent.max_drafts, self.parent.acceptance_weight
                ),
            )
        key = self.parent.key(len(req_ids), context, mixed)
        state = self.state(req_ids, key)
        state.decisions += 1
        state.debt *= 0.99
        cohort = frozenset(req_ids)
        if state.cohort != cohort:
            state.cohort = cohort
            state.probing = None
            state.selected = 0
            state.attempts = 0
        if self.parent.prefer_default and (
            mixed
            or any(
                r not in self.core.requests or self.core.requests[r].observations < 32
                for r in req_ids
            )
        ):
            state.probing = None
            return self.core.choose(req_ids, context, mixed)
        if self.parent.prefer_default and len(state.samples.get(3, ())) < 3:
            state.probing = None
            state.selected = 0
            self.core.choose(req_ids, context, mixed)
            return self.parent.default
        if (
            self.parent.prefer_default
            and state.selected > 3
            and state.probing is None
            and state.decisions
            - max(state.incumbent_updated, state.incumbent_refreshed)
            >= self.amortization
        ):
            state.incumbent_refreshed = state.decisions
            state.refresh_remaining = 4
        if state.refresh_remaining and state.probing is None:
            self.core.choose(req_ids, context, mixed)
            state.refresh_remaining -= 1
            return self.parent.default
        if (
            state.probing is not None
            and (
                state.stable < 3
                or (
                    self.parent.prefer_default
                    and self.probability(req_ids, state.probing) is None
                )
            )
            and state.attempts < 12
        ):
            state.attempts += 1
            return state.probing
        finished = state.probing
        state.probing = None
        fallback = self.core.choose(req_ids, context, mixed)
        costs = self.core._costs(req_ids, key)
        required = 4 if len(req_ids) == 1 else 3
        lengths = (3,) if self.parent.prefer_default else self.core.lengths
        if any(len(costs.samples.get(k, ())) < required for k in lengths):
            return fallback
        measured = {
            k: statistics.median(state.samples.get(k, v))
            if self.parent.prefer_default and len(state.samples.get(k, ())) >= 3
            else statistics.median(v)
            for k, v in costs.samples.items()
            if k in self.core.lengths and len(v) >= 3
        }
        baseline = max((self.yields(req_ids, k) or 0.0) / measured[k] for k in measured)
        parent = self.parent._costs(req_ids, key)
        high: dict[int, float] = {}
        for k in range(4, self.parent.max_drafts + 1):
            y = self.yields(req_ids, k)
            samples = (
                state.samples if self.parent.prefer_default else parent.samples
            ).get(k, ())
            if y is not None and len(samples) >= 3:
                high[k] = y / (
                    statistics.median(samples) + state.debt / self.amortization
                )
        prefer_default = self.parent.prefer_default
        growth = 1.05 if prefer_default else 1.03
        minimum_score = baseline * 1.01 if prefer_default else baseline / 1.01
        if (
            self.parent.prefer_default
            and state.selected > 3
            and state.selected not in high
        ) or (
            state.selected in high
            and (
                high[state.selected] < minimum_score
                or self.probability(req_ids, 3) is None
            )
        ):
            state.selected = 0
            state.changed = state.decisions
        best = max(high, key=high.__getitem__) if high else 0
        promoted = False
        current = high.get(state.selected, baseline)
        margin = 1.03 if best > state.selected else 1.01
        if (
            best
            and high[best] > baseline * growth
            and high[best] > current * margin
            and state.decisions - state.changed >= 2
        ):
            promoted = state.selected != best
            state.selected = best
            state.changed = state.decisions
        if finished:
            profitable = finished in high and high[finished] > baseline * growth
            state.interval = 128 if profitable else min(1024, state.interval * 2)
            state.next_probe = state.decisions + state.interval
        if promoted:
            state.interval = 128
            state.next_probe = state.decisions + 64
        p3 = self.probability(req_ids, 3)
        if p3 is not None:
            recovered = p3 >= self.prefix_threshold and state.prefix_floor < 0.25
            stale = all(
                (observed := self.observed.get(r, {}).get(4))
                and self.clocks[r] - observed[-1][0] >= 64
                for r in req_ids
            )
            improved = p3 >= 0.75 and p3 - state.prefix_floor >= 0.25 and stale
            if recovered or (
                improved
                and self._probe_score(
                    req_ids,
                    4,
                    measured,
                    parent,
                    state.debt,
                    samples=state.samples if prefer_default else None,
                )
                > baseline * 1.01
            ):
                state.next_probe = state.decisions
                state.interval = 128
                state.prefix_floor = p3
                for r in req_ids:
                    for position in range(4, self.parent.max_drafts + 1):
                        self.observed.get(r, {}).pop(position, None)
            else:
                state.prefix_floor = min(state.prefix_floor, p3)
        if (
            p3 is not None
            and (prefer_default or p3 >= self.prefix_threshold)
            and state.selected < self.parent.max_drafts
        ):
            candidate = 4
            for drafts in range(5, self.parent.max_drafts + 1):
                prefix = self.probability(req_ids, drafts - 1)
                if prefix is None or (not prefer_default and prefix < 0.35):
                    break
                candidate = drafts
            reference = max(baseline, high.get(state.selected, 0.0))
            ceiling = self._probe_score(
                req_ids,
                candidate,
                measured,
                parent,
                state.debt,
                samples=state.samples if prefer_default else None,
            )
            ready = state.decisions >= state.next_probe
            followup = (
                finished == candidate - 1
                and state.stable >= 3
                and (prefer_default or p3 >= self.prefix_threshold)
            )
            if ready or followup:
                margin = 1.01 if followup else growth
                refresh = (
                    ready
                    and candidate > max(4, state.selected)
                    and ceiling <= reference * margin
                )
                if refresh:
                    candidate = 4
                    ceiling = self._probe_score(
                        req_ids,
                        candidate,
                        measured,
                        parent,
                        state.debt,
                        samples=state.samples if prefer_default else None,
                    )
                # A rare high-prefix probe can discover a non-linear cheap shape.
                nonlinear = (
                    ready
                    and candidate == 4
                    and p3 >= 0.75
                    and state.decisions >= (256 if prefer_default else 128)
                )
                if candidate != state.selected and (
                    ceiling > reference * margin or nonlinear
                ):
                    if refresh:
                        for r in req_ids:
                            for position in range(4, self.parent.max_drafts + 1):
                                self.observed.get(r, {}).pop(position, None)
                    state.probing = candidate
                    state.stable = 0
                    state.attempts = 1
                    return candidate
        return state.selected or fallback

    def scheduled(
        self,
        step: int,
        started: float,
        context: int,
        mixed: bool,
        drafts: dict[str, int],
        *,
        proposal_drafts: int | None = None,
    ) -> None:
        proposal = -1 if any(k > 3 for k in drafts.values()) else proposal_drafts
        self.core.scheduled(
            step,
            started,
            context,
            mixed,
            {r: min(3, k) for r, k in drafts.items()},
            proposal_drafts=proposal,
        )

    def observe_completion(
        self,
        pending: tuple[float, tuple[int, int, bool], dict[str, int], int | None] | None,
        now: float,
        sampled: dict[str, int],
    ) -> None:
        if pending:
            started, key, drafts, proposal = pending
            req_ids = list(drafts)
            for r, k in drafts.items():
                if k >= 3 and r in self.parent.requests and sampled.get(r, 0) > 0:
                    observed = self.observed.setdefault(r, {})
                    accepted = min(k, sampled[r] - 1)
                    # Short rounds must not evict evidence for unverified suffixes.
                    for position in range(3, k + 1):
                        observed.setdefault(position, deque(maxlen=32)).append(
                            (self.clocks.get(r, 0), accepted)
                        )
            if all(
                r in self.parent.requests and sampled.get(r, 0) > 0 for r in req_ids
            ):
                state = self.state(req_ids, key)
                elapsed = now - started
                if (
                    frozenset(req_ids) == state.cohort
                    and proposal is not None
                    and math.isfinite(elapsed)
                    and elapsed > 0
                    and set(drafts.values()) == {proposal}
                ):
                    state.samples.setdefault(proposal, deque(maxlen=6)).append(elapsed)
                    if proposal == 3:
                        state.incumbent_updated = state.decisions
                if (
                    state.probing
                    and math.isfinite(elapsed)
                    and elapsed > 0
                    and set(drafts.values()) == {state.probing}
                    and proposal == state.probing
                ):
                    state.stable += 1
                if (proposal and proposal > 3) or any(k > 3 for k in drafts.values()):
                    costs = self.core._costs(req_ids, key)
                    measured = {
                        k: statistics.median(state.samples.get(k, v))
                        if self.parent.prefer_default
                        and len(state.samples.get(k, ())) >= 3
                        else statistics.median(v)
                        for k, v in costs.samples.items()
                        if len(v) >= 3
                    }
                    if measured and math.isfinite(elapsed) and elapsed > 0:
                        rate = max(
                            (self.yields(req_ids, k) or 0.0) / c
                            for k, c in measured.items()
                        )
                        state.debt = max(
                            0.0,
                            state.debt
                            + elapsed
                            - sum(sampled[r] for r in req_ids) / rate,
                        )

    def retain_requests(self, req_ids: Collection[str]) -> None:
        self.core.retain_requests(req_ids)
        for r in self.observed.keys() - req_ids:
            del self.observed[r]
        for r in self.clocks.keys() - req_ids:
            del self.clocks[r]
        retained = set(req_ids)
        for identity, key in list(self.extensions):
            if (isinstance(identity, str) and identity not in retained) or (
                isinstance(identity, frozenset) and not identity <= retained
            ):
                del self.extensions[(identity, key)]
