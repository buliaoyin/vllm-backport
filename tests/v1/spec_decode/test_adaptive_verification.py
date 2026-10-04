# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections import Counter, deque
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.spec_decode.dynamic.adaptive import (
    AdaptiveDraftBudget,
    ConfidenceDraftBudget,
    MTPDraftBudget,
    prefers_default_mtp_budget,
)
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
)
from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
def test_scheduler_budget_follows_acceptance_when_task_changes(budget_cls):
    """An inexpensive extra verification is useful only while drafts are accepted."""
    budget = budget_cls(7)
    budget.choose(["request"], 32000, False)
    for phase, accepted in enumerate((7, 0, 7)):
        for step in range(60):
            k = budget.lengths[step % len(budget.lengths)]
            stamp = phase * 100 + step
            budget.scheduled(stamp, 0.0, 32000, False, {"request": k})
            budget.complete(
                stamp,
                0.017 + 0.003 * k,
                {"request": 1 + min(accepted, k)},
            )
        assert budget.choose(["request"], 32000, False) == (7 if accepted else 1)


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
def test_scheduler_budget_does_not_treat_unverified_suffix_as_rejected(budget_cls):
    budget = budget_cls(7)
    budget.choose(["request"], 32000, False)
    budget.scheduled(1, 0.0, 32000, False, {"request": 7})
    budget.complete(1, 0.03, {"request": 8})
    suffix = budget.requests["request"].conditional[1:]
    budget.scheduled(2, 0.1, 32000, False, {"request": 1})
    budget.complete(2, 0.12, {"request": 1})
    assert budget.requests["request"].conditional[1:] == suffix


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
def test_scheduler_budget_recovers_after_short_prefix_acceptance_improves(budget_cls):
    budget = budget_cls(7)
    budget.choose(["request"], 32000, False)
    for step in range(60):
        k = budget.lengths[step % len(budget.lengths)]
        budget.scheduled(step, 0.0, 32000, False, {"request": k})
        budget.complete(step, 0.017 + 0.003 * k, {"request": 1})
    assert budget.choose(["request"], 32000, False) == 1

    # Rejections at the first position must not poison unobserved later ones.
    for step in range(60, 100):
        budget.scheduled(step, 0.0, 32000, False, {"request": 3})
        budget.complete(step, 0.026, {"request": 4})
    assert budget.choose(["request"], 32000, False) > 3


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
def test_scheduler_budget_ignores_stale_output_and_reclaims_request_state(budget_cls):
    budget = budget_cls(7)
    budget.choose(["aborted", "active"], 32000, False)
    budget.scheduled(1, 0.0, 32000, False, {"aborted": 7})
    budget.complete(1, 0.04, {})
    assert not budget.counts
    assert not budget.pending
    budget.retain_requests({"active"})
    assert set(budget.requests) == {"active"}
    assert not budget.costs[budget.key(2, 32000, False)].samples


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
@pytest.mark.parametrize("best_budget", [2, 4, 5, 6])
def test_scheduler_budget_selects_middle_budget_when_longer_is_expensive(
    budget_cls, best_budget
):
    """An efficient middle budget must not be rounded to an old preset."""
    budget = budget_cls(7)
    budget.choose(["request"], 32000, False)
    for step in range(120):
        k = budget.lengths[step % len(budget.lengths)]
        budget.scheduled(step, 0.0, 32000, False, {"request": k})
        budget.complete(
            step,
            0.017 + 0.003 * k if k <= best_budget else 0.060,
            {"request": min(k, best_budget) + 1},
        )
    assert budget.choose(["request"], 32000, False) == best_budget


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
def test_scheduler_confidence_rejects_stale_and_invalid_feedback(budget_cls):
    budget = budget_cls(3)
    budget.observe_confidences("request", 8, [0.8, 0.7, 0.6])
    budget.observe_confidences("request", 7, [0.1, 0.1, 0.1])
    budget.observe_confidences("request", 9, [float("nan"), 0.9, 0.9])
    budget.observe_confidences("request", 10, [0.9])
    assert budget.confidences["request"] == (8, [0.8, 0.7, 0.6])
    budget.retain_requests(set())
    assert not budget.confidences


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, MTPDraftBudget])
def test_mtp_budget_transition_updates_acceptance_without_mixing_costs(budget_cls):
    budget = budget_cls(3)
    budget.choose(["request"], 4096, False)
    budget.scheduled(1, 0, 4096, False, {"request": 3}, proposal_drafts=1)
    budget.complete(1, 0.03, {"request": 1})
    assert budget.requests["request"].conditional[0] < 0.75
    assert budget.counts[3] == 1
    costs = budget._costs(["request"], budget.key(1, 4096, False))
    assert not costs.samples
    budget.scheduled(2, 1, 4096, False, {"request": 1}, proposal_drafts=1)
    budget.complete(2, 1.02, {"request": 2})
    assert list(costs.samples[1]) == pytest.approx([0.02])


def test_mtp_budget_does_not_stay_short_after_rejection_bursts():
    """K=3 has the best average yield, despite periodic runs of rejections."""
    budget = MTPDraftBudget(3)
    pattern = [3] * 8 + [2] * 4 + [0] * 6
    selected = []
    now = 0.0
    for step in range(360):
        k = budget.choose(["request"], 32768, False)
        selected.append(k)
        budget.scheduled(step, now, 32768, False, {"request": k}, proposal_drafts=k)
        now += 0.028 + 0.006 * k
        budget.complete(
            step, now, {"request": 1 + min(k, pattern[step % len(pattern)])}
        )
    assert selected[-180:].count(3) >= 160


@pytest.mark.parametrize("num_reqs", [1, 8])
@pytest.mark.parametrize("prefer_default", [False, True])
def test_mtp_budget_still_adapts_when_acceptance_changes(num_reqs, prefer_default):
    budget = MTPDraftBudget(5, prefer_default=prefer_default)
    req_ids = [str(i) for i in range(num_reqs)]
    now = 0.0
    for phase, accepted in enumerate((5, 0, 5)):
        choices = []
        for iteration in range(160):
            k = budget.choose(req_ids, 32768, False)
            choices.append(k)
            step = phase * 160 + iteration
            budget.scheduled(
                step, now, 32768, False, dict.fromkeys(req_ids, k), proposal_drafts=k
            )
            now += 0.02 + 0.002 * k
            budget.complete(step, now, dict.fromkeys(req_ids, 1 + min(k, accepted)))
        assert choices[-32:].count(5 if accepted else 1) >= 30


@pytest.mark.parametrize(
    "quantization,pipeline_parallel_size,expected",
    [("exl3", 1, True), ("modelopt_fp4", 1, False), ("modelopt_fp4", 2, True)],
)
def test_qwen_mtp_incumbent_policy_matches_quantization_and_pipeline(
    quantization, pipeline_parallel_size, expected
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen4_exp_text"),
            quantization=quantization,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        speculative_config=SimpleNamespace(
            method="mtp",
            enable_adaptive_verification=True,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="qwen4_exp_mtp",
                    architectures=["Qwen4ExpMTP"],
                    n_predict=1,
                )
            ),
        ),
    )
    assert prefers_default_mtp_budget(config) is expected
    config.speculative_config.enable_adaptive_verification = False
    assert not prefers_default_mtp_budget(config)


def test_mtp_reuses_shape_costs_but_resets_request_acceptance():
    budget = MTPDraftBudget(3)
    budget.choose(["old"], 32768, False)
    for step in range(12):
        k = step % 3 + 1
        budget.scheduled(step, 0.0, 32768, False, {"old": k}, proposal_drafts=k)
        budget.complete(step, 0.02 + 0.04 * k, {"old": k + 1})
    budget.retain_requests(set())
    assert not budget.requests and not budget.request_costs
    # The expensive longer shapes need not be measured again on a new request.
    assert budget.choose(["new"], 32768, False) == 1
    assert budget.requests["new"].conditional == [0.75] * 3
    # Different contexts, batches and prefill overlap still need measurements.
    assert budget.choose(["long"], 65536, False) == 3
    assert budget.choose(["mixed"], 32768, True) == 3
    assert budget.choose(["a", "b"], 32768, False) == 1


def test_mtp_batch_feedback_uses_verified_batch_size():
    """A future single-request proposal must not slow the pending batch feedback."""
    budget = MTPDraftBudget(5)
    budget.choose(["a", "b"], 4096, False)
    for step in range(20):
        budget.scheduled(step, float(step), 4096, False, {"a": 3, "b": 3})
        budget.choose(["a"], 4096, False)
        budget.complete(step, step + 0.05, {"a": 1, "b": 1})
    assert budget.requests["a"].expected(1) < 1.1
    assert budget.requests["b"].expected(1) < 1.1


def test_mtp_batch_probes_measure_shapes_with_nonlinear_costs():
    """A probe must outlast the old-K verification to discover a cheap shape."""
    budget = MTPDraftBudget(5)
    req_ids = [str(i) for i in range(8)]
    now, previous, choices = 0.0, 3, []
    latencies = {1: 0.03, 2: 0.06, 3: 0.10, 4: 0.03, 5: 0.13}
    for step in range(400):
        k = budget.choose(req_ids, 4096, False)
        choices.append(k)
        budget.scheduled(
            step, now, 4096, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        now += latencies[k]
        budget.complete(step, now, dict.fromkeys(req_ids, min(previous, 4) + 1))
        previous = k
    assert choices[-64:].count(4) >= 58


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_batch_refreshes_previously_measured_costs(feedback_delay: int) -> None:
    """Periodic probes must measure again when a calibrated K becomes faster."""
    budget = MTPDraftBudget(3)
    req_ids = [str(i) for i in range(4)]
    latencies = {1: 0.020, 2: 0.019, 3: 0.018}
    pending: deque[tuple[int, float, int]] = deque()
    now, previous = 0.0, 3
    choices: list[int] = []
    for step in range(424):
        if step == 24:
            assert choices[-1] == 3
            latencies[2] = 0.010
        k = budget.choose(req_ids, 4096, False)
        choices.append(k)
        budget.scheduled(
            step, now, 4096, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        now += latencies[k]
        pending.append((step, now, previous + 1))
        if len(pending) > feedback_delay:
            completed, finished, sampled = pending.popleft()
            budget.complete(completed, finished, dict.fromkeys(req_ids, sampled))
        previous = k
    assert choices[-64:].count(2) >= 56


def test_mtp_batch_rejecting_drafts_starts_with_short_calibration():
    budget = MTPDraftBudget(5)
    req_ids = [str(i) for i in range(8)]
    choices = []
    for step in range(16):
        k = budget.choose(req_ids, 4096, False)
        choices.append(k)
        budget.scheduled(step, float(step), 4096, False, dict.fromkeys(req_ids, k))
        budget.complete(step, step + 0.01 + 0.005 * k, dict.fromkeys(req_ids, 1))
    assert set(choices) == {1, 2, 3}
    assert choices[-4:] == [1] * 4


@pytest.mark.parametrize("num_reqs", [1, 4])
def test_mtp_default_policy_preserves_short_request_budget(num_reqs):
    """Short requests must not pay to calibrate every available draft length."""
    budget = MTPDraftBudget(5, prefer_default=True)
    req_ids = [str(i) for i in range(num_reqs)]
    for step in range(32):
        k = budget.choose(req_ids, 4096, False)
        assert k == 3
        budget.scheduled(
            step,
            float(step),
            4096,
            False,
            dict.fromkeys(req_ids, k),
            proposal_drafts=k,
        )
        budget.complete(step, step + 0.03, dict.fromkeys(req_ids, 4))


def test_mtp_default_policy_avoids_trials_during_prefill_overlap():
    """Queued work must not cause extra exploratory drafting on active requests."""
    budget = MTPDraftBudget(5, prefer_default=True)
    for step in range(160):
        k = budget.choose(["request"], 4096, True)
        assert k == 3
        budget.scheduled(
            step, float(step), 4096, True, {"request": k}, proposal_drafts=k
        )
        budget.complete(step, step + 0.03, {"request": 4})


def test_mtp_default_policy_measures_short_shape_before_promoting_it():
    """Low acceptance cannot justify a slower small-M kernel."""
    budget = MTPDraftBudget(5, prefer_default=True)
    choices = []
    for step in range(160):
        k = budget.choose(["request"], 4096, False)
        choices.append(k)
        budget.scheduled(
            step, float(step), 4096, False, {"request": k}, proposal_drafts=k
        )
        budget.complete(step, step + (0.06 if k < 3 else 0.03), {"request": 1})
    assert choices.count(1) >= 3
    assert choices[-32:] == [3] * 32


def test_mtp_default_policy_refreshes_censored_third_position():
    """A good first two drafts must not hide recovery of the third draft."""
    budget = MTPDraftBudget(3, prefer_default=True)
    latencies = {1: 0.04, 2: 0.032, 3: 0.036}
    for phase, accepted in enumerate((2, 3)):
        choices = []
        for iteration in range(192):
            step = phase * 192 + iteration
            k = budget.choose(["request"], 4096, False)
            choices.append(k)
            budget.scheduled(
                step, float(step), 4096, False, {"request": k}, proposal_drafts=k
            )
            budget.complete(
                step, step + latencies[k], {"request": min(k, accepted) + 1}
            )
        assert choices[-64:].count(2 if accepted == 2 else 3) >= 48


@pytest.mark.parametrize("num_reqs", [1, 4])
@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_default_policy_bounds_refresh_cost_for_profitable_short_budget(
    num_reqs, feedback_delay
):
    """Refreshing a censored suffix must preserve a measured short-budget gain."""
    budget = MTPDraftBudget(3, prefer_default=True)
    req_ids = [str(i) for i in range(num_reqs)]
    latencies = {1: 0.024, 2: 0.030, 3: 0.036}
    pending: deque[tuple[int, float, int]] = deque()
    measured = []
    now, previous = 0.0, 3
    for step in range(384):
        k = budget.choose(req_ids, 4096, False)
        budget.scheduled(
            step, now, 4096, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        elapsed = (latencies[previous] + latencies[k]) / 2
        now += elapsed
        sampled = min(previous, 2) + 1
        measured.append((sampled, elapsed))
        pending.append((step, now, sampled))
        if len(pending) > feedback_delay:
            completed, finished, sampled = pending.popleft()
            budget.complete(completed, finished, dict.fromkeys(req_ids, sampled))
        previous = k
    observed = sum(tokens for tokens, _ in measured[-192:]) / sum(
        elapsed for _, elapsed in measured[-192:]
    )
    assert observed >= 0.98 * 3 / latencies[2]


def test_mtp_default_policy_scores_observed_prefix_yield():
    """Rejection bursts must not distort the fixed-three throughput reference."""
    budget = MTPDraftBudget(5, prefer_default=True)
    extension = budget._extension
    assert extension is not None
    for step in range(32):
        budget.choose(["request"], 4096, False)
        budget.scheduled(
            step, float(step), 4096, False, {"request": 3}, proposal_drafts=3
        )
        budget.complete(step, step + 0.03, {"request": 4 if step % 2 else 1})
    assert extension.yields(["request"], 3) == pytest.approx(2.5)
    assert extension.yields(["request"], 2) == pytest.approx(2.0)
    assert extension.core.requests["request"].expected(3) == pytest.approx(2.5)
    assert extension.core.requests["request"].expected(2) == pytest.approx(2.0)


def test_mtp_default_policy_retains_unverified_prefix_yields():
    """Short proposals must not bias the measured yield of longer budgets."""
    budget = MTPDraftBudget(3, prefer_default=True)
    budget.choose(["request"], 4096, False)
    for step in range(40):
        k = 3 if step < 8 else 1
        budget.scheduled(
            step, float(step), 4096, False, {"request": k}, proposal_drafts=k
        )
        budget.complete(step, step + 0.03, {"request": k + 1})
    acceptance = budget.requests["request"]
    assert acceptance.expected(1) == pytest.approx(2.0)
    assert acceptance.expected(3) == pytest.approx(4.0)


def test_mtp_default_policy_returns_to_three_when_short_gain_is_marginal():
    """A measured short budget must also yield to the default within the margin."""
    budget = MTPDraftBudget(3, prefer_default=True)
    latencies = {1: 0.04, 2: 0.032, 3: 0.036}
    for phase in range(2):
        choices = []
        for iteration in range(192):
            step = phase * 192 + iteration
            k = budget.choose(["request"], 4096, False)
            choices.append(k)
            accepted = 2 if not phase or iteration % 3 else 3
            budget.scheduled(
                step, float(step), 4096, False, {"request": k}, proposal_drafts=k
            )
            budget.complete(
                step, step + latencies[k], {"request": min(k, accepted) + 1}
            )
        assert choices[-64:].count(2 if not phase else 3) >= 48


@pytest.mark.parametrize("prefer_default", [False, True])
def test_mtp_default_policy_avoids_marginal_long_budget_gains(prefer_default):
    """A four-percent estimate must not displace the preferred three drafts."""
    budget = MTPDraftBudget(5, prefer_default=prefer_default)
    latencies = {1: 0.022, 2: 0.024, 3: 0.026, 4: 0.03125, 5: 0.0375}
    previous, now, choices = 3, 0.0, []
    for step in range(400):
        k = budget.choose(["request"], 4096, False)
        choices.append(k)
        budget.scheduled(
            step, now, 4096, False, {"request": previous}, proposal_drafts=k
        )
        now += (latencies[previous] + latencies[k]) / 2
        budget.complete(step, now, {"request": previous + 1})
        previous = k
    expected = {3} if prefer_default else {4, 5}
    assert sum(k in expected for k in choices[-64:]) >= 60


@pytest.mark.parametrize("num_reqs", [1, 4])
def test_mtp_default_policy_discovers_efficient_suffix_with_moderate_survival(num_reqs):
    """Cheap suffixes can improve throughput even when most prefixes reject."""
    budget = MTPDraftBudget(4, prefer_default=True)
    req_ids = [str(i) for i in range(num_reqs)]
    latencies = {1: 0.028, 2: 0.029, 3: 0.030, 4: 0.03114}
    acceptance = [4, 4, 4, 3, 0, 0, 0, 0, 0, 0]
    pending: deque[tuple[int, float, int]] = deque()
    measured = []
    now, previous = 0.0, 3
    for step in range(768):
        k = budget.choose(req_ids, 4096, False)
        budget.scheduled(
            step, now, 4096, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        elapsed = (latencies[previous] + latencies[k]) / 2
        now += elapsed
        accepted = acceptance[step % len(acceptance)]
        sampled, reference = min(previous, accepted) + 1, min(3, accepted) + 1
        measured.append((sampled, reference, elapsed))
        pending.append((step, now, sampled))
        if len(pending) > 2:
            completed, finished, sampled = pending.popleft()
            budget.complete(completed, finished, dict.fromkeys(req_ids, sampled))
        previous = k
    observed = sum(tokens for tokens, _, _ in measured[-256:]) / sum(
        elapsed for _, _, elapsed in measured[-256:]
    )
    reference_rate = sum(tokens for _, tokens, _ in measured[-256:]) / (
        256 * latencies[3]
    )
    assert observed > 1.05 * reference_rate


def _mtp_prefix_probe_steps(
    budget, feedback_delay, accepted, latencies=None, max_steps=80
):
    pending: deque[tuple[int, float, int, int, int]] = deque()
    now, previous, completed = 0.0, 3, None
    latencies = latencies or {1: 0.029, 2: 0.032, 3: 0.035, 4: 0.038}
    for step in range(max_steps):
        proposed = budget.choose(["request"], 8192, False)
        stats = budget.extension_stats(["request"], 8192, False)
        yield proposed, stats, completed
        budget.scheduled(
            step, now, 8192, False, {"request": previous}, proposal_drafts=proposed
        )
        step_latencies = latencies(step) if callable(latencies) else latencies
        now += (step_latencies[previous] + step_latencies[proposed]) / 2
        count = accepted(previous)
        pending.append((step, now, previous, proposed, count))
        completed = None
        if len(pending) > feedback_delay:
            done, finished, verified, proposal, count = pending.popleft()
            budget.complete(done, finished, {"request": count + 1})
            completed = verified, proposal, count
        previous = proposed


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_default_trial_waits_for_conditional_prefix_opportunities(feedback_delay):
    """Three cheap costs cannot decide a suffix before its prefix is observed."""
    budget = MTPDraftBudget(4, prefer_default=True)
    long_rounds = 0

    def accepted(drafts):
        nonlocal long_rounds
        if drafts == 4:
            long_rounds += 1
            return 2 if long_rounds <= 4 else 4
        return drafts

    stable, opportunities, held, promoted = 0, 0, False, False
    for proposed, stats, completed in _mtp_prefix_probe_steps(
        budget, feedback_delay, accepted
    ):
        if completed is not None:
            verified, proposal, count = completed
            stable += int(verified == proposal == 4)
            opportunities += int(verified == 4 and count >= 3)
        if stable >= 3 and opportunities < 3:
            assert proposed == stats["trial_k"] == 4
            held = True
        if proposed == 4 and stats["trial_k"] is None:
            assert stable >= 3 and opportunities >= 3
            promoted = True
            break
    assert held and promoted


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_default_trial_backs_off_when_prefix_stays_unobserved(feedback_delay):
    """Missing conditional evidence must not prolong a trial beyond its cap."""
    budget = MTPDraftBudget(4, prefer_default=True)
    trials, returned = 0, 0
    for proposed, stats, _ in _mtp_prefix_probe_steps(
        budget, feedback_delay, lambda drafts: 2 if drafts == 4 else drafts
    ):
        if stats["trial_k"] == 4:
            assert proposed == 4
            trials += 1
        elif trials:
            assert proposed == 3 and stats["next_trial_in"] > 0
            returned += 1
            if returned == 8:
                break
    assert trials == 12 and returned == 8


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_default_policy_retains_profitable_suffix_when_gain_shrinks(feedback_delay):
    """A promoted suffix survives a smaller gain, then exits when unprofitable."""
    budget = MTPDraftBudget(4, prefer_default=True)
    gains = (0.08, 0.03, -0.05)

    def latencies(step):
        return {
            1: 0.029,
            2: 0.032,
            3: 0.035,
            4: 0.035 * 1.25 / (1 + gains[step // 384]),
        }

    choices: list[list[int]] = [[], [], []]
    for step, (proposed, _, _) in enumerate(
        _mtp_prefix_probe_steps(
            budget,
            feedback_delay,
            lambda drafts: drafts,
            latencies=latencies,
            max_steps=3 * 384,
        )
    ):
        phase = step // 384
        choices[phase].append(proposed)
    assert choices[0][-128:].count(4) >= 120
    assert choices[1][-128:].count(4) >= 120
    assert choices[2][-128:].count(3) >= 120


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_default_policy_remeasures_long_costs_for_new_batch_cohort(feedback_delay):
    """Another cohort's cheap shape cannot justify prolonged expensive drafting."""
    budget = MTPDraftBudget(4, prefer_default=True)
    now, step = 0.0, 0
    for cohort in ("cheap", "expensive"):
        req_ids = [f"{cohort}-{i}" for i in range(4)]
        budget.retain_requests(set(req_ids))
        latencies = {1: 0.028, 2: 0.029, 3: 0.030}
        latencies[4] = 0.031 if cohort == "cheap" else 0.080
        pending: deque[tuple[int, float, int]] = deque()
        choices = []
        previous = 3
        for _ in range(256):
            k = budget.choose(req_ids, 4096, False)
            choices.append(k)
            budget.scheduled(
                step,
                now,
                4096,
                False,
                dict.fromkeys(req_ids, previous),
                proposal_drafts=k,
            )
            now += (latencies[previous] + latencies[k]) / 2
            pending.append((step, now, previous + 1))
            if len(pending) > feedback_delay:
                completed, finished, sampled = pending.popleft()
                budget.complete(completed, finished, dict.fromkeys(req_ids, sampled))
            previous = k
            step += 1
        for completed, finished, sampled in pending:
            budget.complete(completed, finished, dict.fromkeys(req_ids, sampled))
        if cohort == "cheap":
            assert choices[-64:].count(4) >= 60
        else:
            assert choices[:64].count(4) <= 4 + feedback_delay
            assert choices[-64:].count(3) >= 60


@pytest.mark.parametrize("num_reqs", [1, 8])
def test_mtp_long_budget_matches_three_drafts_without_prefix_evidence(num_reqs):
    """A larger maximum must not force suffix work on rejecting requests."""
    budgets = [MTPDraftBudget(3), MTPDraftBudget(5)]
    now, previous = 0.0, 3
    for step in range(384):
        req_ids = [f"{step // 96}-{i}" for i in range(num_reqs)]
        context, mixed = 32768 * (step % 2), step % 17 < 3
        for budget in budgets:
            budget.retain_requests(set(req_ids))
        choices = [budget.choose(req_ids, context, mixed) for budget in budgets]
        assert choices[0] == choices[1]
        for budget in budgets:
            budget.scheduled(
                step,
                now,
                context,
                mixed,
                dict.fromkeys(req_ids, previous),
                proposal_drafts=choices[0],
            )
            budget.complete(step, now + 0.03, dict.fromkeys(req_ids, 1))
        now += 0.03
        previous = choices[0]


@pytest.mark.parametrize("num_reqs", [1, 8])
@pytest.mark.parametrize("prefer_default", [False, True])
def test_mtp_expensive_long_trials_back_off(num_reqs, prefer_default):
    """High acceptance alone cannot justify recurring expensive suffix work."""
    budget = MTPDraftBudget(5, prefer_default=prefer_default)
    req_ids = [str(i) for i in range(num_reqs)]
    latencies = {1: 0.03, 2: 0.034, 3: 0.038, 4: 0.095, 5: 0.15}
    now, previous, choices = 0.0, 3, []
    for step in range(768):
        k = budget.choose(req_ids, 32768, False)
        choices.append(k)
        budget.scheduled(
            step, now, 32768, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        now += (latencies[k] + latencies[previous]) / 2
        budget.complete(step, now, dict.fromkeys(req_ids, previous + 1))
        previous = k
    assert sum(k > 3 for k in choices) < 16
    assert choices[-128:].count(3) > 120


@pytest.mark.parametrize("num_reqs", [1, 8])
@pytest.mark.parametrize("prefer_default", [False, True])
def test_mtp_keeps_suffix_evidence_until_next_long_trial(num_reqs, prefer_default):
    """Short rounds must not hide a profitable K5 after a marginal K4 trial."""
    budget = MTPDraftBudget(5, prefer_default=prefer_default)
    req_ids = [str(i) for i in range(num_reqs)]
    latencies = {1: 0.035, 2: 0.041, 3: 0.048, 4: 0.058, 5: 0.065}
    now, previous, choices = 0.0, 3, []
    for step in range(384):
        k = budget.choose(req_ids, 32768, False)
        choices.append(k)
        budget.scheduled(
            step, now, 32768, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        now += (latencies[k] + latencies[previous]) / 2
        budget.complete(step, now, dict.fromkeys(req_ids, previous + 1))
        previous = k
    assert choices[-64:].count(5) >= 60
    assert choices.index(5) < 64


@pytest.mark.parametrize("num_reqs", [1, 8])
def test_mtp_remeasures_early_suffix_rejection_before_retrying_longer_drafts(num_reqs):
    """Two early K4 rejections must not strand a later profitable K5 at K3."""
    budget = MTPDraftBudget(5)
    req_ids = [str(i) for i in range(num_reqs)]
    latencies = {1: 0.064, 2: 0.075, 3: 0.089, 4: 0.107, 5: 0.118}
    now, previous, rejections, choices = 0.0, 3, 2, []
    for step in range(384):
        k = budget.choose(req_ids, 32768, False)
        choices.append(k)
        budget.scheduled(
            step, now, 32768, False, dict.fromkeys(req_ids, previous), proposal_drafts=k
        )
        now += (latencies[k] + latencies[previous]) / 2
        accepted = previous
        if previous == 4 and rejections:
            accepted = 3
            rejections -= 1
        budget.complete(step, now, dict.fromkeys(req_ids, accepted + 1))
        previous = k
    assert 5 in choices[:320]
    assert choices[-64:].count(5) >= 60


@pytest.mark.parametrize("num_reqs", [1, 8])
def test_mtp_refreshes_suffix_when_only_later_acceptance_changes(num_reqs):
    """Stale suffix rejection must not prevent recovery with a reliable prefix."""
    budget = MTPDraftBudget(5)
    req_ids = [str(i) for i in range(num_reqs)]
    latencies = {1: 0.035, 2: 0.041, 3: 0.048, 4: 0.054, 5: 0.060}
    now, previous, step = 0.0, 3, 0
    for accepted, length in ((5, 160), (3, 1536), (5, 1536)):
        choices = []
        for _ in range(length):
            k = budget.choose(req_ids, 32768, False)
            choices.append(k)
            budget.scheduled(
                step,
                now,
                32768,
                False,
                dict.fromkeys(req_ids, previous),
                proposal_drafts=k,
            )
            now += (latencies[k] + latencies[previous]) / 2
            budget.complete(
                step, now, dict.fromkeys(req_ids, min(previous, accepted) + 1)
            )
            previous = k
            step += 1
        assert choices[-32:].count(5 if accepted == 5 else 3) >= 30


@pytest.mark.parametrize("verified,proposal", [(4, 3), (3, 4), (5, 5)])
def test_mtp_long_shapes_preserve_short_policy_timing(verified, proposal):
    budget = MTPDraftBudget(5)
    budget.choose(["request"], 4096, False)
    budget.scheduled(0, 0.0, 4096, False, {"request": 3}, proposal_drafts=3)
    budget.complete(0, 0.03, {"request": 4})
    core = budget._extension.core
    costs = core._costs(["request"], budget.key(1, 4096, False))
    before = list(costs.samples[3])
    budget.scheduled(
        1, 1.0, 4096, False, {"request": verified}, proposal_drafts=proposal
    )
    budget.complete(1, 1.8, {"request": verified + 1})
    assert list(costs.samples[3]) == before
    assert core.requests["request"].conditional[0] > 0.75


@pytest.mark.parametrize("num_reqs", [1, 8])
def test_mtp_suffix_yield_follows_current_prefix_acceptance(num_reqs):
    """A changing prefix must not turn 50% suffix acceptance into certainty."""
    budget = MTPDraftBudget(5)
    req_ids = [str(i) for i in range(num_reqs)]
    for step in range(48):
        budget.choose(req_ids, 4096, False)
        k = 1 if 32 <= step < 40 else 5
        accepted = (3 if step % 2 else 5) if step < 32 else 0
        budget.scheduled(
            step,
            float(step),
            4096,
            False,
            dict.fromkeys(req_ids, k),
            proposal_drafts=k,
        )
        budget.complete(step, step + 0.05, dict.fromkeys(req_ids, accepted + 1))
        if step in (39, 47):
            prefix = budget.extension_stats(req_ids, 4096, False)["prefix_acceptance"]
            assert prefix[4] == pytest.approx(prefix[3] * 0.5, abs=0.002)
            assert prefix[5] == pytest.approx(prefix[4], abs=0.002)


def test_mtp_batch_suffix_score_averages_request_survival_probabilities():
    """A reliable request's suffix must not inherit another request's prefix."""
    budget = MTPDraftBudget(5)
    req_ids = ["stable", "changing"]
    for step in range(48):
        budget.choose(req_ids, 4096, False)
        if step < 32:
            drafts = dict.fromkeys(req_ids, 5)
            sampled = {"stable": 4 if step % 2 else 6, "changing": 6}
        else:
            drafts = {"stable": 3, "changing": 1}
            sampled = {"stable": 4, "changing": 1}
        budget.scheduled(step, float(step), 4096, False, drafts)
        budget.complete(step, step + 0.05, sampled)
    batched = budget.extension_stats(req_ids, 4096, False)["prefix_acceptance"]
    individual = []
    for req_id in req_ids:
        budget.choose([req_id], 4096, False)
        individual.append(
            budget.extension_stats([req_id], 4096, False)["prefix_acceptance"]
        )
    for position in (4, 5):
        assert batched[position] == pytest.approx(
            sum(p[position] for p in individual) / len(individual), abs=0.002
        )


def test_mtp_cancelled_long_request_cannot_recreate_feedback():
    budget = MTPDraftBudget(5)
    budget.choose(["request"], 4096, False)
    budget.scheduled(0, 0.0, 4096, False, {"request": 5}, proposal_drafts=5)
    budget.retain_requests(set())
    budget.complete(0, 0.5, {})
    assert not budget.requests and not budget.counts
    assert not budget._extension.observed and not budget._extension.extensions
    assert not budget._extension.clocks
    assert not budget._extension.core.requests


def test_calibrated_budget_follows_high_low_high_acceptance():
    budget = ConfidenceDraftBudget(5)
    now = 0.0
    for phase, accepted in enumerate((5, 0, 5)):
        selected = []
        for iteration in range(100):
            k = budget.choose(["request"], 32000, False)
            step = phase * 100 + iteration
            budget.scheduled(step, now, 32000, False, {"request": k})
            now += 0.02 + 0.004 * k
            budget.complete(step, now, {"request": min(k, accepted) + 1})
            budget.observe_confidences(
                "request", step, [0.99 if accepted else 0.05] * 5
            )
            if iteration >= 80:
                selected.append(k)
        assert set(selected) == {5 if accepted else 1}


def test_first_confidence_feedback_can_raise_the_initial_budget():
    budget = ConfidenceDraftBudget(7)
    assert budget.choose(["request"], 32000, False) == 3
    budget.scheduled(1, 0.0, 32000, False, {"request": 3})
    budget.complete(1, 0.03, {"request": 4})
    budget.observe_confidences("request", 1, [0.99] * 4 + [float("nan")] * 3)
    assert budget.choose(["request"], 32000, False) > 3


def test_single_request_confidence_budget_is_stable_under_timing_noise():
    """Routing latency must not change choices for the same acceptance feedback."""
    choices = []
    for noisy in (False, True):
        budget = ConfidenceDraftBudget(7)
        now, selected = 0.0, []
        for step in range(80):
            k = budget.choose(["request"], 32000, False)
            selected.append(k)
            accepted = 7 if step < 30 or step >= 60 else 0
            budget.scheduled(step, now, 32000, False, {"request": k})
            now += (0.02 + 0.002 * k) * (8 if noisy and step % 3 == 0 else 1)
            budget.complete(step, now, {"request": 1 + min(k, accepted)})
            budget.observe_confidences(
                "request", step, [0.99 if accepted else 0.05] * 7
            )
        choices.append(selected)
    assert choices[0] == choices[1]
    assert min(choices[0]) == 1
    assert max(choices[0]) == 7


def test_calibration_only_uses_observed_prefix_from_matching_step():
    budget = ConfidenceDraftBudget(5)
    budget.choose(["request"], 32000, False)
    budget.scheduled(1, 0.0, 32000, False, {"request": 1})
    budget.complete(1, 0.02, {"request": 1})
    budget.observe_confidences("request", 1, [0.8] * 5)
    calibrated = list(budget.confidence_bias["request"])
    assert calibrated[0] < 0
    assert calibrated[1:] == [0.0] * 4
    budget.observe_confidences("request", 2, [0.9] * 5)
    assert budget.confidence_bias["request"] == calibrated
    budget.retain_requests(set())
    assert not budget.confidence_bias
    assert not budget.smoothed_confidences
    assert not budget.last_observations


@pytest.mark.parametrize("num_reqs,mixed", [(2, False), (1, True)])
def test_batched_or_mixed_budget_uses_measured_nonlinear_costs(num_reqs, mixed):
    budget = ConfidenceDraftBudget(5)
    req_ids = [str(index) for index in range(num_reqs)]
    now, selected = 0.0, []
    for step in range(120):
        k = budget.choose(req_ids, 32000, mixed)
        budget.scheduled(step, now, 32000, mixed, dict.fromkeys(req_ids, k))
        now += 0.0195 + 0.0005 * k if k <= 3 else 0.100
        budget.complete(step, now, dict.fromkeys(req_ids, k + 1))
        for req_id in req_ids:
            budget.observe_confidences(req_id, step, [0.99] * 5)
        selected.append(k)
    # High acceptance must not override the expensive five-token batch shape.
    assert selected[-64:].count(3) >= 54


@pytest.mark.parametrize("num_reqs,mixed", [(4, False), (1, True)])
def test_low_acceptance_batch_does_not_explore_every_long_budget(num_reqs, mixed):
    budget = ConfidenceDraftBudget(7)
    req_ids = [str(index) for index in range(num_reqs)]
    now, selected = 0.0, []
    for step in range(12):
        k = budget.choose(req_ids, 32000, mixed)
        budget.scheduled(step, now, 32000, mixed, dict.fromkeys(req_ids, k))
        now += 0.01 + 0.005 * k
        budget.complete(step, now, dict.fromkeys(req_ids, 1))
        selected.append(k)
    assert set(selected) == {1, 3}
    assert selected[-3:] == [1, 1, 1]


@pytest.mark.parametrize("budget_cls", [AdaptiveDraftBudget, ConfidenceDraftBudget])
def test_short_proposal_confidence_keeps_unobserved_suffix(budget_cls):
    budget = budget_cls(7)
    budget.choose(["request"], 32000, False)
    budget.observe_confidences("request", 1, [0.9] * 7)
    budget.observe_confidences("request", 2, [0.6, 0.5] + [float("nan")] * 5)
    assert budget.confidences["request"] == (2, [0.6, 0.5])
    if isinstance(budget, ConfidenceDraftBudget):
        assert budget.smoothed_confidences["request"][2:] == [0.9] * 5
    budget.observe_confidences("request", 3, [float("nan")] * 7)
    assert budget.confidences["request"][0] == 2


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager.cost_tables = (np.zeros(num_reqs + 1), verify_cost_ms)
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    return manager


def test_manager_scopes_varlen_check_without_weakening_runner_cg_mode(monkeypatch):
    class Backend:
        @classmethod
        def supports_device_cpu_query_lens_mismatch(cls):
            return True

    class Builder:
        def __init__(self, support):
            self.support = support

        def get_cudagraph_support(self, *_args):
            return self.support

    def group(layer_name, support):
        builder = Builder(support)
        return SimpleNamespace(
            layer_names=[layer_name],
            backend=Backend,
            kv_cache_spec=None,
            get_metadata_builder=lambda _index: builder,
        )

    groups = [
        [
            group("target", AttentionCGSupport.ALWAYS),
            group("draft", AttentionCGSupport.UNIFORM_BATCH),
        ]
    ]
    runner_support = AttentionCGSupportInfo(
        AttentionCGSupport.UNIFORM_BATCH, "DraftBackend"
    )
    created = object()
    monkeypatch.setattr(
        adaptive_module,
        "AdaptiveVerificationManager",
        lambda *_args, **_kwargs: created,
    )

    manager = maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=True,
        attn_groups=groups,
        attn_cg_support=runner_support,
        req_states=object(),
        query_start_loc=object(),
        num_bonus_tokens=1,
        max_total_logits=1,
        vllm_config=None,
        target_layer_names={"target"},
    )

    assert manager is created
    assert runner_support.min_cg_support == AttentionCGSupport.UNIFORM_BATCH


def test_budget_stops_where_marginal_drafts_stop_paying_for_themselves():
    # Verification is cheap up to two extra tokens, then jumps 100x; only the
    # highest-confidence draft is worth the cheap slot.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    manager.set_cost_curves = lambda draft, verify: curves.update(
        draft=draft, verify=verify
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_tokens"] // 8,
            # Only the captured sizes replay a graph; the tail sizes run eager.
            full_cudagraph=batch["num_tokens"] <= 1024,
        )
        for batch in manager.batches_to_profile([8, 1024])
    ]
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    assert curves["verify"] == [
        (8, 8.0),
        (1024, 1024.0),
        (1536, 1536.0),
        (2048, 2048.0),
        (4096, 4096.0),
    ]
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(1, 1.0), (128, 1.0)]


def test_compact_batch_preserves_totals_and_bounds():
    # The CPU placeholder layout must keep the batch total equal to the GPU
    # total and every verification row within decode_query_len, or downstream
    # CPU metadata desyncs from the reallocated GPU boundaries.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.array([1.0] * 44 + [100.0] * 3),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    cu_num_logits_np = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, _ = manager.compact_batch(drafts, scheduled, cu_num_logits_np)

    assert int(compacted.sum()) == num_tokens
    num_steps = manager.num_speculative_steps
    assert (compacted[:2] <= 1 + num_steps).all()
    assert compacted[2] == 40


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1


def test_zero_budget_rebuilds_cpu_cu_num_logits():
    # When one bonus row per request already overflows a verification chunk, the
    # budget clamps to zero but the batch still needs chunking. Every capacity is
    # zeroed on device, so the CPU can name that layout exactly -- and must, since
    # _iter_request_chunks slices the compacted logits with these offsets.
    #
    # The third request is a chunked prefill (no drafts, still mid-prompt). The
    # runner gives *every* request num_bonus_tokens logits rows regardless
    # (num_logits = num_draft_tokens_per_req + num_bonus_tokens), so the rebuilt
    # offsets stay uniform rather than skipping non-verification rows.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 0

    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    scheduled_cu_num_logits = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, cu_num_logits_np = manager.compact_batch(
        drafts, scheduled, scheduled_cu_num_logits
    )

    # One bonus row per request, matching cumsum(capacities + num_bonus_tokens)
    # with every capacity zeroed -- the prefill row included.
    expected = np.arange(4, dtype=np.int32) * manager.num_bonus_tokens
    assert np.array_equal(cu_num_logits_np, expected)
    assert cu_num_logits_np.dtype == scheduled_cu_num_logits.dtype
    # The prefill keeps its scheduled tokens; only drafts are dropped.
    assert np.array_equal(compacted, np.array([1, 1, 40], dtype=np.int32))


def test_zero_budget_keeps_one_grammar_row_per_scheduled_draft():
    # The scheduler sizes the grammar bitmask from the *scheduled* drafts
    # (len(drafts) + 1 rows per request), but a zero budget rewrites
    # cu_num_logits_np to bonus-only. Deriving the bitmask -> logits mapping
    # from those rewritten offsets drops rows and trips the
    # `num_masks == len(mapping)` assert in apply_grammar_bitmask.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    scheduled_spec_decode_tokens = {"low": [1, 2], "high": [3, 4]}
    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40}, scheduled_spec_decode_tokens
    )
    assert manager._batch_budget[2] == 0

    req_ids = ["low", "high", "prefill"]
    num_draft_tokens_per_req = np.array([2, 2, 0], dtype=np.int32)
    _, cu_num_logits_np = manager.compact_batch(
        num_draft_tokens_per_req,
        np.array([3, 3, 40], dtype=np.int32),
        np.array([0, 3, 6, 7], dtype=np.int32),
    )

    mask_stride = manager.num_speculative_steps + manager.num_bonus_tokens
    mapping = _build_grammar_mapping(
        req_ids,
        req_ids,
        cu_num_logits_np,
        num_draft_tokens_per_req,
        manager.num_bonus_tokens,
        mask_stride,
    )

    num_bitmask_rows = sum(
        len(scheduled_spec_decode_tokens.get(req_id, ())) + 1 for req_id in req_ids
    )
    assert len(mapping) == num_bitmask_rows
    # (request, position) keys, so the kernel can mask rows the compacted
    # device layout no longer has room for.
    assert mapping == [0, 1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("local_cost,expected", [(0.030, 4), (0.012, 3)])
def test_mtp_default_policy_measures_new_cohort_incumbent(local_cost, expected):
    """Other cohorts' K3 costs cannot promote a locally measured long budget."""
    budget = MTPDraftBudget(4, prefer_default=True)
    context, step, now = 8192, 0, 0.0

    def complete(req_ids, verified, proposal, elapsed):
        nonlocal step, now
        budget.scheduled(
            step,
            now,
            context,
            False,
            dict.fromkeys(req_ids, verified),
            proposal_drafts=proposal,
        )
        now += elapsed
        budget.complete(step, now, dict.fromkeys(req_ids, verified + 1))
        step += 1

    for _ in range(16):
        budget.choose(["previous"], context, False)
        complete(["previous"], 3, 3, 0.020)
    budget.retain_requests(set())
    cohort = ["a", "b", "c", "d"]
    for _ in range(32):
        budget.choose(cohort, context, False)
        complete(cohort, 3, 3, 0.020)

    assert budget.choose(["a"], context, False) == 3
    for _ in range(3):
        complete(["a"], 4, 4, 0.018)
        assert budget.choose(["a"], context, False) == 3

    complete(["a"], 3, 4, 0.500)
    assert budget.choose(["a"], context, False) == 3
    for _ in range(3):
        complete(["a"], 3, 3, local_cost)
    assert budget.choose(["a"], context, False) == expected


@pytest.mark.parametrize("prefer_default", [False, True])
def test_mtp_extension_debt_uses_local_incumbent_only_when_preferred(prefer_default):
    """A cheap cohort must not charge debt to another cohort's profitable suffix."""
    budget = MTPDraftBudget(4, prefer_default=prefer_default)
    now, step = 0.0, 0

    def complete(req_id, drafts, elapsed):
        nonlocal now, step
        budget.choose([req_id], 4096, False)
        budget.scheduled(
            step, now, 4096, False, {req_id: drafts}, proposal_drafts=drafts
        )
        now += elapsed
        budget.complete(step, now, {req_id: drafts + 1})
        step += 1

    for _ in range(4):
        complete("slow", 3, 0.030)
    for _ in range(12):
        complete("cheap", 3, 0.012)
    complete("slow", 4, 0.031)
    excess = budget.extension_stats(["slow"], 4096, False)["excess_ms_per_step"]
    if prefer_default:
        assert excess == 0.0
    else:
        assert excess > 0.0


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_default_policy_releases_long_budget_when_prefix_evidence_expires(
    feedback_delay,
):
    """An unscored long budget must not block the core's cheaper short proposal."""
    budget = MTPDraftBudget(4, prefer_default=True)
    rounds = 0

    def accepted(drafts):
        nonlocal rounds
        count = drafts if rounds < 96 else 0
        rounds += 1
        return count

    latencies = {1: 0.020, 2: 0.025, 3: 0.035, 4: 0.028}
    choices = []
    for proposed, stats, _ in _mtp_prefix_probe_steps(
        budget,
        feedback_delay,
        accepted,
        latencies=latencies,
        max_steps=192,
    ):
        choices.append(proposed)
    assert choices[64:96].count(4) >= 30
    assert stats["prefix_acceptance"][4] is None
    assert choices[-32:].count(1) >= 30


def _mtp_incumbent_cost_trace(budget, feedback_delay, drift, steps=1024):
    pending: deque[tuple[int, float, int]] = deque()
    previous, now, rows = 3, 0.0, []
    for step in range(steps):
        latency = {
            1: 0.029,
            2: 0.032,
            3: 0.024 if drift and step >= 256 else 0.035,
            4: 0.038,
            5: 0.060,
        }
        proposed = budget.choose(["request"], 8192, False)
        budget.scheduled(
            step, now, 8192, False, {"request": previous}, proposal_drafts=proposed
        )
        elapsed = (latency[previous] + latency[proposed]) / 2
        now += elapsed
        rows.append((proposed, previous + 1, elapsed))
        pending.append((step, now, previous + 1))
        if len(pending) > feedback_delay:
            completed, finished, sampled = pending.popleft()
            budget.complete(completed, finished, {"request": sampled})
        previous = proposed
    return rows


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_long_policy_remeasures_incumbent_when_routing_cost_drops(feedback_delay):
    """Unchanged acceptance must not retain K4 against a newly cheaper K3."""
    budget = MTPDraftBudget(4, prefer_default=True)
    rows = _mtp_incumbent_cost_trace(budget, feedback_delay, drift=True)
    assert sum(k == 4 for k, _, _ in rows[64:128]) >= 60
    observed = sum(n for _, n, _ in rows[-512:]) / sum(
        elapsed for _, _, elapsed in rows[-512:]
    )
    assert observed > 0.99 * (4 / 0.024)
    assert budget.extension_stats(["request"], 8192, False)["step_ms"][3] == 24.0


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_incumbent_refresh_keeps_profitable_long_budget_overhead_bounded(
    feedback_delay,
):
    """Periodic incumbent measurements should cost under one percent at steady K4."""
    budget = MTPDraftBudget(4, prefer_default=True)
    rows = _mtp_incumbent_cost_trace(budget, feedback_delay, drift=False)
    last = rows[-512:]
    assert sum(k == 3 for k, _, _ in last) <= 16
    observed = sum(n for _, n, _ in last) / sum(elapsed for _, _, elapsed in last)
    assert observed > 0.99 * (5 / 0.038)


def test_mtp_incumbent_refresh_waits_between_trials_when_feedback_is_delayed():
    """Missing PP completions must not turn each four-step refresh into another."""
    budget = MTPDraftBudget(4, prefer_default=True)
    rows = _mtp_incumbent_cost_trace(budget, 0, drift=False, steps=256)
    previous = rows[-1][0]
    now = sum(elapsed for _, _, elapsed in rows)
    choices = []
    for step in range(256, 856):
        proposed = budget.choose(["request"], 8192, False)
        choices.append(proposed)
        budget.scheduled(
            step, now, 8192, False, {"request": previous}, proposal_drafts=proposed
        )
        now += 0.038
        previous = proposed
    assert choices.count(3) <= 20
    consecutive = longest = 0
    for proposed in choices:
        consecutive = consecutive + 1 if proposed == 3 else 0
        longest = max(longest, consecutive)
    assert longest == 4


def test_mtp_incumbent_refresh_preserves_an_active_longer_probe():
    """A deferred incumbent refresh must yield to an ongoing K5 trial."""
    budget = MTPDraftBudget(5, prefer_default=True)
    _mtp_incumbent_cost_trace(budget, 0, drift=False, steps=256)
    extension = budget._extension
    state = extension.state(["request"], budget.key(1, 8192, False))
    state.selected = 4
    state.probing, state.stable, state.attempts = 5, 0, 0
    state.refresh_remaining = 4
    state.decisions = max(state.incumbent_updated, state.incumbent_refreshed) + 127
    assert budget.choose(["request"], 8192, False) == 5
    assert budget.extension_stats(["request"], 8192, False)["trial_k"] == 5


def _mtp_previous_confidence_trace(
    feedback_delay,
    *,
    use_confidence=True,
    feature_mode="correlated",
    prefer_default=True,
    noise=0.0,
    steps=1536,
    budget=None,
):
    """Two interleaved PP cohorts verify their previous producer's actual block."""
    if budget is None:
        budget = MTPDraftBudget(4, prefer_default=prefer_default, use_confidence=True)
    requests = ["cohort0", "cohort1"]
    producers = {
        r: (3, 0.8 if i == 0 else 0.05, 4 if i == 0 else 0)
        for i, r in enumerate(requests)
    }
    local: Counter[str] = Counter()
    pending: deque[tuple[int, float, str, int, float]] = deque()
    costs = {1: 0.021, 2: 0.028, 3: 0.035, 4: 0.038}
    now = tokens = fixed_tokens = 0.0
    choices = []
    feature_random = random.Random(77)
    cost_random = random.Random(19)

    def complete(entry):
        step, finished, r, count, feature = entry
        budget.complete(step, finished, {r: count})
        if use_confidence:
            budget.observe_confidences(r, step, [feature, *[float("nan")] * 3])

    for step in range(steps):
        cohort = step % 2
        r = requests[cohort]
        high = ((local[r] + cohort * 16) // 16) % 2 == 0
        truth = 4 if high else 0
        feature = 0.8 if high else 0.05
        if feature_mode == "random":
            feature = 0.8 if feature_random.random() < 0.5 else 0.05
        elif feature_mode == "nan":
            feature = float("nan")
        elif feature_mode == "constant":
            feature = 0.3
        proposed = budget.choose([r], 8192, False)
        verified, old_feature, old_truth = producers[r]
        budget.scheduled(
            step, now, 8192, False, {r: verified}, proposal_drafts=proposed
        )
        factor = 1 + cost_random.uniform(-noise, noise)
        now += 0.5 * (costs[verified] + costs[proposed]) * factor
        tokens += 1 + min(verified, old_truth)
        fixed_tokens += 1 + min(3, old_truth)
        pending.append((step, now, r, 1 + min(verified, old_truth), old_feature))
        if len(pending) > feedback_delay:
            complete(pending.popleft())
        producers[r] = (proposed, feature, truth)
        local[r] += 1
        choices.append(proposed)
    while pending:
        complete(pending.popleft())
    # The cost stream is independent of budget decisions, so this is the same
    # token acceptance and timing counterfactual for a uniform fixed K=3.
    cost_random = random.Random(19)
    fixed_time = sum(
        costs[3] * (1 + cost_random.uniform(-noise, noise)) for _ in range(steps)
    )
    return (
        budget,
        producers,
        {
            "rate": tokens / now,
            "fixed_rate": fixed_tokens / fixed_time,
            "choices": choices,
        },
    )


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_confidence_is_opt_in_without_changing_empirical_policy(feedback_delay):
    """Confidence packets cannot train or alter a default empirical controller."""
    default = MTPDraftBudget(4, prefer_default=True)
    _, _, measured = _mtp_previous_confidence_trace(feedback_delay, budget=default)
    empirical = MTPDraftBudget(4, prefer_default=True)
    _, _, expected = _mtp_previous_confidence_trace(
        feedback_delay, use_confidence=False, budget=empirical
    )
    assert measured == expected
    for r in ("cohort0", "cohort1"):
        stats = default.extension_stats([r], 8192, False)
        assert stats["confidence_forecast"] == {}
        assert stats == empirical.extension_stats([r], 8192, False)


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_previous_confidence_forecast_improves_changing_prefix_latency(
    feedback_delay,
):
    """A calibrated past-block feature saves time when acceptance changes."""
    budget, _, measured = _mtp_previous_confidence_trace(feedback_delay, noise=0.15)
    _, _, empirical = _mtp_previous_confidence_trace(
        feedback_delay, use_confidence=False, noise=0.15
    )
    assert measured["rate"] > empirical["rate"] * 1.015
    assert measured["rate"] > measured["fixed_rate"] * 1.05
    for i, r in enumerate(["cohort0", "cohort1"]):
        stats = budget.extension_stats([r], 8192, False)["confidence_forecast"]
        assert stats["training_pairs"] > 500
        assert stats["weighted_decisions"] > 100
        feature, producer, consumer = stats["last_pair_steps"][r]
        assert feature < producer < consumer
        assert feature % 2 == producer % 2 == consumer % 2 == i


@pytest.mark.parametrize("feedback_delay", [0, 2])
@pytest.mark.parametrize("feature_mode", ["random", "nan", "constant"])
def test_mtp_previous_confidence_falls_back_without_predictive_evidence(
    feedback_delay, feature_mode
):
    """NaNs, unchanging scores and unrelated scores retain the empirical policy."""
    _, _, measured = _mtp_previous_confidence_trace(
        feedback_delay, feature_mode=feature_mode
    )
    _, _, empirical = _mtp_previous_confidence_trace(
        feedback_delay, feature_mode=feature_mode, use_confidence=False
    )
    assert measured == empirical


@pytest.mark.parametrize("feedback_delay", [0, 2])
def test_mtp_previous_confidence_does_not_change_nonpreferred_policy(feedback_delay):
    _, _, measured = _mtp_previous_confidence_trace(
        feedback_delay, prefer_default=False
    )
    _, _, empirical = _mtp_previous_confidence_trace(
        feedback_delay, prefer_default=False, use_confidence=False
    )
    assert measured == empirical


@pytest.mark.parametrize("cancel", ["preempt", "missing_output"])
def test_mtp_previous_confidence_discards_cancelled_producer_and_finished_request(
    cancel,
):
    """Stale producer metadata cannot train a resumed or reused request ID."""
    budget, producers, _ = _mtp_previous_confidence_trace(2, steps=384)
    r = "cohort0"
    before = budget.extension_stats([r], 8192, False)["confidence_forecast"]
    verified, _, _ = producers[r]
    proposed = budget.choose([r], 8192, False)
    budget.scheduled(1000, 10.0, 8192, False, {r: verified}, proposal_drafts=proposed)
    if cancel == "preempt":
        budget.invalidate_confidence(r)
        # Even an unexpected late successful output carries an older epoch.
        budget.complete(1000, 10.035, {r: verified + 1})
    else:
        budget.complete(1000, 10.035, {})
    budget.observe_confidences(r, 1000, [0.8, *[float("nan")] * 3])
    budget.choose([r], 8192, False)
    budget.scheduled(1002, 10.1, 8192, False, {r: 3}, proposal_drafts=3)
    budget.complete(1002, 10.135, {r: 4})
    budget.observe_confidences(r, 1002, [0.8, *[float("nan")] * 3])
    after = budget.extension_stats([r], 8192, False)["confidence_forecast"]
    assert after["training_pairs"] == before["training_pairs"]
    assert after["ready_requests"] == 0
    budget.retain_requests([])
    assert budget.choose([r], 8192, False) == 3
    fresh = budget.extension_stats([r], 8192, False)["confidence_forecast"]
    assert fresh["training_pairs"] == fresh["weighted_decisions"] == 0
    assert fresh["conditional_samples"] == {}


def test_mtp_previous_confidence_labels_only_prefix_and_first_rejection():
    """A rank-zero feature does not label unverified future suffix positions."""
    budget = MTPDraftBudget(4, prefer_default=True, use_confidence=True)
    for step in range(20):
        budget.choose(["request"], 8192, False)
        budget.scheduled(
            step, step * 0.1, 8192, False, {"request": 3}, proposal_drafts=3
        )
        budget.complete(step, step * 0.1 + 0.035, {"request": 2})
        budget.observe_confidences(
            "request", step, [0.8 if step % 2 else 0.05, *[float("nan")] * 3]
        )
    stats = budget.extension_stats(["request"], 8192, False)["confidence_forecast"]
    assert stats["training_pairs"] == 18
    assert stats["conditional_samples"]["request"] == [18, 18, 0, 0]
