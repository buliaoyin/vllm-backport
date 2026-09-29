# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.spec_decode.dynamic.adaptive import (
    AdaptiveDraftBudget,
    ConfidenceDraftBudget,
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
