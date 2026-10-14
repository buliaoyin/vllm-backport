# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share stable Marlin slots between layers on the same GPU and PP worker."""

import heapq
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CapacityPolicy:
    interval: int = 128
    max_swaps: int = 16
    min_experts: int = 8
    decay: float = 0.5
    resize: bool = True
    capacity_only: bool = True
    history: bool = True
    online: bool = True
    horizon: int | None = None
    safety_factor: float | None = None

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (self.interval, self.max_swaps, self.min_experts)
            )
            or not math.isfinite(self.decay)
            or not 0 <= self.decay < 1
            or type(self.resize) is not bool
            or type(self.capacity_only) is not bool
            or type(self.history) is not bool
            or type(self.online) is not bool
            or (
                self.horizon is not None
                and (type(self.horizon) is not int or self.horizon < 1)
            )
            or (
                self.safety_factor is not None
                and (
                    isinstance(self.safety_factor, bool)
                    or not math.isfinite(self.safety_factor)
                    or self.safety_factor <= 0
                )
            )
        ):
            raise ValueError("Invalid adaptive expert capacity policy")

    def plan(
        self, caches, scores, horizon, protected, *, cpu_costs=None, diagnostics=None
    ):
        """Trade the least valuable resident slot for a more valuable expert."""
        if horizon <= 0:
            return []
        cpu_costs = cpu_costs or {
            cache: cache.dynamic_policy.cpu_call_ms for cache in caches
        }

        def record(reason, count=1):
            if diagnostics is not None:
                diagnostics[reason] = diagnostics.get(reason, 0) + int(count)

        selections = {cache: set(cache.selected) for cache in caches}
        transfers = {
            cache: cache._transfer_costs(cache.dynamic_policy) for cache in caches
        }
        margins = {
            cache: (
                cache.dynamic_policy.safety_factor
                if self.safety_factor is None
                else self.safety_factor
            )
            for cache in caches
        }
        incoming: list[tuple[float, int, int]] = []
        for i, cache in enumerate(caches):
            benefit = np.asarray(scores[cache]) * cpu_costs[cache]
            eligible = benefit > 0
            eligible[cache.selected] = False
            record("candidates", np.count_nonzero(eligible))
            affordable = benefit * horizon > margins[cache] * transfers[cache]
            record("gross_payback", np.count_nonzero(eligible & ~affordable))
            incoming.extend(
                (float(benefit[e]), i, int(e))
                for e in np.flatnonzero(eligible & affordable)
            )
        if not incoming:
            return []
        incoming.sort(reverse=True)
        victims = [
            sorted(
                (float(scores[cache][expert] * cpu_costs[cache]), expert)
                for expert in cache.selected
                if expert not in protected[cache]
            )
            for cache in caches
        ]
        positions = [0] * len(caches)
        heap: list[tuple[float, int, int]] = []

        def head(i):
            if positions[i] == len(victims[i]):
                return None
            loss, expert = victims[i][positions[i]]
            return loss, i, expert

        def offer(i):
            candidate = head(i)
            if candidate is not None and len(selections[caches[i]]) > self.min_experts:
                heapq.heappush(heap, candidate)

        for i in range(len(caches)):
            offer(i)
        planned: list[tuple] = []
        admitted = dict.fromkeys(caches, 0)
        spent = dict.fromkeys(caches, 0.0)
        for index, (benefit, i, expert) in enumerate(incoming):
            target = caches[i]
            if len(planned) >= self.max_swaps:
                record("pool_limit", len(incoming) - index)
                break
            cost = transfers[target][expert]
            if admitted[target] >= target.dynamic_policy.decode_max_swaps:
                record("layer_limit")
                continue
            if spent[target] + cost > target.dynamic_policy.decode_budget_ms:
                record("transfer_budget")
                continue
            while heap and (
                heap[0] != head(heap[0][1])
                or len(selections[caches[heap[0][1]]]) <= self.min_experts
            ):
                heapq.heappop(heap)
            choice = heap[0] if heap and self.resize else None
            local = head(i)
            if local is not None and (choice is None or local[0] <= choice[0]):
                choice = local
            if choice is None:
                record("no_victim")
            else:
                loss, j, outgoing = choice
                donor = caches[j]
                if self.capacity_only and donor is target:
                    record("local_replacement")
                    continue
                saving = float((benefit - loss) * horizon)
                cost = transfers[target][expert]
                if saving <= margins[target] * cost:
                    record("net_payback")
                    continue
                selections[donor].remove(outgoing)
                was_at_floor = len(selections[target]) == self.min_experts
                selections[target].add(expert)
                planned.append((donor, outgoing, target, expert, saving))
                record("accepted")
                admitted[target] += 1
                spent[target] += cost
                positions[j] += 1
                offer(j)
                if donor is not target and was_at_floor:
                    offer(i)
        return planned


def capacity_settings(value):
    if value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("adaptive_cache must be a boolean or policy object")
    try:
        return asdict(CapacityPolicy(**value))
    except TypeError as error:
        raise ValueError("Invalid adaptive expert capacity policy") from error


class SharedExpertPool:
    def __init__(self, capacity, policy):
        self.capacity = capacity
        self.policy = policy
        self.caches = []
        self.initial = {}
        self.owners = {}
        self.packed = None
        self.enabled = True
        self.reset_stats()

    def reset_stats(self):
        self.snapshots = {}
        self.scores = {}
        self.last_step = 0
        self.cpu_snapshots = {}
        self.cpu_costs = {}
        self.stats: dict[str, Any] = dict(
            checks=0,
            swaps=0,
            cross_layer_swaps=0,
            seconds=0.0,
            history_swaps=0,
            decode_swaps=0,
            snapshot_seconds=0.0,
            planning_seconds=0.0,
            transfer_seconds=0.0,
            history_decisions={},
            decode_decisions={},
        )

    def register(self, cache, layer, experts):
        offset = sum(len(value[1]) for value in self.initial.values())
        slots = list(range(offset, offset + len(experts)))
        if (
            not experts
            or len(experts) < self.policy.min_experts
            or slots[-1] >= self.capacity
        ):
            raise ValueError("Shared expert pool cannot fit its initial selections")
        cache.capacity_pool = self
        cache.layer_idx = layer
        cache.slot_ids = slots
        cache.sort_map = torch.full_like(cache.expert_map, -1)
        self.initial[cache] = (slots.copy(), list(experts))
        self.caches.append(cache)

    def select(self, cache, experts, preserve_slots=True):
        if len(experts) != cache.capacity or len(set(experts)) != cache.capacity:
            raise ValueError("GPU expert cache selection must fill every slot once")
        if any(not 0 <= expert < cache.num_experts for expert in experts):
            raise ValueError("GPU expert cache selection contains an invalid expert")
        wanted = set(experts)
        if preserve_slots and wanted == set(cache.selected):
            return
        additions = iter(expert for expert in experts if expert not in cache.selected)
        selected = [
            expert if expert in wanted else next(additions) for expert in cache.selected
        ]
        if not selected or not preserve_slots:
            selected = list(experts)
        started = time.perf_counter()
        self._wait(cache)
        with torch.cuda.stream(cache.stream):
            for slot, expert in zip(cache.slot_ids, selected):
                if self.owners.get(slot) == (cache, expert):
                    continue
                self._write(cache, slot, expert)
        cache.stream.synchronize()
        cache.packed = self.packed
        cache.selected = selected
        cache.set_enabled(cache.enabled)
        cache.reload_seconds += time.perf_counter() - started
        cache.reload_updates += 1

    @staticmethod
    def _wait(cache):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Shared expert updates must run between graph replays")
        cache.stream.wait_event(torch.cuda.current_stream(cache.origin).record_event())
        cache.stream.synchronize()

    def _write(self, cache, slot, expert):
        packed = cache._pack_expert(expert)
        if self.packed is None:
            self.packed = tuple(
                torch.empty(
                    (self.capacity, *part.shape[1:]),
                    dtype=part.dtype,
                    device=cache.device,
                )
                for part in packed
            )
        for destination, source in zip(self.packed, packed):
            if (
                destination.shape[1:] != source.shape[1:]
                or destination.dtype != source.dtype
            ):
                raise ValueError("Shared experts must use identical Marlin layouts")
            destination[slot : slot + 1].copy_(source, non_blocking=True)
        self.owners[slot] = (cache, expert)
        cache.reloaded_experts += 1

    def move(self, donor, outgoing, target, incoming, saving=0.0, *, publish=True):
        """Keep storage fixed and publish both maps after the weight copy."""
        if incoming in target.selected or outgoing not in donor.selected:
            raise ValueError("Invalid shared expert slot transfer")
        if publish:
            self._wait(donor)
            self._wait(target)
            for cache in {donor, target}:
                if cache.decode_feedback is not None:
                    snapshot = cache.decode_feedback.snapshot()
                    cache._track_decode_snapshot(snapshot)
        index = donor.selected.index(outgoing)
        slot = donor.slot_ids[index]
        started = time.perf_counter()
        with torch.cuda.stream(target.stream):
            self._write(target, slot, incoming)
        target.stream.synchronize()
        elapsed_ms = 1000 * (time.perf_counter() - started)
        if donor is target:
            target.selected[index] = incoming
        else:
            donor.slot_ids.pop(index)
            donor.selected.pop(index)
            donor.capacity -= 1
            target.slot_ids.append(slot)
            target.selected.append(incoming)
            target.capacity += 1
            if publish:
                donor.set_enabled(donor.enabled)
        if publish:
            target.set_enabled(target.enabled)
        target.dynamic_stats["swaps"] += 1
        target.dynamic_stats["updates"] += 1
        target.dynamic_stats["update_seconds"] += elapsed_ms / 1000
        target.reload_seconds += elapsed_ms / 1000
        target.reload_updates += 1
        target.dynamic_stats["estimated_saved_ms"] += float(saving)
        self.stats["swaps"] += 1
        self.stats["cross_layer_swaps"] += int(donor is not target)

    def refresh(self, step, remaining_steps, *, history=False):
        if not self.enabled or not (
            self.policy.history if history else self.policy.online
        ):
            return
        if not history and step - self.last_step < self.policy.interval:
            return
        caches = [c for c in self.caches if c.dynamic_enabled and c.enabled]
        if not caches:
            return
        started = time.perf_counter()
        if history:
            scores = {
                c: c.feedback_history for c in caches if c.feedback_history is not None
            }
        else:
            self.last_step = step
            for cache in caches:
                snapshot = cache.decode_feedback.snapshot()
                previous = self.snapshots.get(cache, np.zeros_like(snapshot))
                delta = snapshot - previous
                if delta[-2] < 0:
                    delta = snapshot
                self.snapshots[cache] = snapshot
                if delta[-2] > 0:
                    recent = (
                        delta[cache.num_experts : 2 * cache.num_experts] / delta[-2]
                    )
                    old = self.scores.get(cache, recent)
                    self.scores[cache] = (
                        self.policy.decay * old + (1 - self.policy.decay) * recent
                    )
            scores = self.scores
        if len(scores) != len(caches):
            return
        for cache in caches:
            self._observe_cpu_cost(cache)
        protected = {cache: set(cache.dynamic_pinned) for cache in caches}
        horizon = self.policy.horizon or min(
            c.dynamic_policy.decode_horizon_steps for c in caches
        )
        if remaining_steps is not None:
            horizon = min(horizon, max(0, remaining_steps))
        if history:
            horizon *= min(c.dynamic_policy.feedback_horizon_requests for c in caches)
        cpu_costs = {
            cache: self.cpu_costs.get(cache, cache.dynamic_policy.cpu_call_ms)
            for cache in caches
        }
        observed = time.perf_counter()
        phase = "history" if history else "decode"
        plan = self.policy.plan(
            caches,
            scores,
            horizon,
            protected,
            cpu_costs=cpu_costs,
            diagnostics=self.stats[f"{phase}_decisions"],
        )
        planned = time.perf_counter()
        swaps_before = self.stats["swaps"]
        touched: set = set()
        for donor, outgoing, target, incoming, saving in plan:
            if donor is not target and donor.capacity <= self.policy.min_experts:
                continue
            for cache in {donor, target} - touched:
                self._wait(cache)
                snapshot = (
                    cache.decode_feedback.snapshot()
                    if history
                    else self.snapshots[cache]
                )
                cache._track_decode_snapshot(snapshot)
                touched.add(cache)
            self.move(donor, outgoing, target, incoming, saving, publish=False)
        for cache in touched:
            cache.set_enabled(cache.enabled)
        finished = time.perf_counter()
        self.stats["checks"] += 1
        self.stats[f"{phase}_swaps"] += self.stats["swaps"] - swaps_before
        self.stats["snapshot_seconds"] += observed - started
        self.stats["planning_seconds"] += planned - observed
        self.stats["transfer_seconds"] += finished - planned
        self.stats["seconds"] += finished - started

    def _observe_cpu_cost(self, cache):
        reader = getattr(cache, "cpu_stats", None)
        if reader is None:
            return
        stats = reader()
        if stats is None:
            return
        current = (float(stats["native_seconds"]), int(sum(stats["expert_counts"])))
        previous = self.cpu_snapshots.get(cache, current)
        seconds, misses = current[0] - previous[0], current[1] - previous[1]
        if misses >= 32 and seconds > 0:
            sample = 1000 * seconds / misses
            old = self.cpu_costs.get(cache, cache.dynamic_policy.cpu_call_ms)
            self.cpu_costs[cache] = 0.8 * old + 0.2 * sample
        self.cpu_snapshots[cache] = current

    def begin_request(self):
        self.snapshots = {}
        self.scores = {}
        self.last_step = 0
        for cache in self.caches:
            self._observe_cpu_cost(cache)

    def restore_initial(self):
        for cache, (slots, experts) in self.initial.items():
            cache.slot_ids = slots.copy()
            cache.capacity = len(slots)
            cache.selected = []
        for cache, (_, experts) in self.initial.items():
            self.select(cache, experts, preserve_slots=False)
        self.reset_stats()
