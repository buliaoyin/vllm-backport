# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A fixed-capacity GPU expert cache with stable addresses across graph replays."""

import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import TypedDict

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types

from .cache_policy import TailCachePolicy


class DynamicCacheStats(TypedDict):
    host_bytes: int
    prepare_seconds: float
    observations: int
    decode_observations: int
    updates: int
    swaps: int
    repacked_experts: int
    update_seconds: float
    observe_seconds: float
    estimated_saved_ms: float
    last_admitted: list[int]
    last_evicted: list[int]
    last_group_counts: list[int]
    lru_host_bytes: int
    lru_hits: int
    lru_fills: int
    feedback_requests: int
    feedback_steps: int
    feedback_tokens: int
    feedback_routes: list[int]
    feedback_calls: list[int]


class GPUExpertCache:
    def __init__(self, capacity, num_experts, device, limit, top_k, weight_source=None):
        self.capacity = capacity
        self.num_experts = num_experts
        self.device = torch.device("cuda", device)
        self.origin = torch.device("cuda", torch.accelerator.current_device_index())
        self.stream = torch.cuda.Stream(device=self.device)
        self.weights = {}
        self.weight_shape: tuple[int, int] | None = None
        self.loaded: set[tuple[int, int]] = set()
        self.weight_source = weight_source
        self.packed: tuple[torch.Tensor, ...] | None = None
        self.selected = []
        self.enabled = True
        self.calibrate_from_prompt = True
        self.calibrations = 0
        self.reloaded_experts = 0
        self.reload_updates = 0
        self.reload_seconds = 0.0
        self.dynamic_policy: TailCachePolicy | None = None
        self.dynamic_enabled = False
        self.dynamic_pinned: set[int] = set()
        self.host_packed: dict[int, tuple[torch.Tensor, ...]] = {}
        self.host_lru: tuple[torch.Tensor, ...] | None = None
        self.host_lru_slots: OrderedDict[int, int] = OrderedDict()
        self.decode_feedback = None
        self.feedback_enabled = False
        self.feedback_history = None
        self.feedback_mass = 0.0
        self.feedback_requests = 0
        self._reset_decode_accounting()
        self.dynamic_stats: DynamicCacheStats = {
            "host_bytes": 0,
            "prepare_seconds": 0.0,
            "observations": 0,
            "decode_observations": 0,
            "updates": 0,
            "swaps": 0,
            "repacked_experts": 0,
            "update_seconds": 0.0,
            "observe_seconds": 0.0,
            "estimated_saved_ms": 0.0,
            "last_admitted": [],
            "last_evicted": [],
            "last_group_counts": [],
            "lru_host_bytes": 0,
            "lru_hits": 0,
            "lru_fills": 0,
            "feedback_requests": 0,
            "feedback_steps": 0,
            "feedback_tokens": 0,
            "feedback_routes": [],
            "feedback_calls": [],
        }
        self.limit = limit
        self.top_k = top_k
        self.device_io: tuple[torch.Tensor, ...] | None = None
        self.prefill_io: tuple[torch.Tensor, ...] | None = None
        self.memory_pool: torch.cuda.MemPool | None = None
        self.expert_map = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=self.device
        )
        self.membership = torch.zeros(num_experts, dtype=torch.bool, device=self.origin)
        with torch.cuda.stream(self.stream):
            self.workspace = marlin_make_workspace_new(self.device, 4)
            if self.device != self.origin:
                self.memory_pool = torch.cuda.MemPool()

    def load_weight(self, expert, projection, weight, scale):
        self.loaded.add((expert, projection))
        if projection == 0:
            self.weight_shape = tuple(weight.shape)
        if self.weight_source is None:
            self.weights[expert, projection] = (
                weight.view(torch.uint8),
                scale.view(torch.uint8),
            )

    def _pack_expert(self, expert):
        if expert in self.host_packed:
            return self.host_packed[expert]
        if self.feedback_enabled and expert in self.host_lru_slots:
            slot = self.host_lru_slots.pop(expert)
            self.host_lru_slots[expert] = slot
            self.dynamic_stats["lru_hits"] += 1
            assert self.host_lru is not None
            return tuple(part[slot : slot + 1] for part in self.host_lru)
        self.dynamic_stats["repacked_experts"] += 1
        weights = {
            p: self.weight_source(expert, p)
            if self.weight_source is not None
            else self.weights[expert, p]
            for p in range(3)
        }
        raw = [
            torch.cat([weights[p][part] for p in projections], dim=0)
            .unsqueeze(0)
            .to(self.device)
            for projections, part in (
                ((0, 2), 0),
                ((1,), 0),
                ((0, 2), 1),
                ((1,), 1),
            )
        ]
        w13, w2, s13, s2 = raw
        packed = prepare_moe_mxfp4_layer_for_marlin(
            SimpleNamespace(params_dtype=torch.bfloat16),
            w13,
            w2,
            s13,
            s2,
            None,
            None,
            inplace=True,
        )[:4]
        if self.feedback_enabled and self.host_lru is not None:
            capacity = self.host_lru[0].shape[0]
            if len(self.host_lru_slots) == capacity:
                _, slot = self.host_lru_slots.popitem(last=False)
            else:
                slot = len(self.host_lru_slots)
            for destination, source in zip(self.host_lru, packed):
                destination[slot : slot + 1].copy_(source, non_blocking=True)
            self.host_lru_slots[expert] = slot
            self.dynamic_stats["lru_fills"] += 1
        return packed

    def set_feedback_enabled(self, enabled):
        self.feedback_enabled = enabled and self.decode_feedback is not None
        if self.decode_feedback is not None:
            self.decode_feedback.set_enabled(self.feedback_enabled)

    def learning_state(self):
        return {
            "history": None
            if self.feedback_history is None
            else self.feedback_history.tolist(),
            "requests": self.feedback_requests,
            "mass": self.feedback_mass,
            "decode_snapshot": self._decode_snapshot.tolist(),
            "decode_totals": [
                self._decode_hits,
                self._decode_routes,
                self._decode_steps,
            ],
            "last_decode_check": self._last_decode_check,
            "pending": None
            if self.decode_feedback is None
            else self.decode_feedback.counts.cpu().tolist(),
        }

    def restore_learning_state(self, state=None):
        state = state or {}
        self._reset_decode_accounting()
        if (snapshot := state.get("decode_snapshot")) is not None:
            self._decode_snapshot = np.asarray(snapshot, dtype=np.int64)
        self._decode_hits, self._decode_routes, self._decode_steps = state.get(
            "decode_totals", (0, 0, 0)
        )
        self._last_decode_check = state.get("last_decode_check", 0)
        history = state.get("history")
        self.feedback_history = None if history is None else np.array(history)
        self.feedback_mass = state.get("mass", 0.0)
        self.feedback_requests = state.get("requests", 0)
        self.dynamic_stats["feedback_requests"] = self.feedback_requests
        if self.decode_feedback is not None:
            pending = state.get("pending")
            if pending is None:
                self.decode_feedback.counts.zero_()
            else:
                self.decode_feedback.counts.copy_(torch.tensor(pending))

    def select(self, experts, *, preserve_slots=True):
        selected = set(experts)
        if len(experts) != self.capacity or len(selected) != self.capacity:
            raise ValueError("GPU expert cache selection must fill every slot once")
        if any(e < 0 or e >= self.num_experts for e in experts):
            raise ValueError("GPU expert cache selection contains an invalid expert")
        if (preserve_slots and selected == set(self.selected)) or list(
            experts
        ) == self.selected:
            return
        start = time.perf_counter()
        additions = iter(e for e in experts if e not in self.selected)
        slots = [e if e in selected else next(additions) for e in self.selected]
        if not slots or not preserve_slots:
            slots = list(experts)
        self.stream.synchronize()
        # Let StreamContext restore both devices' streams. An outer device guard
        # would leave vLLM's cached current stream pointing at the cache GPU.
        with torch.cuda.stream(self.stream):
            changes = [
                (slot, expert)
                for slot, expert in enumerate(slots)
                if not self.selected or self.selected[slot] != expert
            ]
            if self.feedback_enabled:
                changes.sort(key=lambda item: item[1] not in self.host_lru_slots)
            for slot, expert in changes:
                packed = self._pack_expert(expert)
                if self.packed is None:
                    self.packed = tuple(
                        torch.empty(
                            (self.capacity, *part.shape[1:]),
                            dtype=part.dtype,
                            device=self.device,
                        )
                        for part in packed
                    )
                for destination, source in zip(self.packed, packed):
                    destination[slot : slot + 1].copy_(source, non_blocking=True)
                del packed
                self.reloaded_experts += 1
        self.stream.synchronize()
        self.selected = slots
        self.set_enabled(self.enabled)
        self.reload_seconds += time.perf_counter() - start
        self.reload_updates += 1

    def set_enabled(self, enabled):
        self.enabled = enabled
        mapping = torch.full((self.num_experts,), -1, dtype=torch.int32, device="cpu")
        if enabled:
            mapping[self.selected] = torch.arange(
                len(self.selected), dtype=torch.int32, device="cpu"
            )
        with torch.cuda.stream(self.stream):
            self.expert_map.copy_(mapping)
            self.membership.copy_(mapping >= 0)
        self.stream.synchronize()

    def select_static(self, experts):
        self.select(experts)
        self.calibrate_from_prompt = False

    def prepare_dynamic(self, settings):
        policy = TailCachePolicy(**settings)
        if self.dynamic_policy is not None:
            if self.dynamic_policy != policy:
                raise ValueError("Cannot change a prepared dynamic cache policy")
            return
        mutable = set(policy.mutable_experts)
        if not mutable or not mutable <= set(self.selected):
            raise ValueError("Dynamic slots must belong to the initial cache")
        assert self.packed is not None
        start = time.perf_counter()
        self.dynamic_pinned = set(self.selected) - mutable
        expert_bytes = sum(
            part[0].numel() * part.element_size() for part in self.packed
        )
        lru_capacity = policy.host_lru_experts if policy.feedback_weight else 0
        prepack_capacity = self.num_experts
        if policy.host_cache_bytes is not None:
            host_capacity = policy.host_cache_bytes // expert_bytes
            lru_capacity = min(lru_capacity, host_capacity)
            prepack_capacity = host_capacity - lru_capacity
        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            for expert in range(self.num_experts):
                if len(self.host_packed) >= prepack_capacity:
                    break
                if expert in self.dynamic_pinned:
                    continue
                if expert in self.selected:
                    if not policy.prepack_resident:
                        continue
                    slot = self.selected.index(expert)
                    packed = tuple(part[slot : slot + 1] for part in self.packed)
                else:
                    packed = self._pack_expert(expert)
                host = tuple(
                    torch.empty_like(part, device="cpu", pin_memory=True)
                    for part in packed
                )
                for dest, source in zip(host, packed):
                    dest.copy_(source, non_blocking=True)
                self.stream.synchronize()
                self.host_packed[expert] = host
                self.dynamic_stats["host_bytes"] += sum(
                    part.numel() * part.element_size() for part in host
                )
        if policy.feedback_weight:
            from .cache_feedback import DecodeRouteFeedback

            self.decode_feedback = DecodeRouteFeedback(self.num_experts, self.origin)
            self.feedback_enabled = True
            if lru_capacity:
                from .host_memory import empty_registered

                self.host_lru = tuple(
                    empty_registered((lru_capacity, *part.shape[1:]), part.dtype)[0]
                    for part in self.packed
                )
                self.dynamic_stats["lru_host_bytes"] = sum(
                    part.numel() * part.element_size() for part in self.host_lru
                )
        self.dynamic_stats["prepare_seconds"] = time.perf_counter() - start
        self.dynamic_policy = policy
        self.dynamic_enabled = True
        self.calibrate_from_prompt = False

    def _reset_decode_accounting(self):
        self._decode_snapshot = np.zeros(2 * self.num_experts + 2, dtype=np.int64)
        self._decode_hits = self._decode_routes = self._decode_steps = 0
        self._last_decode_check = 0

    def _track_decode_snapshot(self, snapshot, *, consumed=False):
        delta = snapshot - self._decode_snapshot
        self._decode_hits += int(delta[: self.num_experts][self.selected].sum())
        self._decode_routes += int(delta[: self.num_experts].sum())
        self._decode_steps += int(delta[-2])
        self._decode_snapshot = np.zeros_like(snapshot) if consumed else snapshot
        return delta

    def _transfer_costs(self, policy):
        return np.array(
            [
                policy.transfer_ms
                if e in self.host_packed or e in self.host_lru_slots
                else policy.repack_transfer_ms
                for e in range(self.num_experts)
            ]
        )

    def refresh_decode(self, step, *, remaining_steps=None):
        """Update one layer outside graph replay after its routes have completed."""
        policy = self.dynamic_policy
        if (
            not self.dynamic_enabled
            or not self.enabled
            or not self.feedback_enabled
            or self.decode_feedback is None
            or policy is None
            or step - self._last_decode_check < policy.decode_interval
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Decode expert admission must run outside graph capture")
        self._last_decode_check = step
        start = time.perf_counter()
        delta = self._track_decode_snapshot(self.decode_feedback.snapshot())
        steps = int(delta[-2])
        if not steps:
            return
        self.dynamic_stats["decode_observations"] += 1
        selected, saving, counts = policy.plan_decode(
            delta[self.num_experts : 2 * self.num_experts],
            steps,
            self.selected,
            self.dynamic_pinned,
            costs=self._transfer_costs(policy),
            remaining_steps=remaining_steps,
        )
        self._apply_admission(selected, saving, counts, start)

    def finish_request(self, outputs_per_step=None):
        if self.decode_feedback is None:
            return None
        self.decode_feedback.set_enabled(False)
        snapshot = self.decode_feedback.consume()
        self._track_decode_snapshot(snapshot, consumed=True)
        steps = int(snapshot[-2])
        calls = snapshot[self.num_experts : 2 * self.num_experts]
        policy = self.dynamic_policy
        if steps and policy is not None:
            self.feedback_history, self.feedback_mass = policy.update_history(
                self.feedback_history, self.feedback_mass, calls, steps
            )
            self.feedback_requests += 1
            self.dynamic_stats["feedback_requests"] = self.feedback_requests
            if outputs_per_step is not None:
                self.dynamic_policy = replace(
                    policy,
                    expected_tokens_per_step=max(1.0, outputs_per_step),
                )
        return self._decode_hits, self._decode_routes, self._decode_steps

    def begin_request(self, max_tokens):
        self._reset_decode_accounting()
        if self.decode_feedback is not None:
            self.decode_feedback.set_enabled(False)
            self.decode_feedback.counts.zero_()
        if self.dynamic_policy is not None and max_tokens is not None:
            self.dynamic_policy = replace(
                self.dynamic_policy, future_tokens=max(1, max_tokens - 1)
            )

    def adapt(self, ids):
        if not self.dynamic_enabled or not self.enabled:
            return
        assert self.dynamic_policy is not None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dynamic expert admission requires eager execution")
        start = time.perf_counter()
        routes = ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        policy = self.dynamic_policy
        costs = None
        if self.feedback_enabled:
            assert self.decode_feedback is not None
            snapshot = self.decode_feedback.consume()
            self._track_decode_snapshot(snapshot, consumed=True)
            steps, tokens = snapshot[-2:]
            self.dynamic_stats["feedback_steps"] = int(steps)
            self.dynamic_stats["feedback_tokens"] = int(tokens)
            self.dynamic_stats["feedback_routes"] = snapshot[
                : self.num_experts
            ].tolist()
            calls = snapshot[self.num_experts : 2 * self.num_experts]
            self.dynamic_stats["feedback_calls"] = calls.tolist()
            if steps:
                self.feedback_history, self.feedback_mass = policy.update_history(
                    self.feedback_history, self.feedback_mass, calls, steps
                )
                self.feedback_requests += 1
                self.dynamic_stats["feedback_requests"] = self.feedback_requests
            self.decode_feedback.set_enabled(True)
            costs = self._transfer_costs(policy)
        selected, saving, counts = policy.plan(
            routes,
            self.selected,
            self.dynamic_pinned,
            self.num_experts,
            history=self.feedback_history,
            costs=costs,
            feedback=self.feedback_enabled,
        )
        self._apply_admission(selected, saving, counts, start)

    def _apply_admission(self, selected, saving, counts, start):
        self.dynamic_stats["observations"] += 1
        self.dynamic_stats["last_group_counts"] = counts
        incoming = sorted(set(selected) - set(self.selected))
        self.dynamic_stats["last_admitted"] = incoming
        self.dynamic_stats["last_evicted"] = sorted(set(self.selected) - set(selected))
        self.dynamic_stats["observe_seconds"] += time.perf_counter() - start
        if incoming:
            start = time.perf_counter()
            self.select(selected)
            self.dynamic_stats["update_seconds"] += time.perf_counter() - start
            self.dynamic_stats["updates"] += 1
            self.dynamic_stats["swaps"] += len(incoming)
            self.dynamic_stats["estimated_saved_ms"] += saving

    def _allocation_context(self):
        if self.memory_pool is None:
            return nullcontext()
        return torch.cuda.use_mem_pool(self.memory_pool, device=self.device)

    def _allocate_io(self, tokens=16):
        assert self.weight_shape is not None
        intermediate, packed_hidden = self.weight_shape
        hidden = packed_hidden * 2
        options = {"device": self.device, "dtype": torch.bfloat16}
        with torch.cuda.stream(self.stream), self._allocation_context():
            return (
                torch.empty((tokens, hidden), **options),
                torch.empty(
                    (tokens, self.top_k), device=self.device, dtype=torch.int32
                ),
                torch.empty(
                    (tokens, self.top_k), device=self.device, dtype=torch.float32
                ),
                torch.empty((tokens, hidden), **options),
                torch.empty(
                    (tokens * self.top_k * max(2 * intermediate, hidden),), **options
                ),
                torch.empty((tokens * self.top_k, intermediate), **options),
            )

    def finalize(self, decode_tokens=16):
        if len(self.loaded) != self.num_experts * 3:
            raise RuntimeError("GPU expert cache checkpoint weights are incomplete")
        if self.packed is None:
            self.select(list(range(self.capacity)))
        if self.device_io is None:
            self.device_io = self._allocate_io(decode_tokens)

    @torch.no_grad()
    def warmup(self):
        """Allocate remote scratch before pipeline sends can wait on this rank."""
        assert self.weight_shape is not None
        hidden = self.weight_shape[1] * 2
        x = torch.zeros(128, hidden, dtype=torch.bfloat16, device=self.origin)
        ids = torch.tensor(self.selected[: self.top_k], device=self.origin)
        ids = ids.expand(128, -1).contiguous()
        routes = torch.full((128, self.top_k), 1 / self.top_k, device=self.origin)
        output = self.launch(x, ids, routes)
        self.join(output)
        self.stream.synchronize()
        torch.cuda.current_stream(self.origin).synchronize()

    def calibrate(self, ids):
        if not self.calibrate_from_prompt:
            return
        valid = ids.flatten().long()
        valid = valid[valid >= 0]
        counts = torch.bincount(valid, minlength=self.num_experts)
        experts = counts.argsort(descending=True, stable=True)[: self.capacity]
        self.select(experts.tolist())
        self.calibrations += 1

    def launch(self, hidden, ids, routes):
        assert self.packed is not None and self.device_io is not None
        tokens = hidden.shape[0]
        if tokens > 128:
            raise ValueError("GPU expert cache supports at most 128 tokens")
        io = self.device_io
        if tokens > self.device_io[0].shape[0]:
            if self.prefill_io is None:
                # Decode graphs retain the original buffers' addresses.
                self.prefill_io = self._allocate_io(128)
            io = self.prefill_io
        x, selected, routing, output = (part[:tokens] for part in io[:4])
        result = torch.empty_like(hidden) if self.device != self.origin else output
        origin_stream = torch.cuda.current_stream(hidden.device)
        self.stream.wait_stream(origin_stream)
        # Keep remote allocations alive independently of the origin GPU's graph
        # pool. Later draft captures must not recycle their captured addresses.
        with torch.cuda.stream(self.stream), self._allocation_context():
            x.copy_(hidden, non_blocking=True)
            selected.copy_(ids.int(), non_blocking=True)
            routing.copy_(routes, non_blocking=True)
            w13, w2, s13, s2 = self.packed
            fused_marlin_moe(
                x,
                w13,
                w2,
                None,
                None,
                s13,
                s2,
                routing,
                selected,
                scalar_types.float4_e2m1f.id,
                global_num_experts=self.num_experts,
                expert_map=self.expert_map,
                workspace=self.workspace,
                intermediate_cache13=io[4],
                intermediate_cache2=io[5],
                output=output,
                activation_config=ApplyMoEActivationConfig(clamp_limit=self.limit),
            )
        return result

    def join(self, output):
        assert self.device_io is not None
        io = (
            self.device_io
            if output.shape[0] <= self.device_io[0].shape[0]
            else self.prefill_io
        )
        assert io is not None
        if self.device != output.device:
            # Enqueue this only after the CPU branch. PyTorch's peer copy waits
            # on both GPUs' streams, so an early copy would serialize the branches.
            with torch.cuda.stream(self.stream):
                output.copy_(io[3][: output.shape[0]], non_blocking=True)
        stream = torch.cuda.current_stream(output.device)
        stream.wait_stream(self.stream)
        output.record_stream(stream)
