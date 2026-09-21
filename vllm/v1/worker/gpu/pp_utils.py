# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@triton.jit
def _scatter_draft_tokens_kernel(
    dst,
    src,
    indices,
    dst_stride: tl.constexpr,
    src_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(indices + row * idx_stride)
    if index < 0:
        return
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + row * src_stride + offsets, offsets < width, other=0)
    tl.store(dst + index * dst_stride + offsets, values, offsets < width)


def scatter_draft_tokens(
    dst: torch.Tensor, src: torch.Tensor, indices: torch.Tensor
) -> None:
    """Scatter request rows, skipping padding without a device-to-host sync."""
    rows, width = src.shape
    assert dst.shape[1] == width and indices.shape == (rows,)
    assert dst.stride(1) == src.stride(1) == 1
    if rows == 0 or width == 0:
        return
    _scatter_draft_tokens_kernel[(rows,)](
        dst,
        src,
        indices,
        dst.stride(0),
        src.stride(0),
        indices.stride(0),
        width,
        BLOCK=triton.next_power_of_2(width),
    )


class PPRecvBufferState(Enum):
    FREE = "free"
    RECEIVING = "receiving"
    MODEL = "model"
    DSPARK = "dspark"


class PPRecvBufferGuard:
    """Tracks enqueue-time ownership of a reused PP receive buffer."""

    def __init__(self) -> None:
        self._state = PPRecvBufferState.FREE
        self._generation = 0
        self._owner_stream: object | None = None

    @property
    def state(self) -> PPRecvBufferState:
        return self._state

    @property
    def is_stream_tracked(self) -> bool:
        return self._owner_stream is not None

    def begin_receive(self, stream: object | None = None) -> int:
        if self._owner_stream is not None:
            raise RuntimeError(
                "Cannot start a receive while the PP receive buffer still has "
                f"owner stream {self._owner_stream}."
            )
        self._transition(
            PPRecvBufferState.FREE,
            PPRecvBufferState.RECEIVING,
            "start a receive",
        )
        self._generation += 1
        self._owner_stream = stream
        return self._generation

    def finish_receive(self, generation: int, stream: object | None = None) -> None:
        if generation != self._generation:
            raise RuntimeError(
                "Stale PP receive completion for buffer generation "
                f"{generation}; current generation is {self._generation}."
            )
        self._assert_stream(stream, "finish a receive")
        self._transition(
            PPRecvBufferState.RECEIVING,
            PPRecvBufferState.MODEL,
            "finish a receive",
        )

    def begin_dspark(self, stream: object | None = None) -> None:
        self._assert_stream(stream, "start DSpark")
        self._transition(
            PPRecvBufferState.MODEL,
            PPRecvBufferState.DSPARK,
            "start DSpark",
        )

    def release(self, stream: object | None = None) -> None:
        self._assert_stream(stream, "release the buffer")
        if self._state not in (
            PPRecvBufferState.MODEL,
            PPRecvBufferState.DSPARK,
        ):
            raise RuntimeError(
                f"Cannot release PP receive buffer while it is {self._state.value}."
            )
        self._state = PPRecvBufferState.FREE
        self._owner_stream = None

    def _assert_stream(self, stream: object | None, action: str) -> None:
        if self._owner_stream is not None and stream != self._owner_stream:
            raise RuntimeError(
                f"Cannot {action} on CUDA stream {stream}; PP receive buffer "
                f"is owned by stream {self._owner_stream}."
            )

    def _transition(
        self,
        expected: PPRecvBufferState,
        new: PPRecvBufferState,
        action: str,
    ) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"Cannot {action} while PP receive buffer is {self._state.value}; "
                f"expected {expected.value}."
            )
        self._state = new


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_spec_tokens]


def _pad_sampled_tokens_for_pp(
    sampled_token_ids: torch.Tensor, max_sample_len: int
) -> torch.Tensor:
    width = sampled_token_ids.shape[-1]
    if width == max_sample_len:
        return sampled_token_ids
    if width > max_sample_len:
        raise ValueError(
            f"Sampled token width {width} exceeds PP receive width {max_sample_len}."
        )
    padded = sampled_token_ids.new_full(
        (sampled_token_ids.shape[0], max_sample_len), -1
    )
    padded[:, :width] = sampled_token_ids
    return padded


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(
                    slot.idx_mapping_np[keep], device=self.device
                )
            scatter_draft_tokens(
                draft_tokens_to_update, draft_tokens, draft_idx_mapping
            )

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = _pad_sampled_tokens_for_pp(
                sampled_token_ids, self.max_sample_len
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
