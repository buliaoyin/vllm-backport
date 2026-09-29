# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry, and the PPHandler
sampled-token / draft-token relay under spec decode."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import PPHandler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PPHandler drives a side CUDA stream"
)


@requires_cuda
def test_send_staging_preserves_reused_sources_and_inflight_slots():
    """Delayed consumers must see each generation after both buffers wrap."""
    staging = pp_utils.PPSendStaging(
        {"hidden_states": ((257, 128), torch.float32)}, torch.device("cuda:0")
    )
    source = torch.empty((257, 128), device="cuda:0")
    consumer = torch.cuda.Stream()
    received, expected, pointers = [], [], set()
    allocation = staging.nbytes

    def send(tensors):
        tensor = tensors["hidden_states"]
        pointers.add(tensor.data_ptr())
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(200000)
            received.append(tensor.clone())
            done = torch.cuda.Event()
            done.record()
        return [Mock(), pp_utils._CopyComplete(done)]

    for generation in range(12):
        rows = (1, 17, 257)[generation % 3]
        view = source[:rows].fill_(generation)
        handles = staging.isend({"hidden_states": view}, send)
        for handle in handles[1:]:
            handle.wait()
        source.fill_(-1)
        expected.append(generation)
    torch.accelerator.synchronize()
    for tensor, value in zip(received, expected):
        torch.testing.assert_close(tensor, torch.full_like(tensor, value))
    assert len(pointers) == 2
    assert staging.nbytes == allocation == 2 * source.numel() * source.element_size()


@requires_cuda
def test_send_staging_falls_back_without_growing_for_unsupported_layout():
    staging = pp_utils.PPSendStaging(
        {"hidden_states": ((2, 8), torch.float32)}, torch.device("cuda:0")
    )
    tensors = {"hidden_states": torch.zeros((3, 8), device="cuda:0")}
    handles = [Mock(), Mock()]
    send = Mock(return_value=handles)
    assert staging.isend(tensors, send) is handles
    assert send.call_args.args[0] is tensors
    assert staging.nbytes == 2 * 2 * 8 * 4


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


# ---------------------------------------------------------------------------
# PPHandler relay: sender and receiver must stay in lockstep on every step
# ---------------------------------------------------------------------------


def make_handler(
    monkeypatch,
    *,
    is_last_rank: bool,
    num_speculative_steps: int,
    world_size: int = 2,
) -> PPHandler:
    """Build a real PPHandler with the PP group stubbed out."""
    pp_group = SimpleNamespace(
        is_last_rank=is_last_rank,
        last_rank=world_size - 1,
        world_size=world_size,
        make_sibling_device_group=lambda group_desc: object(),
    )
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: pp_group)
    return PPHandler(
        max_num_reqs=8,
        num_speculative_steps=num_speculative_steps,
        device=torch.device("cuda"),
    )


def record_broadcasts(monkeypatch, fill_value: int | None = None) -> list:
    """Capture every tensor handed to the collective, in call order.

    With `fill_value`, also fill the tensor so a receiver observes "data" as if
    the last rank had sent it."""
    calls: list[torch.Tensor] = []

    def fake_broadcast(tensor, src, group):
        calls.append(tensor)
        if fill_value is not None:
            tensor.fill_(fill_value)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    return calls


def make_input_batch(num_reqs: int = 3, *, needs_sample: bool = True):
    # compute_need_sampled_mask only reads the three numpy fields. With
    # needs_sample=False every request is a non-final prefill chunk, so no
    # sample is produced this step.
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len_np=np.full(num_reqs, 4 if needs_sample else 4096, dtype=np.int32),
        num_scheduled_tokens=np.full(num_reqs, 4, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device="cuda"),
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
    )


def send_step(handler, input_batch, *, width: int, with_draft: bool):
    num_reqs = input_batch.num_reqs
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device="cuda")
    counts = torch.zeros(num_reqs, dtype=torch.int32, device="cuda")
    handler.broadcast(sampled, counts, counts, input_batch)
    if with_draft:
        # The runner relays its persistent [max_num_reqs, num_spec] table.
        draft_tokens = torch.zeros(
            8, handler.num_speculative_steps, dtype=torch.int64, device="cuda"
        )
        handler.broadcast_drafts(draft_tokens, input_batch)


@requires_cuda
@pytest.mark.parametrize("width,num_spec", [(1, 1), (1, 3), (2, 3)])
def test_broadcast_pads_sampled_tokens_to_max_sample_len(monkeypatch, width, num_spec):
    """The sampler emits width 1 on steps with no draft tokens (prefill, first
    decode) and num_spec+1 only after rejection sampling. The receiver always
    posts a max_sample_len buffer, so an unpadded send is a count mismatch."""
    handler = make_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=num_spec
    )
    calls = record_broadcasts(monkeypatch)
    input_batch = make_input_batch()

    send_step(handler, input_batch, width=width, with_draft=False)

    sent_sampled = calls[0]
    assert sent_sampled.shape == (input_batch.num_reqs, handler.max_sample_len)
    # The real columns come first; the padding columns are ignored by
    # post_update, which advances each request by its own num_sampled count.
    assert (sent_sampled[:, :width] == 0).all()


@requires_cuda
def test_send_and_recv_op_counts_match_with_speculator(monkeypatch):
    """With a speculator the step is three broadcasts: sampled, combined, draft,
    matched strictly by order on the communicator."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=3)
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(), width=1, with_draft=True)
    assert len(calls) == 3
    assert calls[0].shape == (3, sender.max_sample_len)
    assert calls[2].shape == (3, sender.num_speculative_steps)

    receiver = make_handler(monkeypatch, is_last_rank=False, num_speculative_steps=3)
    calls.clear()
    assert receiver.receive(make_input_batch())
    assert len(calls) == 3
    assert calls[0].shape == (3, sender.max_sample_len)
    assert calls[2].shape == (3, sender.num_speculative_steps)


@requires_cuda
def test_send_and_recv_op_counts_match_without_speculator(monkeypatch):
    """Without spec decode the step is two broadcasts on both sides."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=0)
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(), width=1, with_draft=False)
    assert len(calls) == 2

    receiver = make_handler(monkeypatch, is_last_rank=False, num_speculative_steps=0)
    calls.clear()
    assert receiver.receive(make_input_batch())
    assert len(calls) == 2
    assert receiver.queue[-1].draft_tokens is None


@requires_cuda
def test_both_ranks_skip_when_no_request_needs_sampling(monkeypatch):
    """The skip gate must be symmetric, or the ranks desynchronize."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=3)
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(needs_sample=False), width=1, with_draft=True)
    assert calls == []

    receiver = make_handler(monkeypatch, is_last_rank=False, num_speculative_steps=3)
    calls.clear()
    assert not receiver.receive(make_input_batch(needs_sample=False))
    assert calls == []


@requires_cuda
def test_relayed_draft_tokens_are_scattered_on_consume(monkeypatch):
    """Non-last ranks verify against req_states.draft_tokens, which only the
    last rank can compute. The relayed drafts must land in that table when the
    deferred entry is consumed pp_size steps later, or the earlier stages verify
    against a zero-initialised table (near-zero acceptance, corrupt output)."""
    receiver = make_handler(
        monkeypatch, is_last_rank=False, num_speculative_steps=3, world_size=2
    )
    record_broadcasts(monkeypatch, fill_value=7)
    input_batch = make_input_batch(num_reqs=2)
    assert receiver.receive(input_batch)

    draft_tokens = torch.zeros(8, 3, dtype=torch.int64, device="cuda")
    # The queue is pre-seeded with pp_size placeholders: the first consume is
    # a no-op, the second returns the entry pushed above.
    assert receiver.get_prev_sampled_outputs(draft_tokens) is None
    outputs = receiver.get_prev_sampled_outputs(draft_tokens)
    torch.accelerator.synchronize()

    assert outputs is not None
    assert (draft_tokens[:2] == 7).all()
    assert (draft_tokens[2:] == 0).all()
    assert (outputs["sampled_tokens"] == 7).all()


@requires_cuda
@pytest.mark.parametrize("num_spec", [3, 7])
def test_mixed_draft_relay_skips_prefill_and_reused_slots_without_host_sync(
    monkeypatch, num_spec
):
    receiver = make_handler(
        monkeypatch, is_last_rank=False, num_speculative_steps=num_spec
    )
    record_broadcasts(monkeypatch, fill_value=7)
    batch = make_input_batch(num_reqs=3)
    batch.idx_mapping_np = np.array([4, 2, 6], dtype=np.intp)
    batch.idx_mapping = torch.as_tensor(batch.idx_mapping_np, device="cuda")
    batch.prefill_len_np[1] = 4096
    assert not receiver.receive(batch)
    receiver.on_req_idx_freed(6)

    drafts = torch.full((8, num_spec), -9, dtype=torch.int64, device="cuda")
    # Compile the scatter before checking that consuming a mixed step is async.
    pp_utils.scatter_draft_tokens(drafts, drafts[:3], batch.idx_mapping)
    torch.accelerator.synchronize()
    assert receiver.get_prev_sampled_outputs(drafts) is None
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        outputs = receiver.get_prev_sampled_outputs(drafts)
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    expected = torch.full_like(drafts, -9)
    expected[4] = 7
    torch.testing.assert_close(drafts, expected)
    assert outputs is not None
    assert outputs["idx_mapping"].tolist() == [4, -1, -1]
