# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
    num_query_tokens: int | None = None,
    max_num_reqs: int = 4,
    num_reqs: int = 1,
):
    device = torch.device("cuda")
    max_num_tokens = max(16, max_num_reqs * (num_query_tokens or 3))
    num_speculative_steps = 3
    state_indices = [
        (min(2, max_num_reqs - 1) + i) % max_num_reqs for i in range(num_reqs)
    ]

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device),
        query_start_loc=torch.full(
            (max_num_reqs + 1,), -1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.full(num_reqs, 4, dtype=np.int32),
        positions=torch.tensor(
            target_positions * num_reqs, dtype=torch.int64, device=device
        ),
        query_start_loc=torch.arange(
            0, 4 * num_reqs + 1, 4, dtype=torch.int32, device=device
        ),
        idx_mapping=torch.tensor(state_indices, dtype=torch.int32, device=device),
    )
    query_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    sample_count = max_num_reqs * num_speculative_steps
    sample_storage = [
        torch.full((sample_count + 4,), -77, dtype=dtype, device=device)
        for dtype in (torch.int64, torch.int64, torch.int32)
    ]
    sample_indices, sample_pos, sample_idx_mapping = (
        tensor[:sample_count] for tensor in sample_storage
    )
    for tensor in (sample_indices, sample_pos, sample_idx_mapping):
        tensor.fill_(-1)
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    input_temperature = torch.zeros_like(temperature)
    input_seeds = torch.zeros_like(seeds)
    last_sampled = torch.zeros_like(seeds)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    for index, state_idx in enumerate(state_indices):
        input_temperature[state_idx] = 1.0 + index
        input_seeds[state_idx] = 17 + index
        last_sampled[state_idx] = 99 + index
        next_prefill_tokens[state_idx] = 199 + index
    block_table = torch.tensor(
        [
            [value + 16 * i if value else 0 for value in block_table_values]
            for i in range(num_reqs)
        ],
        dtype=torch.int32,
        device=device,
    )

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        torch.tensor(
            [int(i % 2 == 0) for i in range(num_reqs)], dtype=torch.int32, device=device
        ),
        torch.tensor(
            [2 if i % 2 == 0 else 0 for i in range(num_reqs)],
            dtype=torch.int32,
            device=device,
        ),
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        4,
        cp_rank,
        cp_size,
        cp_interleave,
        123,
        num_query_tokens or num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        sample_guards=[tensor[sample_count:].cpu() for tensor in sample_storage],
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


def test_prepare_dflash_inputs_excludes_rejected_context_suffix():
    # Positions 10/11 use physical block 7. Rejected positions 12/13 would use
    # block 8, but must be PAD context rather than contaminating draft KV.
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]

    # The replacement query starts immediately after the two valid rows and
    # advances from the last accepted position (11).
    assert out.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert out.input_buffers.positions[:3].cpu().tolist() == [12, 13, 14]
    assert out.query_slot_mapping[:3].tolist() == [32, 33, 34]
    assert out.sample_indices[:3].tolist() == [0, 1, 2]
    assert out.sample_pos[:3].tolist() == [13, 14, 15]
    assert out.sample_idx_mapping[:3].tolist() == [2, 2, 2]
    assert out.temperature[2].item() == 1.0
    assert out.seeds[2].item() == 17


def test_prepare_dflash_inputs_excludes_rejected_context_suffix_with_dcp():
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 7, 8, 9],
        cp_rank=1,
        cp_size=2,
        cp_interleave=2,
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [28, 29, PAD_SLOT_ID, PAD_SLOT_ID]
    assert out.query_slot_mapping[:3].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID, 30]


def test_prepare_dflash_inputs_never_writes_the_null_block():
    # The valid context uses logical block 0 and the replacement query uses
    # logical block 1. Both map to the null block and must remain unwritable.
    out = _run_prepare(
        target_positions=[2, 3, 4, 5],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_slot_mapping[:4].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert out.query_slot_mapping[:3].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]


def test_dspark_query_prefix_keeps_tail_kv_without_sampling_past_output():
    # Five backbone queries cross a KV block boundary, but only the first
    # three become proposals. Guard cells expose writes past the sample buffers.
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
        num_query_tokens=5,
        max_num_reqs=1,
    )
    assert out.input_buffers.positions[:5].cpu().tolist() == [12, 13, 14, 15, 16]
    assert out.input_buffers.input_ids[:5].cpu().tolist() == [99, 123, 123, 123, 123]
    assert out.input_buffers.query_start_loc.cpu().tolist() == [0, 5]
    assert out.input_buffers.seq_lens.cpu().tolist() == [17]
    assert out.query_slot_mapping[:5].tolist() == [32, 33, 34, 35, 36]
    assert out.sample_indices.tolist() == [0, 1, 2]
    assert out.sample_pos.tolist() == [13, 14, 15]
    assert out.sample_idx_mapping.tolist() == [0, 0, 0]
    for guard in out.sample_guards:
        assert guard.tolist() == [-77] * 4


@pytest.mark.parametrize("num_reqs", [3, 16])
def test_dspark_query_prefix_keeps_concurrent_request_slots_separate(num_reqs):
    """Five-query drafts must respect reordered requests and rejected suffixes."""
    blocks = [0, 0, 7, 8, 9, 10, 11, 12]
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=blocks,
        num_query_tokens=5,
        max_num_reqs=16,
        num_reqs=num_reqs,
    )
    for index in range(num_reqs):
        start = 12 if index % 2 == 0 else 14
        anchor = 99 + index if index % 2 == 0 else 199 + index
        query = slice(5 * index, 5 * (index + 1))
        sample = slice(3 * index, 3 * (index + 1))
        assert out.input_buffers.positions[query].cpu().tolist() == list(
            range(start, start + 5)
        )
        assert out.input_buffers.input_ids[query].cpu().tolist() == [anchor] + [123] * 4
        assert out.query_slot_mapping[query].tolist() == [
            (blocks[position // 4] + 16 * index) * 4 + position % 4
            for position in range(start, start + 5)
        ]
        assert out.sample_indices[sample].tolist() == list(
            range(5 * index, 5 * index + 3)
        )
        assert out.sample_pos[sample].tolist() == list(range(start + 1, start + 4))
        assert out.sample_idx_mapping[sample].tolist() == [(2 + index) % 16] * 3
    assert out.input_buffers.query_start_loc.cpu().tolist() == [
        min(index, num_reqs) * 5 for index in range(17)
    ]
    assert out.sample_idx_mapping[3 * num_reqs :].tolist() == [-1] * (
        3 * (16 - num_reqs)
    )
    assert out.query_slot_mapping[5 * num_reqs :].tolist() == [PAD_SLOT_ID] * (
        80 - 5 * num_reqs
    )
    for guard in out.sample_guards:
        assert guard.tolist() == [-77] * 4
