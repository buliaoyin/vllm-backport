# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookup and decode the ExLlamaV3 160-dimensional n-gram row format."""

from functools import lru_cache

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _exl3_ngram_lookup(
    packed,
    ids,
    bias,
    output,
    codebook,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    WORDS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ids + row).to(tl.int64)
    valid = (index >= 0) & (index < ROWS)
    columns = tl.arange(0, BLOCK)
    mask = (columns < 160) & valid
    scale_word = tl.load(packed + index * WORDS, valid, other=0)
    scale = scale_word.to(tl.float16, bitcast=True).to(tl.float32)
    state = tl.full((BLOCK,), 0, tl.uint32)
    for bit in tl.static_range(16):
        position = (columns - bit // BITS + 160) % 160
        source_bit = position * BITS + bit % BITS
        word = (
            tl.load(packed + index * WORDS + 1 + source_bit // 16, mask, other=0)
            .to(tl.uint16)
            .to(tl.uint32)
        )
        state |= ((word >> (source_bit % 16)) & 1) << bit
    code = tl.load(codebook + state).to(tl.float32)
    head_bias = tl.load(bias + (row % HEADS) * 160 + columns, mask, other=0).to(
        tl.float32
    )
    values = tl.where(valid, code * scale + head_bias, 0)
    tl.store(output + row * 160 + columns, values, columns < 160)


@lru_cache
def _ngram_codebook(device: torch.device) -> torch.Tensor:
    states = torch.arange(65536, dtype=torch.int64, device="cpu")
    product = states * 0x83DCD12D & 0xFFFFFFFF
    total = 1024 + sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    inverse = torch.tensor(0x1EEE, dtype=torch.uint16, device="cpu").view(torch.float16)
    offset = torch.tensor(0xC931, dtype=torch.uint16, device="cpu").view(torch.float16)
    return (total.float() * inverse.float() + offset.float()).half().to(device)


def exl3_ngram_lookup(
    packed: torch.Tensor,
    ids: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode selected compressed rows from device or pinned-host storage."""
    if output is None:
        output = torch.empty((*ids.shape, 160), dtype=torch.float16, device=ids.device)
    if ids.numel():
        _exl3_ngram_lookup[(ids.numel(),)](
            packed,
            ids,
            bias,
            output,
            _ngram_codebook(ids.device),
            ROWS=packed.shape[0],
            HEADS=bias.shape[0],
            WORDS=packed.shape[1],
            BITS=(packed.shape[1] - 1) // 10,
            BLOCK=256,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return output
