# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused n-gram hashing and lookup for GPU-addressable PLE tables."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ple_lookup_kernel(
    input_ids,
    query_start_loc,
    context,
    multipliers,
    sizes,
    offsets,
    weight,
    weight_scale,
    output,
    num_reqs,
    context_stride,
    tp_start,
    tp_end,
    EOS: tl.constexpr,
    NGRAM: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DEQUANTIZE_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    num_heads: tl.constexpr = (NGRAM - 1) * HEADS_PER_NGRAM
    token = row // num_heads
    head = row % num_heads
    valid = token < tl.load(query_start_loc + num_reqs)
    lo = 0
    hi = num_reqs
    while lo < hi:
        mid = (lo + hi) // 2
        boundary = tl.load(query_start_loc + mid)
        before = boundary <= token
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    req = tl.minimum(lo - 1, num_reqs - 1)
    start = tl.load(query_start_loc + req, mask=valid, other=0)
    mixed = tl.load(input_ids + token, mask=valid, other=EOS).to(tl.int64)
    mixed *= tl.load(multipliers)
    active = valid
    for shift in tl.static_range(1, NGRAM):
        pos = token - start - shift
        prev_input = tl.load(
            input_ids + token - shift, mask=valid & (pos >= 0), other=EOS
        )
        prev_context = tl.load(
            context + req * context_stride + NGRAM - 1 + pos,
            mask=valid & (pos < 0),
            other=EOS,
        )
        previous = tl.where(pos >= 0, prev_input, prev_context)
        active = active & (previous != EOS)
        previous = tl.where(active, previous, EOS).to(tl.int64)
        product = previous * tl.load(multipliers + shift)
        mixed = tl.where(shift < head // HEADS_PER_NGRAM + 2, mixed ^ product, mixed)
    size = tl.load(sizes + head)
    index = mixed % size
    index = tl.where(index < 0, index + size, index) + tl.load(offsets + head)
    owned = valid & (index >= tp_start) & (index < tp_end)
    columns = tl.arange(0, BLOCK_D)
    values = tl.load(
        weight + (index - tp_start) * DIM + columns,
        mask=owned & (columns < DIM),
        other=0,
    )
    if DEQUANTIZE_FP8:
        # E4M3FN bytes are portable to SM80, which has no native FP8 conversion.
        bits = values.to(tl.uint32)
        magnitude = bits & 0x7F
        normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
        value = tl.where(magnitude < 8, magnitude.to(tl.float32) / 512.0, normal)
        value = tl.where(magnitude == 127, float("nan"), value)
        signed = value.to(tl.uint32, bitcast=True) | ((bits & 0x80) << 24)
        value = signed.to(tl.float32, bitcast=True)
        scale = tl.load(weight_scale).to(output.dtype.element_ty).to(tl.float32)
        values = value * scale
    tl.store(output + row * DIM + columns, values, mask=columns < DIM)


def fused_ple_lookup(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    *,
    eos_token_id: int,
    heads_per_ngram: int,
    tp_start: int,
    tp_end: int,
    output: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Look up rows without constructing a requests-by-context workspace."""
    num_tokens = input_ids.numel()
    ngram = multipliers.numel()
    dim = weight.shape[1]
    heads = (ngram - 1) * heads_per_ngram
    if output is None:
        output = torch.empty(
            (num_tokens, heads * dim), device=input_ids.device, dtype=weight.dtype
        )
    num_reqs = query_start_loc.numel() - 1
    if not num_tokens:
        return output
    if num_reqs < 1:
        return output.zero_()
    fp8 = weight.dtype == torch.float8_e4m3fn
    dequantize = fp8 and output.dtype != torch.float8_e4m3fn
    if dequantize and weight_scale is None:
        raise ValueError("FP8 PLE lookup requires its checkpoint scale")
    _ple_lookup_kernel[(num_tokens * heads,)](
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers,
        sizes,
        offsets,
        weight.view(torch.uint8) if fp8 else weight,
        weight_scale,
        output.view(torch.uint8) if fp8 and not dequantize else output,
        num_reqs,
        ngram_context.stride(0),
        tp_start,
        tp_end,
        EOS=eos_token_id,
        NGRAM=ngram,
        HEADS_PER_NGRAM=heads_per_ngram,
        DIM=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        DEQUANTIZE_FP8=dequantize,
    )
    return output
