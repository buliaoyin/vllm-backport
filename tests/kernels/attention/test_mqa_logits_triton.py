# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton MQA logits kernels."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops import mqa_logits_triton as mqa_logits_mod
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton MQA logits kernels require CUDA/ROCm",
)


def _quantize_k_per_row(
    k_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    amax = k_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = amax / 448.0
    k_fp8 = (k_bf16.float() / sf).to(torch.float8_e4m3fn)
    return k_fp8, sf.squeeze(-1)


def _pack_paged_kv(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack BF16 KV into the layout produced by `indexer_k_quant_and_cache`.

    Physical bytes per block are segregated: all `block_size * head_dim` fp8 K
    bytes first, then `block_size * 4` fp32 scale bytes. The outer
    `[NB, BS, 1, D+4]` shape matches the production cache allocation.
    """
    num_blocks, block_size, head_dim = kv_bf16.shape
    amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = (amax / 448.0).to(torch.float32)
    k_fp8 = (kv_bf16.float() / sf).to(torch.float8_e4m3fn)
    packed = torch.empty(
        (num_blocks, block_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device=kv_bf16.device,
    )
    flat = packed.view(num_blocks, -1)
    k_end = block_size * head_dim
    flat[:, :k_end] = k_fp8.reshape(num_blocks, -1).view(torch.uint8)
    flat[:, k_end:] = sf.reshape(num_blocks, -1).view(torch.uint8)
    return packed


# References adapted from DeepGEMM (test_attention.py) — used as the spec
# the Triton kernels must agree with.
def _fp8_mqa_logits_ref(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    arange = torch.arange(0, seq_len_kv, device=q.device)[None, :]
    mask = (arange >= cu_seqlen_ks[:, None]) & (arange < cu_seqlen_ke[:, None])
    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def _fp8_paged_mqa_logits_ref(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * dim
    kv_data = (
        flat[:, :k_end].reshape(num_blocks, block_size, 1, dim).view(fp8_dtype).float()
    )
    scale = flat[:, k_end:].view(torch.float32).reshape(num_blocks, block_size, 1, 1)
    q = q.float()
    kv_data = kv_data * scale
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_data[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device=q.device,
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = (torch.relu(s) * weight_slice[..., None]).sum(dim=0)
            block_start = block_rk * block_size
            block_end = min((block_rk + 1) * block_size, max_model_len)
            block_width = block_end - block_start
            logits[
                i * next_n : (i + 1) * next_n,
                block_start:block_end,
            ] = torch.where(
                k_offsets[None, :block_width] <= q_offsets[:, None],
                s[:, :block_width],
                float("-inf"),
            )
    return logits


# Looser tolerance to accommodate FP8 rounding and the paged torch
# reference using fp32 matmul while the triton kernel uses bf16 matmul
# (with an fp32 accumulator, matching the DeepGEMM path).
_ATOL = 1.0
_RTOL = 0.2


# (8, 16640) crosses only _KV_GROUP_MIN_N and remains on the ungrouped path;
# it is the long-context regression shape for the new M gate.
@pytest.mark.parametrize("M,N", [(64, 64), (128, 256), (256, 512), (8, 16640)])
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("partial_mask", [False, True])
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_mqa_logits_triton_matches_torch(
    M, N, num_heads, partial_mask, clean_logits
):
    """`clean_logits=True` requires full-tensor agreement with the reference;
    `clean_logits=False` is only contractually correct on the in-range
    `[ks, ke)` slots, which is what `finite` already masks here (the reference
    writes -inf outside that range)."""
    torch.manual_seed(0)
    head_dim = 128
    device = "cuda"

    q_bf16 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(N, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scales = _quantize_k_per_row(k_bf16)

    if partial_mask:
        # Exercise the non-trivial ks/ke masking branch (chunked prefill).
        ks = torch.arange(M, dtype=torch.int32, device=device) % (N // 4)
        ke = ks + torch.randint(1, N // 2, (M,), dtype=torch.int32, device=device)
        ke = torch.minimum(ke, torch.tensor(N, dtype=torch.int32, device=device))
    else:
        ks = torch.zeros(M, dtype=torch.int32, device=device)
        ke = torch.full((M,), N, dtype=torch.int32, device=device)

    out_torch = _fp8_mqa_logits_ref(q_fp8, (k_fp8, k_scales), weights, ks, ke)
    out_triton = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=clean_logits
    )

    if clean_logits:
        assert torch.equal(
            torch.isinf(out_torch) & (out_torch < 0),
            torch.isinf(out_triton) & (out_triton < 0),
        )
    finite = ~torch.isinf(out_torch)
    if finite.any():
        torch.testing.assert_close(
            out_triton[finite], out_torch[finite], atol=_ATOL, rtol=_RTOL
        )


@pytest.mark.parametrize(
    "is_sm80,M,N,expected",
    [
        # The gate is on the GROUPED grid's CTA count, not on M: grouping
        # divides the grid's N dimension by KV_GROUP, so a small M with a wide
        # N still fills the machine while a small M with a narrow N does not.
        # Every row below is a measured corner (see the module's table).
        (True, 240, 61440, 8),  # the served 256k shape; M>=512 missed it
        (True, 8, 16384, 1),  # 128 grouped CTAs < one wave: grouping loses
        (True, 8, 61440, 8),  # 480 grouped CTAs: same M, wide N, wins
        (True, 32, 16384, 8),  # 512 grouped CTAs
        (True, 8, 16640, 1),
        (True, 512, 16383, 1),  # N gate still binds independently
        (True, 512, 16384, 8),
        (True, 2048, 28672, 8),
        (False, 2048, 28672, 1),
    ],
)
def test_fp8_mqa_logits_group_dispatch_boundaries(monkeypatch, is_sm80, M, N, expected):
    monkeypatch.setattr(mqa_logits_mod, "_IS_SM80", is_sm80)
    # Pin the wave size so this asserts the gate's arithmetic, not the SM
    # count of whichever GPU runs the suite (A100 = 108 x 3 = 324).
    monkeypatch.setattr(mqa_logits_mod, "_kv_group_min_ctas", lambda _idx: 324)
    assert mqa_logits_mod._select_prefill_kv_group(M, N) == expected


@pytest.mark.parametrize(
    "is_sm80,expected_calls",
    [
        (False, [(8, 8192, None)]),
        (True, [(8, 8192, None), (8, 16384, 8)]),
    ],
)
def test_fp8_mqa_logits_warmup_forces_grouped_specialization(
    monkeypatch, is_sm80, expected_calls
):
    calls = []

    def record_shape(num_heads, head_dim, m, n, device, kv_group_override=None) -> None:
        calls.append((m, n, kv_group_override))

    monkeypatch.setattr(mqa_logits_mod, "_warmup_fp8_mqa_logits_shape", record_shape)
    monkeypatch.setattr(mqa_logits_mod, "_IS_SM80", is_sm80)
    mqa_logits_mod.warmup_fp8_mqa_logits_triton(64, 128, torch.device("cuda:0"))
    assert calls == expected_calls


def test_fp8_mqa_logits_triton_clean_logits_false_overwrites_masked():
    """Regression test for PR #38476 comment 4398225404: when `clean_logits=False`
    skips the `-inf` pre-fill, the kernel itself must still write `-inf` to
    masked positions. Otherwise the K-tile early-exit branch leaves
    uninitialized memory in the output, downstream top-k reads garbage, and the
    model collapses to repeating a single token.

    Trigger: a narrow per-row `[ks, ke)` so that BLOCK_N tiles past every row's
    `ke` hit the early-exit path. We assert that `clean_logits=False` produces
    bit-identical output to `clean_logits=True` on every position (the flag is
    a perf opt, not a behavior change)."""
    torch.manual_seed(0)
    M, N, num_heads, head_dim = 256, 1024, 32, 128
    device = "cuda"

    q_fp8 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    k_bf16 = torch.randn(N, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    k_fp8, k_scales = _quantize_k_per_row(k_bf16)

    # Narrow per-row range: most rows' `[ks, ke)` cover ~1/16 of N, so most
    # K-tiles past `ke` hit the kernel's early-exit branch.
    ks = torch.arange(M, dtype=torch.int32, device=device) % (N // 8)
    ke = ks + N // 16
    ke = torch.minimum(ke, torch.full_like(ke, N))

    out_clean = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=True
    )
    out_dirty = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scales), weights, ks, ke, clean_logits=False
    )

    # Every position the clean run wrote `-inf` to (i.e. outside `[ks, ke)`)
    # must also be `-inf` in the dirty run — otherwise the early-exit left
    # uninitialized memory and downstream top-k will pick from garbage.
    inf_mask = torch.isinf(out_clean) & (out_clean < 0)
    bad = inf_mask & ~(torch.isinf(out_dirty) & (out_dirty < 0))
    if bad.any():
        # Surface a useful error: how many positions, and a sample of the
        # leaked values so it's obvious they're not `-inf`.
        leaked_count = int(bad.sum().item())
        sample = out_dirty[bad].flatten()[:8].tolist()
        raise AssertionError(
            f"clean_logits=False left {leaked_count} positions un-initialised "
            f"(should be -inf). Sample leaked values: {sample}. "
            "Likely cause: the kernel's K-tile early-exit branch returns "
            "without writing -inf, so downstream top-k reads garbage."
        )


@pytest.mark.parametrize(
    "batch_size,next_n,context_len",
    [
        (1, 1, 128),
        (1, 1, 512),
        (2, 1, 256),
        (1, 4, 512),  # speculative decoding with next_n=4
        (2, 4, 130),  # active max length is not block-aligned
    ],
)
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_paged_mqa_logits_triton_matches_torch(
    batch_size, next_n, context_len, num_heads, clean_logits
):
    """`clean_logits=True` requires whole-tensor agreement; `clean_logits=False`
    is only correct on `[:, :context_len]` (downstream topk reads only that
    range), so all comparisons below are sliced accordingly."""
    torch.manual_seed(0)
    head_dim = 128
    block_size = 64
    device = "cuda"

    total_blocks = 64
    max_blocks = (context_len + block_size - 1) // block_size + 4

    kv_bf16 = torch.randn(
        total_blocks, block_size, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_packed = _pack_paged_kv(kv_bf16)

    q_fp8 = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )

    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    block_tables = torch.randint(
        0,
        total_blocks,
        (batch_size, max_blocks),
        dtype=torch.int32,
        device=device,
    )

    max_model_len = context_len

    out_torch = _fp8_paged_mqa_logits_ref(
        q_fp8, kv_packed, weights, context_lens, block_tables, max_model_len
    )
    out_triton = fp8_paged_mqa_logits_triton(
        q_fp8,
        kv_packed,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        clean_logits=clean_logits,
    )

    inf_torch = (torch.isinf(out_torch) & (out_torch < 0))[:, :context_len]
    inf_triton = (torch.isinf(out_triton) & (out_triton < 0))[:, :context_len]
    assert torch.equal(inf_torch, inf_triton)
    finite = ~inf_torch
    if finite.any():
        torch.testing.assert_close(
            out_triton[:, :context_len][finite],
            out_torch[:, :context_len][finite],
            atol=_ATOL,
            rtol=_RTOL,
        )


def test_fp8_paged_mqa_logits_triton_strided_pool_no_int32_overflow():
    """Unified-KV-pool layer views have a large block stride; with enough
    blocks, int32 `block_idx * stride` exceeds 2**31 and wraps to a negative
    offset (IMA, or silent corruption when the wrapped address is mapped).
    Uses a padded pool whose stride puts the referenced blocks past the
    overflow threshold and checks parity against the torch reference."""
    torch.manual_seed(0)
    head_dim = 128
    block_size = 64
    batch_size, next_n, num_heads = 2, 1, 16
    context_len = 256
    device = "cuda"

    # Row payload is block_size * (head_dim + 4) = 8448 B; pad the pool block
    # stride to 2 MiB so int32 wraps at block_idx >= 2**31 / 2**21 = 1024.
    pool_stride = 2**21
    total_blocks = 1040

    row_elems = block_size * (head_dim + 4)
    kv_bf16 = torch.randn(
        total_blocks, block_size, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_contig = _pack_paged_kv(kv_bf16)

    backing = torch.zeros(total_blocks * pool_stride, dtype=torch.uint8, device=device)
    kv_strided = backing.as_strided(
        (total_blocks, block_size, 1, head_dim + 4),
        (pool_stride, head_dim + 4, head_dim + 4, 1),
    )
    kv_strided.copy_(kv_contig.view(total_blocks, block_size, 1, head_dim + 4))
    assert kv_strided.view(total_blocks, -1).stride(0) == pool_stride
    assert kv_strided.view(total_blocks, -1).shape[1] == row_elems

    q_fp8 = torch.randn(
        batch_size, next_n, num_heads, head_dim, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )
    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    # Every referenced block sits past the int32 overflow threshold.
    max_blocks = (context_len + block_size - 1) // block_size
    block_tables = torch.randint(
        1024,
        total_blocks,
        (batch_size, max_blocks),
        dtype=torch.int32,
        device=device,
    )

    out_torch = _fp8_paged_mqa_logits_ref(
        q_fp8, kv_strided, weights, context_lens, block_tables, context_len
    )
    out_triton = fp8_paged_mqa_logits_triton(
        q_fp8,
        kv_strided,
        weights,
        context_lens,
        block_tables,
        context_len,
        clean_logits=True,
    )

    inf_torch = torch.isinf(out_torch) & (out_torch < 0)
    inf_triton = torch.isinf(out_triton) & (out_triton < 0)
    assert torch.equal(inf_torch, inf_triton)
    finite = ~inf_torch
    torch.testing.assert_close(
        out_triton[finite], out_torch[finite], atol=_ATOL, rtol=_RTOL
    )


def _spread_k_scales(n: int, device: torch.device) -> torch.Tensor:
    """Per-row quantization scales spanning orders of magnitude.

    Canon rule 34: a distribution that makes the property trivially true makes
    the test unable to fail. Real activations of this model drive per-row fp8
    scales apart by ~2 decades, and it is exactly that spread that decides
    whether reordering the head sum's scaling can move a near-tie. Uniform
    scales cannot fail this test.
    """
    exponent = torch.linspace(-2.0, 2.0, n, device=device)
    return (10.0**exponent).to(torch.float32)


@pytest.mark.parametrize("kv_group", [1, 8])
def test_factored_k_scale_keeps_the_relu_active_set_and_the_selection(
    monkeypatch, kv_group
):
    """K7: hoisting `k_scale` out of the relu.

    `out = sum_h w_h * relu(k_scale * s)` becomes
    `out = k_scale * sum_h w_h * relu(s)`, exact in real arithmetic because
    `k_scale >= 0` and relu is positively homogeneous.

    What that argument does and does not buy, measured here rather than
    assumed:

    * The relu's ACTIVE SET is untouched -- a non-negative scale cannot move a
      sign. That is exact and asserted exactly.
    * The SELECTION is not guaranteed. The scaling happens once instead of
      BLOCK_H times, so the head sum rounds differently, and on inputs with
      the real spread (scales over ~4 decades, signed `weights_proj` output)
      the perturbation reaches ~3e-3 against a tightest top-k boundary gap of
      ~9e-4. The perturbation is therefore large enough to cross a near-tie
      even though none of these rows does. The relative error is cancellation
      in the signed head sum, not the factoring: with non-negative weights the
      same inputs agree to 2.8e-7.

    So this change carries a quality gate, not a selection-identity gate --
    which is the opposite of what a uniform-scale test would have suggested
    (canon rule 34).
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    M, N, H, D = 64, 4096, 64, 128

    q_fp8 = torch.randn(M, H, D, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    k_fp8 = torch.randn(N, D, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    k_scales = _spread_k_scales(N, device)
    weights = torch.randn(M, H, dtype=torch.float32, device=device)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    outs = {}
    for factored in (False, True):
        monkeypatch.setattr(
            mqa_logits_mod.envs, "VLLM_INDEXER_LOGITS_FACTOR_K_SCALE", factored
        )
        outs[factored] = mqa_logits_mod._fp8_mqa_logits_triton_impl(
            q_fp8, (k_fp8, k_scales), weights, ks, ke, kv_group
        ).clone()

    ref, factored_out = outs[False], outs[True]
    # Which entries the relu zeroed is the observable half of "no sign moved",
    # and it is exact in both variants.
    assert torch.equal(ref == 0, factored_out == 0)

    # The logits move by rounding only: bound the change against the size of
    # the summands, not against the (cancelling) sum.
    summand_scale = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    assert ((ref - factored_out).abs() / summand_scale).max() < 1e-4

    # Fixed seed, so this is a deterministic regression guard on the property
    # the indexer actually consumes -- not a proof that it cannot move.
    k = 512
    sel_ref = torch.topk(ref, k, dim=-1).indices.sort(dim=-1).values
    sel_new = torch.topk(factored_out, k, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(sel_ref, sel_new)


def test_paged_q_lut_hoist_is_bit_identical(monkeypatch):
    """K1: decoding q on the host must reproduce the in-kernel LUT exactly.

    Same 256-entry table (NaN pinned to +-480), same bf16 operands into the
    same `tl.dot`, same fp32 accumulate -- so this is an equality assert, not
    a tolerance. The NaN encodings are included deliberately: they are the
    only bytes where the table and a plain fp8->bf16 cast disagree, so a
    version that used `.to(bfloat16)` instead of the table would fail here.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    B, next_n, H, D = 3, 6, 64, 128
    block_size, num_blocks = 64, 32

    q_bytes = torch.randint(0, 256, (B, next_n, H, D), dtype=torch.uint8, device=device)
    q_bytes[0, 0, 0, :2] = torch.tensor([0x7F, 0xFF], dtype=torch.uint8, device=device)
    q = q_bytes.view(torch.float8_e4m3fn)
    kv_cache = _pack_paged_kv(
        torch.randn(num_blocks, block_size, D, dtype=torch.bfloat16, device=device)
    )
    weights = torch.randn(B * next_n, H, dtype=torch.float32, device=device)
    context_lens = torch.full(
        (B,), num_blocks * block_size // 2, dtype=torch.int32, device=device
    )
    block_tables = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .view(1, -1)
        .repeat(B, 1)
        .contiguous()
    )

    outs = {}
    for hoisted in (False, True):
        monkeypatch.setattr(mqa_logits_mod.envs, "VLLM_INDEXER_PAGED_Q_BF16", hoisted)
        outs[hoisted] = fp8_paged_mqa_logits_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len=num_blocks * block_size,
        ).clone()

    assert torch.equal(outs[False], outs[True])


@torch.inference_mode()
def test_fp8_paged_mqa_logits_torch_next_n_reads_planar_cache():
    """The torch fallback's next_n>1 branch must read the planar block layout.

    Regression for the layout mismatch class reported in
    vllm-project/vllm#50576 (CuritisSun): the cache writer is block-planar
    (all fp8 K bytes, then all fp32 scales), while a reader slicing the
    nominal ``[NB, BS, 1, D+4]`` shape token-interleaved silently drifts K by
    4 bytes/token and bitcasts value bytes as scales. Ground truth is derived
    from the source K/scale arrays, never from a second read of the cache.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, dim, heads = 6, 8, 32, 4
    batch_size, next_n = 2, 3
    context_lens = torch.tensor([13, 7], dtype=torch.int32, device=device)
    max_model_len = 16
    # Non-monotonic block tables: logical order != physical order.
    block_tables = torch.tensor([[3, 0], [5, 1]], dtype=torch.int32, device=device)

    kv_bf16 = torch.randn(
        num_blocks, block_size, dim, dtype=torch.bfloat16, device=device
    )
    packed = _pack_paged_kv(kv_bf16)
    # Source-array dequant (writer's definition of the content).
    amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = (amax / 448.0).to(torch.float32)
    k_fp8 = (kv_bf16.float() / sf).to(torch.float8_e4m3fn)
    kv_deq = (k_fp8.float() * sf).view(num_blocks, block_size, 1, dim)

    q = torch.randn(batch_size, next_n, heads, dim, dtype=torch.bfloat16, device=device)
    weights = torch.rand(batch_size * next_n, heads, dtype=torch.float32, device=device)

    out = fp8_paged_mqa_logits_torch(
        q, packed, weights, context_lens, block_tables, max_model_len
    )

    # Ground truth: replicate the branch's math from the SOURCE arrays.
    qf = q.float()
    expected = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len_i = int(context_lens[i].item())
        q_offsets = torch.arange(context_len_i - next_n, context_len_i, device=device)
        context_limit = torch.full(
            (next_n,), context_len_i, dtype=torch.int32, device=device
        )
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len_i, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = qf[i], kv_deq[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    expected.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            expected[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))

    assert torch.equal(out, expected)


@pytest.mark.parametrize("M,N", [(1, 127), (4, 8193), (128, 65537), (128, 524288)])
def test_bucketed_prefill_allocations_preserve_logits_and_topk(M, N):
    """Rounded storage must preserve masked logits and the strided top-k consumer."""
    from vllm import _custom_ops as ops

    torch.manual_seed(7)
    q = torch.randn(M, 64, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(N, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.logspace(-4, 4, N, device="cuda")
    weights = torch.randn(M, 64, device="cuda")
    ks = torch.arange(M, dtype=torch.int32, device="cuda") % (N // 4)
    ke = torch.full_like(ks, N - 1)
    args = q, (k, scales), weights, ks, ke
    exact = fp8_mqa_logits_triton(*args)
    rounded = fp8_mqa_logits_triton(*args, round_allocations=True)
    torch.testing.assert_close(rounded, exact, atol=0, rtol=0)
    outputs = []
    for logits in (exact, rounded):
        out = torch.empty(M, 8, dtype=torch.int32, device="cuda")
        ops.top_k_per_row_prefill(
            logits, ks, ke, out, M, logits.stride(0), logits.stride(1), 8
        )
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
