# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

import vllm.envs as envs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


# Below this token count the shard is not worth a collective: the GEMM is
# already small and all_gatherv would add a barrier per boundary. The cuBLAS
# route itself starts at 32 tokens; sharding wants a prefill-sized shape.
#
# The value is also load-bearing for a second reason: cudagraph capture tops
# out at min(max_num_seqs * 2, 512) = 256 at the default max_num_seqs=128, so a
# threshold at or above 512 guarantees every sharded batch runs eager and no
# collective is ever captured into a decode graph. Lowering it below the
# capture cap would put an all_gatherv inside a replayed graph.
_PRENORM_SHARD_MIN_TOKENS = 512


@triton.jit
def _row_sqrsum_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    K,
    BLOCK_K: tl.constexpr,
):
    """out[row] = sum(x[row].float() ** 2): the fp32 sqrsum the prenorm GEMM
    kernels produce as a side output, as a standalone one-pass reduction."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + row * stride_row + k0 + offs,
            mask=k0 + offs < K,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += x * x
    tl.store(out_ptr + row, tl.sum(acc))


def _prenorm_shard_rows(num_rows: int, tp_size: int) -> list[int] | None:
    """Row counts per TP rank, or None if this shape should stay replicated.

    Sizes are returned rather than assumed even so ``all_gatherv`` can
    reassemble any token count without a padding row -- there is then nothing
    written-but-unread that could reach the sinkhorn.
    """
    if tp_size < 2 or num_rows < _PRENORM_SHARD_MIN_TOKENS:
        return None
    base, rem = divmod(num_rows, tp_size)
    if base == 0:
        return None
    return [base + (r < rem) for r in range(tp_size)]


def hc_prenorm_gemm_cublas(
    x: Tensor,
    fn: Tensor,
    out: Tensor,
    sqrsum: Tensor | None,
) -> None:
    """Prenorm GEMM as cuBLAS bf16 plus a companion sqrsum reduction.

    The tilelang prenorm kernels re-read ``fn`` from every token tile, which
    makes them L2-bound at large T; here cuBLAS reads ``x`` once for the GEMM
    and ``_row_sqrsum_kernel`` reads it a second time — two passes total
    instead of the fused kernel's fn re-reads.

    Numerics: ``fn`` is rounded to bf16 (~3 mantissa bits below the fp32/tf32
    reference) and the GEMM result is rounded to bf16 before the fp32 upcast
    into ``out`` (cuBLAS will not emit fp32 from bf16 inputs via torch.mm).
    The parity test in tests/kernels/test_mhc_kernels.py bounds both.

    ``sqrsum`` may be None when the caller has already produced it -- the
    kernel that writes ``x`` can accumulate the same reduction for free, which
    saves re-reading ``x`` a second time.
    """
    assert out.shape[0] == 1
    fn_bf16 = getattr(fn, "_hc_prenorm_bf16", None)
    if fn_bf16 is None:
        # Cached on the weight tensor itself so lifetime and identity track
        # the weight; inference weights are never mutated in place.
        fn_bf16 = fn.to(torch.bfloat16)
        fn._hc_prenorm_bf16 = fn_bf16

    num_rows, k = x.shape
    rows = tp = None
    # `sqrsum is None` is exactly the condition that the caller already produced
    # it -- i.e. VLLM_MHC_POST_FUSE_SQRSUM is on -- and it is also what makes
    # this shard worth doing: one gather per boundary instead of two. Sharding
    # while still owing a sqrsum costs a second collective and gives most of the
    # win back, so the pairing is enforced here rather than left to whoever
    # writes the serve flags.
    #
    # Order matters: the token threshold is checked before anything reaches for
    # the TP group, because get_tp_group() asserts when the group is not
    # initialized and this kernel is also exercised single-process by the tests.
    if (
        envs.VLLM_MHC_PRENORM_SHARD
        and sqrsum is None
        and num_rows >= _PRENORM_SHARD_MIN_TOKENS
    ):
        from vllm.distributed.parallel_state import (
            get_tp_group,
            model_parallel_is_initialized,
        )

        if model_parallel_is_initialized():
            tp = get_tp_group()
            rows = _prenorm_shard_rows(num_rows, tp.world_size)

    if rows is None:
        out[0].copy_(x @ fn_bf16.t())
        if sqrsum is not None:
            assert sqrsum.shape[0] == 1
            _row_sqrsum_kernel[(num_rows,)](
                x, sqrsum[0], x.stride(0), k, BLOCK_K=1024, num_warps=4
            )
        return

    # Every rank already holds the whole residual (mhc_post is replicated), so
    # only the work is divided and nothing is scattered first. The gather
    # carries the 24-wide output rather than the 16384-wide input, which is
    # the entire reason this pays.
    start = sum(rows[: tp.rank_in_group])
    x_shard = x[start : start + rows[tp.rank_in_group]]
    out[0].copy_(tp.all_gatherv(x_shard @ fn_bf16.t(), dim=0, sizes=rows))


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_collapse_triton(x: Tensor, pre_mix: Tensor) -> Tensor:
    """Collapse BF16 residual streams with FP32 pre-mix coefficients."""
    assert x.ndim == 3 and x.dtype == torch.bfloat16
    num_tokens, hc_mult, hidden_size = x.shape
    assert pre_mix.shape == (num_tokens, hc_mult)
    assert pre_mix.dtype == torch.float32
    out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out

    block_h = 1024
    _hc_head_reduce_store_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        x,
        out,
        hidden_size,
        hc_mult,
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
        # Preserve the separate FP32 multiply and sum in the Torch reference.
        enable_fp_fusion=False,
    )
    return out


def _hc_collapse_triton_fake(x: Tensor, pre_mix: Tensor) -> Tensor:
    return torch.empty(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)


direct_register_custom_op(
    op_name="hc_collapse_triton",
    op_func=hc_collapse_triton,
    mutates_args=[],
    fake_impl=_hc_collapse_triton_fake,
)
