# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row quantization and grouped INT8 GEMM with affine EXL3 codebook scaling."""

from vllm.triton_utils import tl, triton


@triton.jit
def quantize(A, Q, Scales, Sums, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    values = tl.load(A + row * K + cols, mask=cols < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 127.0, 1e-20)
    q = tl.extra.cuda.libdevice.nearbyint(values / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + cols, q, mask=cols < K)
    tl.store(Scales + row, scale)
    tl.store(Sums + row, tl.sum(values, 0))


@triton.jit
def grouped(
    A,
    B,
    C,
    Scales,
    Sums,
    Sorted,
    Experts,
    Padded,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    ALPHA4: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(EM, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N)
    group_id = pid // (GROUP_SIZE_M * nn)
    first = group_id * GROUP_SIZE_M
    group_size = tl.minimum(nm - first, GROUP_SIZE_M)
    pm = first + pid % group_size
    pn = (pid % (GROUP_SIZE_M * nn)) // group_size
    if pm * BLOCK_SIZE_M >= tl.load(Padded):
        return
    expert = tl.load(Experts + pm).to(tl.int64)
    if expert < 0:
        return
    slots = tl.load(Sorted + pm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(
        tl.int64
    )
    valid = slots < M
    cols = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    ks = tl.arange(0, BLOCK_SIZE_K)
    ap = A + slots[:, None] * K + ks[None, :]
    bp = B + expert * K * N + ks[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int32)
    for block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            ap, mask=valid[:, None] & (ks[None, :] + block * BLOCK_SIZE_K < K), other=0
        )
        b = tl.load(
            bp,
            mask=(cols[None, :] < N) & (ks[:, None] + block * BLOCK_SIZE_K < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        ap += BLOCK_SIZE_K
        bp += BLOCK_SIZE_K * N
    scale = tl.load(Scales + slots, mask=valid, other=0)
    row_sum = tl.load(Sums + slots, mask=valid, other=0)
    result = acc.to(tl.float32) * (scale[:, None] * ALPHA4) + row_sum[:, None] * BETA
    tl.store(
        C + slots[:, None] * N + cols[None, :],
        result.to(tl.float16),
        mask=valid[:, None] & (cols[None, :] < N),
    )
