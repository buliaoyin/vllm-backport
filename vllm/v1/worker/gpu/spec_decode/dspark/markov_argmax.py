# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused vocab-sharded Markov step for DSpark's sequential drafting.

One draft position of ``DSparkSpeculator._sample_sequential`` under
``use_local_argmax_reduction`` is, in eager PyTorch::

    markov_embed = markov_w1(prev)  # [B, r]
    logits = F.linear(markov_embed, markov_w2.weight)  # [B, N]  N = shard width
    logits = logits + base_shard_logits[:, step]  # [B, N]
    logits[..., -num_pad:] = -inf  # padded-column mask
    val, idx = logits.max(dim=-1)  # [B], [B]
    pair = stack([val.float(), (idx + vocab_start).float()])
    gathered = all_gather(pair)  # [B, 2 * tp]
    prev = reduce_global_argmax(...)  # [B]

Seven kernels and three round-trips of the ``[B, N]`` row through HBM, five
times in sequence, all inside the captured draft graph. The row is never
*read* by anything but the argmax, so it never has to exist: this module fuses
the embedding lookup, the GEMV, the base-logit add, the padding mask and the
argmax into a single pass over ``markov_w2`` that emits only a (value, global
id) pair per request.

Three kernels per step, not one: the argmax spans the whole shard, so the
block-level maxima need a grid-wide reduction, and the cross-rank collective
sits between the local argmax and the final id. The chain is

    1. ``_markov_gemv_blockmax``  -- [B, nblk] packed block maxima
    2. ``_markov_reduce_scatter`` -- shard argmax, written into this rank's
       slot of a zeroed ``[B, 2 * tp]`` buffer (all other lanes zero)
    3. (collective) sum-all_reduce of that buffer
    4. ``_markov_finalize``       -- ``reduce_global_argmax`` semantics, the
       draft-to-target map, and the write of both ``draft_tokens[:, step]``
       and the ``prev`` buffer the next step reads

Everything writes into preallocated buffers and takes no data-dependent
shapes, so the chain is CUDA-graph capturable, which the DSpark draft step
requires.

PACKING. Steps 1-2 carry (value, index) through a single int64 so the
block-level and grid-level reductions are plain integer maxima::

    packed = monotone_i32(value) << 32 | (INT32_MAX - shard_local_index)

``monotone_i32`` is the standard order-preserving float-to-int map (flip the
magnitude bits of negatives), so integer order matches float order. The index
is stored complemented, which makes a *larger* packed word mean a *smaller*
index, so the integer max resolves ties to the lowest index -- the same rule
:func:`reduce_global_argmax` applies across shards, and the one that matches a
replicated ``argmax`` over the concatenated vocab. NaNs are mapped to a
canonical positive-NaN key above +inf so that a NaN wins its row the way
``torch.max`` propagates one; :func:`reduce_global_argmax`'s NaN fallback then
sees the same input it would have seen from the eager path.
"""

from dataclasses import dataclass

import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Wrapped in tl.constexpr because a @triton.jit body cannot read a plain
# module-level Python int.
#
# Sentinel below every packable word: monotone_i32(-inf) is 0x807FFFFF, so the
# smallest reachable packed value is 0x807FFFFF << 32 = -9.185e18, which is
# above INT64_MIN but *below* -(2**62) -- an all--inf row would let a -(2**62)
# sentinel win and return a garbage token id.
_PACKED_EMPTY = tl.constexpr(-(2**63))
_INT32_MAX = tl.constexpr(0x7FFFFFFF)
# Canonical positive quiet NaN. Above monotone_i32(+inf) = 0x7F800000, so a
# NaN lane wins its row under an integer max.
_NAN_KEY = tl.constexpr(0x7FC00000)


@triton.jit
def _monotone_bits(val):
    """Order-preserving int32 key for an fp32 value (NaN sorts highest)."""
    bits = val.to(tl.int32, bitcast=True)
    bits = tl.where(bits < 0, bits ^ _INT32_MAX, bits)
    return tl.where(val != val, _NAN_KEY, bits)


@triton.jit(do_not_specialize=["step", "num_valid", "shard_width"])
def _markov_gemv_blockmax(
    prev_ptr,  # [B] int64, previously drafted token id (TARGET vocab)
    w1_ptr,  # [target_vocab, R] Markov embedding table
    w2_ptr,  # [shard_width, R] this rank's Markov projection rows
    base_ptr,  # [B, n_spec, shard_width] base draft logits (this shard)
    part_ptr,  # [B, nblk] int64 out: packed block maxima
    stride_w1_v,
    stride_w2_n,
    stride_base_b,
    stride_base_s,
    step,
    num_valid,  # shard_width - num_org_vocab_padding
    shard_width,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    pid = tl.program_id(1)
    nblk = tl.num_programs(1)

    tok = tl.load(prev_ptr + b)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs < shard_width

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # R and BLOCK_R are both constexpr, so this unrolls; a [BLOCK_N, R] tile
    # in one shot is more registers than an SM will give at any useful BLOCK_N.
    for r0 in range(0, R, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        rmask = r < R
        e = tl.load(w1_ptr + tok * stride_w1_v + r, mask=rmask, other=0.0)
        w = tl.load(
            w2_ptr + offs[:, None] * stride_w2_n + r[None, :],
            mask=nmask[:, None] & rmask[None, :],
            other=0.0,
        )
        acc += tl.sum(w.to(tl.float32) * e.to(tl.float32)[None, :], axis=1)

    # Round the GEMV result to the head dtype before the add, then round the
    # sum again: that is what `F.linear(...) + base` does elementwise, and the
    # rounding is what the eager argmax actually compares.
    ety = base_ptr.dtype.element_ty
    prod = acc.to(ety).to(tl.float32)
    base = tl.load(
        base_ptr + b * stride_base_b + step * stride_base_s + offs,
        mask=nmask,
        other=0.0,
    ).to(tl.float32)
    val = (prod + base).to(ety).to(tl.float32)

    # Columns past org_vocab_size on this shard are zero-filled weight rows;
    # `get_shard_logits` masks them and so must this.
    val = tl.where(offs < num_valid, val, float("-inf"))

    packed = (_monotone_bits(val).to(tl.int64) << 32) | (
        (_INT32_MAX - offs).to(tl.int64)
    )
    tl.store(part_ptr + b * nblk + pid, tl.max(packed, axis=0))


@triton.jit(do_not_specialize=["nblk", "vocab_start", "tp_rank", "tp_size"])
def _markov_reduce_scatter(
    part_ptr,  # [B, nblk] int64 packed block maxima
    pair_ptr,  # [B, 2 * tp_size] fp32 out
    stride_pair_b,
    nblk,
    vocab_start,  # this shard's org_vocab_start_index
    tp_rank,
    tp_size,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)

    k = tl.arange(0, BLOCK_K)
    packed = tl.load(part_ptr + b * nblk + k, mask=k < nblk, other=_PACKED_EMPTY)
    best = tl.max(packed, axis=0)

    # Split the packed word by subtraction rather than a mask: `best & 0xFFFF
    # FFFF` would depend on whether Triton types that literal as int32 (where
    # it is -1 and the mask is a no-op) or int64. The high word is signed, the
    # low word is in [0, 2^31), so `best == (best >> 32) * 2^32 + low` exactly.
    high = best >> 32
    low = best - (high << 32)
    bits = high.to(tl.int32)
    bits = tl.where(bits < 0, bits ^ _INT32_MAX, bits)
    val = bits.to(tl.float32, bitcast=True)
    gid = ((_INT32_MAX - low).to(tl.int32) + vocab_start).to(tl.float32)

    # Write (val, global id) into this rank's two lanes and zero everywhere
    # else, so a sum-all_reduce reconstructs what an all_gather would have.
    # Exact: every other rank contributes exactly +0.0 to these lanes.
    # Broadcast by multiplying by ones, not by adding zeros: 0.0 + -0.0 is
    # +0.0, and a shard whose winning logit is -0.0 should stay -0.0.
    lane = tl.arange(0, BLOCK_L)
    out = tl.where(lane == 2 * tp_rank, tl.full([BLOCK_L], 1.0, tl.float32) * val, 0.0)
    out = tl.where(
        lane == 2 * tp_rank + 1, tl.full([BLOCK_L], 1.0, tl.float32) * gid, out
    )
    tl.store(pair_ptr + b * stride_pair_b + lane, out, mask=lane < 2 * tp_size)


@triton.jit(do_not_specialize=["step", "tp_size", "draft_limit", "target_limit"])
def _markov_finalize(
    pair_ptr,  # [B, 2 * tp_size] fp32, post-all_reduce
    d2t_ptr,  # [draft_vocab] int64 draft->target offsets (unused if no map)
    prev_ptr,  # [B] int64 out: next step's input token
    draft_ptr,  # [max_num_reqs, n_spec] int64 out
    stride_pair_b,
    stride_draft_b,
    step,
    tp_size,
    draft_limit,  # draft vocab size, for the d2t gather bound
    target_limit,  # markov_w1 row count, for the next step's embedding bound
    HAS_D2T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    smask = s < tp_size

    vals = tl.load(pair_ptr + b * stride_pair_b + 2 * s, mask=smask, other=0.0)
    ids = tl.load(pair_ptr + b * stride_pair_b + 2 * s + 1, mask=smask, other=0.0)

    # `reduce_global_argmax` semantics, lane for lane:
    #   best     = values.max(-1)                       (NaN propagates)
    #   fallback = global_ids.amax(-1)
    #   result   = where(values == best, ids, fallback).min(-1)
    # A NaN row makes `values == best` false everywhere, so every candidate
    # collapses to the fallback -- an id some shard actually offered, which is
    # the hardening that keeps a NaN row out of an out-of-range embedding
    # lookup. Reproduced here explicitly because tl.max does not promise
    # torch's NaN propagation.
    # Kept as an int, not a bool: `not <tl.tensor>` is a compile-time Python
    # negation and silently does the wrong thing on a runtime value.
    any_nan = tl.max(tl.where(smask & (vals != vals), 1, 0), axis=0)
    best = tl.max(tl.where(smask, vals, float("-inf")), axis=0)
    fallback = tl.max(tl.where(smask, ids, float("-inf")), axis=0)
    hit = smask & (vals == best) & (any_nan == 0)
    cand = tl.where(hit, ids, fallback)
    cand = tl.where(smask, cand, float("inf"))
    draft_id = tl.min(cand, axis=0).to(tl.int64)

    # Bound the id before it is used as an index. On real values this is a
    # no-op -- every candidate is some shard's `local_argmax + vocab_start`.
    # It matters during the allocation-pattern warm-up vLLM runs inside
    # `graph_capture()`, where a non-capturing custom all-reduce returns
    # `torch.empty_like` (custom_all_reduce.py:387-391), i.e. uninitialized
    # bytes. The eager path turns those into a device-side assert in
    # `F.embedding`; unclamped, this kernel would instead do an unchecked
    # out-of-bounds read off markov_w1, which is strictly worse. Same class of
    # hardening as the INT64_MAX sentinel fix in ed1d242587.
    draft_id = tl.minimum(tl.maximum(draft_id, 0), draft_limit - 1)
    if HAS_D2T:
        draft_id = draft_id + tl.load(d2t_ptr + draft_id)
        draft_id = tl.minimum(tl.maximum(draft_id, 0), target_limit - 1)
    tl.store(prev_ptr + b, draft_id)
    tl.store(draft_ptr + b * stride_draft_b + step, draft_id)


@dataclass
class MarkovFusionOperands:
    """The tensors and shard geometry the fused kernels need.

    Produced by the draft model (see ``DSparkMarkovHead.fusion_operands``) so
    the speculator does not reach into module internals, and so a head whose
    layout the kernels do not cover can decline by returning ``None``.
    """

    w1: torch.Tensor  # [target_vocab, R]
    w2: torch.Tensor  # [shard_width, R]
    num_valid: int  # shard_width - num_org_vocab_padding
    vocab_start: int
    tp_size: int
    tp_rank: int


def _pick_block_n(shard_width: int, num_sms: int) -> int:
    """Block width for the GEMV pass.

    The pass is bandwidth-bound on ``markov_w2`` (8.27 MB at TP=8 for the
    DeepSeek-V4-Flash shard), so the only thing that matters is filling the
    device: pick the largest block that still leaves at least two waves of
    CTAs, and never go below 32 (narrower blocks stop coalescing the row).

    Swept on an A100 at the shipped shape (16160 x 256), 5-step chain, us:

        BLOCK_N   32     64    128    256
        B=1     50.8   52.1   60.9   61.9
        B=8    115.0  107.8  100.6  107.1

    which this rule resolves to 64 (253 blocks, 2.3 waves over 108 SMs). No
    single width is best at both batches -- 32 wins B=1 by 2.5% and loses B=8
    by 14%, 128 the reverse -- and 64 is the only one within 7% of best at
    both. Not worth a per-call switch: the buffer and the reduce kernel's
    BLOCK_K are both sized from it, so varying it per batch would compile a
    second variant that warmup might not cover before a capture.
    """
    for block_n in (256, 128, 64, 32):
        if triton.cdiv(shard_width, block_n) >= 2 * num_sms:
            return block_n
    return 32


class FusedMarkovSampler:
    """Drives the fused chain for one draft step.

    Holds every buffer the kernels write, so the whole sequence is a fixed set
    of launches over fixed addresses and can be captured.
    """

    def __init__(
        self,
        operands: MarkovFusionOperands,
        d2t: torch.Tensor | None,
        max_num_reqs: int,
        n_spec: int,
        device: torch.device,
    ) -> None:
        self.op = operands
        self.d2t = d2t
        self.n_spec = n_spec
        shard_width = operands.w2.shape[0]
        self.rank_dim = operands.w2.shape[1]
        # Draft-vocab bound for the selected id: the shards tile it exactly.
        self.draft_limit = shard_width * operands.tp_size

        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.block_n = _pick_block_n(shard_width, num_sms)
        self.nblk = triton.cdiv(shard_width, self.block_n)
        # 256 is the DeepSeek-V4-Flash markov_rank; a full-rank tile at
        # BLOCK_N=32..256 is more registers than an A100 SM will give, so the
        # reduction is chunked.
        self.block_r = min(64, triton.next_power_of_2(self.rank_dim))

        self.part = torch.empty(
            (max_num_reqs, self.nblk), dtype=torch.int64, device=device
        )
        self.pair = torch.empty(
            (max_num_reqs, 2 * operands.tp_size), dtype=torch.float32, device=device
        )
        self.prev = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)

    def sample(
        self,
        num_reqs: int,
        base_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        """Sample the supplied steps into the prefix of ``draft_tokens[:num_reqs]``.

        ``base_logits`` is ``[num_reqs, num_steps, shard_width]`` and
        ``anchor_ids`` the ``[num_reqs]`` bonus tokens the chain starts from.
        """
        op = self.op
        assert base_logits.stride(-1) == 1
        assert 0 < base_logits.shape[1] <= self.n_spec
        prev = self.prev[:num_reqs]
        prev.copy_(anchor_ids)

        part = self.part[:num_reqs]
        pair = self.pair[:num_reqs]
        shard_width = op.w2.shape[0]

        for step in range(base_logits.shape[1]):
            _markov_gemv_blockmax[(num_reqs, self.nblk)](
                prev,
                op.w1,
                op.w2,
                base_logits,
                part,
                op.w1.stride(0),
                op.w2.stride(0),
                base_logits.stride(0),
                base_logits.stride(1),
                step,
                op.num_valid,
                shard_width,
                R=self.rank_dim,
                BLOCK_R=self.block_r,
                BLOCK_N=self.block_n,
            )
            _markov_reduce_scatter[(num_reqs,)](
                part,
                pair,
                pair.stride(0),
                self.nblk,
                op.vocab_start,
                op.tp_rank,
                op.tp_size,
                BLOCK_K=triton.next_power_of_2(self.nblk),
                BLOCK_L=triton.next_power_of_2(2 * op.tp_size),
            )
            # Sum of exact zeros: lane 2r holds rank r's value and lane 2r+1
            # its global id, unchanged. Same content an all_gather would
            # produce, but eligible for the one-shot custom all-reduce path.
            # Safe to hand it the persistent buffer even if the collective is
            # in-place -- the next step's kernel 2 rewrites every lane before
            # it is read again.
            reduced = tensor_model_parallel_all_reduce(pair) if op.tp_size > 1 else pair
            _markov_finalize[(num_reqs,)](
                reduced,
                # Triton still wants a real pointer for the unused argument.
                self.d2t if self.d2t is not None else prev,
                prev,
                draft_tokens,
                reduced.stride(0),
                draft_tokens.stride(0),
                step,
                op.tp_size,
                self.draft_limit,
                op.w1.shape[0],
                HAS_D2T=self.d2t is not None,
                BLOCK_S=triton.next_power_of_2(op.tp_size),
            )


def build_fused_markov_sampler(
    model: torch.nn.Module,
    max_num_reqs: int,
    n_spec: int,
    device: torch.device,
) -> FusedMarkovSampler | None:
    """Build the fused sampler, or ``None`` if this model is not covered.

    Declining is always safe -- the caller keeps the eager chain -- but it is
    logged, because a silent fallback is how a "measured" optimization stops
    being one.
    """
    hook = getattr(model, "markov_fusion_operands", None)
    if hook is None:
        logger.warning_once(
            "DSpark: %s does not expose markov_fusion_operands(); keeping the "
            "unfused Markov chain.",
            model.__class__.__name__,
        )
        return None
    operands = hook()
    if operands is None:
        return None
    d2t = getattr(model, "draft_id_to_target_id", None)
    sampler = FusedMarkovSampler(operands, d2t, max_num_reqs, n_spec, device)
    logger.info(
        "DSpark: fused Markov step (shard width %d, rank %d, BLOCK_N=%d, "
        "%d blocks); the [B, %d] logit row is never materialized.",
        operands.w2.shape[0],
        sampler.rank_dim,
        sampler.block_n,
        sampler.nblk,
        operands.w2.shape[0],
    )
    return sampler
