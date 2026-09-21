# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete decode route counts, including GPU cache hits, on a stable stream."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _record_decode_routes(
    Ids,
    Padding,
    Enabled,
    Stats,
    T: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    N: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BINS: tl.constexpr,
):
    if tl.load(Enabled):
        t = tl.arange(0, BT)
        k = tl.arange(0, BK)
        ids = tl.load(
            Ids + t[:, None] * S0 + k[None, :] * S1,
            (t[:, None] < T) & (k[None, :] < K),
            other=-1,
        ).to(tl.int32)
        valid = (ids >= 0) & (ids < N)
        if HAS_PADDING:
            padding = tl.load(Padding + t, t < T, other=1)
            valid &= ~padding[:, None]
        bins = tl.reshape(tl.where(valid, ids, N), (BT * BK,))
        counts = tl.histogram(bins, BINS)
        e = tl.arange(0, BINS)
        previous = tl.load(Stats + e, e < N, other=0)
        groups = tl.load(Stats + N + e, e < N, other=0)
        tl.store(Stats + e, previous + counts, e < N)
        tl.store(Stats + N + e, groups + (counts > 0), e < N)
        tokens = tl.sum(tl.max(valid.to(tl.int32), axis=1), axis=0)
        old_steps = tl.load(Stats + 2 * N)
        old_tokens = tl.load(Stats + 2 * N + 1)
        tl.store(Stats + 2 * N, old_steps + (tokens > 0))
        tl.store(Stats + 2 * N + 1, old_tokens + tokens)


class DecodeRouteFeedback:
    def __init__(self, num_experts, device):
        self.num_experts = num_experts
        self.counts = torch.zeros(2 * num_experts + 2, dtype=torch.int64, device=device)
        self.enabled = torch.zeros((), dtype=torch.int32, device=device)
        self.active = False

    def set_enabled(self, enabled):
        if self.active != enabled:
            self.enabled.fill_(int(enabled))
            self.active = enabled

    def record(self, ids, padding=None):
        tokens, top_k = ids.shape
        if not 0 < tokens <= 128:
            raise ValueError("Decode feedback supports 1 to 128 physical rows")
        _record_decode_routes[(1,)](
            ids,
            padding if padding is not None else ids,
            self.enabled,
            self.counts,
            tokens,
            top_k,
            *ids.stride(),
            self.num_experts,
            padding is not None,
            triton.next_power_of_2(tokens),
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(self.num_experts + 1),
            num_warps=4,
        )

    def snapshot(self):
        return self.counts.cpu().numpy().copy()

    def consume(self):
        result = self.snapshot()
        self.counts.zero_()
        return result
