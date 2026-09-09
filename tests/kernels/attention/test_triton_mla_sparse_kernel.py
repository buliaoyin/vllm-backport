# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton sparse MLA kernel.

Compares split-KV against the single-pass (`num_kv_splits=1`) path
produced by the same kernel — both paths must agree to within bf16 ULPs.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    triton_mla_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton sparse MLA kernel requires CUDA/ROCm",
)


@pytest.fixture(scope="module")
def kv_cache():
    torch.manual_seed(0)
    return torch.randn(32768, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")


def _assert_split_matches_single_pass(
    num_tokens: int,
    num_heads: int,
    topk: int,
    num_kv_splits: int | None,
    kv_cache: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(
        0, kv_cache.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
    )
    out_ref = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=1,
    )
    out = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=num_kv_splits,
    )
    torch.testing.assert_close(
        out.float(),
        out_ref.float(),
        atol=5e-2,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    "num_tokens,num_heads",
    [(1, 16), (1, 128), (8, 32), (32, 128), (128, 16)],
)
@pytest.mark.parametrize("topk", [1024, 2048, 4096])
@pytest.mark.parametrize("num_kv_splits", [2, 4, 8])
def test_split_kv_matches_single_pass(
    num_tokens, num_heads, topk, num_kv_splits, kv_cache
):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads,
        topk,
        num_kv_splits,
        kv_cache,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_auto_split_matches_single_pass(num_tokens, kv_cache):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads=128,
        topk=2048,
        num_kv_splits=None,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_short_prefill_no_nan(num_kv_splits, kv_cache):
    """Regression: short prefill where most topk slots are -1 sentinels.

    The indexer fills 2048 topk positions with only a handful of valid
    indices; the rest are -1. Before the NEG_LARGE sentinel fix, the online
    softmax produced NaN via `max(-inf, -inf) = -inf` and
    `exp2(-inf − -inf) = NaN`, poisoning every split.
    """
    torch.manual_seed(0)
    num_tokens, num_heads, topk = 5, 16, 2048
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda")
    # Only the first `t+1` slots of each query hold valid indices; the
    # remaining ~2045 slots are -1, producing many all-invalid BLOCK_N tiles.
    for t in range(num_tokens):
        indices[t, 0, : t + 1] = torch.arange(
            64, 64 + t + 1, dtype=torch.int32, device="cuda"
        )
    out = triton_mla_sparse_attention(
        q, kv_cache, indices, sm_scale=0.0417, num_kv_splits=num_kv_splits
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize("splits", [1, 4])
def test_padding_patterns_match_attention_after_graph_replay(head_dim, splits):
    """Padding density may change between replays of the same captured graph."""
    torch.manual_seed(42)
    q = torch.randn(3, 16, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(1024, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((3, 1, 2176), -1, dtype=torch.int32, device="cuda")
    patterns = indices.clone()
    patterns[0, 0, :600] = torch.arange(600, device="cuda")
    patterns[1, 0, :32] = torch.arange(32, device="cuda")
    patterns[1, 0, 2048:2051] = torch.tensor([63, 64, 65], device="cuda")
    patterns[1, 0, 511] = kv.shape[0]  # invalid positive probe, like other padding
    indices.copy_(patterns)
    triton_mla_sparse_attention(q, kv, indices, 0.0625, num_kv_splits=splits)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = triton_mla_sparse_attention(
            q, kv, indices, 0.0625, num_kv_splits=splits
        )
    for shift in range(3):
        indices.copy_(patterns.roll(shift, dims=0))
        graph.replay()
        for row in range(3):
            selected = indices[row, 0]
            selected = selected[(selected >= 0) & (selected < kv.shape[0])].long()
            if selected.numel() == 0:
                assert torch.count_nonzero(output[row]).item() == 0
                continue
            keys = kv[selected, 0].float()
            probabilities = torch.softmax((q[row].float() @ keys.T) * 0.0625, dim=-1)
            expected = probabilities @ keys[:, :512]
            torch.testing.assert_close(
                output[row].float(), expected, rtol=0.02, atol=0.01
            )
