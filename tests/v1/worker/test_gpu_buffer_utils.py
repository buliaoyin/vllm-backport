# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.worker.gpu.buffer_utils import PinnedStagingPool


def test_staging_growth_preserves_inflight_copies():
    """Growing a pool must not overwrite host pages used by an earlier DMA."""
    pool = PinnedStagingPool(torch.int32, max_concurrency=3)
    pool.reserve(4)
    stream = torch.cuda.Stream()
    outputs = []
    inputs = [torch.arange(n, dtype=torch.int32) + i for i, n in enumerate((4, 9, 2))]
    with torch.cuda.stream(stream):
        for value in inputs:
            outputs.append(pool.copy_to_gpu(value, device=torch.device("cuda")))
    stream.synchronize()
    for actual, expected in zip(outputs, inputs):
        torch.testing.assert_close(actual.cpu(), expected)


def test_staging_reuses_capacity_after_consumers_finish():
    pool = PinnedStagingPool(torch.int32, max_concurrency=2)
    pool.reserve(8)
    pointers = []
    for i in range(6):
        staged = pool.stage(torch.full((4,), i, dtype=torch.int32))
        pointers.append(staged.data_ptr())
        actual = staged.to("cuda", non_blocking=True)
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual.cpu(), torch.full((4,), i, dtype=torch.int32))
    assert pointers[0] != pointers[1]
    assert pointers == pointers[:2] * 3
