# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the external FreeToken ds_fp4 executor against the same inputs."""

import importlib.util
from pathlib import Path

import torch


class FreeTokenCPUExperts:
    def __init__(self, config, experts, hidden, intermediate, top_k, limit, max_tokens):
        path = str(Path(config.library_path).resolve())
        spec = importlib.util.spec_from_file_location("_cpu_moe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.extension = module
        self.config = config
        self.geometry = experts, hidden, intermediate, top_k, limit, max_tokens
        self.weights = torch.empty(
            (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8
        )
        self.scales = torch.empty(
            (experts, 2 * intermediate, hidden // 32), dtype=torch.uint8
        )
        self.down = torch.empty((experts, hidden, intermediate // 2), dtype=torch.uint8)
        self.down_scales = torch.empty(
            (experts, hidden, intermediate // 32), dtype=torch.uint8
        )
        self.executor = None
        self.io = {}

    def load_expert(self, expert, projection, weight, scales):
        intermediate = self.geometry[2]
        if projection == 1:
            self.down[expert].copy_(weight)
            self.down_scales[expert].copy_(scales)
        else:
            start = 0 if projection == 0 else intermediate
            self.weights[expert, start : start + intermediate].copy_(weight)
            self.scales[expert, start : start + intermediate].copy_(scales)

    def prepare(self):
        if self.executor is not None:
            return
        experts, hidden, intermediate, top_k, limit, max_tokens = self.geometry
        self.tables = [
            torch.tensor([weight.data_ptr()], dtype=torch.uint64)
            for weight in (self.weights, self.down, self.scales, self.down_scales)
        ]
        # ds_fp4 uses checkpoint row-major E2M1/E8M0 and DeepSeek's clamped SiLU.
        # The separate gpt-oss MXFP4 branch has a different layout and epilogue.
        self.executor = self.extension.CpuMoeExecutor(
            self.config.num_threads,
            1,
            experts,
            top_k,
            hidden,
            intermediate,
            max_tokens,
            0,
            0,
            3,
            self.tables[0].data_ptr(),
            self.tables[1].data_ptr(),
            self.tables[2].data_ptr(),
            0,
            self.tables[3].data_ptr(),
            0,
            0,
            0,
            1.0,
            limit,
            list(range(self.config.num_threads)),
        )

    def forward(self, hidden, ids, routes):
        self.prepare()
        tokens = hidden.shape[0]
        if tokens not in self.io:
            x = torch.empty_like(hidden, dtype=torch.bfloat16)
            indices = torch.empty_like(ids, dtype=torch.int32)
            weights = torch.empty_like(routes, dtype=torch.float32)
            output = torch.empty_like(x)
            task = self.executor.create_task(
                0,
                tokens,
                x.data_ptr(),
                indices.data_ptr(),
                weights.data_ptr(),
                output.data_ptr(),
            )
            self.io[tokens] = x, indices, weights, output, task
        x, indices, weights, output, task = self.io[tokens]
        x.copy_(hidden)
        indices.copy_(ids)
        weights.copy_(routes)
        self.executor.run_task(task)
        return output

    def close(self):
        self.executor = None
