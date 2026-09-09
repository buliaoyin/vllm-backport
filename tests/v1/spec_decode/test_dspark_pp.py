# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.model import _needs_mtp_hidden_states
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


@pytest.mark.cpu_test
def test_dspark_does_not_require_mtp_target_hidden_states():
    dspark = SimpleNamespace(
        use_dspark=lambda: True,
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
    )
    mtp = SimpleNamespace(
        use_dspark=lambda: False,
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
    )

    assert not _needs_mtp_hidden_states(dspark)
    assert _needs_mtp_hidden_states(mtp)


@pytest.mark.cpu_test
def test_dspark_packs_aux_states_into_pp_workspace():
    class CombineModel:
        packed_hidden_states: torch.Tensor | None = None

        def combine_hidden_states(
            self, packed_hidden_states: torch.Tensor
        ) -> torch.Tensor:
            self.packed_hidden_states = packed_hidden_states
            width = packed_hidden_states.shape[-1] // 2
            return packed_hidden_states[:, :width] + packed_hidden_states[:, width:]

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.model = CombineModel()
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    speculator._target_hidden_states_workspace = None
    workspace = torch.empty(32)
    assert speculator.set_target_hidden_states_workspace(workspace)
    aux_hidden_states = [
        torch.arange(6, dtype=torch.float32).view(3, 2),
        torch.full((3, 2), 10.0),
    ]

    hidden_states = speculator._prepare_context_hidden_states(
        torch.empty((3, 2)),
        aux_hidden_states,
        num_target_tokens=2,
    )

    packed = speculator.model.packed_hidden_states
    assert packed is not None
    assert packed.data_ptr() == speculator._target_hidden_states_workspace.data_ptr()
    torch.testing.assert_close(hidden_states, (aux_hidden_states[0] + 10)[:2])


@pytest.mark.cpu_test
def test_dspark_does_not_overwrite_aux_states_aliasing_workspace():
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    workspace = torch.arange(16, dtype=torch.float32)
    speculator.set_target_hidden_states_workspace(workspace)
    speculator.model = SimpleNamespace(combine_hidden_states=lambda x: x)
    aux = [workspace[4:8].view(2, 2), workspace[:4].view(2, 2)]
    expected = torch.cat(aux, dim=-1)
    result = speculator._prepare_context_hidden_states(torch.empty(2, 4), aux, 2)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(workspace, torch.arange(16, dtype=torch.float32))
