# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.glm5next.nvidia import model as glm5_model
from vllm.sequence import IntermediateTensors


class _DeferredMhcLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_state = None
        self.post_state = None

    def forward(self, positions, hidden_states, residual, post, comb):
        self.forward_state = (residual, post, comb)
        return (
            hidden_states + 1,
            hidden_states + 2,
            hidden_states + 3,
            hidden_states + 4,
        )

    def hc_post(self, hidden_states, residual, post, comb):
        self.post_state = (hidden_states, residual, post, comb)
        return hidden_states + residual + post + comb


def _make_stage(layer: nn.Module) -> glm5_model.Glm5NextModel:
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model._active_layers = nn.ModuleList([layer])
    model.is_sequence_parallel = False
    return model


def test_glm5next_pp_intermediate_tensor_shape():
    model = glm5_model.Glm5NextModel.__new__(glm5_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=8,
        mhc=True,
        mhc_num_residual_streams=4,
    )

    intermediate = model.make_empty_intermediate_tensors(
        batch_size=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert set(intermediate.tensors) == {"hidden_states"}
    assert intermediate["hidden_states"].shape == (3, 4, 8)
    assert intermediate["hidden_states"].dtype == torch.bfloat16


def test_glm5next_pp_materializes_deferred_mhc_state(monkeypatch):
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    layer = _DeferredMhcLayer()
    model = _make_stage(layer)
    hidden_states = torch.ones(2, 3)

    output = model(
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=None,
        inputs_embeds=hidden_states,
    )

    assert isinstance(output, IntermediateTensors)
    assert set(output.tensors) == {"hidden_states"}
    torch.testing.assert_close(output["hidden_states"], hidden_states * 4 + 10)
    assert layer.post_state is not None


def test_glm5next_pp_receiving_stage_starts_new_mhc_fusion(monkeypatch):
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    layer = _DeferredMhcLayer()
    model = _make_stage(layer)
    hidden_states = torch.ones(2, 4, 3)

    model(
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=IntermediateTensors({"hidden_states": hidden_states}),
    )

    assert layer.forward_state == (None, None, None)
