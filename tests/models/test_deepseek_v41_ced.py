# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED suffix retention and separation of global KV from bounded SWA replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.ced import CEDMetadata, SuffixBuffer
from vllm.models.deepseek_v4_1.pp_kv import SharedKVTransferPlan
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


@pytest.mark.parametrize("chunks", [[1024] * 32, [128, 2], [31, 31, 31, 31, 9], [17]])
def test_ced_suffix_survives_chunk_boundaries_and_reused_input_buffers(chunks):
    suffix = SuffixBuffer(128)
    consumed = 0
    for count in chunks:
        positions = torch.arange(consumed, consumed + count)
        saved = suffix.append("first", consumed, positions=positions)
        positions.zero_()
        consumed += count
        torch.testing.assert_close(
            saved["positions"], torch.arange(max(0, consumed - 128), consumed)
        )
        assert saved["positions"].untyped_storage().nbytes() <= 128 * 8
    saved = suffix.append("second", 0, positions=torch.arange(3))
    torch.testing.assert_close(saved["positions"], torch.arange(3))


def test_ced_suffix_rejects_missing_encoder_history():
    suffix = SuffixBuffer(128)
    with pytest.raises(ValueError, match="contiguous"):
        suffix.append("request", 1024, positions=torch.arange(128))


def test_ced_replay_limits_swa_without_shortening_global_context(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
    )
    state = CEDMetadata(config, torch.device("cpu"))
    state.groups = []
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        return {
            "decoder.swa": DeepseekSparseSWAMetadata(
                block_table=torch.zeros(1, 8, dtype=torch.int32),
                slot_mapping=kwargs["slot_mappings"][0],
                block_size=32,
                num_prefills=1,
                num_prefill_tokens=kwargs["num_tokens"],
                prefill_seq_lens=kwargs["seq_lens"],
                prefill_gather_lens=torch.tensor([255]),
            )
        }

    monkeypatch.setattr("vllm.models.deepseek_v4_1.ced.build_attn_metadata", build)
    for start, count in [(0, 1024), (1024, 2)]:
        positions = torch.arange(start, start + count)
        batch = SimpleNamespace(
            has_prefill=True,
            num_reqs=1,
            is_prefilling_np=np.array([True]),
            num_computed_prefill_tokens_np=np.array([start]),
            prefill_len_np=np.array([1026]),
            num_tokens=count,
            req_ids=["request"],
            positions=positions,
            input_ids=positions + 100,
            seq_lens=torch.tensor([start + count]),
            seq_lens_cpu_upper_bound=torch.tensor([start + count]),
        )
        step = state.prepare(batch, (), positions[None, :], [], None)
        if not start:
            assert not step.final and not captured
    assert len(captured) == 1
    assert captured[0]["num_tokens"] == 128
    assert captured[0]["max_seq_len"] == 1026
    torch.testing.assert_close(step.positions, torch.arange(898, 1026))
    torch.testing.assert_close(step.input_ids, torch.arange(998, 1126))
    torch.testing.assert_close(captured[0]["slot_mappings"][0], step.positions)
    metadata = step.replay_metadata["decoder.swa"]
    assert metadata.prefill_gather_lens.item() == 128
    assert metadata.prefill_seq_lens.item() == 1026


@pytest.mark.parametrize("prompt_logprobs", [None, 0, 1])
@pytest.mark.parametrize("with_image", [False, True])
def test_ced_preserves_prompt_logprobs_by_running_the_full_prompt(
    monkeypatch, prompt_logprobs, with_image
):
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    full_metadata = object()
    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", lambda *args: full_metadata)
    state = object.__new__(DeepseekV41ModelState)
    state.requires_eager_prefill = True
    state.ced_disabled_requests = set()
    state.ced_metadata = Mock()
    request = SimpleNamespace(
        req_id="request",
        sampling_params=SimpleNamespace(prompt_logprobs=prompt_logprobs),
        mm_features=[object()] if with_image else [],
    )
    state.add_request(0, request)
    batch = SimpleNamespace(req_ids=[request.req_id])
    assert state.prepare_attn(batch, None, None, None, None, None) is full_metadata
    if prompt_logprobs is None:
        assert state.ced_step is state.ced_metadata.prepare.return_value
    else:
        state.ced_metadata.prepare.assert_not_called()
        assert state.ced_step is None
    state.remove_request(request.req_id)
    request.sampling_params.prompt_logprobs = None
    state.add_request(0, request)
    state.prepare_attn(batch, None, None, None, None, None)
    assert state.ced_step is state.ced_metadata.prepare.return_value


def test_dspark_ced_restores_the_entire_suffix_before_a_short_final_chunk(monkeypatch):
    from vllm.models.deepseek_v4_1.ced import CEDDraftContext, CEDPrefill
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    ced = CEDPrefill(128)
    positions = torch.arange(898, 1026)
    auxiliary = [torch.full((128, 4), float(layer)) for layer in (37, 38, 39)]
    slots = torch.stack([positions + group * 2048 for group in range(6)])
    ced.draft_context = CEDDraftContext("request", positions, slots, auxiliary)
    events = []

    def insert(states, saved_positions, layer_slots):
        torch.testing.assert_close(states, torch.cat(auxiliary, dim=-1))
        torch.testing.assert_close(saved_positions, positions)
        for actual, expected_group in zip(layer_slots, (5, 2, 5)):
            torch.testing.assert_close(actual, slots[expected_group])
        events.append("suffix")

    def ordinary(self, batch, **kwargs):
        events.append("chunk")
        assert batch.num_tokens == 2
        return "draft"

    monkeypatch.setattr(DFlashSpeculator, "propose", ordinary)
    speculator = object.__new__(DSparkSpeculator)
    speculator._ced_context_source = ced.take_draft_context
    speculator.draft_kv_cache_group_ids = [2, 5]
    speculator._layer_group_idx = [1, 0, 1]
    speculator.model = SimpleNamespace(
        combine_hidden_states=lambda value: value,
        precompute_and_store_context_kv=insert,
    )
    batch = SimpleNamespace(req_ids=["request"], num_tokens=2)
    assert speculator.propose(batch, dummy_run=True) == "draft"
    assert ced.draft_context is not None
    events.clear()
    assert speculator.propose(batch) == "draft"
    assert events == ["suffix", "chunk"]
    assert ced.draft_context is None
    events.clear()
    speculator.propose(batch)
    assert events == ["chunk"]


def test_v41_auxiliary_pp_support_requires_all_sources_on_the_last_stage(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

    model = model_module.DeepseekV4Model.__new__(model_module.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=40)
    model.start_layer = 14
    pp = SimpleNamespace(world_size=3, is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: pp)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "8,6,26")
    model._set_aux_hidden_state_layers((37, 38, 39))
    assert model.supports_aux_hidden_states_over_pp
    assert model.aux_hidden_state_layers == (37, 38, 39)
    with pytest.raises(ValueError, match="last stage"):
        model._set_aux_hidden_state_layers((8, 37, 39))


@pytest.mark.parametrize("has_engram", [False, True])
@pytest.mark.parametrize("has_hashes", [False, True])
def test_v41_engram_injection_is_independent_of_pp_boundary(
    monkeypatch, has_engram, has_hashes
):
    """A PP boundary must preserve injection before the next attention pre-mix."""
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

    inputs = []

    def pre(stream, *args, **kwargs):
        inputs.append(stream.clone())
        return None, None, stream.mean(dim=1), kwargs.get("pre_mix")

    def post(x, residual, *args):
        return residual + x.unsqueeze(1)

    class Engram:
        layer_hash_index = 1

        def __call__(self, stream, hashes, mask):
            return stream + hashes.unsqueeze(-1) * mask[:, None, None]

    engram = Engram()
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", pre)
    monkeypatch.setattr(model_module, "mhc_post_tilelang", post)
    layer = SimpleNamespace(
        use_sequence_parallel=False,
        engram=engram if has_engram else None,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=20,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        ffn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        attn=lambda positions, x, unused: x,
        ffn=lambda x, ids: x,
    )
    residual = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    x = torch.full((2, 4), 0.25)
    stream = post(x, residual)
    hashes = torch.tensor([[[90.0], [3.0]], [[80.0], [7.0]]])
    mask = torch.tensor([True, False])
    kwargs = dict(
        positions=torch.arange(2),
        input_ids=None,
        pre_mix=torch.tensor([1.0]),
        engram_hashes=hashes if has_hashes else None,
        engram_mask=mask,
    )
    forward = model_module.DeepseekV4DecoderLayer.forward
    ordinary = forward(layer, x, residual=residual, **kwargs)
    boundary = forward(layer, stream, **kwargs)
    expected = (
        engram(stream, hashes[:, 1], mask) if has_engram and has_hashes else stream
    )
    torch.testing.assert_close(inputs[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(inputs[2], expected, atol=0, rtol=0)
    for a, b in zip(ordinary, boundary):
        if a is not None:
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["auto", "portable", "sm120_decode"])
def test_heterogeneous_pp_selects_attention_for_the_worker_device(monkeypatch, mode):
    from vllm.models.deepseek_v4_1.nvidia import model
    from vllm.platforms.interface import DeviceCapability

    devices = []

    def capability(device=0):
        devices.append(device)
        return DeviceCapability(12 if device == 3 else 8, 0)

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(model.current_platform, "get_device_capability", capability)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        additional_config={"dsv41_attention": mode},
    )
    selected = model._select_dsv4_attn_cls(config)
    assert devices == [3]
    assert selected.__name__ == (
        "DeepseekV41SM120DecodeAttention"
        if mode == "sm120_decode"
        else "DeepseekV41AmpereMLAAttention"
    )


def test_sm120_decode_padding_keeps_valid_lengths_and_shared_metadata(monkeypatch):
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41SM120DecodeAttention,
    )
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferSM120Attention,
    )

    attention = object.__new__(DeepseekV41SM120DecodeAttention)
    attention._decode_widths = (128, 512)
    indices = torch.arange(4 * 192).reshape(4, 1, 192).int()
    lengths = torch.tensor([128, 129, 130, 131], dtype=torch.int32)
    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.zeros(1, 8, dtype=torch.int32),
        slot_mapping=torch.arange(4),
        block_size=64,
        num_decodes=1,
        num_decode_tokens=4,
        decode_swa_indices=indices,
        decode_swa_lens=lengths,
    )
    seen = []

    def native(self, q, kv_cache, swa, attn, swa_only, output):
        seen.append(swa)
        assert swa.decode_swa_lens is lengths
        torch.testing.assert_close(swa.decode_swa_indices[..., :192], indices)
        assert (swa.decode_swa_indices[..., 192:] == -1).all()

    monkeypatch.setattr(DeepseekV4FlashInferSM120Attention, "_forward_decode", native)
    attention._forward_decode(None, None, metadata, None, False, None)
    assert len(seen) == 1 and seen[0] is not metadata
    assert metadata.decode_swa_indices is indices


def test_cpu_phase_threads_switch_only_after_pending_callbacks_finish(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    events: list[str | int] = []

    class Backend:
        def __init__(self):
            self.config = SimpleNamespace(num_threads=16)

        def set_num_threads(self, threads):
            assert events[-1] == "sync" or events[-1] == threads
            self.config.num_threads = threads
            events.append(threads)

    state = object.__new__(DeepseekV41ModelState)
    state.cpu_phase_threads = [16, 8]
    state.cpu_async_modules = [SimpleNamespace(backend=Backend()) for _ in range(2)]
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append("sync")),
    )
    state._set_cpu_phase_threads(True)
    assert events == []
    state._set_cpu_phase_threads(False)
    state._set_cpu_phase_threads(False)
    assert events == ["sync", 8, 8]
    state._set_cpu_phase_threads(True)
    assert events == ["sync", 8, 8, "sync", 16, 16]


@pytest.mark.parametrize("hybrid", [False, True])
def test_ced_publishes_encoder_kv_with_a_compatible_prefill_subclass(
    monkeypatch, hybrid
):
    from vllm.models.deepseek_v4_1 import ced
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
        DeepseekV41SM120DecodeAttention,
    )

    attention_cls = (
        DeepseekV41SM120DecodeAttention if hybrid else DeepseekV41AmpereMLAAttention
    )
    attention = object.__new__(attention_cls)
    layer = SimpleNamespace(
        attn=attention,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6),
    )
    model = SimpleNamespace(
        layers=[None] * 20 + [layer],
        config=SimpleNamespace(hidden_size=2),
        aux_hidden_state_layers=[],
        _mtp_hidden_buffer=None,
    )
    hidden = torch.zeros(3, 4, 2)
    published = []
    monkeypatch.setattr(ced, "mhc_post_tilelang", lambda *args: hidden)
    monkeypatch.setattr(
        ced,
        "mhc_pre_delayed_tilelang",
        lambda *args, **kwargs: (None, None, hidden.mean(1), None),
    )
    monkeypatch.setattr(ced, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        ced,
        "publish_decoder_kv",
        lambda attn, normalized, positions: published.append((attn, positions)),
    )
    replay = ced.CEDPrefill(128)
    step = ced.CEDStep(
        "request",
        0,
        3,
        False,
        requests=[ced.CEDRequest("request", 0, 0, 3, True, True, False)],
        source_positions=torch.arange(3),
    )
    output = replay.forward(model, step, hidden, None, None, None, torch.zeros(3, 4))
    assert output.shape == (3, 2) and not output.any()
    assert replay.encoder_tokens == 3 and replay.decoder_tokens == 0
    assert len(published) == 1 and published[0][0] is attention
    torch.testing.assert_close(published[0][1], torch.arange(3))


def _pp_kv_topology():
    return SimpleNamespace(
        num_hidden_layers=40,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
    )


@pytest.mark.parametrize(
    "start,end,incoming,outgoing",
    [
        (0, 4, None, 2),
        (4, 9, 2, 8),
        (9, 14, 8, None),
        (14, 40, None, None),
        (0, 2, None, None),
        (2, 8, None, None),
    ],
)
def test_transfer_plan_crosses_only_required_groups(start, end, incoming, outgoing):
    assert SharedKVTransferPlan.for_stage(
        _pp_kv_topology(), start, end
    ) == SharedKVTransferPlan(incoming, outgoing)


def test_transfer_plan_rejects_decoder_indexer_dependencies():
    with pytest.raises(ValueError, match="encoder group"):
        SharedKVTransferPlan.for_stage(_pp_kv_topology(), 20, 26)


@pytest.mark.parametrize("owns_vision", [True, False])
def test_checkpoint_stream_releases_tensors_before_finalizing_experts(owns_vision):
    """Interleaved text/vision shards must not retain the full checkpoint."""
    import weakref

    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV41LLMForCausalLM
    from vllm.models.deepseek_v4_1.nvidia.vl_model import DeepseekV41ForCausalLM

    finalized = []

    class LanguageModel(torch.nn.Module):
        load_weights = DeepseekV41LLMForCausalLM.load_weights

        def __init__(self):
            super().__init__()
            self.hf_to_vllm_mapper = WeightsMapper()
            self.blocks = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(1)) for _ in range(8)]
            )

        def process_weights_after_loading(self):
            assert [p.item() for p in self.blocks] == list(range(1, 9))
            finalized.append(True)

    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.language_model = LanguageModel()
    model.owns_vision = owns_vision
    model.vision = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.zeros(1)) for _ in range(8)]
    )
    model.hf_to_vllm_mapper = WeightsMapper()
    live: list[weakref.ReferenceType[torch.Tensor]] = []
    expected = set()

    def weights():
        for i in range(8):
            for prefix in ("language_model.blocks", "vision"):
                assert not finalized
                assert sum(ref() is not None for ref in live) <= 2
                name = f"{prefix}.{i}"
                if owns_vision or prefix.startswith("language_model"):
                    expected.add(name)
                tensor = torch.tensor([float(i + 1)])
                live.append(weakref.ref(tensor))
                yield name, tensor
        assert not finalized

    assert model.load_weights(weights()) == expected
    expected_vision = list(range(1, 9)) if owns_vision else [0] * 8
    assert [p.item() for p in model.vision] == expected_vision
    assert finalized == [True]
    assert all(ref() is None for ref in live)
    model.process_weights_after_loading()
    assert finalized == [True]


def test_hybrid_feedback_ignores_warmup_requests(monkeypatch):
    """Sampler warmup must not seed real-request cache history or output budgets."""
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    state = object.__new__(DeepseekV41ModelState)
    state.requires_eager_prefill = False
    state.hybrid_requests = {}
    state.hybrid_ready = False
    request = SimpleNamespace(
        req_id="warmup", sampling_params=SimpleNamespace(max_tokens=4)
    )
    state.add_request(0, request)
    assert not state.hybrid_requests
    state.hybrid_ready = True
    request = SimpleNamespace(
        req_id="real", sampling_params=SimpleNamespace(max_tokens=256)
    )
    state.add_request(0, request)
    assert state.hybrid_requests == {"real": 256}


def test_ced_interleaved_requests_keep_separate_suffixes_and_decode_rows(monkeypatch):
    """A mixed batch must select each request's own history and block table."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
    )
    state = CEDMetadata(config, torch.device("cpu"))
    state.groups = []
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        return {}

    monkeypatch.setattr("vllm.models.deepseek_v4_1.ced.build_attn_metadata", build)

    def prepare(rows, disabled=frozenset()):
        positions = torch.cat(
            [torch.arange(start, start + n) for _, start, n, _, _ in rows]
        )
        counts = [row[2] for row in rows]
        batch = SimpleNamespace(
            has_prefill=True,
            num_reqs=len(rows),
            req_ids=[row[0] for row in rows],
            is_prefilling_np=np.array([row[4] for row in rows]),
            num_computed_prefill_tokens_np=np.array([row[1] for row in rows]),
            prefill_len_np=np.array([row[3] for row in rows]),
            query_start_loc_np=np.cumsum([0, *counts]),
            num_tokens=sum(counts),
            positions=positions,
            input_ids=positions + 100,
            seq_lens=torch.tensor([row[1] + row[2] for row in rows]),
            seq_lens_cpu_upper_bound=torch.tensor([row[1] + row[2] for row in rows]),
        )
        tables = [torch.arange(len(rows))[:, None] + 1000]
        return state.prepare(batch, tables, positions[None], [], None, disabled)

    first = prepare([("a", 0, 128, 130, True), ("b", 0, 130, 136, True)])
    assert not first.final
    mixed = prepare(
        [
            ("decode", 40, 4, 30, False),
            ("b", 130, 3, 136, True),
            ("a", 128, 2, 130, True),
        ]
    )
    assert [r.replay_count for r in mixed.requests] == [4, 0, 128]
    torch.testing.assert_close(
        mixed.positions, torch.cat([torch.arange(40, 44), torch.arange(2, 130)])
    )
    assert captured[-1]["query_start_loc_cpu"].tolist() == [0, 4, 132]
    assert captured[-1]["block_tables"][0].flatten().tolist() == [1000, 1002]
    assert captured[-1]["is_prefilling"].tolist() == [False, True]
    assert set(state.suffixes) == {"b"}
    final = prepare([("b", 133, 3, 136, True)])
    torch.testing.assert_close(final.positions, torch.arange(8, 136))
    assert not state.suffixes
    prepare([("cancelled", 0, 32, 256, True)])
    state.remove_request("cancelled")
    assert not state.suffixes
    mixed_logprobs = prepare(
        [("ced", 0, 96, 128, True), ("logprobs", 0, 4, 4, True)],
        disabled={"logprobs"},
    )
    assert [r.replay_count for r in mixed_logprobs.requests] == [0, 4]
    assert mixed_logprobs.positions.tolist() == [0, 1, 2, 3]
    assert set(state.suffixes) == {"ced"}
    state.remove_request("ced")


def test_hybrid_keeps_feedback_until_all_interleaved_requests_finish(monkeypatch):
    """Removing one request must not reset another request's pending statistics."""
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    state = object.__new__(DeepseekV41ModelState)
    state.ced_disabled_requests = set()
    state.ced_metadata = None
    state.hybrid_requests = {"a": 256, "b": 256}
    state.hybrid_active = "a"
    state.hybrid_generated = 12
    state.hybrid_steps = 3
    state.hybrid_request_steps = 6
    cache = Mock()
    cache.finish_request.return_value = (7, 12, 3)
    state.cpu_async_modules = [SimpleNamespace(gpu_cache=cache)]
    state.remove_request("a")
    cache.finish_request.assert_not_called()
    assert state.hybrid_requests == {"b": 256}
    state.remove_request("b")
    cache.finish_request.assert_called_once_with(2.0)
    assert state.hybrid_active is None and state.hybrid_request_steps == 0


def test_pipeline_loader_skips_remote_experts_before_accessing_tensors(monkeypatch):
    """A stage must not inspect or report weights owned by a different stage."""
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)
    model = SimpleNamespace(
        named_parameters=lambda: (),
        get_expert_mapping=lambda: (),
        start_layer=4,
        end_layer=8,
        config=SimpleNamespace(num_attention_heads=1),
        quant_config=None,
        use_sequence_parallel=False,
    )
    weights = [
        (f"layers.{layer}.ffn.experts.0.w1.weight_scale", None) for layer in (3, 8)
    ]
    assert model_module.DeepseekV4Model.load_weights(model, weights) == set()
