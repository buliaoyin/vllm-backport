# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


@pytest.mark.parametrize(("hc_mult", "expected"), [(None, 64), (4, 256)])
def test_speculator_uses_draft_model_hidden_size(monkeypatch, hc_mult, expected):
    # Qwen4Exp targets expose multi-stream HC residuals to the drafter.
    monkeypatch.setattr(base_spec_module, "_target_feeds_hc_residual", lambda _: True)
    hf_config = SimpleNamespace()
    if hc_mult is not None:
        hf_config.hc_mult = hc_mult
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        get_hidden_size=lambda: 64,
        get_vocab_size=lambda: 32,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=3,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=32,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == expected


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        (
            "Draft model _TextOnlyDraftModel does not support external multimodal "
            "embeddings. Embeddings from the target model will not be passed to the "
            "drafter; using text-only draft inputs instead."
        )
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("max_drafts", [3, 5])
@pytest.mark.parametrize("record_confidence", [False, True])
def test_adaptive_mtp_changes_work_and_clears_unused_drafts(
    monkeypatch, fused, max_drafts, record_confidence
):
    """Switching K must skip model calls and never return a previous draft tail."""
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    spec = object.__new__(MTPSpeculator)
    spec.num_speculative_steps = max_drafts
    spec.dynamic_draft = True
    spec._decode_managers = {}
    spec.max_model_len = 32
    spec.max_num_reqs = 1
    spec.dp_size, spec.dp_rank = 1, 0
    spec.hidden_states = torch.zeros(2, 3)
    spec.draft_tokens = torch.full((1, max_drafts), 999, dtype=torch.int64)
    spec.record_scheduler_confidence = record_confidence
    spec.draft_token_confidence_probs = torch.full((1, max_drafts), 0.99)
    spec.last_token_indices = torch.zeros(1, dtype=torch.int64)
    spec.idx_mapping = torch.zeros(1, dtype=torch.int64)
    spec.current_draft_step = torch.tensor(0)
    spec.sample_src_positions = torch.zeros(1, dtype=torch.int64)
    spec.input_buffers = SimpleNamespace(
        positions=torch.zeros(1), query_start_loc=torch.tensor([0, 1])
    )
    spec.use_fused_multi_step_decode = fused
    spec.prefill_cudagraph_manager = spec.decode_cudagraph_manager = None
    spec._copy_request_inputs = spec._prepare_eplb_forward = lambda *args: None
    spec.share_mtp_topk_indices = True
    hooks = SimpleNamespace(set_skip_topk=Mock(), compact_topk_indices=Mock())
    confidence_head = Mock(
        side_effect=lambda hidden: (
            hidden[:, 0].long() + 100,
            torch.full((hidden.shape[0],), 0.5),
        )
    )
    greedy_head = Mock(side_effect=lambda hidden: hidden[:, 0].long() + 100)
    spec.use_local_argmax_reduction = True
    spec.model = SimpleNamespace(
        model=hooks,
        get_top_tokens_with_confidence=confidence_head,
        get_top_tokens=greedy_head,
    )
    steps = []

    def generate(step):
        steps.append(step)
        spec.draft_tokens[0, step] = 100 + step
        if record_confidence:
            tokens = spec.sample_draft(
                torch.full((1, 3), float(step)),
                spec.sample_src_positions,
                spec.idx_mapping,
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int64),
                torch.tensor(step),
                None,
            )
            spec.draft_tokens[0, step] = tokens[0]

    spec._prefill = lambda *args, **kwargs: generate(0)
    spec._generate_draft = lambda *args, **kwargs: generate(
        spec.current_draft_step.item()
    )
    for name in ("prepare_prefill_inputs", "prepare_decode_inputs"):
        monkeypatch.setattr(spec_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        spec_module,
        "dispatch_cg_and_sync_dp",
        lambda manager, reqs, tokens, *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.NONE, tokens, reqs),
            None,
        ),
    )
    batch = SimpleNamespace(
        num_tokens=2,
        num_tokens_after_padding=2,
        num_reqs=1,
        num_scheduled_tokens=torch.tensor([2]),
        seq_lens_cpu_upper_bound=torch.tensor([4]),
        seq_lens=torch.tensor([4]),
        has_prefill=False,
        idx_mapping=spec.idx_mapping,
    )
    for budget in (max_drafts, 1, 2, max_drafts - 1, max_drafts, 1):
        spec.set_draft_budget(budget)
        steps.clear()
        hooks.set_skip_topk.reset_mock()
        hooks.compact_topk_indices.reset_mock()
        confidence_head.reset_mock()
        greedy_head.reset_mock()
        result = spec.propose(
            batch,
            {},
            {},
            torch.ones(2, 3),
            None,
            *[torch.ones(1, dtype=torch.int64) for _ in range(6)],
            skip_attn_for_dummy_run=True,
            dummy_run=True,
        )
        assert steps == list(range(budget))
        assert result.tolist() == [
            list(range(100, 100 + budget)) + [-1] * (max_drafts - budget)
        ]
        if record_confidence:
            assert confidence_head.call_count == 1
            assert greedy_head.call_count == budget - 1
            torch.testing.assert_close(
                spec.draft_token_confidence_probs[0, 0], torch.tensor(0.5)
            )
            assert torch.isnan(spec.draft_token_confidence_probs[0, 1:]).all()
            assert not spec._collect_scheduler_confidence
        assert hooks.compact_topk_indices.call_count == int(budget > 1)
        assert [call.args[0] for call in hooks.set_skip_topk.call_args_list] == (
            [False, True, False] if budget > 1 else [False]
        )
    for invalid in (0, max_drafts + 1):
        with pytest.raises(ValueError, match="budget"):
            spec.set_draft_budget(invalid)


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None


@pytest.mark.parametrize(
    "field,value,expected",
    [
        (None, None, True),
        ("additional_config", {}, False),
        ("additional_config", None, False),
        ("additional_config", {"mtp_confidence_forecast": False}, False),
        ("quantization", "exl3", False),
        ("pipeline_parallel_size", 1, False),
        ("tensor_parallel_size", 2, False),
        ("model_type", "qwen3_5_text", False),
        ("enable_adaptive_verification", False, False),
        ("use_local_argmax_reduction", False, False),
        ("draft_sample_method", "probabilistic", False),
    ],
)
def test_mtp_scheduler_confidence_scope(monkeypatch, field, value, expected):
    """Confidence work must be restricted to the measured Qwen NVFP4 PP2 path."""
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    monkeypatch.setattr(base_spec_module, "_target_feeds_hc_residual", lambda _: False)
    draft_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp_mtp", architectures=["Qwen4ExpMTP"], n_predict=1
        ),
        get_hidden_size=lambda: 4,
        get_vocab_size=lambda: 16,
    )
    spec_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=4,
        draft_model_config=draft_config,
        use_local_argmax_reduction=True,
        draft_sample_method="greedy",
        enable_adaptive_verification=True,
    )
    model = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen4_exp_text"),
        quantization="modelopt_fp4",
        max_model_len=32,
        head_dtype=torch.float32,
        dtype=torch.float32,
        use_fp64_gumbel=False,
    )
    parallel = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )
    if field == "quantization":
        model.quantization = value
    elif field == "model_type":
        model.hf_text_config.model_type = value
    elif field in ("pipeline_parallel_size", "tensor_parallel_size"):
        setattr(parallel, field, value)
    elif field is not None and field != "additional_config":
        setattr(spec_config, field, value)
    config = SimpleNamespace(
        speculative_config=spec_config,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8),
        model_config=model,
        parallel_config=parallel,
        additional_config=(
            value if field == "additional_config" else {"mtp_confidence_forecast": True}
        ),
    )
    speculator = MTPSpeculator(config, torch.device("cpu"))
    assert speculator.record_scheduler_confidence is expected
    if expected:
        assert speculator.scheduler_confidence_table.shape == (2, 4)
        assert torch.isnan(speculator.scheduler_confidence_table).all()
        assert torch.isnan(speculator.draft_token_confidence_probs).all()
    else:
        assert speculator.scheduler_confidence_table is None
        assert speculator.draft_token_confidence_probs is None


def test_mtp_confidence_capture_separates_prefill_and_decode_work():
    """Prefill capture records one feature head; decode captures only greedy work."""
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    spec = object.__new__(MTPSpeculator)
    spec.record_scheduler_confidence = True
    spec.draft_token_confidence_probs = torch.full((1, 4), 0.99)
    spec.use_local_argmax_reduction = True
    spec.dynamic_draft = True
    spec.num_speculative_steps = 4
    spec.max_num_reqs = 1
    spec.share_mtp_topk_indices = False
    spec._decode_managers = {}
    spec.use_fused_multi_step_decode = True
    spec.last_token_indices = torch.zeros(1, dtype=torch.int64)
    spec.idx_mapping = torch.zeros(1, dtype=torch.int64)
    tokens = torch.tensor([7])
    feature_head = Mock(return_value=(tokens, torch.tensor([0.8])))
    greedy_head = Mock(return_value=tokens)
    spec.model = SimpleNamespace(
        get_top_tokens_with_confidence=feature_head, get_top_tokens=greedy_head
    )
    phases = []

    def sample(step):
        phases.append(spec._collect_scheduler_confidence)
        assert (
            spec.sample_draft(
                torch.ones(1, 4),
                torch.zeros(1),
                spec.idx_mapping,
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int64),
                torch.tensor(step),
                None,
            )
            is tokens
        )

    spec._prefill = lambda *args, **kwargs: sample(0)
    spec._generate_fused_drafts = lambda *args, **kwargs: [
        sample(step) for step in range(1, 4)
    ]
    manager = SimpleNamespace(
        use_breakable_cg=False,
        capture=Mock(side_effect=lambda fn, *args, **kwargs: fn()),
    )
    spec.prefill_cudagraph_manager = spec.decode_cudagraph_manager = manager
    for name in (
        "model_state",
        "target_input_buffers",
        "block_tables",
        "target_attn_groups",
        "kv_cache_config",
        "input_buffers",
        "attn_groups",
    ):
        setattr(spec, name, None)
    spec.capture()
    assert phases == [True, False, False, False]
    assert feature_head.call_count == 1
    assert greedy_head.call_count == 3
    assert not spec._collect_scheduler_confidence
    torch.testing.assert_close(
        spec.draft_token_confidence_probs[0, 0], torch.tensor(0.8)
    )
    assert torch.isnan(spec.draft_token_confidence_probs[0, 1:]).all()


def test_mtp_confidence_excludes_stochastic_and_padded_rows():
    """Target sampling mode must not contaminate greedy acceptance calibration."""
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    spec = object.__new__(MTPSpeculator)
    spec.record_scheduler_confidence = True
    spec._collect_scheduler_confidence = True
    spec.draft_token_confidence_probs = torch.full((3, 4), float("nan"))
    tokens = torch.tensor([4, 5, 6])
    spec.model = SimpleNamespace(
        get_top_tokens_with_confidence=Mock(
            return_value=(tokens, torch.tensor([0.6, 0.7, 0.8]))
        )
    )
    sampled = spec.sample_draft(
        torch.ones(3, 4),
        torch.zeros(3),
        torch.tensor([2, 0, -1]),
        torch.tensor([0.8, 0.0, 0.0]),
        torch.zeros(3, dtype=torch.int64),
        torch.tensor(0),
        None,
    )
    assert sampled is tokens
    torch.testing.assert_close(
        spec.draft_token_confidence_probs[0, 0], torch.tensor(0.6)
    )
    assert torch.isnan(spec.draft_token_confidence_probs[1:]).all()
    assert torch.isnan(spec.draft_token_confidence_probs[0, 1:]).all()
