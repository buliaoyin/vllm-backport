# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


@pytest.mark.parametrize("mode", ["immediate", "deferred", "failed"])
def test_sampling_finishes_deferred_checks_after_draft_before_output(monkeypatch, mode):
    """A failed host completion cannot deliver sampled output or cache stats."""
    calls = []
    runner = GPUModelRunner.__new__(GPUModelRunner)
    batch = SimpleNamespace(
        req_ids=["request"], idx_mapping=torch.tensor([0]), query_start_loc=None
    )
    runner.execute_model_state = SimpleNamespace(
        input_batch=batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=torch.zeros(1, 1),
        aux_hidden_states=None,
        dp_sync=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        routed_experts=None,
        scheduler_step_id=0,
        num_spec_tokens_to_schedule=1,
        cudagraph_stats=None,
    )
    runner.is_last_pp_rank = True
    runner.pp_handler = runner.pcp_manager = runner.adaptive_verification = None
    runner.main_stream = runner.output_copy_stream = None
    runner.check_ep_fault = None
    runner._draft_workspace_lane = 0
    runner._begin_pp_recv_buffer_dspark = lambda: None
    runner.num_speculative_steps = 1
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    buffer = SimpleNamespace(gpu=None, np=None)
    runner.req_states = SimpleNamespace(
        all_token_ids=buffer,
        num_computed_tokens=buffer,
        prompt_len=buffer,
        last_sampled_tokens=None,
        next_prefill_tokens=None,
        draft_tokens=torch.zeros(1, 1, dtype=torch.int64),
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(temperature=buffer, seeds=buffer)
    )
    runner.sample = lambda *args: (SimpleNamespace(sampled_token_ids=None), None, None)
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {}
    )
    runner.model = SimpleNamespace(compute_logits=None)
    runner.kv_connector = SimpleNamespace(post_forward=lambda _: None)
    runner.eplb = SimpleNamespace(step=lambda **kwargs: calls.append("eplb"))

    def propose(*args, **kwargs):
        calls.append("draft")
        return torch.zeros(1, 1, dtype=torch.int64)

    def cache_stats():
        calls.append("stats")
        return "cache-stats"

    runner.speculator = SimpleNamespace(supports_mm_inputs=False, propose=propose)
    runner.model_state = SimpleNamespace(take_expert_cache_stats=cache_stats)
    monkeypatch.setattr(
        model_runner_module, "use_workspace_lane", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        model_runner_module, "AsyncOutput", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    def complete(output):
        assert output.model_runner_output.req_ids == ["request"]
        calls.append("complete")
        if mode == "failed":
            raise RuntimeError("callback failed")

    runner.postprocess_sampled = lambda *args: None if mode == "immediate" else complete
    if mode == "failed":
        with pytest.raises(RuntimeError, match="callback failed"):
            runner.sample_tokens(None)
        assert calls == ["draft", "complete"]
    else:
        output = runner.sample_tokens(None)
        assert output.model_runner_output.expert_cache_stats == "cache-stats"
        assert calls == (
            ["stats", "draft", "eplb"]
            if mode == "immediate"
            else ["draft", "complete", "stats", "eplb"]
        )


@pytest.mark.parametrize("with_drafter", [False, True])
def test_kv_cache_specs_carry_draft_ownership(monkeypatch, with_drafter):
    """The worker RPC must retain ownership for attention and QSA side caches."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.speculator = None
    if with_drafter:
        runner.speculator = DSparkSpeculator.__new__(DSparkSpeculator)
        runner.speculator.draft_attn_layer_names = {"b", "c", "uncached"}
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    specs = {"a": attention, "b": attention, "c": ring}
    monkeypatch.setattr(
        model_runner_module, "get_kv_cache_spec", lambda _: specs.copy()
    )

    result = runner.get_kv_cache_spec()

    assert {name for name, spec in result.items() if spec.is_draft} == (
        {"b", "c"} if with_drafter else set()
    )
    assert result == specs
    assert not attention.is_draft
    assert not ring.is_draft


@pytest.mark.parametrize("spec_kind", ["circular", "kpool_tail"])
def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch, spec_kind):
    """Ring-buffer caches (QSA circular buffer, GLM-5.3 kpool tail) hold one
    block per request and compute their own slot mapping; the generic
    position-indexed mapping would index far past their 1-block table row."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    if spec_kind == "circular":
        raw_spec = CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
    else:
        raw_spec = KpoolTailSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=8,
        )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


@pytest.mark.parametrize("with_draft", [False, True])
def test_auto_fit_updates_attention_capture_and_draft_limits(with_draft):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 1048576
    runner.req_states = SimpleNamespace(max_model_len=1048576)
    runner.model_state = SimpleNamespace(max_model_len=1048576)
    runner.speculator = None
    if with_draft:
        runner.speculator = DSparkSpeculator.__new__(DSparkSpeculator)
        runner.speculator.max_model_len = 1048576
        runner.speculator.draft_max_seq_len = 1048576

    runner.update_max_model_len(466432)

    assert runner.max_model_len == runner.req_states.max_model_len == 466432
    assert runner.model_state.max_model_len == 466432
    if with_draft:
        assert runner.speculator.max_model_len == 466432
        assert runner.speculator.draft_max_seq_len == 466432


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.parametrize("num_reqs", [1, 16])
def test_dummy_profile_preserves_requested_batch_shape(num_reqs):
    """Memory profiling must exercise one long query as well as many short ones."""
    from unittest.mock import Mock

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_num_reqs = 16
    runner.kv_connector = Mock()
    runner.is_first_pp_rank = True
    runner.lora_config = None
    runner.maybe_dummy_run_with_lora = lambda *args, **kwargs: contextlib.nullcontext()
    captured = []

    class ForwardReached(Exception):
        pass

    def forward(batch, *args, **kwargs):
        captured.append(batch.num_scheduled_tokens)
        raise ForwardReached

    runner.execute_model = forward
    with pytest.raises(ForwardReached):
        runner._dummy_run(2048, num_reqs=num_reqs, skip_eplb=True)
    assert list(captured[0].values()) == [2048 // num_reqs] * num_reqs
