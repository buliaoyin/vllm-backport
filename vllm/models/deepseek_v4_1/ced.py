# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-isolated CED prefill with bounded decoder SWA replay."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup


def ced_prefill_enabled(config: VllmConfig) -> bool:
    extra = config.additional_config
    enabled = isinstance(extra, dict) and bool(extra.get("ced_prefill", False))
    if not enabled:
        return False
    if not config.use_v2_model_runner:
        raise ValueError("CED prefill requires the v2 model runner")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.use_ubatching:
        raise ValueError("CED prefill currently requires TP1 without microbatches")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("CED prefill currently requires prefix caching disabled")
    if (
        config.speculative_config is not None
        and config.speculative_config.method != "dspark"
    ):
        raise ValueError("CED prefill currently supports only DSpark speculation")
    return True


class SuffixBuffer:
    """Keep a suffix across arbitrary chunk boundaries without retaining the chunk."""

    def __init__(self, width: int):
        self.width = width
        self.request_id: str | None = None
        self.next_position = 0
        self.tensors: dict[str, torch.Tensor] = {}

    def append(self, request_id: str, start: int, **tensors) -> dict[str, torch.Tensor]:
        if start == 0:
            self.request_id = request_id
            self.next_position = 0
            self.tensors = {}
        if request_id != self.request_id or start != self.next_position:
            raise ValueError("CED prefill chunks must be contiguous from position zero")
        lengths = {value.shape[0] for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("CED suffix tensors have different token counts")
        count = lengths.pop()
        for name, value in tensors.items():
            if count < self.width and name in self.tensors:
                value = torch.cat((self.tensors[name], value))
            self.tensors[name] = value[-self.width :].clone()
        self.next_position += count
        return self.tensors


@dataclass
class CEDRequest:
    request_id: str
    start: int
    offset: int
    num_tokens: int
    is_prefilling: bool
    use_ced: bool
    final: bool
    replay_count: int = 0


@dataclass
class CEDStep:
    request_id: str
    start: int
    num_tokens: int
    final: bool
    replay_metadata: dict[str, Any] | None = None
    positions: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    replay_slots: torch.Tensor | None = None
    requests: list[CEDRequest] | None = None
    source_positions: torch.Tensor | None = None


@dataclass
class CEDDraftContext:
    request_id: str
    positions: torch.Tensor
    slots: torch.Tensor
    auxiliary: list[torch.Tensor]
    request_ids: tuple[str, ...] = ()


class CEDMetadata:
    def __init__(self, config: VllmConfig, device: torch.device):
        self.config = config
        self.device = device
        self.width = config.model_config.hf_config.sliding_window
        self.suffixes: dict[str, SuffixBuffer] = {}
        self.groups: list[list[AttentionGroup]] | None = None

    def remove_request(self, request_id):
        self.suffixes.pop(request_id, None)

    def prepare(
        self,
        batch,
        block_tables,
        slot_mappings,
        groups,
        kv_config,
        disabled_requests=frozenset(),
    ):
        if not batch.has_prefill:
            return None
        requests = []
        selected = []
        suffixes = []
        starts = getattr(batch, "query_start_loc_np", [0, batch.num_tokens])
        for index, request_id in enumerate(batch.req_ids):
            offset, end = int(starts[index]), int(starts[index + 1])
            start = int(batch.num_computed_prefill_tokens_np[index])
            prefilling = bool(batch.is_prefilling_np[index])
            use_ced = prefilling and request_id not in disabled_requests
            final = start + end - offset == int(batch.prefill_len_np[index])
            request = CEDRequest(
                request_id, start, offset, end - offset, prefilling, use_ced, final
            )
            values = dict(
                slots=slot_mappings[:, offset:end].T,
                positions=batch.positions[offset:end],
                input_ids=batch.input_ids[offset:end],
            )
            if use_ced:
                buffer = self.suffixes.setdefault(request_id, SuffixBuffer(self.width))
                values = buffer.append(request_id, start, **values)
            if not use_ced or final:
                request.replay_count = values["positions"].shape[0]
                selected.append(index)
                suffixes.append(values)
                self.suffixes.pop(request_id, None)
            requests.append(request)
        first = requests[0]
        step = CEDStep(
            first.request_id,
            first.start,
            batch.num_tokens,
            bool(selected),
            requests=requests,
            source_positions=batch.positions,
        )
        if not selected:
            return step
        if self.groups is None:
            self.groups = []
            for collection in groups:
                cloned = []
                for group in collection:
                    names = [
                        name
                        for name in group.layer_names
                        if 20 <= extract_layer_index(name) < 40
                    ]
                    if not names:
                        continue
                    new = AttentionGroup(
                        group.backend,
                        names,
                        group.kv_cache_spec,
                        group.kv_cache_group_id,
                    )
                    new.create_metadata_builders(
                        self.config,
                        self.device,
                        kernel_block_size=group.get_metadata_builder().kernel_block_size,
                    )
                    cloned.append(new)
                self.groups.append(cloned)
        suffix = {
            name: torch.cat([values[name] for values in suffixes])
            for name in ("positions", "input_ids", "slots")
        }
        active = [requests[index] for index in selected]
        lengths = [request.replay_count for request in active]
        query_cpu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int()
        replay_count = sum(lengths)
        indices = torch.tensor(selected, dtype=torch.long, device=self.device)
        seq_lens = batch.seq_lens.index_select(0, indices)
        seq_cpu = batch.seq_lens_cpu_upper_bound[selected]
        step.replay_metadata = build_attn_metadata(
            attn_groups=self.groups,
            num_reqs=len(selected),
            num_tokens=replay_count,
            query_start_loc_gpu=query_cpu.to(self.device),
            query_start_loc_cpu=query_cpu,
            max_query_len=max(lengths),
            seq_lens=seq_lens,
            max_seq_len=int(seq_cpu.max()),
            block_tables=[table.index_select(0, indices) for table in block_tables],
            slot_mappings=suffix["slots"].T.contiguous(),
            kv_cache_config=kv_config,
            seq_lens_cpu_upper_bound=seq_cpu,
            positions=suffix["positions"],
            is_prefilling=torch.tensor([r.is_prefilling for r in active]),
        )
        for metadata in step.replay_metadata.values():
            if (
                isinstance(metadata, DeepseekSparseSWAMetadata)
                and metadata.prefill_gather_lens is not None
            ):
                bounds = [
                    r.replay_count if r.use_ced else int(seq_cpu[i])
                    for i, r in enumerate(active)
                    if r.is_prefilling
                ]
                metadata.prefill_gather_lens = torch.minimum(
                    metadata.prefill_gather_lens,
                    metadata.prefill_gather_lens.new_tensor(bounds),
                )
        step.positions = suffix["positions"]
        step.input_ids = suffix["input_ids"]
        step.replay_slots = suffix["slots"].T.contiguous()
        return step


def publish_decoder_kv(attn, normalized, positions):
    compressor = attn.compressor
    kv_score = torch.mm(
        normalized, compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32
    )
    latent = compressor(kv_score, positions)
    compressor.insert_cache(latent, positions, attn.rotary_emb)
    attn.indexer._produce_k(latent, positions, attn.indexer_rotary_emb)
    return latent


class CEDPrefill:
    def __init__(self, width: int):
        self.width = width
        self.suffixes: dict[str, SuffixBuffer] = {}
        self.encoder_tokens = 0
        self.decoder_tokens = 0
        self.replay_calls = 0
        self.draft_context: CEDDraftContext | None = None

    def remove_request(self, request_id):
        self.suffixes.pop(request_id, None)

    def take_draft_context(self):
        context, self.draft_context = self.draft_context, None
        return context

    def forward(self, model, step, hidden, residual, post_mix, res_mix, pre_mix):
        physical_tokens = hidden.shape[0]
        self.draft_context = None
        encoder_hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
        layer = model.layers[20]
        from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
            DeepseekV41AmpereMLAAttention,
        )

        if not isinstance(layer.attn, DeepseekV41AmpereMLAAttention):
            raise ValueError(
                "CED replay currently requires the Ampere attention backend"
            )
        count = step.num_tokens
        _, _, normalized, _ = mhc_pre_delayed_tilelang(
            encoder_hidden,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            norm_weight=layer.attn_norm.weight,
            norm_eps=layer.attn_norm.variance_epsilon,
        )
        attn = layer.attn
        context = get_forward_context()
        # The source at layer 20 projects every encoder token into the decoder's
        # global main/index KV. It performs no decoder attention or expert work.
        saved_hidden = []
        saved_mix = []
        assert step.requests is not None
        for request in step.requests:
            begin = request.offset
            end = begin + request.num_tokens
            values = dict(hidden=encoder_hidden[begin:end], pre_mix=pre_mix[begin:end])
            if request.use_ced:
                suffix = self.suffixes.setdefault(
                    request.request_id, SuffixBuffer(self.width)
                )
                values = suffix.append(request.request_id, request.start, **values)
            if request.replay_count:
                saved_hidden.append(values["hidden"])
                saved_mix.append(values["pre_mix"])
                self.suffixes.pop(request.request_id, None)
        positions = step.source_positions
        publish_decoder_kv(attn, normalized, positions)
        self.encoder_tokens += count
        output = torch.zeros(
            (physical_tokens, model.config.hidden_size),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        auxiliary_output = [
            torch.zeros_like(output) for _ in model.aux_hidden_state_layers
        ]
        if model._mtp_hidden_buffer is not None:
            model._mtp_hidden_buffer[:physical_tokens].zero_()
        if not step.final:
            return (output, auxiliary_output) if auxiliary_output else output
        original_metadata = context.attn_metadata
        original_slots = context.slot_mapping
        original_padding = context.is_padding
        try:
            context.attn_metadata = step.replay_metadata
            context.slot_mapping = {
                name: metadata.slot_mapping
                for name, metadata in step.replay_metadata.items()
                if hasattr(metadata, "slot_mapping")
            }
            context.is_padding = None
            hidden = torch.cat(saved_hidden)
            pre_mix = torch.cat(saved_mix)
            residual = post_mix = res_mix = None
            auxiliary = []
            for index, layer in enumerate(model.layers[20:40], start=20):
                hidden, residual, post_mix, res_mix, pre_mix = layer(
                    hidden,
                    step.positions,
                    step.input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                )
                if index + 1 in model.aux_hidden_state_layers:
                    auxiliary.append(
                        mhc_post_tilelang(hidden, residual, post_mix, res_mix).mean(
                            dim=1
                        )
                    )
            hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
            collapsed = model.norm(hc_collapse_triton(hidden, pre_mix))
            replay_offset = 0
            for request in step.requests:
                if not request.replay_count:
                    continue
                replay_end = replay_offset + request.replay_count
                returned = min(request.num_tokens, request.replay_count)
                dest_end = request.offset + request.num_tokens
                destination = slice(dest_end - returned, dest_end)
                source = slice(replay_end - returned, replay_end)
                output[destination].copy_(collapsed[source])
                if model._mtp_hidden_buffer is not None:
                    model._mtp_hidden_buffer[destination].copy_(
                        hidden[source].flatten(1)
                    )
                for padded, value in zip(auxiliary_output, auxiliary):
                    padded[destination].copy_(value[source])
                replay_offset = replay_end
            self.decoder_tokens += hidden.shape[0]
            self.replay_calls += 1
            if auxiliary_output:
                if len(auxiliary) != len(auxiliary_output):
                    raise ValueError(
                        "CED auxiliary states must come from decoder layers"
                    )
                self.draft_context = CEDDraftContext(
                    step.request_id,
                    step.positions,
                    step.replay_slots,
                    auxiliary,
                    tuple(r.request_id for r in step.requests if r.replay_count),
                )
                return output, auxiliary_output
            return output
        finally:
            context.attn_metadata = original_metadata
            context.slot_mapping = original_slots
            context.is_padding = original_padding
