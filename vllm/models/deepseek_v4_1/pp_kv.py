# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental compressed KV replicas at encoder pipeline boundaries."""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode


@dataclass(frozen=True)
class SharedKVTransferPlan:
    incoming_source: int | None
    outgoing_source: int | None

    @classmethod
    def for_stage(cls, config, start: int, end: int):
        sources = tuple(config.kv_source_layer_ids)
        index_sources = tuple(config.index_source_layer_ids)
        decoder_start = config.candidate_source_layer_id

        def crossing(boundary):
            if boundary in (0, config.num_hidden_layers) or boundary in sources:
                return None
            if not config.compress_ratios[boundary]:
                return None
            source = max(layer for layer in sources if layer < boundary)
            group_end = min(
                (layer for layer in sources if layer > source),
                default=config.num_hidden_layers,
            )
            if boundary >= decoder_start or any(
                source < layer < group_end for layer in index_sources
            ):
                raise ValueError(
                    "PP KV replicas require an encoder group with one index source"
                )
            return source

        return cls(crossing(start), crossing(end))


@triton.jit
def _copy_packed_kv_rows(
    cache,
    slots,
    rows,
    CACHE_STRIDE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    SCATTER: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    byte = tl.arange(0, 1024)
    slot = tl.load(slots + row, mask=row < NUM_SLOTS, other=-1).to(tl.int64)
    block = slot // PAGE_ROWS
    offset = slot % PAGE_ROWS
    # Each page stores all 576-byte payloads before its 8-byte scale rows.
    address = block * CACHE_STRIDE + tl.where(
        byte < 576,
        offset * 576 + byte,
        PAGE_ROWS * 576 + offset * 8 + byte - 576,
    )
    valid = (slot >= 0) & (byte < 584)
    if SCATTER:
        value = tl.load(rows + row * 584 + byte, mask=valid, other=0)
        tl.store(cache + address, value, mask=valid)
    else:
        value = tl.load(cache + address, mask=valid, other=0)
        tl.store(rows + row * 584 + byte, value, mask=byte < 584)


def pack_kv_rows(cache: torch.Tensor, slots: torch.Tensor, num_tokens: int):
    rows = torch.empty((num_tokens, 584), dtype=torch.uint8, device=cache.device)
    if num_tokens:
        _copy_packed_kv_rows[(num_tokens,)](
            cache,
            slots,
            rows,
            CACHE_STRIDE=cache.stride(0),
            PAGE_ROWS=cache.shape[1],
            NUM_SLOTS=slots.numel(),
            SCATTER=False,
            num_warps=4,
        )
    return rows


def scatter_kv_rows(rows: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor):
    if rows.shape[0]:
        _copy_packed_kv_rows[(rows.shape[0],)](
            cache,
            slots,
            rows,
            CACHE_STRIDE=cache.stride(0),
            PAGE_ROWS=cache.shape[1],
            NUM_SLOTS=slots.numel(),
            SCATTER=True,
            num_warps=4,
        )


class SharedKVReplica(nn.Module, AttentionLayerBase):
    def __init__(self, vllm_config: VllmConfig, prefix: str, source: int, backend):
        super().__init__()
        self.prefix = prefix
        self.source = source
        self.backend_cls = backend
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate KV replica: {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return self.backend_cls

    def bind_kv_cache(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_config
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=config.head_dim,
            dtype=torch.uint8,
            tokens_per_state=config.compress_ratios[self.source],
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
            state_content_bytes=584,
        )


class IncrementalPPKV(nn.Module):
    """Carry new cache rows and current indices in existing PP messages."""

    def __init__(self, vllm_config, prefix, start, end, attention_cls):
        super().__init__()
        if not attention_cls.use_fp8_ds_mla_layout:
            raise ValueError("PP KV replicas currently require fp8_ds_mla")
        config = vllm_config.model_config.hf_config
        self.plan = SharedKVTransferPlan.for_stage(config, start, end)
        self.index_topk = config.index_topk
        self._context = vllm_config.compilation_config.static_forward_context
        self.incoming_prefix = self.outgoing_prefix = None
        if self.plan.incoming_source is not None:
            source = self.plan.incoming_source
            self.incoming_prefix = f"{prefix}.layers.{source}.attn"
            self.replica = SharedKVReplica(
                vllm_config, self.incoming_prefix, source, attention_cls.backend_cls
            )
        if self.plan.outgoing_source is not None:
            source = self.plan.outgoing_source
            self.outgoing_prefix = f"{prefix}.layers.{source}.attn"

    def make_empty_inputs(self, num_tokens, device):
        if self.incoming_prefix is None:
            return {}
        return {
            "shared_kv_rows": torch.zeros(
                (num_tokens, 584), dtype=torch.uint8, device=device
            ),
            "shared_topk_indices": torch.zeros(
                (num_tokens, self.index_topk), dtype=torch.int32, device=device
            ),
        }

    def _metadata(self, prefix):
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ValueError("PP KV replicas do not support DBO")
        return metadata[prefix]

    def import_chunk(self, tensors, topk_indices_buffer):
        if self.incoming_prefix is None:
            return
        metadata = self._metadata(self.incoming_prefix)
        if metadata is None:
            return
        rows = tensors["shared_kv_rows"]
        cache = self._context[self.incoming_prefix].kv_cache
        scatter_kv_rows(rows, cache, metadata.slot_mapping)
        topk_indices_buffer[: rows.shape[0]].copy_(tensors["shared_topk_indices"])

    def export_chunk(self, num_tokens, topk_indices_buffer):
        if self.outgoing_prefix is None:
            return {}
        metadata = self._metadata(self.outgoing_prefix)
        if metadata is None:
            rows = torch.zeros(
                (num_tokens, 584), dtype=torch.uint8, device=topk_indices_buffer.device
            )
        else:
            cache = self._context[self.outgoing_prefix].kv_cache
            rows = pack_kv_rows(cache, metadata.slot_mapping, num_tokens)
        return {
            "shared_kv_rows": rows,
            # The next chunk can reuse the indexer's buffer before send completes.
            "shared_topk_indices": topk_indices_buffer[:num_tokens].clone(),
        }
