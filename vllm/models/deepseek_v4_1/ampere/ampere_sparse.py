# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/A6000).

Reuses the ROCm Triton sparse-MLA implementation wholesale, exactly like the
V4 Ampere layer: its kernels, ragged metadata builders and bf16 o_proj
reference path are plain Triton/torch (the aiter-only preshuffle GEMMs and the
fused aiter norm+quant self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.
"""

from dataclasses import replace

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterMLAAttention,
)
from vllm.models.deepseek_v4_1.ampere.prefill_metadata import (
    combine_topk_swa_indices,
)
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.platforms.interface import DeviceCapability


class PrefillSubmission:
    """Keep at most one attention interval ahead of the previous checkpoint."""

    def __init__(self):
        self.events: dict[torch.cuda.Stream, torch.cuda.Event] = {}

    def wait(self):
        stream = torch.cuda.current_stream()
        event = self.events.get(stream)
        if event is None:
            event = self.events[stream] = torch.cuda.Event()
        else:
            event.synchronize()
        event.record(stream)


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in (8, 12)


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend

    @staticmethod
    def _combine_prefill_indices(*args, **kwargs):
        if not torch.cuda.is_current_stream_capturing():
            submission = None
            if is_forward_context_available():
                submission = get_forward_context().additional_kwargs.get(
                    "dsv41_prefill_submission"
                )
            # Bound live eager intermediates while overlapping host submission.
            if submission is not None:
                submission.wait()
            else:
                torch.cuda.current_stream().synchronize()
        return combine_topk_swa_indices(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The ROCm layer routes wo_a through the ordinary MXFP8 linear kernel
        # (is_bmm=False) because its bf16 einsum dequantizes the raw fp8 weight
        # itself. On CUDA that ordinary list is Marlin, whose repack would turn
        # the weight into packed garbage for the einsum. Keep the grouped-BMM
        # kernel list instead: on SM8x it resolves to the emulation kernel,
        # which dequantizes wo_a to a plain bf16 [g*r, d] weight at load time;
        # _get_cached_wo_a_bf16 then only views it (no second dequant).
        self.wo_a.is_bmm = True


class DeepseekV41SM120DecodeAttention(DeepseekV41AmpereMLAAttention):
    """Portable prefill with FlashInfer SM120 sparse decode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FlashInfer's SM120 decode specializes its primary cache to 64 rows.
        self.swa_cache_layer.block_size = 64
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config

        self._decode_widths = tuple(
            width
            for width in (128, 192, 256, 512, 1024)
            if has_flashinfer_sparse_mla_sm120_config(self.padded_heads, width)
        )
        if not self._decode_widths:
            raise RuntimeError(
                "FlashInfer has no compatible SM120 sparse decode kernel"
            )
        self._get_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    _get_workspace = staticmethod(DeepseekV4FlashInferSM120Attention._get_workspace)
    _as_sparse_cache = staticmethod(DeepseekV4FlashInferSM120Attention._as_sparse_cache)
    _prepare_query = DeepseekV4FlashInferSM120Attention._prepare_query

    def _forward_decode(
        self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
    ):
        indices = swa_metadata.decode_swa_indices
        width = indices.shape[-1]
        native_width = next((w for w in self._decode_widths if w >= width), None)
        if native_width is None:
            return super()._forward_decode(
                q, kv_cache, swa_metadata, attn_metadata, swa_only, output
            )
        if native_width != width:
            indices = torch.nn.functional.pad(
                indices, (0, native_width - width), value=-1
            )
            swa_metadata = replace(swa_metadata, decode_swa_indices=indices)
        DeepseekV4FlashInferSM120Attention._forward_decode(
            self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
        )
