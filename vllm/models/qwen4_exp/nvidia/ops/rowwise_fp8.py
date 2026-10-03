# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional SM120 weight-only FP8 projections for Qwen4Exp."""

from collections.abc import Callable
from functools import wraps

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_CHUNK_BYTES = 64 * 1024 * 1024


def rowwise_fp8_head_enabled(vllm_config: VllmConfig) -> bool:
    return _rowwise_fp8_enabled(vllm_config, "sm120_rowwise_fp8_head", head=True)


def rowwise_fp8_hc_enabled(vllm_config: VllmConfig) -> bool:
    return _rowwise_fp8_enabled(vllm_config, "sm120_rowwise_fp8_hc", head=False)


def rowwise_fp8_output_enabled(vllm_config: VllmConfig) -> bool:
    return _rowwise_fp8_enabled(vllm_config, "sm120_rowwise_fp8_output", head=False)


def _rowwise_fp8_enabled(vllm_config: VllmConfig, flag: str, *, head: bool) -> bool:
    model = vllm_config.model_config
    text = model.hf_text_config
    requested = getattr(
        text,
        flag,
        getattr(model.hf_config, flag, False),
    )
    if requested is False:
        return False
    if requested is not True:
        raise ValueError(f"{flag} must be a boolean")
    parallel = vllm_config.parallel_config
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability(120)
        or model.dtype != torch.bfloat16
        or (head and model.head_dtype != torch.bfloat16)
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or (head and text.tie_word_embeddings)
        or vllm_config.lora_config is not None
    ):
        raise ValueError(
            f"{flag} requires SM120, BF16 model/head, TP1/PP1, "
            "untied embeddings, and no LoRA"
        )
    return True


def quantize_rowwise_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("rowwise FP8 head requires a 2-D BF16 weight")
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    rows = max(1, _CHUNK_BYTES // (4 * weight.shape[1]))
    for start in range(0, weight.shape[0], rows):
        block = weight[start : start + rows].float()
        scale = block.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 448.0
        quantized[start : start + rows] = (
            (block / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        scales[start : start + rows] = scale.squeeze(-1)
    return quantized, scales


# Adapted from SGLang's Apache-2.0 sm120_online_fp8.py at 12846e83153f.
@triton.jit
def _rowwise_fp8_projection(
    x_ptr,
    w_ptr,
    s_ptr,
    bf16_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    sxm,
    sxk,
    swn,
    swk,
    BF16_START: tl.constexpr,
    BF16_ROWS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    split = tl.program_id(1)
    split_width = tl.cdiv(K, BLOCK_K * SPLIT_K) * BLOCK_K
    for start in range(0, split_width, BLOCK_K):
        rk = split * split_width + start + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rm[:, None] * sxm + rk[None, :] * sxk,
            (rm[:, None] < M) & (rk[None, :] < K),
            0.0,
        )
        w = tl.load(
            w_ptr + rn[:, None] * swn + rk[None, :] * swk,
            (rn[:, None] < N) & (rk[None, :] < K),
            0.0,
        ).to(tl.bfloat16)
        if BF16_ROWS:
            bf16 = tl.load(
                bf16_ptr + (rn[:, None] - BF16_START) * K + rk[None, :],
                (rn[:, None] >= BF16_START)
                & (rn[:, None] < BF16_START + BF16_ROWS)
                & (rk[None, :] < K),
                0.0,
            )
            w = tl.where(
                (rn[:, None] >= BF16_START) & (rn[:, None] < BF16_START + BF16_ROWS),
                bf16,
                w,
            )
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    if SPLIT_K == 1:
        scale = tl.load(s_ptr + rn, rn < N, 0)
        acc *= scale[None, :]
    tl.store(
        out_ptr + (split * M + rm[:, None]) * N + rn[None, :],
        acc.to(out_ptr.dtype.element_ty),
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _rowwise_fp8_reduce(
    partial_ptr,
    scale_ptr,
    output_ptr,
    MN,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT_K):
        total += tl.load(partial_ptr + split * MN + r, r < MN, 0.0)
    scale = tl.load(scale_ptr + r % N, r < MN, 0.0)
    tl.store(output_ptr + r, total * scale, r < MN)


@triton.jit
def _rowwise_fp8_simt(
    x_ptr,
    weight_ptr,
    scale_ptr,
    bf16_ptr,
    output_ptr,
    M,
    N,
    K: tl.constexpr,
    SXM,
    SXK,
    BF16_START: tl.constexpr,
    BF16_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    weight = tl.load(weight_ptr + n * K + k, k < K, 0.0).to(tl.float32)
    if BF16_ROWS:
        bf16 = tl.load(
            bf16_ptr + (n - BF16_START) * K + k,
            (n >= BF16_START) & (n < BF16_START + BF16_ROWS) & (k < K),
            0.0,
        ).to(tl.float32)
        weight = tl.where(
            (n >= BF16_START) & (n < BF16_START + BF16_ROWS), bf16, weight
        )
    scale = tl.load(scale_ptr + n)
    for m in range(BLOCK_M):
        x = tl.load(x_ptr + m * SXM + k * SXK, (m < M) & (k < K), 0.0).to(tl.float32)
        total = tl.sum(x * weight, 0)
        tl.store(output_ptr + m * N + n, total * scale, m < M)


def rowwise_fp8_logits(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bf16_weight: torch.Tensor | None = None,
    bf16_start: int = 0,
) -> torch.Tensor:
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or weight.ndim != 2
        or scale.shape != weight.shape[:1]
        or x.shape[-1] != weight.shape[1]
        or x.device != weight.device
        or scale.device != weight.device
    ):
        raise ValueError("rowwise FP8 head input/weight/scale mismatch")
    bf16_rows = 0 if bf16_weight is None else bf16_weight.shape[0]
    if bf16_weight is not None and (
        bf16_weight.dtype != torch.bfloat16
        or bf16_weight.ndim != 2
        or bf16_weight.shape[1] != weight.shape[1]
        or bf16_weight.device != weight.device
        or not bf16_weight.is_contiguous()
        or bf16_start < 0
        or bf16_start + bf16_rows > weight.shape[0]
    ):
        raise ValueError("rowwise FP8 BF16 rows do not match the weight")
    flat = x.reshape(-1, x.shape[-1])
    m, n = flat.shape[0], weight.shape[0]
    output = x.new_empty((m, n))
    if m == 0:
        return output.reshape(*x.shape[:-1], n)
    hc_up_prefill = bf16_weight is not None and m > 32 and weight.shape == (10240, 320)
    if x.is_cuda and (m <= 32 or hc_up_prefill):
        if n <= 512 and weight.shape[1] >= 8192 and m == 1 and weight.is_contiguous():
            _rowwise_fp8_simt[(n,)](
                flat,
                weight,
                scale,
                x if bf16_weight is None else bf16_weight,
                output,
                m,
                n,
                weight.shape[1],
                flat.stride(0),
                flat.stride(1),
                BF16_START=bf16_start,
                BF16_ROWS=bf16_rows,
                BLOCK_M=triton.next_power_of_2(m),
                BLOCK_K=triton.next_power_of_2(weight.shape[1]),
                num_warps=16,
            )
            return output.reshape(*x.shape[:-1], n)
        split_k = 16 if n <= 512 and weight.shape[1] >= 8192 else 1
        partials = (
            torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
            if split_k > 1
            else output
        )
        block_m = 64 if hc_up_prefill else max(16, triton.next_power_of_2(m))
        block_n = 128 if hc_up_prefill else 32
        _rowwise_fp8_projection[
            (triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))
        ](
            flat,
            weight,
            scale,
            x if bf16_weight is None else bf16_weight,
            partials,
            m,
            n,
            weight.shape[1],
            flat.stride(0),
            flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            BF16_START=bf16_start,
            BF16_ROWS=bf16_rows,
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=64 if hc_up_prefill else 128,
            num_warps=4,
            num_stages=3 if hc_up_prefill else 4,
        )
        if split_k > 1:
            _rowwise_fp8_reduce[(triton.cdiv(m * n, 256),)](
                partials, scale, output, m * n, n, SPLIT_K=split_k, BLOCK=256
            )
    else:
        rows = max(1, _CHUNK_BYTES // (2 * weight.shape[1] + 4 * m))
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            dense = weight[start:stop].bfloat16()
            if bf16_weight is not None and bf16_rows:
                lo, hi = max(start, bf16_start), min(stop, bf16_start + bf16_rows)
                if lo < hi:
                    dense[lo - start : hi - start] = bf16_weight[
                        lo - bf16_start : hi - bf16_start
                    ]
            if x.is_cuda:
                logits = torch.mm(flat, dense.t(), out_dtype=torch.float32)
            else:
                logits = flat.float() @ dense.float().t()
            output[:, start:stop] = logits * scale[start:stop]
    return output.reshape(*x.shape[:-1], n)


def _rowwise_fp8_logits_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bf16_weight: torch.Tensor | None = None,
    bf16_start: int = 0,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_rowwise_fp8_logits",
    op_func=rowwise_fp8_logits,
    fake_impl=_rowwise_fp8_logits_fake,
)


def rowwise_fp8_output(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if (
        weight.shape != (2560, 6144)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or scale.shape != (2560,)
        or x.shape[-1] != 6144
        or x.device != weight.device
        or scale.device != weight.device
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "rowwise FP8 output requires BF16 inputs and (2560, 6144) weights"
        )
    if not x.is_cuda:
        return rowwise_fp8_logits(x, weight, scale)
    flat = x.reshape(-1, 6144)
    m = flat.shape[0]
    if m > 32:
        quantized, input_scale = ops.scaled_fp8_quant(
            flat.contiguous(), use_per_token_if_dynamic=True
        )
        output = ops.cutlass_scaled_mm(
            quantized, weight.t(), input_scale, scale, torch.bfloat16
        )
    else:
        output = x.new_empty((m, 2560))
        if m:
            partials = torch.empty((8, m, 2560), dtype=torch.float32, device=x.device)
            _rowwise_fp8_projection[(40, 8, 1)](
                flat,
                weight,
                scale,
                x,
                partials,
                m,
                2560,
                6144,
                flat.stride(0),
                flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                BF16_START=0,
                BF16_ROWS=0,
                SPLIT_K=8,
                BLOCK_M=max(16, triton.next_power_of_2(m)),
                BLOCK_N=64,
                BLOCK_K=64 if m in (1, 16) else 128,
                num_warps=4,
                num_stages=3,
            )
            _rowwise_fp8_reduce[(triton.cdiv(m * 2560, 256),)](
                partials, scale, output, m * 2560, 2560, SPLIT_K=8, BLOCK=256
            )
    return output.reshape(*x.shape[:-1], 2560)


direct_register_custom_op(
    op_name="qwen4_exp_rowwise_fp8_output",
    op_func=rowwise_fp8_output,
    fake_impl=_rowwise_fp8_logits_fake,
)


class RowwiseFP8HeadMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.dtype == torch.float8_e4m3fn:
            if layer.weight_scale.shape != layer.weight.shape[:1]:
                raise ValueError("rowwise FP8 head scale does not match its weight")
        else:
            weight, scale = quantize_rowwise_fp8(layer.weight)
            replace_parameter(layer, "weight", weight)
            layer.weight_scale.copy_(scale)
        loader = getattr(layer.weight, "weight_loader", None)
        if loader is not None and not getattr(loader, "_rowwise_fp8_guard", False):
            layer.weight.weight_loader = _guard_resident_reload(loader)

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        projection = (
            torch.ops.vllm.qwen4_exp_rowwise_fp8_logits
            if x.is_cuda
            else rowwise_fp8_logits
        )
        logits = projection(x, layer.weight, layer.weight_scale)
        return logits if bias is None else logits + bias


class RowwiseFP8OutputMethod(RowwiseFP8HeadMethod):
    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        projection = (
            torch.ops.vllm.qwen4_exp_rowwise_fp8_output
            if x.is_cuda
            else rowwise_fp8_output
        )
        output = projection(x, layer.weight, layer.weight_scale)
        return output if bias is None else output + bias


def install_rowwise_fp8_output(layer: nn.Module) -> None:
    if (
        layer.weight.shape != (2560, 6144)
        or layer.weight.dtype != torch.bfloat16
        or not isinstance(layer.quant_method, UnquantizedLinearMethod)
        or isinstance(layer.quant_method, RowwiseFP8HeadMethod)
    ):
        raise ValueError(
            "sm120_rowwise_fp8_output requires an unquantized BF16 "
            "(2560, 6144) output linear"
        )
    layer.register_buffer(
        "weight_scale",
        torch.ones(2560, dtype=torch.float32, device=layer.weight.device),
    )
    layer.quant_method = RowwiseFP8OutputMethod()


def _guard_resident_reload(loader: Callable) -> Callable:
    @wraps(loader)
    def load(parameter, *args, **kwargs):
        if parameter.dtype == torch.float8_e4m3fn:
            raise ValueError("rowwise FP8 heads require BF16 layerwise weight reload")
        return loader(parameter, *args, **kwargs)

    load.__dict__["_rowwise_fp8_guard"] = True
    return load


class RowwiseFP8HCMethod(RowwiseFP8HeadMethod):
    def __init__(self, bf16_start: int, bf16_rows: int, pad_rows: int) -> None:
        self.bf16_start = bf16_start
        self.bf16_rows = bf16_rows
        self.pad_rows = pad_rows

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.dtype == torch.bfloat16 and self.bf16_rows:
            layer.bf16_weight.copy_(
                layer.weight[self.bf16_start : self.bf16_start + self.bf16_rows]
            )
        super().process_weights_after_loading(layer)
        if self.bf16_rows:
            rows = slice(self.bf16_start, self.bf16_start + self.bf16_rows)
            layer.weight.data[rows].zero_()
            layer.weight_scale[rows] = 1
        if self.pad_rows:
            layer.weight.data[-self.pad_rows :].zero_()
            layer.weight_scale[-self.pad_rows :] = 1

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        projection = (
            torch.ops.vllm.qwen4_exp_rowwise_fp8_logits
            if x.is_cuda
            else rowwise_fp8_logits
        )
        logits = projection(
            x, layer.weight, layer.weight_scale, layer.bf16_weight, self.bf16_start
        )
        return logits if bias is None else logits + bias


def install_rowwise_fp8_hc(
    layer: nn.Module, *, bf16_start: int = 0, bf16_rows: int = 0, pad_rows: int = 0
) -> None:
    if (
        bf16_start < 0
        or bf16_rows < 0
        or pad_rows < 0
        or bf16_start + bf16_rows > layer.weight.shape[0] - pad_rows
    ):
        raise ValueError("rowwise FP8 HC preserved rows/padding must not overlap")
    if layer.weight.dtype != torch.bfloat16 or not isinstance(
        layer.quant_method, UnquantizedLinearMethod
    ):
        raise ValueError("sm120_rowwise_fp8_hc requires an unquantized BF16 linear")
    layer.register_buffer(
        "weight_scale",
        torch.ones(
            layer.weight.shape[0], dtype=torch.float32, device=layer.weight.device
        ),
    )
    layer.register_buffer(
        "bf16_weight",
        layer.weight.new_empty((bf16_rows, layer.weight.shape[1])),
    )
    layer.quant_method = RowwiseFP8HCMethod(bf16_start, bf16_rows, pad_rows)


def install_rowwise_fp8_head(head: ParallelLMHead) -> None:
    if isinstance(head.quant_method, RowwiseFP8HeadMethod):
        return
    if head.weight.dtype != torch.bfloat16 or not isinstance(
        head.quant_method, (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)
    ):
        raise ValueError("sm120_rowwise_fp8_head requires an unquantized BF16 head")
    head.register_buffer(
        "weight_scale",
        torch.ones(
            head.weight.shape[0], dtype=torch.float32, device=head.weight.device
        ),
    )
    head.quant_method = RowwiseFP8HeadMethod()
