# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 checkpoints, using the optional MIT-licensed ExLlamaV3 CUDA extension."""

import importlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

_DTYPES = {
    "I16": torch.int16,
    "I32": torch.int32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}
_COMPONENTS = {"trellis", "suh", "svh", "su", "sv", "mul1", "mcg", "weight"}


@lru_cache(maxsize=1)
def _extension():
    try:
        return importlib.import_module("exllamav3_ext")
    except ImportError as exc:
        raise ImportError(
            "EXL3 requires exllamav3>=1.4.8 compiled for the installed PyTorch "
            "and CUDA versions. See docs/features/quantization/exl3.md."
        ) from exc


@dataclass(frozen=True)
class Exl3Tensor:
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class Exl3Matrix:
    name: str
    tensors: dict[str, Exl3Tensor]

    @property
    def quantized(self):
        return "trellis" in self.tensors

    @property
    def dimensions(self):
        if self.quantized:
            shape = self.tensors["trellis"].shape
            return shape[0] * 16, shape[1] * 16
        shape = self.tensors["weight"].shape
        return shape[1], shape[0]

    def validate(self):
        if not self.quantized:
            return
        t = self.tensors["trellis"]
        if (
            len(t.shape) != 3
            or t.dtype != torch.int16
            or t.shape[-1] not in range(16, 129, 16)
        ):
            raise ValueError(f"Invalid EXL3 trellis: {self.name}: {t}")
        k, n = self.dimensions
        if k % 128 or n % 128:
            raise ValueError(f"EXL3 requires dimensions divisible by 128: {self.name}")
        for unpacked, packed, size in (("suh", "su", k), ("svh", "sv", n)):
            spec = self.tensors.get(unpacked)
            if spec is not None:
                if spec.shape != (size,) or spec.dtype != torch.float16:
                    raise ValueError(f"Invalid EXL3 {unpacked}: {self.name}")
            elif self.tensors.get(packed) != Exl3Tensor((size // 16,), torch.int16):
                raise ValueError(
                    f"Missing or invalid EXL3 {unpacked}/{packed}: {self.name}"
                )
        if "mcg" in self.tensors and "mul1" in self.tensors:
            raise ValueError(f"Conflicting EXL3 codebooks: {self.name}")


class Exl3Config(QuantizationConfig):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.matrices: dict[str, Exl3Matrix] = {}
        self.workspaces: dict[tuple, list[torch.Tensor]] = {}

    @classmethod
    def get_name(cls):
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        if self.matrices:
            return
        root = Path(model_name)
        if not root.is_dir():
            raise ValueError("EXL3 currently requires a local safetensors checkpoint.")
        files = sorted(root.glob("*.safetensors"))
        index = root / "model.safetensors.index.json"
        if index.exists():
            file_names = set(json.loads(index.read_text())["weight_map"].values())
            files = [root / name for name in sorted(file_names)]
        for file in files:
            with file.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if size > 100_000_000:
                    raise ValueError(f"Invalid safetensors header size in {file}")
                header = json.loads(stream.read(size))
            for key, value in header.items():
                if "." not in key or key == "__metadata__":
                    continue
                name, component = key.rsplit(".", 1)
                if component not in _COMPONENTS or value["dtype"] not in _DTYPES:
                    continue
                if component == "weight" and len(value["shape"]) != 2:
                    continue
                spec = Exl3Tensor(tuple(value["shape"]), _DTYPES[value["dtype"]])
                matrix = self.matrices.setdefault(name, Exl3Matrix(name, {}))
                if component in matrix.tensors:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                matrix.tensors[component] = spec
        self.matrices = {
            name: spec
            for name, spec in self.matrices.items()
            if spec.quantized or "weight" in spec.tensors
        }
        for matrix in self.matrices.values():
            matrix.validate()
        if not any(spec.quantized for spec in self.matrices.values()):
            raise ValueError("No EXL3 trellis tensors found in checkpoint.")

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        mapped = {}
        for name, spec in self.matrices.items():
            new_name = hf_to_vllm_mapper._map_name(name + ".trellis")
            if new_name is not None:
                new_name = new_name.removesuffix(".trellis")
                mapped[new_name] = Exl3Matrix(new_name, spec.tensors)
        self.matrices = mapped

    def resolve(self, prefix):
        if prefix in self.matrices:
            return [self.matrices[prefix]]
        parent, _, leaf = prefix.rpartition(".")
        parts = self.packed_modules_mapping.get(leaf, [])
        names = [f"{parent}.{part}" if parent else part for part in parts]
        if names and all(name in self.matrices for name in names):
            return [self.matrices[name] for name in names]
        return []

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(self, layer.moe_config, prefix)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            parts = self.resolve(prefix)
            if any(part.quantized for part in parts):
                return Exl3LinearMethod(parts)
            return UnquantizedLinearMethod()
        return None


def _unpack_signs(value):
    bits = value.to(torch.int32).unsqueeze(-1)
    shifts = torch.arange(16, device=value.device)
    return (1 - 2 * ((bits >> shifts) & 1)).flatten().half()


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, parts):
        self.parts = parts
        self.views = {}
        self.loaded = set()
        self.shards = {}

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if getattr(layer, "tp_size", 1) != 1:
            raise ValueError("EXL3 currently supports TP=1; use pipeline parallelism.")
        expected = (input_size_per_partition, sum(output_partition_sizes))
        if (
            any(part.dimensions[0] != expected[0] for part in self.parts)
            or sum(part.dimensions[1] for part in self.parts) != expected[1]
        ):
            raise ValueError(
                f"EXL3 shape mismatch for {self.parts[0].name}: {expected}"
            )
        shard = 0
        for i, part in enumerate(self.parts):
            start, width = shard, 0
            while width < part.dimensions[1] and shard < len(output_partition_sizes):
                width += output_partition_sizes[shard]
                shard += 1
            if width != part.dimensions[1]:
                raise ValueError(f"EXL3 cannot split a rotated matrix: {part.name}")
            ids = tuple(range(start, shard))
            self.shards[ids if len(ids) > 1 else ids[0]] = i
        for component in sorted(set().union(*(p.tensors for p in self.parts))):
            specs = [
                (i, part.tensors[component])
                for i, part in enumerate(self.parts)
                if component in part.tensors
            ]
            dtype = params_dtype if component == "weight" else specs[0][1].dtype
            if component != "weight" and any(spec.dtype != dtype for _, spec in specs):
                raise ValueError(f"EXL3 mixed storage dtype for {component}")
            count = sum(math.prod(spec.shape) for _, spec in specs)
            param = nn.Parameter(torch.empty(count, dtype=dtype), requires_grad=False)
            offset = 0
            for i, spec in specs:
                length = math.prod(spec.shape)
                self.views[i, component] = (offset, length, spec.shape)
                offset += length
            set_weight_attrs(param, {"weight_loader": self._loader(component)})
            layer.register_parameter(component, param)

    def _loader(self, component):
        def load(param, tensor, shard_id=None):
            if shard_id is None:
                if len(self.parts) != 1:
                    raise ValueError(
                        "EXL3 fused projections require a shard identifier"
                    )
                part = 0
            else:
                if isinstance(shard_id, str):
                    shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                part = self.shards[shard_id]
            offset, size, shape = self.views[part, component]
            if (
                tensor.shape != shape
                or tensor.dtype != self.parts[part].tensors[component].dtype
            ):
                raise ValueError(
                    f"Unexpected EXL3 tensor: {self.parts[part].name}.{component}"
                )
            param.data.narrow(0, offset, size).copy_(tensor.reshape(-1))
            self.loaded.add((part, component))

        return load

    def process_weights_after_loading(self, layer):
        missing = self.views.keys() - self.loaded
        if missing:
            raise ValueError(
                f"Incomplete EXL3 weights for {self.parts[0].name}: {missing}"
            )
        self.weights = []
        for i, part in enumerate(self.parts):
            weights = {}
            for component in part.tensors:
                offset, size, shape = self.views[i, component]
                weights[component] = (
                    getattr(layer, component).data.narrow(0, offset, size).view(shape)
                )
            for unpacked, packed in (("suh", "su"), ("svh", "sv")):
                if unpacked not in weights and packed in weights:
                    weights[unpacked] = _unpack_signs(weights[packed])
            self.weights.append(weights)
        _extension()

    def apply(self, layer, x, bias=None):
        outputs = []
        for part, weights in zip(self.parts, self.weights):
            if part.quantized:
                y = torch.ops.vllm.exl3_linear(
                    x,
                    weights["trellis"],
                    weights["suh"],
                    weights["svh"],
                    "mcg" in weights,
                    "mul1" in weights,
                )
            else:
                y = torch.nn.functional.linear(x, weights["weight"].to(x.dtype))
            outputs.append(y)
        result = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
        return result if bias is None else result + bias


def _exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    ext = _extension()
    shape = x.shape[:-1] + (svh.numel(),)
    inp = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
    out = torch.empty((inp.shape[0], svh.numel()), device=x.device, dtype=torch.float16)
    if inp.shape[0] == 0:
        return out.view(shape).to(x.dtype)
    rotated = torch.empty_like(inp)
    bits = trellis.shape[-1] // 16
    if inp.shape[0] <= 144:
        ext.exl3_gemm(inp, trellis, out, suh, rotated, svh, -1, mcg, mul1, 0)
    else:
        fused = inp.shape[0] >= 1024
        if not fused:
            ext.had_r_128(inp, rotated, suh, None, 1.0)
        for start in range(0, svh.numel(), 32768):
            width = min(32768, svh.numel() - start)
            weight = torch.empty(
                (suh.numel(), width), device=x.device, dtype=torch.float16
            )
            if fused:
                ext.reconstruct_had_slice(
                    weight, trellis, suh, svh[start:], bits, mcg, mul1, start
                )
            else:
                ext.reconstruct_slice(weight, trellis, bits, mcg, mul1, start)
            ext.hgemm(inp if fused else rotated, weight, out[:, start : start + width])
        if not fused:
            ext.had_r_128(out, out, None, svh, 1.0)
    return out.view(shape).to(x.dtype)


def _exl3_linear_fake(x, trellis, suh, svh, mcg, mul1):
    return x.new_empty(x.shape[:-1] + (svh.numel(),))


direct_register_custom_op("exl3_linear", _exl3_linear, fake_impl=_exl3_linear_fake)


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config, moe, prefix):
        super().__init__(moe)
        self.config = config
        self.prefix = prefix.removesuffix(".routed_experts")
        self.loaded = set()

    @property
    def supports_multi_stream(self) -> bool:
        # ExLlamaV3 uses a device-wide lock/scratch area for cooperative GEMMs.
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        pc = self.moe.moe_parallel_config
        if pc.tp_size != 1 or pc.ep_size != 1:
            raise ValueError("EXL3 MoE currently requires TP=EP=1; PP is supported.")
        if str(self.moe.activation) not in ("silu", "MoEActivation.SILU"):
            raise ValueError("EXL3 MoE currently supports SiLU activation only.")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        self.parts = {}
        for kind, name in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            parts = [
                self.config.matrices[f"{self.prefix}.{expert}.{name}"]
                for expert in range(num_experts)
            ]
            spec = parts[0]
            expected = (
                (intermediate_size_per_partition, hidden_size)
                if kind == "w2"
                else (hidden_size, intermediate_size_per_partition)
            )
            if not spec.quantized or spec.dimensions != expected:
                raise ValueError(f"Invalid EXL3 expert shape: {spec.name}")
            if any(part.tensors != spec.tensors for part in parts):
                raise ValueError(
                    "EXL3 MoE requires uniform shapes and codebooks across experts."
                )
            if not {"suh", "svh"}.issubset(spec.tensors):
                raise ValueError("EXL3 MoE requires unpacked suh/svh sign vectors.")
            self.parts[kind] = spec
        self.views = {}
        for base, kinds in (("w13", ("w1", "w3")), ("w2", ("w2",))):
            components = set().union(*(self.parts[k].tensors for k in kinds))
            for component in sorted(components):
                specs = [
                    (kind, self.parts[kind].tensors[component])
                    for kind in kinds
                    if component in self.parts[kind].tensors
                ]
                dtype = specs[0][1].dtype
                count = sum(math.prod(spec.shape) * num_experts for _, spec in specs)
                param = nn.Parameter(
                    torch.empty(count, dtype=dtype), requires_grad=False
                )
                offset = 0
                for kind, spec in specs:
                    size = math.prod(spec.shape)
                    self.views[kind, component] = (
                        base + "_" + component,
                        offset,
                        size,
                        spec.shape,
                    )
                    offset += size * num_experts
                set_weight_attrs(param, {"weight_loader": self._loader(component)})
                layer.register_parameter(base + "_" + component, param)

    def _loader(self, component):
        def load(
            param,
            tensor,
            weight_name=None,
            shard_id=None,
            expert_id=None,
            return_success=False,
        ):
            _, offset, size, shape = self.views[shard_id, component]
            if expert_id is None or not 0 <= expert_id < self.num_experts:
                raise ValueError(f"Invalid EXL3 expert id: {expert_id}")
            if (
                tensor.shape != shape
                or tensor.dtype != self.parts[shard_id].tensors[component].dtype
            ):
                raise ValueError(f"Unexpected EXL3 expert tensor: {weight_name}")
            param.data.narrow(0, offset + expert_id * size, size).copy_(
                tensor.reshape(-1)
            )
            self.loaded.add((shard_id, component, expert_id))
            return True if return_success else None

        return load

    def process_weights_after_loading(self, layer):
        expected = {
            (kind, component, expert)
            for kind, component in self.views
            for expert in range(self.num_experts)
        }
        if missing := expected - self.loaded:
            raise ValueError(
                f"Incomplete EXL3 experts: {self.prefix}: {len(missing)} tensors"
            )
        ext = _extension()
        self.ptrs = []
        self.bits = []
        self.flags = []
        for kind in ("w1", "w3", "w2"):
            spec = self.parts[kind]
            self.bits.append(spec.tensors["trellis"].shape[-1] // 16)
            self.flags.extend(["mcg" in spec.tensors, "mul1" in spec.tensors])
            for component in ("trellis", "suh", "svh"):
                name, offset, size, _ = self.views[kind, component]
                param = getattr(layer, name)
                pointers = [
                    param.data_ptr() + (offset + i * size) * param.element_size()
                    for i in range(self.num_experts)
                ]
                self.ptrs.append(
                    torch.tensor(pointers, dtype=torch.int64, device=param.device)
                )
        if not any(self.flags):
            raise ValueError("EXL3 MoE requires the mul1 or mcg codebook.")
        if len(set(zip(self.flags[::2], self.flags[1::2]))) != 1:
            raise ValueError(
                "EXL3 fused MoE requires the same codebook for gate/up/down."
            )
        device = self.ptrs[0].device
        capacity = envs.VLLM_EXL3_MOE_MAX_TOKENS
        if capacity <= 0:
            raise ValueError("VLLM_EXL3_MOE_MAX_TOKENS must be positive")
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            capacity = min(
                capacity, vllm_config.scheduler_config.max_num_batched_tokens
            )
        key = (device, self.hidden_size, self.intermediate_size, capacity)
        if key not in self.config.workspaces:
            concurrency = ext.exl3_moe_max_concurrency(device.index)
            self.config.workspaces[key] = [
                torch.empty(
                    (concurrency, capacity, width), device=device, dtype=torch.float16
                )
                for width in (
                    self.hidden_size,
                    self.hidden_size,
                    self.intermediate_size,
                    self.intermediate_size,
                )
            ]
        self.workspace = self.config.workspaces[key]

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        return torch.ops.vllm.exl3_moe(
            x,
            topk_weights,
            topk_ids,
            self.ptrs,
            self.workspace,
            self.bits,
            self.flags,
            float(self.moe.swiglu_limit or 0.0),
        )


def _exl3_moe_decode(x, topk_weights, topk_ids, ptrs, workspace, bits, flags, limit):
    ext = _extension()
    rows, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    slots = rows * top_k
    intermediate = workspace[2].shape[-1]
    hidden = x.to(torch.float16).contiguous()
    inputs = (
        hidden.unsqueeze(1)
        if rows == 1
        else hidden.repeat_interleave(top_k, dim=0).unsqueeze(1)
    )
    rotated = torch.empty((slots, 1, hidden_size), device=x.device, dtype=torch.float16)
    gate = torch.empty((slots, 1, intermediate), device=x.device, dtype=torch.float16)
    up = torch.empty_like(gate)
    activated = torch.empty_like(gate)
    indices = topk_ids.to(torch.int64).reshape(1, slots)
    routing = topk_weights.to(torch.float16).reshape(1, slots)
    for i, target in enumerate((gate, up)):
        ext.exl3_mgemm(
            inputs,
            ptrs[3 * i],
            target,
            ptrs[3 * i + 1],
            rotated,
            ptrs[3 * i + 2],
            indices,
            None,
            bits[i],
            -1,
            flags[2 * i],
            flags[2 * i + 1],
            -1,
            -1,
            0,
            rows,
            None,
            None,
        )
    ext.silu_mul(gate, up, activated, limit)
    output = torch.empty((slots, 1, hidden_size), device=x.device, dtype=torch.float32)
    ext.exl3_mgemm(
        activated,
        ptrs[6],
        output,
        ptrs[7],
        gate,
        ptrs[8],
        indices,
        routing,
        bits[2],
        -1,
        flags[4],
        flags[5],
        -1,
        -1,
        0,
        rows,
        None,
        None,
    )
    return output[:rows, 0].to(x.dtype)


def _exl3_moe_fused(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ptrs: list[torch.Tensor],
    workspace: list[torch.Tensor],
    bits: list[int],
    flags: list[bool],
    limit: float,
) -> torch.Tensor:
    ext = _extension()
    output = torch.empty_like(x)
    experts = ptrs[0].numel()
    # Unique top-k IDs bound each expert count by the workspace capacity.
    capacity = workspace[0].shape[1]
    for start in range(0, x.shape[0], capacity):
        hidden = x[start : start + capacity].to(torch.float16).contiguous()
        ids = topk_ids[start : start + capacity].to(torch.int64).flatten()
        weights = topk_weights[start : start + capacity].to(torch.float16).flatten()
        counts = torch.zeros(experts + 1, dtype=torch.int64, device=x.device)
        counts.scatter_add_(0, ids, torch.ones_like(ids))
        chunk_ptrs = ptrs
        if (
            envs.VLLM_EXL3_MOE_PRIORITY
            and hidden.shape[0] >= 256
            and experts > workspace[0].shape[0]
        ):
            permutation = torch.argsort(counts[:-1], descending=True, stable=True)
            inverse = torch.empty_like(permutation)
            inverse.scatter_(0, permutation, torch.arange(experts, device=x.device))
            ids = inverse[ids]
            counts = torch.cat((counts[:-1][permutation], counts[-1:]))
            chunk_ptrs = [ptr[permutation] for ptr in ptrs]
        order = torch.argsort(ids, stable=True)
        tokens = torch.div(order, topk_ids.shape[1], rounding_mode="floor")
        result = torch.zeros_like(hidden, dtype=torch.float32)
        ext.exl3_moe(
            hidden,
            result,
            counts,
            tokens,
            weights[order],
            *workspace,
            0,
            *bits,
            *chunk_ptrs,
            *flags,
            limit,
            -1,
        )
        output[start : start + hidden.shape[0]].copy_(result)
    return output


def _exl3_moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ptrs: list[torch.Tensor],
    workspace: list[torch.Tensor],
    bits: list[int],
    flags: list[bool],
    limit: float,
) -> torch.Tensor:
    implementation = _exl3_moe_decode if 0 < x.shape[0] <= 8 else _exl3_moe_fused
    return implementation(
        x, topk_weights, topk_ids, ptrs, workspace, bits, flags, limit
    )


def _exl3_moe_fake(x, topk_weights, topk_ids, ptrs, workspace, bits, flags, limit):
    return torch.empty_like(x)


direct_register_custom_op(
    "exl3_moe", _exl3_moe, mutates_args=["workspace"], fake_impl=_exl3_moe_fake
)
