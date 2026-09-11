# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional worker instrumentation for benchmark_exl3.py."""

import regex as re
import torch


class Exl3ProfileWorkerExtension:
    def install_exl3_event_profile(self):
        """Instrument eager expert boundaries after the clean throughput runs."""
        from collections import Counter
        from functools import wraps

        from vllm.model_executor.layers.fused_moe.experts import marlin_moe
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.model_executor.layers.quantization import exl3

        state = {"events": [], "originals": [], "pool": [], "routing": []}
        self._exl3_event_profile = state
        layer_prefixes = {}
        for module in self.get_model().modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, exl3.Exl3MoEMethod):
                layer_prefixes[method.ptrs[0].data_ptr()] = method.prefix
        # Initialize CUDA event handles before entering the measured request.
        for _ in range(16384):
            pair = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            for event in pair:
                event.record()
            state["pool"].append(pair)
        torch.accelerator.synchronize()

        def patch(owner, name, label, routing=False):
            original = getattr(owner, name)
            state["originals"].append((owner, name, original))

            @wraps(original)
            def measured(*args, **kwargs):
                index = len(state["events"])
                if index >= len(state["pool"]):
                    raise RuntimeError("CUDA event profile pool exhausted")
                start, end = state["pool"][index]
                state["events"].append((label, start, end))
                if routing:
                    ids = kwargs.get("topk_ids")
                    if ids is None:
                        ids = args[2]
                    workspace = kwargs.get(
                        "workspace", args[4] if len(args) > 4 else None
                    )
                    pointers = kwargs.get("ptrs", args[3] if len(args) > 3 else None)
                    state["routing"].append(
                        (
                            ids.detach().clone(),
                            workspace[0].shape[1],
                            layer_prefixes[pointers[0].data_ptr()],
                        )
                    )
                start.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    end.record()

            setattr(owner, name, measured)

        patch(MoERunner, "_forward_impl", "moe_runner_inclusive")
        patch(marlin_moe, "fused_marlin_moe", "routed_experts_inclusive")
        patch(marlin_moe.ops, "moe_wna16_marlin_gemm", "marlin_gemm")
        quantization = self.vllm_config.model_config.quantization
        if quantization == "exl3":
            patch(exl3, "_exl3_moe_fused", "routed_experts_inclusive", routing=True)
            ext = exl3._extension()
            for name in (
                "exl3_moe",
                "exl3_gemm",
                "hgemm",
                "reconstruct_slice",
                "reconstruct_had_slice",
                "had_r_128",
            ):
                patch(ext, name, name)
        methods = Counter(
            type(module.quant_method).__name__
            for module in self.get_model().modules()
            if getattr(module, "quant_method", None) is not None
        )
        return {"gpu": torch.cuda.get_device_name(), "quant_methods": dict(methods)}

    def collect_exl3_event_profile(self):
        """Return rank-local stream intervals; nested labels must not be added."""
        from collections import defaultdict

        state = self._exl3_event_profile
        torch.accelerator.synchronize()
        for owner, name, original in reversed(state["originals"]):
            setattr(owner, name, original)
        timings = defaultdict(list)
        for label, start, end in state["events"]:
            timings[label].append(start.elapsed_time(end))
        routing = []
        for ids, capacity, prefix in state["routing"]:
            grouped_counts = torch.stack(
                [
                    torch.bincount(chunk.flatten().long(), minlength=288)
                    for chunk in ids.split(capacity)
                ]
            )
            counts = torch.bincount(ids.flatten().long(), minlength=288)
            chunk_counts = torch.stack(
                [
                    torch.bincount(chunk.flatten().long(), minlength=288)
                    for chunk in ids.split(128)
                ]
            )
            routing.append(
                {
                    "prefix": prefix,
                    "workspace_capacity": capacity,
                    "grouped_active_expert_visits": (grouped_counts > 0).sum().item(),
                    "grouped_rows_in_16row_tiles": (((grouped_counts + 15) // 16) * 16)
                    .sum()
                    .item(),
                    "tokens": ids.shape[0],
                    "topk": ids.shape[1],
                    "max_tokens_per_expert": counts.max().item(),
                    "active_experts": (counts > 0).sum().item(),
                    "chunk128_active_expert_visits": (chunk_counts > 0).sum().item(),
                    "chunk128_max_tokens_per_expert": chunk_counts.max().item(),
                    "chunk128_rows_in_16row_tiles": (((chunk_counts + 15) // 16) * 16)
                    .sum()
                    .item(),
                }
            )
        result = {
            "gpu": torch.cuda.get_device_name(),
            "timer": "CUDA events on current stream, including wrapper launch gaps",
            "nested_labels_are_not_additive": True,
            "timings_ms": dict(timings),
            "routing": routing,
        }
        del self._exl3_event_profile
        return result

    def set_exl3_workspace_capacity(self, capacity):
        """Sweep capacity without reloading weights; eager prefill uses this buffer."""
        from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

        torch.accelerator.synchronize()
        workspaces = {}
        methods = []
        for module in self.get_model().modules():
            method = getattr(module, "quant_method", None)
            if not isinstance(method, Exl3MoEMethod):
                continue
            old = method.workspace
            key = (old[0].device, old[0].shape[2], old[2].shape[2])
            if key not in workspaces:
                workspaces[key] = [
                    t.new_empty((t.shape[0], capacity, t.shape[2])) for t in old
                ]
            method.workspace = workspaces[key]
            methods.append(method)
        return {
            "gpu": torch.cuda.get_device_name(),
            "capacity": capacity,
            "expert_layers": len(methods),
            "workspace_bytes": sum(
                t.numel() * t.element_size()
                for workspace in workspaces.values()
                for t in workspace
            ),
        }

    def start_exl3_route_capture(self, directory):
        """Capture four real prefill chunks at representative depths, outside timing."""
        from pathlib import Path

        from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

        state = {"original": Exl3MoEMethod.apply, "layers": {}, "directory": directory}
        self._exl3_route_capture = state
        Path(directory).mkdir(parents=True, exist_ok=True)

        def capture(method, layer, x, topk_weights, topk_ids, *args, **kwargs):
            layer_id = int(method.prefix.split(".layers.")[1].split(".")[0])
            if layer_id in (3, 22, 44) and x.shape[0] > 8:
                samples = state["layers"].setdefault(method.prefix, [])
                if len(samples) < 4:
                    samples.append(
                        tuple(t.detach().clone() for t in (x, topk_weights, topk_ids))
                    )
            return state["original"](
                method, layer, x, topk_weights, topk_ids, *args, **kwargs
            )

        Exl3MoEMethod.apply = capture

    def finish_exl3_route_capture(self):
        from pathlib import Path

        from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

        state = self._exl3_route_capture
        Exl3MoEMethod.apply = state["original"]
        paths = []
        for prefix, samples in state["layers"].items():
            layer_id = int(prefix.split(".layers.")[1].split(".")[0])
            path = Path(state["directory"]) / f"layer-{layer_id}.pt"
            values = [
                torch.cat([sample[i] for sample in samples]).cpu() for i in range(3)
            ]
            torch.save(
                {
                    "prefix": prefix,
                    "x": values[0],
                    "weights": values[1],
                    "ids": values[2],
                },
                path,
            )
            paths.append(str(path))
        del self._exl3_route_capture
        return paths

    def configure_exl3_optimization(self, options):
        """Experimental switches; the final production path is validated separately."""
        from functools import partial

        from kernels.benchmark_exl3_moe import (
            fused_with_chunk,
            fused_with_reused_buffers,
        )

        from vllm.model_executor.layers.quantization import exl3

        result = self.set_exl3_workspace_capacity(options["capacity"])
        if "priority" in options:
            exl3.envs.VLLM_EXL3_MOE_PRIORITY = bool(options["priority"])
        if not hasattr(self, "_exl3_original_fused"):
            self._exl3_original_fused = exl3._exl3_moe_fused
        if any(
            key in options
            for key in ("groups", "splits", "packing", "longest_first", "reuse_buffers")
        ):
            splits = options.get("splits", 1)
            is_blackwell = torch.cuda.get_device_capability()[0] >= 10
            if options.get("blackwell_only") and not is_blackwell:
                splits = 1
            pointers = {}
            for module in self.get_model().modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, exl3.Exl3MoEMethod):
                    pointers[method.ptrs[0].data_ptr()] = (
                        [p.repeat_interleave(splits) for p in method.ptrs]
                        if splits > 1
                        else method.ptrs
                    )

            fused = (
                partial(fused_with_reused_buffers, pool={})
                if options.get("reuse_buffers")
                else fused_with_chunk
            )

            def candidate(
                x, weights, ids, ptrs, workspace, bits, flags, limit, m32_locks=None
            ):
                inputs = (
                    x,
                    weights,
                    ids,
                    pointers[ptrs[0].data_ptr()],
                    workspace,
                    bits,
                    flags,
                    limit,
                )
                return fused(
                    inputs,
                    workspace[0].shape[1],
                    options.get("groups", -1),
                    splits,
                    options.get("split_threshold", 0),
                    options.get("packing", "sort"),
                    options.get("longest_first", False),
                )

            exl3._exl3_moe_fused = candidate
        else:
            exl3._exl3_moe_fused = self._exl3_original_fused
        result["linear"] = self.set_exl3_linear_experiment(
            options.get("linear_mode", "none"), options.get("shared_overlap", False)
        )
        result["options"] = options
        return result

    def set_exl3_linear_experiment(self, mode, overlap):
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExperts,
            SharedExpertsOrder,
        )
        from vllm.model_executor.layers.quantization import exl3
        from vllm.utils.torch_utils import aux_stream

        if not hasattr(self, "_exl3_original_linear_apply"):
            self._exl3_original_linear_apply = exl3.Exl3LinearMethod.apply
            self._exl3_original_shared_order = (
                SharedExperts._determine_shared_experts_order
            )
            self._exl3_linear_caches = {}
        original = self._exl3_original_linear_apply
        ext = exl3._extension()
        caches = self._exl3_linear_caches
        if mode in ("rotated", "original") and mode not in caches:
            cache = {}
            for module in self.get_model().modules():
                method = getattr(module, "quant_method", None)
                if not isinstance(method, exl3.Exl3LinearMethod):
                    continue
                if any(part.name.endswith("lm_head") for part in method.parts):
                    continue
                values = []
                for part, weights in zip(method.parts, method.weights):
                    if not part.quantized:
                        values.append(None)
                        continue
                    value = torch.empty(
                        part.dimensions,
                        device=weights["trellis"].device,
                        dtype=torch.float16,
                    )
                    bits = weights["trellis"].shape[-1] // 16
                    flags = ("mcg" in weights, "mul1" in weights)
                    if mode == "rotated":
                        ext.reconstruct_slice(
                            value, weights["trellis"], bits, *flags, 0
                        )
                    else:
                        ext.reconstruct_had_slice(
                            value,
                            weights["trellis"],
                            weights["suh"],
                            weights["svh"],
                            bits,
                            *flags,
                            0,
                        )
                    values.append(value)
                cache[id(method)] = values
            caches[mode] = cache

        def calculate(method, layer, x, bias=None):
            rows = x.numel() // x.shape[-1]
            if (
                rows <= 144
                or mode == "none"
                or any(part.name.endswith("lm_head") for part in method.parts)
            ):
                return original(method, layer, x, bias)
            outputs = []
            inp = x.reshape(-1, x.shape[-1]).half().contiguous()
            for index, (part, weights) in enumerate(zip(method.parts, method.weights)):
                if not part.quantized:
                    y = torch.nn.functional.linear(x, weights["weight"].to(x.dtype))
                else:
                    k, n = part.dimensions
                    if mode == "fused":
                        weight = torch.empty(
                            (k, n), device=x.device, dtype=torch.float16
                        )
                        ext.reconstruct_had_slice(
                            weight,
                            weights["trellis"],
                            weights["suh"],
                            weights["svh"],
                            weights["trellis"].shape[-1] // 16,
                            "mcg" in weights,
                            "mul1" in weights,
                            0,
                        )
                    else:
                        weight = caches[mode][id(method)][index]
                    if mode == "rotated":
                        rotated = torch.empty_like(inp)
                        ext.had_r_128(inp, rotated, weights["suh"], None, 1.0)
                        y = torch.mm(rotated, weight)
                        ext.had_r_128(y, y, None, weights["svh"], 1.0)
                    else:
                        y = torch.mm(inp, weight)
                    y = y.view(*x.shape[:-1], n).to(x.dtype)
                outputs.append(y)
            result = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
            return result if bias is None else result + bias

        exl3.Exl3LinearMethod.apply = calculate if mode != "none" else original
        original_order = self._exl3_original_shared_order
        if overlap:
            assert mode in ("original", "rotated")
            for module in self.get_model().modules():
                if isinstance(module, SharedExperts):
                    module._stream = aux_stream()

            def prefill_order(shared, hidden):
                if hidden.shape[0] > 144:
                    return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
                return original_order(shared, hidden)

            SharedExperts._determine_shared_experts_order = prefill_order
        else:
            SharedExperts._determine_shared_experts_order = original_order
        torch.accelerator.synchronize()
        return {
            "mode": mode,
            "overlap": overlap,
            "cache_bytes": sum(
                t.numel() * t.element_size()
                for values in caches.get(mode, {}).values()
                for t in values
                if t is not None
            ),
        }

    def get_exl3_runtime_state(self):
        from vllm import envs
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod
        from vllm.model_executor.models.utils import PPMissingLayer

        layers = []
        decoder_layers = []
        buffers = {}
        for name, module in self.get_model().named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
            if match and not isinstance(module, PPMissingLayer):
                decoder_layers.append(
                    {
                        "name": name,
                        "index": int(match.group(1)),
                        "type": type(module).__name__,
                    }
                )
            method = getattr(module, "quant_method", None)
            if not isinstance(method, Exl3MoEMethod):
                continue
            layers.append(
                {
                    "prefix": method.prefix,
                    "capacity": method.workspace[0].shape[1],
                    "m_tile": method.moe_m_tile,
                    "decode_mode": ("native", "plain", "residual")[method.decode_mode],
                }
            )
            for tensor in method.workspace:
                buffers[tensor.data_ptr()] = tensor.numel() * tensor.element_size()
        return {
            "gpu": torch.cuda.get_device_name(),
            "pp_rank": get_pp_group().rank_in_group,
            "decoder_layers": decoder_layers,
            "priority": envs.VLLM_EXL3_MOE_PRIORITY,
            "workspace_bytes": sum(buffers.values()),
            "layers": layers,
        }
