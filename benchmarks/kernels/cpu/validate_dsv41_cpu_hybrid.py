# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and audit a three-GPU DeepSeek-V4.1 CPU-expert deployment."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def inspect_placement(worker):
    import regex as re
    import torch

    from vllm.distributed import get_pp_group
    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4MoE

    cpu_layers, gpu_layers, engrams = [], [], []
    for name, module in worker.model_runner.model.named_modules():
        if isinstance(module, DeepseekV4MoE):
            layer = int(re.search(r"layers\.(\d+)", name).group(1))
            if isinstance(module.experts, CPUExpertModule):
                assert not list(module.experts.parameters())
                module.experts.finalize()
                cpu_layers.append(layer)
            else:
                assert all(p.is_cuda for p in module.experts.parameters())
                gpu_layers.append(layer)
        if name.endswith("engram.embed_tokens"):
            engrams.append(
                {
                    "name": name,
                    "device": str(module.weight.device),
                    "weight_bytes": module.weight.nbytes,
                    "scale_bytes": module.weight_scale_inv.nbytes,
                }
            )
    return {
        "pp_rank": get_pp_group().rank_in_group,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(
            torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            ).uuid
        ),
        "host_allocator": torch.cuda.memory.host_memory_stats(),
        "process_memory": {
            key: value.strip()
            for line in Path("/proc/self/status").read_text().splitlines()
            for key, value in [line.split(":", 1)]
            if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem")
        },
        "device": torch.accelerator.current_device_index(),
        "cpu_layers": cpu_layers,
        "gpu_layers": gpu_layers,
        "engrams": engrams,
        "cuda_allocated_bytes": torch.accelerator.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
    }


def save_cpu_inputs(worker, directory):
    import regex as re
    from safetensors.torch import save_file

    from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

    saved = []
    for name, module in worker.model_runner.model.named_modules():
        if not isinstance(module, CPUExpertModule):
            continue
        layer = int(re.search(r"layers\.(\d+)", name).group(1))
        if layer not in (20, 30, 39):
            continue
        path = Path(directory) / f"layer-{layer}-last-call.safetensors"
        save_file(
            {
                "hidden": module.host_hidden[:1].clone(),
                "ids": module.host_ids[:1].clone(),
                "routes": module.host_routes[:1].clone(),
                "routed_output": module.host_output[:1].clone(),
            },
            str(path),
            metadata={
                "layer": str(layer),
                "backend": module.backend.config.backend,
                "source_call_tokens": str(module.last_tokens),
                "selected_row": "0",
            },
        )
        saved.append(str(path))
    return saved


class HybridValidationWorker:
    def inspect_dsv41_placement(self):
        return inspect_placement(self)

    def save_dsv41_cpu_inputs(self, directory):
        return save_cpu_inputs(self, directory)

    def trace_dsv41_cpu_calls(self, directory):
        import inspect

        import regex as re
        import torch
        from safetensors.torch import save_file

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        self._dsv41_trace_methods = []

        def wrap(backend, layer):
            original = backend.forward
            calls = 0

            def forward(hidden, ids, routes):
                nonlocal calls
                calls += 1
                tensors = {
                    "hidden": hidden.contiguous(),
                    "ids": ids.contiguous(),
                    "routes": routes.contiguous(),
                }
                path = Path(directory) / f"layer-{layer}-before-call.safetensors"
                save_file(tensors, str(path), metadata={"call": str(calls)})
                if not torch.isfinite(hidden).all() or not torch.isfinite(routes).all():
                    raise RuntimeError(
                        f"Nonfinite CPU MoE input at layer {layer}: {path}"
                    )
                result = original(hidden, ids, routes)
                if not torch.isfinite(result).all():
                    raise RuntimeError(
                        f"Nonfinite CPU MoE output at layer {layer}: {path}"
                    )
                return result

            self._dsv41_trace_methods.append((backend, "forward", original))
            backend.forward = forward

        def wrap_attention(module, name, method):
            original = getattr(module, method)
            signature = inspect.signature(original)

            def forward(*args, **kwargs):
                result = original(*args, **kwargs)
                arguments = signature.bind(*args, **kwargs).arguments
                if not torch.isfinite(arguments["output"]).all():
                    tensors = {}
                    for key, value in arguments.items():
                        if isinstance(value, torch.Tensor):
                            tensors[key] = value.detach().cpu().contiguous().clone()
                        elif value is not None and hasattr(value, "__dict__"):
                            for field, tensor in vars(value).items():
                                if isinstance(tensor, torch.Tensor):
                                    tensors[f"{key}.{field}"] = (
                                        tensor.detach().cpu().contiguous().clone()
                                    )
                    tensors["topk_indices_buffer"] = (
                        module.topk_indices_buffer.detach().cpu().contiguous()
                    )
                    path = Path(directory) / f"{name}-{method}-nonfinite.safetensors"
                    save_file(tensors, str(path))
                    raise RuntimeError(f"Nonfinite attention output: {path}")
                return result

            self._dsv41_trace_methods.append((module, method, original))
            setattr(module, method, forward)

        layers = []
        for name, module in self.model_runner.model.named_modules():
            if type(module).__name__ == "DeepseekV41AmpereMLAAttention":
                for method in ("_forward_prefill", "_forward_decode"):
                    wrap_attention(module, name, method)
            if isinstance(module, CPUExpertModule):
                layer = int(re.search(r"layers\.(\d+)", name).group(1))
                wrap(module.backend, layer)
                layers.append(layer)
        return layers

    def stop_dsv41_trace(self):
        for module, method, original in self._dsv41_trace_methods:
            setattr(module, method, original)
        self._dsv41_trace_methods = []


def output_record(response, prompt, seconds):
    output = response.outputs[0]
    record = {
        "prompt": prompt,
        "prompt_tokens": len(response.prompt_token_ids),
        "text": output.text,
        "token_ids": output.token_ids,
        "prompt_token_ids": response.prompt_token_ids,
        "prompt_logprobs": [
            None
            if step is None
            else {str(token): value.logprob for token, value in step.items()}
            for step in response.prompt_logprobs
        ],
        "logprobs": [
            {str(token): value.logprob for token, value in step.items()}
            for step in output.logprobs
        ],
        "seconds": seconds,
        "finish_reason": output.finish_reason,
    }
    if response.metrics is not None:
        record["metrics"] = asdict(response.metrics)
        decode_seconds = (
            response.metrics.last_token_ts - response.metrics.first_token_ts
        )
        if decode_seconds > 0 and len(output.token_ids) > 1:
            record["decode_tokens_per_second"] = (
                len(output.token_ids) - 1
            ) / decode_seconds
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("kt", "llama", "ik"), default="ik")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--trace-cpu-calls", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 3 or any(not entry for entry in visible):
        parser.error("Set CUDA_VISIBLE_DEVICES to the three intended GPUs explicitly")
    if not any(
        name in os.environ for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "pinned_max_round_threshold_mb:128"
    os.environ["VLLM_PP_LAYER_PARTITION"] = "8,6,26"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1" if args.graph else "0"

    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        pipeline_parallel_size=3,
        distributed_executor_backend="mp",
        worker_extension_cls="validate_dsv41_cpu_hybrid.HybridValidationWorker",
        disable_log_stats=False,
        gpu_memory_utilization=0.95,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        kv_cache_memory_bytes=1024**3,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        limit_mm_per_prompt={"image": 0},
        engram_config={"cpu_offload": True},
        enforce_eager=not args.graph,
        compilation_config={"cudagraph_mode": "PIECEWISE"} if args.graph else None,
        additional_config={
            "cpu_moe": {
                "backend": args.backend,
                "library_path": str(args.library.resolve()),
                "num_threads": args.threads,
                "start_layer": 20,
                "end_layer": 40,
            }
        },
    )
    placement = llm.collective_rpc("inspect_dsv41_placement")
    assert sorted(layer for p in placement for layer in p["cpu_layers"]) == list(
        range(20, 40)
    ), placement
    assert sorted(layer for p in placement for layer in p["gpu_layers"]) == list(
        range(20)
    ), placement
    engrams = [e for rank in placement for e in rank["engrams"]]
    assert len(engrams) == 2 and all(e["device"] == "cpu" for e in engrams), placement
    payload = {
        "backend": args.backend,
        "library": str(args.library.resolve()),
        "model": str(args.model.resolve()),
        "visible_devices": visible,
        "allocator_config": os.environ.get(
            "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        ),
        "placement": placement,
        "load_seconds": time.perf_counter() - started,
        "graph": args.graph,
        "thinking": args.thinking,
        "trace_cpu_calls": args.trace_cpu_calls,
        "outputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    trace_dir = args.output.parent / (args.output.stem + "-inputs")
    trace_dir.mkdir(exist_ok=True)
    prompts = [
        "What is 17 + 25? Give only the number.",
        "用一句话解释为什么白天能看到月亮。",
        "Write a Python function that returns the larger of two numbers.",
    ]
    for tracing in [True, False] if args.trace_cpu_calls else [False]:
        key = "traced_outputs" if tracing else "outputs"
        payload[key] = []
        if tracing:
            llm.collective_rpc(
                "trace_dsv41_cpu_calls", args=(str(trace_dir.resolve()),)
            )
        elif args.trace_cpu_calls:
            llm.collective_rpc("stop_dsv41_trace")
        for prompt in prompts:
            start = time.perf_counter()
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                SamplingParams(
                    temperature=0,
                    max_tokens=args.output_tokens,
                    logprobs=5,
                    prompt_logprobs=1,
                ),
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": args.thinking},
            )[0]
            record = output_record(response, prompt, time.perf_counter() - start)
            payload[key].append(record)
            payload["cpu_input_traces"] = llm.collective_rpc(
                "save_dsv41_cpu_inputs", args=(str(trace_dir.resolve()),)
            )
            args.output.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
