# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure complete single-request prefill and actual output-token throughput."""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

from validate_dsv41_cpu_hybrid import HybridValidationWorker


class HybridBenchmarkWorker(HybridValidationWorker):
    def dsv41_select_cache(self, selections):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        for name, module in self.model_runner.model.named_modules():
            if isinstance(module, CPUExpertModule) and module.gpu_cache is not None:
                layer = name.split(".layers.")[1].split(".")[0]
                module.gpu_cache.select_static(selections[layer])
        return self.dsv41_cache_stats()

    def dsv41_start_routes(self, prefill_only):
        from dsv41_prefill_diagnostics import ExpertRouteRecorder

        self.dsv41_route_recorder = ExpertRouteRecorder(self, prefill_only)

    def dsv41_stop_routes(self):
        result = self.dsv41_route_recorder.finish()
        del self.dsv41_route_recorder
        return result

    def dsv41_set_replay_cache(self, enabled):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        for module in self.model_runner.model.modules():
            if isinstance(module, CPUExpertModule):
                module.use_prefill_cache = enabled

    def dsv41_pipeline_config(self):
        import torch

        config = self.model_runner.vllm_config
        model = self.model_runner.model.language_model.model
        device = torch.accelerator.current_device_index()
        return {
            "async_scheduling": config.scheduler_config.async_scheduling,
            "max_concurrent_batches": config.max_concurrent_batches,
            "use_ubatching": config.parallel_config.use_ubatching,
            "device": device,
            "compute_capability": torch.cuda.get_device_capability(device),
            "layers": [model.start_layer, model.end_layer],
            "kv_cache_layout": config.cache_config.kv_cache_layout,
            "attention_classes": sorted(
                {
                    type(layer.attn).__name__
                    for layer in model.layers[model.start_layer : model.end_layer]
                }
            ),
        }

    def dsv41_set_prefill_index_impl(self, implementation):
        import torch

        from vllm.models.deepseek_v4_1.amd.rocm import (
            combine_topk_swa_indices as torch_combine,
        )
        from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
            DeepseekV41AmpereMLAAttention,
        )

        torch.accelerator.synchronize()
        combine = {
            "torch": torch_combine,
            "fused": DeepseekV41AmpereMLAAttention._combine_prefill_indices,
        }[implementation]
        for module in self.model_runner.model.modules():
            if isinstance(module, DeepseekV41AmpereMLAAttention):
                module._combine_prefill_indices = combine

    def dsv41_start_prefill_timing(self):
        from dsv41_prefill_diagnostics import PrefillTimer

        self.dsv41_prefill_timer = PrefillTimer(self)

    def dsv41_stop_prefill_timing(self):
        result = self.dsv41_prefill_timer.finish()
        del self.dsv41_prefill_timer
        return result

    def dsv41_start_timing(self):
        import torch

        manager = self.model_runner.cudagraph_manager
        self.dsv41_graph_times = []
        original = manager.run_fullgraph
        self.dsv41_original_graph = original

        def timed(batch):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            result = original(batch)
            end.record()
            self.dsv41_graph_times.append((start, end))
            return result

        manager.run_fullgraph = timed

    def dsv41_stop_timing(self):
        import torch

        torch.accelerator.synchronize()
        self.model_runner.cudagraph_manager.run_fullgraph = self.dsv41_original_graph
        times = [start.elapsed_time(end) for start, end in self.dsv41_graph_times]
        self.dsv41_graph_times = []
        return times

    def dsv41_set_cpu_threads(self, threads):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        for module in self.model_runner.model.modules():
            if isinstance(module, CPUExpertModule):
                module.backend.set_num_threads(threads)

    def dsv41_set_cpu_executor(self, mode):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        for module in self.model_runner.model.modules():
            if isinstance(module, CPUExpertModule):
                module.backend.set_execution_mode(mode)

    def dsv41_set_cpu_schedule(self, flags):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        for module in self.model_runner.model.modules():
            if isinstance(module, CPUExpertModule):
                module.backend.set_schedule(flags)

    def dsv41_graph_layout(self):
        manager = self.model_runner.cudagraph_manager
        speculator = getattr(self.model_runner, "speculator", None)
        draft_manager = getattr(speculator, "query_cudagraph_manager", None)
        return {
            "captured_token_counts": manager.captured_token_counts(),
            "decode_query_len": manager.decode_query_len,
            "varlen_decode": manager.varlen_decode,
            "draft_captured_token_counts": draft_manager.captured_token_counts()
            if draft_manager is not None
            else None,
            "draft_query_tokens": getattr(speculator, "num_query_per_req", None),
            "proposed_tokens": getattr(speculator, "num_speculative_steps", None),
            "lookahead_tokens": self.vllm_config.num_lookahead_tokens,
            "draft_attention_speculative_tokens": (
                speculator.attn_vllm_config.num_speculative_tokens
                if speculator is not None
                else None
            ),
        }

    def dsv41_io_stats(self):
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        path = Path("/proc/self/io")
        counters = {
            key: int(value)
            for key, value in (
                line.split(":", 1) for line in path.read_text().splitlines()
            )
        }
        return {
            "pid": os.getpid(),
            **counters,
            "major_faults": usage.ru_majflt,
            "minor_faults": usage.ru_minflt,
            "monotonic_ns": time.monotonic_ns(),
        }

    def dsv41_host_memory(self):
        path = Path("/proc/self/smaps_rollup")
        if not path.exists():
            return {}
        return {
            line.split(":")[0]: line.split(":")[1].strip()
            for line in path.read_text().splitlines()
            if line.startswith(
                ("Rss:", "Pss:", "Pss_Anon:", "Pss_File:", "AnonHugePages:", "Locked:")
            )
        }

    def dsv41_cpu_stats(self):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        return {
            name: module.backend.cuda_stats()
            for name, module in self.model_runner.model.named_modules()
            if isinstance(module, CPUExpertModule)
        }

    def dsv41_cpu_profile_start(self, interval):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        for module in self.model_runner.model.modules():
            if isinstance(module, CPUExpertModule):
                module.backend.set_profile(interval)

    def dsv41_cpu_profile_finish(self, directory):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        result = {}
        for name, module in self.model_runner.model.named_modules():
            if isinstance(module, CPUExpertModule):
                result[name] = module.backend.profile_stats(
                    Path(directory) / f"{name}.bin"
                )
                module.backend.set_profile(0)
        return result

    def dsv41_cache_stats(self):
        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        return {
            name: {
                "device": str(module.gpu_cache.device),
                "weight_source": "cpu_resident"
                if module.gpu_cache.weight_source is not None
                else "checkpoint",
                "checkpoint_view_bytes": sum(
                    part.numel() * part.element_size()
                    for pair in module.gpu_cache.weights.values()
                    for part in pair
                ),
                "prompt_calibration": module.gpu_cache.calibrate_from_prompt,
                "dynamic_enabled": module.gpu_cache.dynamic_enabled,
                "feedback_enabled": module.gpu_cache.feedback_enabled,
                "learning": module.gpu_cache.learning_state(),
                "lru_experts": list(module.gpu_cache.host_lru_slots),
                "dynamic": module.gpu_cache.dynamic_stats,
                "dynamic_pinned": sorted(module.gpu_cache.dynamic_pinned),
                "experts": module.gpu_cache.selected,
                "calibrations": module.gpu_cache.calibrations,
                "reloaded_experts": module.gpu_cache.reloaded_experts,
                "reload_seconds": module.gpu_cache.reload_seconds,
                "resident_bytes": sum(
                    t.numel() * t.element_size() for t in module.gpu_cache.packed
                ),
            }
            for name, module in self.model_runner.model.named_modules()
            if isinstance(module, CPUExpertModule) and module.gpu_cache is not None
        }

    def dsv41_cache_policy(self, policy, reference=None):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        start = time.perf_counter()
        torch.accelerator.synchronize()
        saved = getattr(self, "dsv41_saved_cache_policies", {})
        previous = getattr(self, "dsv41_active_cache_policy", None)
        for name, module in self.model_runner.model.named_modules():
            if not isinstance(module, CPUExpertModule) or module.gpu_cache is None:
                continue
            cache = module.gpu_cache
            if cache.dynamic_policy is not None:
                if previous is not None:
                    saved[previous, name] = {
                        "experts": list(cache.selected),
                        "learning": cache.learning_state(),
                    }
                else:
                    cache.host_lru_slots.clear()
                state = saved.get((policy, name))
                layer = name.split(".layers.")[1].split(".")[0]
                initial = (
                    reference[layer]
                    if reference is not None and policy == "static"
                    else module.backend.config.gpu_cache_static_experts
                )
                experts = state["experts"] if state is not None else initial
                cache.set_feedback_enabled(False)
                cache.select(experts, preserve_slots=False)
                cache.restore_learning_state(
                    state["learning"] if state is not None else None
                )
                if policy == "feedback" and cache.decode_feedback is None:
                    raise ValueError("Feedback policy requires decode feedback buffers")
                cache.dynamic_enabled = policy in ("adaptive", "feedback")
                cache.set_feedback_enabled(policy == "feedback")
            cache.calibrate_from_prompt = policy == "prompt"
            cache.set_enabled(policy != "collect")
        self.dsv41_saved_cache_policies = saved
        self.dsv41_active_cache_policy = policy
        return {
            "previous": previous,
            "policy": policy,
            "restore_seconds": time.perf_counter() - start,
        }

    def dsv41_finish_cache_calibration(self, before):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        torch.accelerator.synchronize()
        result = {}
        for name, module in self.model_runner.model.named_modules():
            if not isinstance(module, CPUExpertModule) or module.gpu_cache is None:
                continue
            previous = next(rank[name] for rank in before if name in rank)
            after = module.backend.cuda_stats()
            counts = torch.tensor(after["expert_counts"]) - torch.tensor(
                previous["expert_counts"]
            )
            if counts.sum().item() == 0:
                raise RuntimeError("No independent calibration routes were collected")
            cache = module.gpu_cache
            selected = counts.argsort(descending=True, stable=True)[: cache.capacity]
            cache.select_static(selected.tolist())
            cache.set_enabled(True)
            layer = name.split(".layers.")[1].split(".")[0]
            result[layer] = {
                "experts": cache.selected,
                "counts": counts.tolist(),
                "calibration_route_coverage": (
                    counts[selected].sum() / counts.sum()
                ).item(),
            }
        return result

    def dsv41_capture_quality_logits(self, enabled):
        if not enabled:
            self.model_runner.sample = self.dsv41_original_sample
            result = self.dsv41_quality_logits
            self.dsv41_quality_logits = None
            return result
        self.dsv41_quality_logits = None
        original = self.model_runner.sample
        self.dsv41_original_sample = original

        def sample(hidden_states, input_batch, grammar_output):
            result = original(hidden_states, input_batch, grammar_output)
            if (
                input_batch.has_prefill
                and input_batch.num_reqs == 1
                and int(input_batch.num_computed_prefill_tokens_np[0])
                + input_batch.num_tokens
                == int(input_batch.prefill_len_np[0])
            ):
                hidden = hidden_states[input_batch.logits_indices]
                logits = self.model_runner.model.compute_logits(hidden)
                self.dsv41_quality_logits = logits[-1].float().cpu().tolist()
            return result

        self.model_runner.sample = sample
        return None

    def dsv41_device_memory(self):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        devices = {torch.accelerator.current_device_index()}
        devices.update(
            module.gpu_cache.device.index
            for module in self.model_runner.model.modules()
            if isinstance(module, CPUExpertModule) and module.gpu_cache is not None
        )
        return {
            device: {
                "allocated": torch.accelerator.memory_allocated(device),
                "reserved": torch.accelerator.memory_reserved(device),
                "peak_allocated": torch.accelerator.max_memory_allocated(device),
                "peak_reserved": torch.accelerator.max_memory_reserved(device),
            }
            for device in sorted(devices)
        }

    def dsv41_ced_stats(self):
        from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4Model

        return [
            {
                "encoder_tokens": module.ced_prefill.encoder_tokens,
                "decoder_replay_tokens": module.ced_prefill.decoder_tokens,
                "replay_calls": module.ced_prefill.replay_calls,
            }
            for module in self.model_runner.model.modules()
            if isinstance(module, DeepseekV4Model) and module.ced_prefill is not None
        ]


def make_prompt(tokenizer, length, run):
    marker = "__REFERENCE_DOCUMENT_INSERTION__"
    message = (
        "Read the reference document below.\n"
        f"The audit code for this document is ORCHID-{731 + run}.\n"
        + marker
        + "\nFirst repeat the document's audit code exactly. Then write a detailed "
        "Python implementation and explanation of an LRU cache with get and put "
        "operations. Include at least eight tests and discuss every edge case."
    )
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = template.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    available = length - len(prefix_ids) - len(suffix_ids)
    if available < 0:
        raise ValueError("The requested prompt is shorter than the benchmark question")
    paragraph = tokenizer.encode(
        "\nReference note: A cache stores recently used values. A dictionary maps "
        "keys to entries. A doubly linked list records access order. Each lookup "
        "moves an existing entry to the front. Insertion evicts the least recently "
        "used entry when capacity is exceeded. Tests should cover updates, missing "
        "keys, capacity one, repeated access, and independent cache instances.\n",
        add_special_tokens=False,
    )
    filler = (paragraph * math.ceil(available / len(paragraph)))[:available]
    tokens = prefix_ids + filler + suffix_ids
    assert len(tokens) == length
    return tokens


def load_prompt_suite(path, length):
    suite = json.loads(path.read_text())
    cases = suite["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Prompt suite requires nonempty, unique case IDs")
    for case in cases:
        tokens = case["prompt_token_ids"]
        if len(tokens) != length or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError(f"Invalid token IDs or prompt length for {case['id']}")
        if not all(case.get(key) for key in ("category", "label", "audit_code")):
            raise ValueError(f"Missing prompt metadata for {case['id']}")
    return suite


def calibrate_static_cache(llm, tokenizer):
    from vllm import SamplingParams

    # Fixed generic tasks are independent of make_prompt and its measured outputs.
    questions = [
        (
            "Explain binary search, write a Python implementation, and give examples "
            "for empty input, odd length, duplicates, and values outside the range."
        ),
        (
            "Write a Python parser for CSV records containing quoted fields and "
            "escaped quotes. Explain its state transitions and demonstrate usage."
        ),
        (
            "Explain why seasons occur, how the water cycle works, and how these "
            "processes interact. Use clear prose with concrete examples."
        ),
        (
            "Compare breadth-first and depth-first graph traversal. Explain their "
            "invariants, complexity, and applications with a worked example."
        ),
        "用中文详细解释如何求解一元二次方程，分别讨论两个实根、重根和复根，给出计算例子。",
        (
            "Write a JSON schema for a bookstore inventory and a Python validator. "
            "Explain validation errors, optional fields, and numeric constraints."
        ),
    ]
    started = time.perf_counter()
    llm.collective_rpc("dsv41_cache_policy", args=("collect",))
    before = llm.collective_rpc("dsv41_cpu_stats")
    samples = []
    for question in questions:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = llm.generate(
            {"prompt_token_ids": tokens},
            SamplingParams(temperature=0, min_tokens=128, max_tokens=128, logprobs=1),
            use_tqdm=False,
        )[0]
        answer = response.outputs[0]
        assert len(answer.token_ids) == 128 and not response.metrics.is_corrupted
        assert all(
            math.isfinite(item.logprob)
            for step in answer.logprobs
            for item in step.values()
        )
        samples.append(
            {
                "question": question,
                "input_tokens": tokens,
                "output_tokens": answer.token_ids,
                "text": answer.text,
            }
        )
    selection = llm.collective_rpc("dsv41_finish_cache_calibration", args=(before,))
    return {
        "setup_seconds": time.perf_counter() - started,
        "samples": samples,
        "layers": {key: value for rank in selection for key, value in rank.items()},
    }


def check_bounded_replay(llm, tokenizer, length, output_path, replay_cache=None):
    import numpy as np
    import torch

    from vllm import SamplingParams

    marker = "__QUALITY_REFERENCE_INSERTION__"
    message = (
        "ARCHIVE_ID=CED-CHECK-8263. Warehouse_A has 17 units. "
        "Warehouse_B has 25 units.\n"
        + marker
        + "\nReply first with the ARCHIVE_ID at the beginning of this document "
        "and the sum of Warehouse_A and Warehouse_B units. Then explain briefly."
    )
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = template.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    available = length - len(prefix_ids) - len(suffix_ids)
    paragraph = tokenizer.encode(
        "Inventory notes record ordinary transactions, receiving dates, product "
        "labels, and shipment destinations. Inspect each record carefully and "
        "preserve the original quantities when computing totals.\n",
        add_special_tokens=False,
    )
    assert available >= 0
    filler = (paragraph * math.ceil(available / len(paragraph)))[:available]
    tokens = prefix_ids + filler + suffix_ids
    runs, logits = [], []
    for full_prompt in (True, False):
        if replay_cache is not None:
            llm.collective_rpc(
                "dsv41_set_replay_cache", args=(replay_cache and not full_prompt,)
            )
        llm.collective_rpc("dsv41_capture_quality_logits", args=(True,))
        try:
            response = llm.generate(
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0,
                    min_tokens=64,
                    max_tokens=64,
                    logprobs=1,
                    prompt_logprobs=0 if full_prompt else None,
                ),
                use_tqdm=False,
            )[0]
        finally:
            captured = llm.collective_rpc("dsv41_capture_quality_logits", args=(False,))
        vector = next(item for item in captured if item is not None)
        logits.append(np.asarray(vector, dtype=np.float32))
        answer = response.outputs[0]
        assert not response.metrics.is_corrupted
        assert all(
            math.isfinite(item.logprob)
            for step in answer.logprobs
            for item in step.values()
        )
        record = {
            "full_prompt": full_prompt,
            "text": answer.text,
            "token_ids": answer.token_ids,
            "audit_code_found": "CED-CHECK-8263" in answer.text,
            "sum_found": "42" in answer.text,
            "selected_logprobs": [
                step[token].logprob
                for token, step in zip(answer.token_ids, answer.logprobs)
            ],
        }
        runs.append(record)
    reference, bounded = (torch.from_numpy(value).double() for value in logits)
    assert torch.isfinite(reference).all() and torch.isfinite(bounded).all()
    reference_logprob, bounded_logprob = (
        value.log_softmax(0) for value in (reference, bounded)
    )
    artifact = output_path.with_name(output_path.stem + f".quality-{length}-logits.npz")
    np.savez_compressed(artifact, full=logits[0], bounded=logits[1])
    return {
        "prompt_tokens": len(tokens),
        "prompt_sha256": hashlib.sha256(bytes(str(tokens), "utf-8")).hexdigest(),
        "first_logit_correlation": np.corrcoef(*logits)[0, 1].item(),
        "first_logit_kl_full_to_bounded": (
            reference_logprob.exp() * (reference_logprob - bounded_logprob)
        )
        .sum()
        .item(),
        "first_logit_argmax_equal": reference.argmax().item()
        == bounded.argmax().item(),
        "logit_artifact": artifact.name,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("llama", "ik", "kt"), default="ik")
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--thread-sweep", type=int, nargs="+")
    parser.add_argument("--cpu-phase-threads", type=int, nargs=2)
    parser.add_argument("--pp-kv-transfer", action="store_true")
    parser.add_argument(
        "--dsv41-attention", choices=("portable", "sm120_decode"), default="portable"
    )
    parser.add_argument("--cpu-executor-sweep", choices=("graph", "compact"), nargs="+")
    parser.add_argument("--cpu-schedule-sweep", type=int, nargs="+")
    parser.add_argument("--cpu-profile-last-block", action="store_true")
    parser.add_argument("--cpu-profile-dir", type=Path)
    parser.add_argument("--cpu-profile-interval", type=int, default=8)
    parser.add_argument("--gpu-cache-experts", type=int, default=0)
    parser.add_argument("--gpu-cache-device", type=int)
    parser.add_argument(
        "--cache-policy-sweep",
        choices=("prompt", "static", "adaptive", "feedback"),
        nargs="+",
    )
    parser.add_argument("--static-cache-selections", type=Path)
    parser.add_argument("--reference-cache-selections", type=Path)
    parser.add_argument("--static-cache-devices", type=Path)
    parser.add_argument("--dynamic-cache-config", type=Path)
    parser.add_argument("--interleave-cache-policies", action="store_true")
    parser.add_argument("--cache-selection-sweep", type=Path, nargs="+")
    parser.add_argument("--record-expert-routes", action="store_true")
    parser.add_argument("--prefill-diagnostic-runs", type=int, default=0)
    parser.add_argument("--diagnostic-case-ids", nargs="+")
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--quality-lengths", type=int, nargs="+")
    parser.add_argument("--quality-after", action="store_true")
    parser.add_argument("--quality-replay-caches", type=int, choices=(0, 1), nargs="+")
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--local-argmax-reduction", action="store_true")
    parser.add_argument("--draft-query-tokens", type=int)
    parser.add_argument("--cuda-library", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--prompt-suite", type=Path)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-batched-tokens", type=int, default=1024)
    parser.add_argument("--pipeline-layers", type=int, nargs="+", default=[8, 6, 26])
    parser.add_argument("--prefill-chunk-sweep", type=int, nargs="+")
    parser.add_argument("--replay-cache-sweep", type=int, choices=(0, 1), nargs="+")
    parser.add_argument(
        "--prefill-index-sweep",
        choices=("torch", "fused"),
        nargs="+",
        help="One implementation per chunk-sweep block, or one for all blocks",
    )
    parser.add_argument("--host-io-stats", action="store_true")
    parser.add_argument("--profile-prefill", action="store_true")
    parser.add_argument("--profile-prefill-last-block", action="store_true")
    parser.add_argument("--profile-prefill-last-blocks", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--ced", action="store_true")
    parser.add_argument("--profile-graphs", action="store_true")
    parser.add_argument("--profile-graphs-last-block", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--graph-mode", choices=("PIECEWISE", "FULL_DECODE_ONLY"), default="PIECEWISE"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompt_suite = (
        load_prompt_suite(args.prompt_suite, args.prompt_tokens)
        if args.prompt_suite
        else None
    )
    chunks = args.prefill_chunk_sweep or [args.max_batched_tokens]
    implementations = args.prefill_index_sweep or [None]
    replay_caches = args.replay_cache_sweep or [0]
    needs_chunk_scheduler = any(chunk != args.max_batched_tokens for chunk in chunks)
    if len(replay_caches) not in (1, len(chunks)):
        parser.error("Use one replay cache mode, or one per chunk block")
    if len(implementations) not in (1, len(chunks)):
        parser.error("Use one prefill implementation, or one per chunk block")
    if args.profile_prefill_last_blocks < 1:
        parser.error("Profile at least one prefill block")
    if args.record_expert_routes and not args.eager:
        parser.error(
            "Full route recording requires --eager; captured graphs skip hooks"
        )
    if args.prefill_diagnostic_runs < 0:
        parser.error("Diagnostic run count must be nonnegative")
    if args.diagnostic_case_ids and (
        prompt_suite is None
        or set(args.diagnostic_case_ids) - {c["id"] for c in prompt_suite["cases"]}
    ):
        parser.error("Diagnostic case IDs must belong to the prompt suite")
    reference = None
    if args.reference_cache_selections:
        if not args.static_cache_selections:
            parser.error("Reference cache requires initial expert selections")
        reference = json.loads(args.reference_cache_selections.read_text())
        initial = json.loads(args.static_cache_selections.read_text())
        if set(reference) != set(initial) or any(
            len(reference[layer]) != len(initial[layer])
            or len(set(reference[layer])) != len(reference[layer])
            or any(type(e) is not int or not 0 <= e < 384 for e in reference[layer])
            for layer in initial
        ):
            parser.error("Reference cache must preserve per-layer capacities")
    cache_paths = args.cache_selection_sweep or [args.static_cache_selections]
    if len(cache_paths) not in (1, len(chunks)):
        parser.error("Use one cache selection file, or one per chunk block")
    cache_selections = {}
    if args.cache_selection_sweep:
        if not args.static_cache_selections:
            parser.error("Cache sweeps require initial fixed selections")
        original = json.loads(args.static_cache_selections.read_text())
        for path in cache_paths:
            selection = json.loads(path.read_text())
            if set(selection) != set(original) or any(
                len(selection[layer]) != len(original[layer])
                or len(set(selection[layer])) != len(selection[layer])
                or any(type(e) is not int or not 0 <= e < 384 for e in selection[layer])
                for layer in original
            ):
                parser.error("Cache sweeps must preserve every layer's capacity")
            cache_selections[str(path)] = selection
    prefill_control = args.output.with_suffix(".prefill-control.json")
    if needs_chunk_scheduler:
        if any(
            not 16 <= chunk <= args.max_batched_tokens
            for chunk in args.prefill_chunk_sweep
        ):
            parser.error("Prefill chunks must be between 16 and max-batched-tokens")
        prefill_control.parent.mkdir(parents=True, exist_ok=True)
        prefill_control.write_text(
            json.dumps({"chunk_tokens": args.prefill_chunk_sweep[0]})
        )
        os.environ["DSV41_PREFILL_CONTROL"] = str(prefill_control.resolve())
    if (
        args.cache_policy_sweep or args.static_cache_selections
    ) and not args.gpu_cache_experts:
        parser.error("Cache policy selection requires a nonzero GPU cache")
    if args.dynamic_cache_config:
        if not args.static_cache_selections or replay_caches != [1]:
            parser.error(
                "Dynamic cache trials require initial static slots and replay cache 1"
            )
        if "prompt" in (args.cache_policy_sweep or []):
            parser.error("Dynamic slots cannot be combined with prompt calibration")
    elif {"adaptive", "feedback"} & set(args.cache_policy_sweep or []):
        parser.error("Adaptive policy requires a dynamic cache config")
    if (
        args.static_cache_selections
        and args.cache_policy_sweep
        and not args.dynamic_cache_config
    ):
        parser.error("Choose a fixed selection file or an online policy comparison")
    if args.static_cache_devices and not args.static_cache_selections:
        parser.error("Per-layer cache placement requires fixed expert selections")
    if args.kv_cache_mib <= 0:
        parser.error("KV cache reservation must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if any(not item for item in visible) or len(visible) < len(args.pipeline_layers):
        parser.error("Select at least one visible GPU per pipeline stage")
    if (
        any(layers <= 0 for layers in args.pipeline_layers)
        or sum(args.pipeline_layers) != 40
    ):
        parser.error("Pipeline layers must be positive and sum to 40")
    model_config = json.loads((args.model / "config.json").read_text())
    text_config = model_config.get("text_config", model_config)
    kv_sources = set(text_config["kv_source_layer_ids"])
    boundary = 0
    for layers in args.pipeline_layers[:-1]:
        boundary += layers
        if (
            not args.pp_kv_transfer
            and text_config["compress_ratios"][boundary] > 0
            and boundary not in kv_sources
        ):
            parser.error(f"Pipeline boundary {boundary} splits a shared KV group")
    if args.ced and boundary >= 20:
        parser.error("CED requires encoder layer 19 and the decoder on the last rank")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "pinned_max_round_threshold_mb:128")
    os.environ["VLLM_PP_LAYER_PARTITION"] = ",".join(map(str, args.pipeline_layers))
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = (
        "1" if not args.eager and args.graph_mode == "PIECEWISE" else "0"
    )
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        pipeline_parallel_size=len(args.pipeline_layers),
        distributed_executor_backend="mp",
        speculative_config={
            "method": "dspark",
            "model": str(args.model.resolve()),
            "num_speculative_tokens": args.speculative_tokens,
            "draft_sample_method": "greedy",
            "enable_adaptive_verification": False,
            "use_local_argmax_reduction": args.local_argmax_reduction,
            "dspark_num_query_tokens": args.draft_query_tokens,
        }
        if args.speculative_tokens
        else None,
        worker_extension_cls="benchmark_dsv41_hybrid.HybridBenchmarkWorker",
        max_model_len=max(1024, args.prompt_tokens + args.output_tokens + 128),
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        scheduler_cls="dsv41_prefill_diagnostics.PrefillBenchmarkScheduler"
        if needs_chunk_scheduler
        else None,
        gpu_memory_utilization=0.95,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        limit_mm_per_prompt={"image": 0},
        engram_config={"cpu_offload": True},
        enforce_eager=args.eager,
        disable_log_stats=False,
        per_request_spec_decode_metrics=(
            "summary" if prompt_suite and args.speculative_tokens else "none"
        ),
        compilation_config=None
        if args.eager
        else {
            "cudagraph_mode": args.graph_mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "max_cudagraph_capture_size": 16,
        },
        additional_config={
            "ced_prefill": args.ced,
            "dsv41_attention": args.dsv41_attention,
            "pp_kv_transfer": args.pp_kv_transfer,
            "cpu_phase_threads": args.cpu_phase_threads,
            "cpu_moe": {
                "backend": args.backend,
                "library_path": str(args.library.resolve()),
                "num_threads": args.threads,
                "gpu_cache_experts": args.gpu_cache_experts,
                "gpu_cache_dynamic": json.loads(args.dynamic_cache_config.read_text())
                if args.dynamic_cache_config
                else None,
                "gpu_cache_prefill": bool(args.dynamic_cache_config),
                "gpu_cache_device": args.gpu_cache_device,
                "gpu_cache_devices": json.loads(args.static_cache_devices.read_text())
                if args.static_cache_devices
                else None,
                "gpu_cache_selections": json.loads(
                    args.static_cache_selections.read_text()
                )
                if args.static_cache_selections
                else None,
                "cuda_library_path": str(args.cuda_library.resolve())
                if args.cuda_library
                else None,
                "start_layer": 20,
                "end_layer": 40,
            },
        },
    )
    payload = {
        "arguments": {
            key: str(value)
            if isinstance(value, Path)
            else [str(item) if isinstance(item, Path) else item for item in value]
            if isinstance(value, list)
            else value
            for key, value in vars(args).items()
        },
        "load_seconds": time.perf_counter() - started,
        "visible_devices": visible,
        "cpu_environment": {
            key: os.environ.get(key)
            for key in (
                "DSV41_CPU_HUGEPAGES",
                "DSV41_CPU_EXECUTOR",
                "CUDA_MPS_PIPE_DIRECTORY",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "OMP_NUM_THREADS",
                "OMP_WAIT_POLICY",
                "GOMP_SPINCOUNT",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "placement": llm.collective_rpc("inspect_dsv41_placement"),
        "host_memory": llm.collective_rpc("dsv41_host_memory"),
        "io_after_load": llm.collective_rpc("dsv41_io_stats")
        if args.host_io_stats
        else None,
        "target_graph_layout": llm.collective_rpc("dsv41_graph_layout"),
        "pipeline_config": llm.collective_rpc("dsv41_pipeline_config"),
        "timing": "prefill = prompt tokens / TTFT, including decoder replay; "
        "decode = (actual output tokens - 1) / (last - first token time)",
        "runs": [],
    }
    if cache_selections:
        payload["cache_selection_sweep"] = {
            path: {
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "experts": selections,
            }
            for path, selections in cache_selections.items()
        }
    if prompt_suite:
        payload["prompt_suite"] = {
            "sha256": hashlib.sha256(args.prompt_suite.read_bytes()).hexdigest(),
            "cases": [
                {key: value for key, value in case.items() if key != "prompt_token_ids"}
                for case in prompt_suite["cases"]
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    save()

    def run_quality():
        if args.quality_lengths:
            if not args.ced:
                raise ValueError("Bounded replay quality comparison requires --ced")
            payload["bounded_replay_quality"] = []
            for replay_cache in args.quality_replay_caches or [replay_caches[-1]]:
                for length in args.quality_lengths:
                    suffix = "" if replay_cache is None else f".cache{replay_cache}"
                    output = args.output.with_name(args.output.stem + suffix + ".json")
                    quality = check_bounded_replay(
                        llm, tokenizer, length, output, replay_cache
                    )
                    quality["replay_cache_enabled"] = replay_cache
                    quality["reference_replay_cache_enabled"] = False
                    payload["bounded_replay_quality"].append(quality)
                    save()
                    print(
                        json.dumps(
                            {
                                "quality_prompt_tokens": length,
                                "first_logit_correlation": quality[
                                    "first_logit_correlation"
                                ],
                                "first_logit_kl": quality[
                                    "first_logit_kl_full_to_bounded"
                                ],
                                "answers_correct": [
                                    item["audit_code_found"] and item["sum_found"]
                                    for item in quality["runs"]
                                ],
                            }
                        ),
                        flush=True,
                    )

    tokenizer = llm.get_tokenizer()
    try:
        if args.ced:
            sanity_tokens = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "What is 17 + 25? Reply with only the number.",
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert len(sanity_tokens) <= 128
            sanity = []
            for prompt_logprobs in (0, None):
                answer = llm.generate(
                    {"prompt_token_ids": sanity_tokens},
                    SamplingParams(
                        temperature=0,
                        max_tokens=8,
                        logprobs=1,
                        prompt_logprobs=prompt_logprobs,
                    ),
                    use_tqdm=False,
                )[0].outputs[0]
                sanity.append(
                    {
                        "full_prompt": prompt_logprobs is not None,
                        "text": answer.text,
                        "token_ids": answer.token_ids,
                        "selected_logprobs": [
                            step[token].logprob
                            for token, step in zip(answer.token_ids, answer.logprobs)
                        ],
                    }
                )
            payload["short_prompt_parity"] = sanity
            save()
            assert sanity[0]["token_ids"] == sanity[1]["token_ids"], sanity
            assert "42" in sanity[1]["text"], sanity
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(
                        sanity[0]["selected_logprobs"], sanity[1]["selected_logprobs"]
                    )
                )
                < 0.05
            ), sanity
        if not args.quality_after:
            run_quality()
        policies = args.cache_policy_sweep or [
            "adaptive"
            if args.dynamic_cache_config
            else "static"
            if args.static_cache_selections
            else "prompt"
        ]
        schedules = args.cpu_schedule_sweep or [None]
        configurations = [
            (
                (block * len(schedules) + schedule_block) * len(chunks) + chunk_block,
                executor,
                schedule,
                chunk,
                implementations[chunk_block % len(implementations)],
                bool(replay_caches[chunk_block % len(replay_caches)]),
                policy,
                threads,
                run,
                case,
                cache_paths[chunk_block % len(cache_paths)],
            )
            for block, executor in enumerate(args.cpu_executor_sweep or [None])
            for schedule_block, schedule in enumerate(schedules)
            for chunk_block, chunk in enumerate(chunks)
            for policy in policies
            for threads in (args.thread_sweep or [args.threads])
            for run in range(
                args.warmup_runs + args.runs + args.prefill_diagnostic_runs
            )
            for case in (prompt_suite["cases"] if prompt_suite else [None])
            if run < args.warmup_runs + args.runs
            or not args.diagnostic_case_ids
            or case["id"] in args.diagnostic_case_ids
        ]
        if args.interleave_cache_policies:
            configurations.sort(
                key=lambda c: (
                    c[0],
                    c[7],
                    c[8],
                    policies.index(c[6])
                    if c[8] % 2 == 0
                    else len(policies) - 1 - policies.index(c[6]),
                )
            )
        active_policy = None
        active_executor = None
        active_cache = str(args.static_cache_selections)
        last_block = max(config[0] for config in configurations)
        for (
            block,
            executor,
            schedule,
            chunk,
            implementation,
            replay_cache,
            policy,
            threads,
            run,
            case,
            cache_path,
        ) in configurations:
            diagnostic = bool(args.cpu_profile_dir) and (
                not args.cpu_profile_last_block or block == last_block
            )
            graph_diagnostic = args.profile_graphs and (
                not args.profile_graphs_last_block or block == last_block
            )
            extra_diagnostic = run >= args.warmup_runs + args.runs
            prefill_diagnostic = (
                extra_diagnostic
                or args.profile_prefill
                and (
                    not args.profile_prefill_last_block
                    or block > last_block - args.profile_prefill_last_blocks
                )
            )
            if str(cache_path) != active_cache:
                llm.collective_rpc(
                    "dsv41_select_cache", args=(cache_selections[str(cache_path)],)
                )
                active_cache = str(cache_path)
            if run == 0:
                llm.collective_rpc("dsv41_set_replay_cache", args=(replay_cache,))
            if implementation is not None and run == 0:
                llm.collective_rpc(
                    "dsv41_set_prefill_index_impl", args=(implementation,)
                )
            if needs_chunk_scheduler and run == 0:
                prefill_control.write_text(json.dumps({"chunk_tokens": chunk}))
            if schedule is not None and run == 0:
                llm.collective_rpc("dsv41_set_cpu_schedule", args=(schedule,))
            if executor is not None and executor != active_executor:
                llm.collective_rpc("dsv41_set_cpu_executor", args=(executor,))
                active_executor = executor
            if policy != active_policy:
                if policy == "static" and not args.static_cache_selections:
                    calibration = calibrate_static_cache(llm, tokenizer)
                    payload["static_cache_calibration"] = calibration
                    selection = {
                        key: value["experts"]
                        for key, value in calibration["layers"].items()
                    }
                    args.output.with_suffix(".selections.json").write_text(
                        json.dumps(selection, indent=2) + "\n"
                    )
                    save()
                io_before_transition = (
                    llm.collective_rpc("dsv41_io_stats") if args.host_io_stats else None
                )
                transition = llm.collective_rpc(
                    "dsv41_cache_policy", args=(policy, reference)
                )
                payload.setdefault("cache_policy_transitions", []).append(
                    {
                        "before_run": len(payload["runs"]),
                        "state_restore": transition,
                        "io_before": io_before_transition,
                        "io_after": llm.collective_rpc("dsv41_io_stats")
                        if args.host_io_stats
                        else None,
                        "timing": "Policy restoration outside timed requests; "
                        "causal dynamic updates remain inside generate and TTFT",
                    }
                )
                active_policy = policy
            if args.thread_sweep and run == 0:
                llm.collective_rpc("dsv41_set_cpu_threads", args=(threads,))
            if diagnostic:
                llm.collective_rpc(
                    "dsv41_cpu_profile_start", args=(args.cpu_profile_interval,)
                )
            tokens = (
                case["prompt_token_ids"]
                if case is not None
                else make_prompt(tokenizer, args.prompt_tokens, run)
            )
            audit_code = case["audit_code"] if case else f"ORCHID-{731 + run}"
            before = llm.collective_rpc("dsv41_ced_stats")
            cpu_before = llm.collective_rpc("dsv41_cpu_stats")
            cache_before = llm.collective_rpc("dsv41_cache_stats")
            if graph_diagnostic:
                llm.collective_rpc("dsv41_start_timing")
            if prefill_diagnostic:
                llm.collective_rpc("dsv41_start_prefill_timing")
            record_routes = args.record_expert_routes or extra_diagnostic
            if record_routes:
                llm.collective_rpc(
                    "dsv41_start_routes", args=(not args.record_expert_routes,)
                )
            io_before = (
                llm.collective_rpc("dsv41_io_stats") if args.host_io_stats else None
            )
            start = time.perf_counter()
            response = llm.generate(
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0,
                    max_tokens=args.output_tokens,
                    min_tokens=args.output_tokens,
                    logprobs=1,
                ),
                use_tqdm=False,
            )[0]
            elapsed = time.perf_counter() - start
            io_after = (
                llm.collective_rpc("dsv41_io_stats") if args.host_io_stats else None
            )
            output = response.outputs[0]
            metrics = response.metrics
            assert len(output.token_ids) == args.output_tokens
            assert not metrics.is_corrupted
            assert all(
                math.isfinite(item.logprob)
                for step in output.logprobs
                for item in step.values()
            )
            decode_seconds = metrics.last_token_ts - metrics.first_token_ts
            record = {
                "warmup": run < args.warmup_runs,
                "diagnostic": extra_diagnostic or args.record_expert_routes,
                "cache_selection_file": str(cache_path) if cache_path else None,
                "full_route_recording": args.record_expert_routes,
                "repetition": run,
                "case_id": case["id"] if case else None,
                "category": case["category"] if case else None,
                "cpu_threads": threads,
                "cpu_executor": executor,
                "cpu_schedule": schedule,
                "cpu_profile_enabled": diagnostic,
                "graph_profile_enabled": graph_diagnostic,
                "prefill_profile_enabled": prefill_diagnostic,
                "prefill_chunk_tokens": chunk,
                "prefill_index_impl": implementation,
                "replay_cache_enabled": replay_cache,
                "executor_block": block,
                "cache_policy": policy,
                "prompt_sha256": hashlib.sha256(
                    bytes(str(tokens), "utf-8")
                ).hexdigest(),
                "prompt_tokens": len(response.prompt_token_ids),
                "output_tokens": len(output.token_ids),
                "ttft_seconds": metrics.first_token_latency,
                "prefill_tokens_per_second": len(tokens) / metrics.first_token_latency,
                "decode_tokens_per_second": (len(output.token_ids) - 1)
                / decode_seconds,
                "elapsed_seconds": elapsed,
                "metrics": asdict(metrics),
                "text": output.text,
                "token_ids": output.token_ids,
                "audit_code_found": audit_code in output.text,
                "finish_reason": output.finish_reason,
                "spec_decode_metrics": output.spec_decode_metrics.to_dict()
                if output.spec_decode_metrics is not None
                else None,
                "ced_before": before,
                "ced_after": llm.collective_rpc("dsv41_ced_stats"),
                "expert_route_histograms": llm.collective_rpc("dsv41_stop_routes")
                if record_routes
                else None,
                "io_before": io_before,
                "io_after": io_after,
                "cpu_before": cpu_before,
                "cpu_after": llm.collective_rpc("dsv41_cpu_stats"),
                "cpu_profile": llm.collective_rpc(
                    "dsv41_cpu_profile_finish",
                    args=(
                        str(
                            args.cpu_profile_dir
                            / f"block{block}-threads{threads}-run{run}"
                        ),
                    ),
                )
                if diagnostic
                else None,
                "gpu_cache_before": cache_before,
                "device_memory": llm.collective_rpc("dsv41_device_memory"),
                "gpu_cache": llm.collective_rpc("dsv41_cache_stats"),
                "decode_stage_graph_ms": llm.collective_rpc("dsv41_stop_timing")
                if graph_diagnostic
                else None,
                "prefill_stage_intervals": llm.collective_rpc(
                    "dsv41_stop_prefill_timing"
                )
                if prefill_diagnostic
                else None,
            }
            payload["runs"].append(record)
            save()
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in (
                            "warmup",
                            "diagnostic",
                            "cache_selection_file",
                            "case_id",
                            "category",
                            "repetition",
                            "cpu_threads",
                            "cpu_executor",
                            "cpu_schedule",
                            "prefill_chunk_tokens",
                            "prefill_index_impl",
                            "replay_cache_enabled",
                            "prefill_profile_enabled",
                            "cpu_profile_enabled",
                            "executor_block",
                            "cache_policy",
                            "prompt_tokens",
                            "output_tokens",
                            "ttft_seconds",
                            "prefill_tokens_per_second",
                            "decode_tokens_per_second",
                            "spec_decode_metrics",
                            "audit_code_found",
                            "ced_before",
                            "ced_after",
                        )
                    }
                ),
                flush=True,
            )
        if args.quality_after:
            if args.dynamic_cache_config:
                quality_policy = "feedback" if "feedback" in policies else "adaptive"
                llm.collective_rpc("dsv41_cache_policy", args=(quality_policy,))
                payload["quality_cache_policy"] = quality_policy
            run_quality()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
