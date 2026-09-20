# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare EXL3 engines using shared token IDs, fixed lengths and zero cache hits.

Run the exllamav3 backend in its own environment with its source on PYTHONPATH.
The input JSON contains model, eos_ids, cases ({name, inputs}) and optional
GSM8K evals ({id, input_ids, answer}). All measurements and completions are saved.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import regex as re
import torch


class VllmBackend:
    def __init__(self, args, data):
        from vllm import LLM

        extension = (
            args.worker_extension_cls
            or args.profile_dir
            or args.event_profile
            or args.moe_workspace_sizes
            or args.moe_variants
            or args.capture_routing_dir
            or args.expected_layer_counts
        )
        if extension:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        self.llm = LLM(
            model=data["model"],
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            pipeline_parallel_size=args.pp,
            max_num_batched_tokens=args.chunk_size,
            enforce_eager=args.eager,
            speculative_config=(
                {"method": "mtp", "num_speculative_tokens": args.mtp}
                if args.mtp
                else None
            ),
            gpu_memory_utilization=0.88,
            kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
            enable_prefix_caching=False,
            disable_log_stats=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            compilation_config=(
                {"cudagraph_capture_sizes": args.cudagraph_capture_sizes}
                if args.cudagraph_capture_sizes
                else {}
            ),
            worker_extension_cls=(
                args.worker_extension_cls
                or (
                    "exl3_profile_worker.Exl3ProfileWorkerExtension"
                    if extension
                    else ""
                )
            ),
            profiler_config=(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(args.profile_dir.resolve()),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": False,
                    "torch_profiler_use_gzip": True,
                }
                if args.profile_dir
                else None
            ),
        )
        self.eos_ids = data["eos_ids"]
        self.synchronize_inputs = args.synchronize_inputs

    def speculative_counters(self):
        from vllm.v1.metrics.reader import Counter

        return {
            metric.name: metric.value
            for metric in self.llm.get_metrics()
            if isinstance(metric, Counter)
            and metric.name.startswith("vllm:spec_decode_")
        }

    def generate(self, inputs, max_tokens, fixed):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            ignore_eos=fixed,
            stop_token_ids=[] if fixed else self.eos_ids,
        )
        start = time.perf_counter()
        prompts = [{"prompt_token_ids": ids} for ids in inputs]
        if self.synchronize_inputs and len(inputs) > 1:
            core = self.llm.llm_engine.engine_core
            core.call_utility("pause_scheduler", "keep", False)
            try:
                self.llm.enqueue(prompts, params, use_tqdm=False)
            finally:
                core.call_utility("resume_scheduler")
            outputs = self.llm.wait_for_completion(use_tqdm=False)
        else:
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        return elapsed, [self.output_row(output) for output in outputs]

    @staticmethod
    def output_row(output):
        stats = output.metrics
        completion = output.outputs[0]
        assert output.num_cached_tokens == 0
        assert stats is not None and not stats.is_corrupted
        return {
            "prompt_tokens": len(output.prompt_token_ids),
            "new_tokens": len(completion.token_ids),
            "token_ids": list(completion.token_ids),
            "text": completion.text,
            "ttft": stats.first_token_latency,
            "prefill": stats.first_token_ts - stats.scheduled_ts,
            "queue": stats.scheduled_ts - stats.queued_ts,
            "first_token_ts": stats.first_token_ts,
            "last_token_ts": stats.last_token_ts,
            "decode": stats.last_token_ts - stats.first_token_ts,
            "cached_tokens": output.num_cached_tokens,
            "finish_reason": completion.finish_reason,
        }

    def generate_staggered(self, case, max_tokens):
        """Admit later requests after the first request has started decoding."""
        from vllm import SamplingParams

        engine = self.llm.llm_engine
        assert not engine.has_unfinished_requests()
        thresholds = case["after_tokens"]
        if thresholds[0] != 0 or max(thresholds) > max_tokens:
            raise ValueError("Invalid staggered admission token thresholds")
        prefix = str(time.perf_counter_ns())
        names = [f"exl3-{prefix}-{i}" for i in range(len(thresholds))]
        arrived, completed, counts, arrivals = {}, {}, {}, {}
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        start = time.perf_counter()
        while len(completed) < len(names):
            for i, name in enumerate(names):
                if name not in arrived and thresholds[i] <= counts.get(names[0], 0):
                    arrived[name] = time.perf_counter() - start
                    engine.add_request(
                        name, {"prompt_token_ids": case["inputs"][i]}, params
                    )
            for output in engine.step():
                name = output.request_id
                count = len(output.outputs[0].token_ids)
                now = time.perf_counter() - start
                if count > counts.get(name, 0):
                    arrivals.setdefault(name, []).append([count, now])
                counts[name] = count
                if output.finished:
                    completed[name] = self.output_row(output)
        rows = []
        for name in names:
            row = completed[name]
            row.update(arrival_offset=arrived[name], token_arrivals=arrivals[name])
            rows.append(row)
        return time.perf_counter() - start, rows


class ExllamaBackend:
    def __init__(self, args, data):
        if args.chunk_size is None:
            args.chunk_size = 512
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer

        config = Config.from_directory(data["model"])
        self.model = Model.from_config(config)
        cache = Cache(
            self.model,
            max_num_tokens=args.exl_cache_tokens,
            max_batch_size=args.batch_size,
        )
        load_args = {
            "max_chunk_size": args.chunk_size,
            "max_batch_size": args.batch_size,
        }
        if args.pp == 1:
            load_args["device"] = "cuda:0"
        else:
            load_args["use_per_device"] = args.exl_memory
        self.model.load(**load_args)
        self.generator = Generator(
            self.model,
            cache,
            Tokenizer(config),
            max_batch_size=args.batch_size,
            max_chunk_size=args.chunk_size,
        )
        self.eos_ids = data["eos_ids"]
        self.module_devices = [
            (getattr(m, "key", type(m).__name__), str(m.device))
            for m in self.model.modules
        ]

    def generate(self, inputs, max_tokens, fixed):
        from exllamav3 import ArgmaxSampler, Job

        gen = self.generator
        assert gen.num_remaining_jobs() == 0
        gen.pagetable.reset_page_table()
        jobs = [
            Job(
                input_ids=torch.tensor([ids], dtype=torch.long),
                max_new_tokens=max_tokens,
                sampler=ArgmaxSampler(),
                stop_conditions=[] if fixed else self.eos_ids,
                identifier=i,
            )
            for i, ids in enumerate(inputs)
        ]
        token_ids = [[] for _ in jobs]
        rows = [None for _ in jobs]
        start = time.perf_counter()
        progress = [0] * len(jobs)
        next_progress = start + 30
        for job in jobs:
            gen.enqueue(job)
        while gen.num_remaining_jobs():
            for result in gen.iterate():
                idx = result["identifier"]
                if result.get("stage") == "prefill":
                    progress[idx] = result["curr_progress"]
                if "token_ids" in result:
                    token_ids[idx].extend(result["token_ids"].flatten().tolist())
                if result.get("eos"):
                    assert result["cached_tokens"] == 0, result
                    rows[idx] = {
                        "prompt_tokens": result["prompt_tokens"],
                        "new_tokens": result["new_tokens"],
                        "token_ids": token_ids[idx],
                        "text": result["full_completion"],
                        "ttft": result["time_enqueued"] + result["time_prefill"],
                        "prefill": result["time_prefill"],
                        "decode": result["time_generate"],
                        "cached_tokens": result["cached_tokens"],
                        "finish_reason": result["eos_reason"],
                    }
            now = time.perf_counter()
            if now >= next_progress:
                print(
                    "PREFILL_PROGRESS",
                    sum(progress),
                    sum(len(ids) - 1 for ids in inputs),
                    flush=True,
                )
                next_progress = now + 30
        return time.perf_counter() - start, rows


def response_before_stop(token_ids, eos_ids, tokenizer):
    """Decode the first response from a fixed-length generation."""
    end = next(
        (i for i, token in enumerate(token_ids) if token in eos_ids), len(token_ids)
    )
    return tokenizer.decode(token_ids[:end], skip_special_tokens=True)


def answer_value(text):
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "exllamav3"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--expected-layer-counts", type=int, nargs="+")
    parser.add_argument("--exl-memory", type=float, nargs="+", default=[31, 40, 40, 48])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-gib", type=float, default=2)
    parser.add_argument("--exl-cache-tokens", type=int, default=16384)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, help="Defaults to --tokens")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--eval-tokens", type=int, default=768)
    parser.add_argument("--eval-batch-size", type=int, help="Defaults to --batch-size")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--mtp", type=int, default=0, help="vLLM MTP draft tokens")
    parser.add_argument(
        "--synchronize-inputs",
        action="store_true",
        help="Queue each input batch before resuming the vLLM scheduler",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        type=int,
        nargs="+",
        help="Override serving graph sizes; omitted uses normal engine defaults",
    )
    parser.add_argument(
        "--profile-case", help="Case name to profile; defaults to first"
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--event-profile", action="store_true")
    parser.add_argument("--profile-tokens", type=int, default=1)
    parser.add_argument("--moe-workspace-sizes", type=int, nargs="+")
    parser.add_argument("--moe-variants", type=Path)
    parser.add_argument("--worker-extension-cls", default="")
    parser.add_argument("--capture-routing-dir", type=Path)
    parser.add_argument("--capture-routing-max-rows", type=int)
    args = parser.parse_args()
    if args.synchronize_inputs and args.backend != "vllm":
        parser.error("--synchronize-inputs requires the vllm backend")
    if args.mtp < 0 or (args.mtp and args.backend != "vllm"):
        parser.error("--mtp requires a nonnegative count and the vllm backend")
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if not 1 <= args.eval_batch_size <= args.batch_size:
        parser.error("--eval-batch-size must be between 1 and --batch-size")
    if args.profile_dir and args.backend != "vllm":
        parser.error("--profile-dir requires the vllm backend")
    if args.expected_layer_counts and (
        len(args.expected_layer_counts) != args.pp
        or any(count <= 0 for count in args.expected_layer_counts)
    ):
        parser.error("--expected-layer-counts requires one positive count per rank")
    payload = args.inputs.read_bytes()
    if args.inputs.suffix == ".gz":
        payload = gzip.decompress(payload)
    data = json.loads(payload)
    for case in data["cases"]:
        if "expected_strings" in case and len(case["expected_strings"]) != len(
            case["inputs"]
        ):
            parser.error("Each input needs one expected string")
        if len(case["inputs"]) > args.batch_size:
            parser.error("Input batch exceeds --batch-size")
        if any(len(ids) + args.tokens > args.max_model_len for ids in case["inputs"]):
            parser.error("Input plus output exceeds --max-model-len")
    profile_case = data["cases"][0] if data["cases"] else None
    if args.profile_case:
        profile_case = next(
            (case for case in data["cases"] if case["name"] == args.profile_case), None
        )
        if profile_case is None:
            parser.error("Unknown --profile-case")
    check_tokenizer = None
    if any("expected_strings" in case for case in data["cases"]):
        from tokenizers import Tokenizer

        check_tokenizer = Tokenizer.from_file(
            str(Path(data["model"]) / "tokenizer.json")
        )
    if (args.profile_dir or args.event_profile or args.capture_routing_dir) and (
        profile_case is None
    ):
        parser.error("Profiling and routing capture require at least one case")
    result = {
        "answer_check_scope": "before_first_stop_token",
        "warmup_new_tokens": args.warmup_tokens or args.tokens,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "model": data["model"],
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(i)
            for i in range(torch.accelerator.device_count())
        ],
        "environment": {
            k: os.environ.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "EXL3_INT8_GEMV",
                "VLLM_EXL3_MOE_MAX_TOKENS",
                "VLLM_EXL3_MOE_PRIORITY",
                "VLLM_EXL3_MOE_DECODE",
                "VLLM_EXL3_MOE_BATCHED_DECODE",
                "VLLM_EXL3_MOE_PREFILL",
                "VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP",
                "VLLM_EXL3_MOE_M_TILE",
                "OMP_NUM_THREADS",
                "VLLM_PP_LAYER_PARTITION",
                "NCCL_P2P_DISABLE",
            ]
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "perf": [],
        "eval": [],
    }

    def save():
        if args.mtp:
            result["speculative_metrics"] = [
                vars(metric)
                for metric in backend.llm.get_metrics()
                if metric.name.startswith("vllm:spec_decode_")
            ]
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temporary.replace(args.output)

    start = time.perf_counter()
    backend = (
        VllmBackend(args, data)
        if args.backend == "vllm"
        else ExllamaBackend(args, data)
    )
    result["load_seconds"] = time.perf_counter() - start
    if args.backend == "exllamav3":
        from exllamav3.version import __version__

        result["exllamav3_version"] = __version__
        result["module_devices"] = backend.module_devices
    else:
        result["resolved_chunk_size"] = (
            backend.llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
        )
        result["resolved_cudagraph_capture_sizes"] = (
            backend.llm.llm_engine.vllm_config.compilation_config.cudagraph_capture_sizes
        )
        result["exllamav3_version"] = importlib.metadata.version("exllamav3")
        if (
            args.worker_extension_cls
            or args.event_profile
            or args.profile_dir
            or args.moe_variants
            or args.moe_workspace_sizes
            or args.capture_routing_dir
            or args.expected_layer_counts
        ):
            result["runtime_before"] = backend.llm.collective_rpc(
                "get_exl3_runtime_state"
            )
    save()
    if args.expected_layer_counts:
        if args.backend == "exllamav3":
            counts = [
                sum(
                    device == f"cuda:{i}" and bool(re.search(r"\.layers\.\d+$", name))
                    for name, device in backend.module_devices
                )
                for i in range(args.pp)
            ]
        else:
            runtimes = sorted(result["runtime_before"], key=lambda r: r["pp_rank"])
            actual = [
                sorted(layer["index"] for layer in rank["decoder_layers"])
                for rank in runtimes
            ]
            expected = []
            offset = 0
            for count in args.expected_layer_counts:
                expected.append(list(range(offset, offset + count)))
                offset += count
            print("DECODER_LAYER_IDS", actual, flush=True)
            if actual != expected:
                raise ValueError(f"Unexpected decoder placement: {actual}")
            counts = [len(indices) for indices in actual]
        print("LAYER_COUNTS", counts, flush=True)
        if counts != args.expected_layer_counts:
            raise ValueError(f"Unexpected decoder placement: {counts}")
    variants = (
        json.loads(args.moe_variants.read_text())
        if args.moe_variants
        else [{"capacity": capacity} for capacity in args.moe_workspace_sizes or [None]]
    )
    for variant in variants:
        capacity = variant["capacity"]
        if capacity is not None:
            result.setdefault("workspace_sweep", []).append(
                backend.llm.collective_rpc(
                    "configure_exl3_optimization", args=(variant,)
                )
            )
        for case in [] if args.skip_perf else data["cases"]:
            # Warm every batch/prefill shape, then reset caches before measuring.
            print("WARMUP", case["name"], flush=True)

            def generate_case(tokens, case=case):
                if "after_tokens" in case:
                    if args.backend != "vllm":
                        raise ValueError("Staggered cases require the vllm backend")
                    return backend.generate_staggered(case, tokens)
                return backend.generate(case["inputs"], tokens, True)

            for _ in range(args.warmups):
                generate_case(args.warmup_tokens or args.tokens)
            for repeat in range(args.repeats):
                print("MEASURE", case["name"], repeat, flush=True)
                previous_spec = backend.speculative_counters() if args.mtp else {}
                elapsed, rows = generate_case(args.tokens)
                assert all(row["new_tokens"] == args.tokens for row in rows)
                record = {
                    "case": case["name"],
                    "workspace_capacity": capacity,
                    "variant": variant,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "throughput": sum(r["new_tokens"] for r in rows) / elapsed,
                    "rows": rows,
                }
                if args.mtp:
                    record["speculative_counters"] = {
                        key: value - previous_spec.get(key, 0)
                        for key, value in backend.speculative_counters().items()
                    }
                if "expected_strings" in case:
                    assert check_tokenizer is not None
                    record["checks"] = []
                    for expected, row in zip(case["expected_strings"], rows):
                        answer = response_before_stop(
                            row["token_ids"], data["eos_ids"], check_tokenizer
                        )
                        pattern = r"\b" + re.escape(expected) + r"\b"
                        record["checks"].append(
                            {
                                "expected": expected,
                                "answer_text": answer,
                                "matched": bool(re.search(pattern, answer)),
                            }
                        )
                    print("CHECK", case["name"], repeat, record["checks"], flush=True)
                record["max_ttft"] = max(row["ttft"] for row in rows)
                result["perf"].append(record)
                print("PERF", case["name"], repeat, record["throughput"], flush=True)
                print("MAX_TTFT", case["name"], record["max_ttft"], flush=True)
                save()
    if args.capture_routing_dir:
        backend.llm.collective_rpc(
            "start_exl3_route_capture",
            args=(
                str(args.capture_routing_dir.resolve()),
                args.capture_routing_max_rows,
            ),
        )
        try:
            backend.generate(profile_case["inputs"], args.profile_tokens, True)
        finally:
            result["routing_samples"] = backend.llm.collective_rpc(
                "finish_exl3_route_capture"
            )
        save()
    if args.profile_dir:
        backend.llm.start_profile()
        try:
            elapsed, rows = backend.generate(
                profile_case["inputs"], args.profile_tokens, True
            )
            result["profile"] = {"seconds": elapsed, "rows": rows}
        finally:
            backend.llm.stop_profile()
        save()
    if args.profile_dir or args.event_profile:
        result["event_profile_install"] = backend.llm.collective_rpc(
            "install_exl3_event_profile"
        )
        try:
            elapsed, rows = backend.generate(
                profile_case["inputs"], args.profile_tokens, True
            )
            result["event_profile"] = {"seconds": elapsed, "rows": rows}
        finally:
            ranks = backend.llm.collective_rpc("collect_exl3_event_profile")
        result["event_profile"]["ranks"] = ranks
        save()
    if not args.skip_eval:
        start = time.perf_counter()
        for offset in range(0, len(data["evals"]), args.eval_batch_size):
            batch = data["evals"][offset : offset + args.eval_batch_size]
            elapsed, rows = backend.generate(
                [r["input_ids"] for r in batch], args.eval_tokens, False
            )
            for request, row in zip(batch, rows):
                predicted = answer_value(row["text"])
                correct = predicted is not None and float(predicted) == float(
                    request["answer"]
                )
                result["eval"].append(
                    {
                        "id": request["id"],
                        "expected": request["answer"],
                        "predicted": predicted,
                        "correct": correct,
                        **row,
                    }
                )
            print(
                "EVAL",
                len(result["eval"]),
                sum(r["correct"] for r in result["eval"]),
                flush=True,
            )
            result["eval_seconds"] = time.perf_counter() - start
            save()
    if "runtime_before" in result:
        result["runtime_after"] = backend.llm.collective_rpc("get_exl3_runtime_state")
        save()
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
