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
import time
from pathlib import Path

import regex as re
import torch


class VllmBackend:
    def __init__(self, args, data):
        from vllm import LLM

        self.llm = LLM(
            model=data["model"],
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            pipeline_parallel_size=args.pp,
            max_num_batched_tokens=args.chunk_size,
            enforce_eager=args.eager,
            gpu_memory_utilization=0.88,
            kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
            enable_prefix_caching=False,
            disable_log_stats=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
        )
        self.eos_ids = data["eos_ids"]

    def generate(self, inputs, max_tokens, fixed):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            ignore_eos=fixed,
            stop_token_ids=[] if fixed else self.eos_ids,
        )
        start = time.perf_counter()
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in inputs], params, use_tqdm=False
        )
        elapsed = time.perf_counter() - start
        rows = []
        for output in outputs:
            stats = output.metrics
            completion = output.outputs[0]
            assert output.num_cached_tokens == 0
            assert stats is not None and not stats.is_corrupted
            rows.append(
                {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "new_tokens": len(completion.token_ids),
                    "token_ids": list(completion.token_ids),
                    "text": completion.text,
                    "ttft": stats.first_token_latency,
                    "prefill": stats.first_token_ts - stats.scheduled_ts,
                    "decode": stats.last_token_ts - stats.first_token_ts,
                    "cached_tokens": output.num_cached_tokens,
                    "finish_reason": completion.finish_reason,
                }
            )
        return elapsed, rows


class ExllamaBackend:
    def __init__(self, args, data):
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
        for job in jobs:
            gen.enqueue(job)
        while gen.num_remaining_jobs():
            for result in gen.iterate():
                idx = result["identifier"]
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
        return time.perf_counter() - start, rows


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
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--eval-tokens", type=int, default=768)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
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
    result = {
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
        result["exllamav3_version"] = importlib.metadata.version("exllamav3")
    save()
    if args.backend == "exllamav3" and args.expected_layer_counts:
        counts = [
            sum(
                device == f"cuda:{i}" and bool(re.search(r"\.layers\.\d+$", name))
                for name, device in backend.module_devices
            )
            for i in range(args.pp)
        ]
        print("LAYER_COUNTS", counts, flush=True)
        if counts != args.expected_layer_counts:
            raise ValueError(f"Unexpected decoder placement: {counts}")
    if not args.skip_perf:
        for case in data["cases"]:
            # Warm every batch/prefill shape, then reset caches before measuring.
            print("WARMUP", case["name"], flush=True)
            for _ in range(args.warmups):
                backend.generate(case["inputs"], 16, True)
            for repeat in range(args.repeats):
                elapsed, rows = backend.generate(case["inputs"], args.tokens, True)
                assert all(row["new_tokens"] == args.tokens for row in rows)
                record = {
                    "case": case["name"],
                    "repeat": repeat,
                    "seconds": elapsed,
                    "throughput": sum(r["new_tokens"] for r in rows) / elapsed,
                    "rows": rows,
                }
                if "expected_strings" in case:
                    record["checks"] = [
                        {
                            "expected": expected,
                            "matched": bool(
                                re.search(
                                    r"\b" + re.escape(expected) + r"\b", row["text"]
                                )
                            ),
                        }
                        for expected, row in zip(case["expected_strings"], rows)
                    ]
                    print("CHECK", case["name"], repeat, record["checks"], flush=True)
                record["max_ttft"] = max(row["ttft"] for row in rows)
                result["perf"].append(record)
                print("PERF", case["name"], repeat, record["throughput"], flush=True)
                print("MAX_TTFT", case["name"], record["max_ttft"], flush=True)
                save()
    if not args.skip_eval:
        start = time.perf_counter()
        for offset in range(0, len(data["evals"]), args.batch_size):
            batch = data["evals"][offset : offset + args.batch_size]
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
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
