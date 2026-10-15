# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare route-tail skipping on one drained, Graph-mode hybrid server.

Serve with VLLM_SERVER_DEV_MODE=1 and the RouteTailSkipBenchmarkWorker extension.
Run quality checks with the repository root on PYTHONPATH.
"""

import argparse
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from benchmark_dsv41_hybrid import HybridBenchmarkWorker
from benchmark_dsv41_serving import request


class RouteTailSkipBenchmarkWorker(HybridBenchmarkWorker):
    def tail_control(self, action, rank=0):
        import torch

        from vllm.models.deepseek_v4_1.cpu_moe import CPUExpertModule

        state = self.model_runner.model_state
        if getattr(state, "hybrid_requests", None):
            raise RuntimeError("Tail-skip controls require drained requests")
        torch.accelerator.synchronize()
        modules = {
            name: module
            for name, module in self.model_runner.model.named_modules()
            if isinstance(module, CPUExpertModule)
        }
        cached = {name: m for name, m in modules.items() if m.gpu_cache is not None}
        if action == "initial" and not hasattr(self, "tail_initial"):
            self.tail_initial = {
                name: {
                    "selected": list(module.gpu_cache.selected),
                    "learning": copy.deepcopy(module.gpu_cache.learning_state()),
                    "policy": copy.deepcopy(module.gpu_cache.dynamic_policy),
                }
                for name, module in cached.items()
            }
            self.tail_pool_initial = {
                pool: {
                    cache: (cache.slot_ids.copy(), list(cache.selected))
                    for cache in pool.caches
                }
                for pool in getattr(state, "hybrid_capacity_pools", [])
            }
        elif action == "restore":
            for pool, saved in self.tail_pool_initial.items():
                initial = pool.initial
                pool.initial = saved
                try:
                    pool.restore_initial()
                finally:
                    pool.initial = initial
            for name, module in cached.items():
                cache = module.gpu_cache
                saved = self.tail_initial[name]
                cache.set_feedback_enabled(False)
                if list(cache.selected) != saved["selected"]:
                    cache.select(saved["selected"], preserve_slots=False)
                cache.restore_learning_state(copy.deepcopy(saved["learning"]))
                cache.dynamic_policy = copy.deepcopy(saved["policy"])
                cache.host_lru_slots.clear()
                cache.dynamic_enabled = True
                cache.set_feedback_enabled(True)
            for module in modules.values():
                module.backend.set_route_tail_skip(rank)
        elif action not in ("initial", "snapshot"):
            raise ValueError(action)
        caches = self.dsv41_cache_stats()
        return {
            "cache": caches,
            "cpu": self.dsv41_cpu_stats(),
            "memory": self.dsv41_device_memory(),
            "pipeline": self.dsv41_pipeline_config(),
            "initial": hasattr(self, "tail_initial"),
            "selection_hash": hashlib.sha256(
                json.dumps(
                    {n: list(m.gpu_cache.selected) for n, m in cached.items()},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "rank": {name: m.backend.route_tail_skip for name, m in modules.items()},
            "task_modes": {
                name: {
                    "decode": sum(bool(key[-1]) for key in m.backend._cuda_tasks),
                    "other": sum(not key[-1] for key in m.backend._cuda_tasks),
                }
                for name, m in modules.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18003")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["zh_story", "en_story", "code_python"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 5, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--quality-questions", type=int, default=16)
    args = parser.parse_args()
    if args.concurrency < 1 or args.prompt_tokens < 608:
        parser.error("Concurrency must be positive; prompt tokens must be at least 608")
    client = httpx.Client(base_url=args.url, timeout=1800)

    def rpc(action, rank=0):
        response = client.post(
            "/collective_rpc",
            json={
                "method": "tail_control",
                "args": [action, rank],
                "timeout": 600,
            },
        )
        if response.status_code == 404:
            raise RuntimeError("The benchmark server needs VLLM_SERVER_DEV_MODE=1")
        response.raise_for_status()
        return response.json()["results"]

    def reset(rank):
        result = rpc("restore", rank)
        response = client.post("/reset_prefix_cache")
        response.raise_for_status()
        if not response.json()["success"]:
            raise RuntimeError("The benchmark requires a successful prefix cache reset")
        return result

    def spec_counters():
        counters = {}
        for line in client.get("/metrics").text.splitlines():
            if line.startswith("vllm:spec_decode_") and "_total" in line:
                key = line.split("{", 1)[0]
                counters[key] = counters.get(key, 0) + float(line.rsplit(" ", 1)[-1])
        return counters

    cases = json.loads(args.suite.read_text())["cases"]
    cases = [c for c in cases if c["id"] in args.cases]
    if len(cases) != len(args.cases):
        parser.error("The suite must contain every selected case exactly once")
    for case in cases:
        ids = case["prompt_token_ids"]
        head, body, tail = ids[:96], ids[96:-512], ids[-512:]
        length = args.prompt_tokens - len(head) - len(tail)
        case["input_ids"] = head + (body * (length // len(body) + 1))[:length] + tail
    common = {
        "model": "local",
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.output_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as file:

        def measure(rank, label, batch):
            before = reset(rank)
            if rank and not any(
                v["decode"] for worker in before for v in worker["task_modes"].values()
            ):
                raise RuntimeError("No captured decode callbacks accept tail skipping")
            spec_before = spec_counters()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                rows = list(
                    pool.map(
                        lambda item: request(
                            client,
                            "/v1/completions",
                            common | {"prompt": item[1]["input_ids"]},
                            f"{label}/{item[0]}/{item[1]['id']}",
                        ),
                        enumerate(batch),
                    )
                )
            if any(len(r["token_ids"]) != args.output_tokens for r in rows):
                raise RuntimeError("Every request must return the requested token IDs")
            if len(rows) == 1:
                row = rows[0]
            else:
                started = min(r["started_monotonic"] for r in rows)
                first = min(r["first_token_monotonic"] for r in rows)
                last = max(r["last_token_monotonic"] for r in rows)
                initial_outputs = sum(
                    max((n for t, n in r["token_events"] if t <= first), default=0)
                    for r in rows
                )
                outputs = sum(r["usage"]["completion_tokens"] for r in rows)
                row = {
                    "label": label,
                    "requests": rows,
                    "usage": {
                        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
                        "completion_tokens": outputs,
                    },
                    "request_seconds": last - started,
                    "ttft_seconds": first - started,
                    "decode_tps": (outputs - initial_outputs) / (last - first),
                    "decode_counting": "batch_first_to_last_output",
                }
            after = rpc("snapshot")
            if rank:

                def skipped(workers):
                    return sum(
                        value.get("tail_skipped_routes", 0)
                        for worker in workers
                        for value in worker["cpu"].values()
                        if value is not None
                    )

                row["removed_routes"] = skipped(after) - skipped(before)
            spec_after = spec_counters()
            row.update(
                rank=rank,
                concurrency=len(rows),
                before=before,
                after=after,
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                speculation={
                    k: v - spec_before.get(k, 0) for k, v in spec_after.items()
                },
            )
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "label",
                            "rank",
                            "usage",
                            "decode_tps",
                            "ttft_seconds",
                            "request_seconds",
                        )
                    }
                ),
                flush=True,
            )

        if all(r["initial"] for r in rpc("snapshot")):
            reset(0)
        for case in cases:
            request(
                client,
                "/v1/completions",
                common | {"prompt": case["input_ids"][:2048], "max_tokens": 128},
                "warmup/" + case["id"],
            )
        rpc("initial")
        if args.quality_dir:
            from tests.evals.gsm8k import gsm8k_eval as evaluation

            evaluation.download_and_cache_file = lambda url, filename=None: str(
                args.quality_dir / url.rsplit("/", 1)[-1]
            )
            prompts, answers = evaluation._build_gsm8k_prompts(
                args.quality_questions, 5
            )
            for rank in args.ranks:
                reset(rank)
                rows = []

                def evaluate(item):
                    i, prompt = item
                    response = client.post(
                        "/v1/completions",
                        json={
                            "model": "local",
                            "prompt": prompt,
                            "temperature": 0,
                            "seed": 0,
                            "max_tokens": 512,
                            "return_token_ids": True,
                            "stop": ["Question", "Assistant:", "<|separator|>"],
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    prediction = evaluation.get_answer_value(body["choices"][0]["text"])
                    return {
                        "index": i,
                        "correct": prediction == answers[i],
                        "answer": answers[i],
                        "prediction": prediction,
                        "response": body,
                    }

                before = rpc("snapshot")
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    for start in range(0, len(prompts), args.concurrency):
                        rows.extend(
                            pool.map(
                                evaluate,
                                enumerate(
                                    prompts[start : start + args.concurrency], start
                                ),
                            )
                        )
                row = {
                    "label": "quality",
                    "rank": rank,
                    "rows": rows,
                    "correct": sum(r["correct"] for r in rows),
                    "questions": len(rows),
                    "concurrency": args.concurrency,
                    "before": before,
                    "after": rpc("snapshot"),
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(
                    json.dumps(
                        {k: row[k] for k in ("label", "rank", "correct", "questions")}
                    ),
                    flush=True,
                )
        else:
            batches = (
                [(case["id"], [case]) for case in cases]
                if args.concurrency == 1
                else [
                    (
                        "mixed",
                        [cases[i % len(cases)] for i in range(args.concurrency)],
                    )
                ]
            )
            for iteration in range(args.rounds):
                ranks = args.ranks if iteration % 2 == 0 else args.ranks[::-1]
                for name, batch in batches:
                    for rank in ranks:
                        measure(rank, f"measure/{iteration}/{name}/{rank}", batch)


if __name__ == "__main__":
    main()
