# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure decode stalls while new long prompts enter a hybrid HTTP server."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx


def summarize(rows):
    first = min(row["first"] for row in rows)
    last = max(row["last"] for row in rows)
    start = min(row["started"] for row in rows)
    outputs = sum(row["usage"]["completion_tokens"] for row in rows)
    decode_tokens = outputs - len(rows)
    result = {
        "requests": len(rows),
        "outputs": outputs,
        "decode_seconds": last - first,
        "aggregate_decode_tps": decode_tokens / (last - first),
        "request_seconds": last - start,
        "output_tps": outputs / (last - start),
        "mean_ttft_seconds": sum(row["first"] - row["started"] for row in rows)
        / len(rows),
        "max_token_gap_seconds": max(row["max_token_gap_seconds"] for row in rows),
    }

    # Count actual streamed tokens, excluding each request's first output.
    def window(begin, end):
        if end <= begin:
            return None
        count = 0
        for row in rows:
            previous = 0
            for timestamp, total in row["token_events"]:
                if begin < timestamp <= end:
                    count += max(0, total - max(1, previous))
                previous = total
        return {"seconds": end - begin, "tokens": count, "tps": count / (end - begin)}

    result["all_decoding"] = window(
        max(row["first"] for row in rows), min(row["last"] for row in rows)
    )
    if len(rows) > 1:
        result["prefill_overlap"] = window(
            max(first, min(row["started"] for row in rows[1:])),
            max(row["first"] for row in rows),
        )
    return result


async def run(args):
    cases = [
        case
        for case in json.loads(args.suite.read_text())["cases"]
        if case["id"] in args.cases
    ]
    if not cases:
        raise ValueError("No matching prompt cases")
    for case in cases:
        ids = case["prompt_token_ids"]
        if args.prompt_tokens is not None:
            if args.prompt_tokens < 608:
                raise ValueError("Prompts must contain at least 608 tokens")
            head, body, tail = ids[:96], ids[96:-512], ids[-512:]
            size = args.prompt_tokens - len(head) - len(tail)
            case["prompt_token_ids"] = (
                head + (body * (size // len(body) + 1))[:size] + tail
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(time.time_ns())
    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=1800,
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        deadline = time.monotonic() + args.wait_ready
        while True:
            try:
                response = await client.get("/v1/models", timeout=3)
                response.raise_for_status()
                model = response.json()["data"][0]["id"]
                break
            except (httpx.HTTPError, KeyError, ValueError):
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(2)

        async def consume(case, label, num_outputs, ready=None):
            payload = {
                "model": model,
                "prompt": case["prompt_token_ids"],
                "temperature": 0,
                "seed": 20260927,
                "max_tokens": num_outputs,
                "ignore_eos": True,
                "cache_salt": hashlib.sha256(f"{run_id}/{label}".encode()).hexdigest(),
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                    "continuous_usage_stats": True,
                },
            }
            started = time.perf_counter()
            first = last = None
            usage = None
            events, text = [], []
            max_gap = 0.0
            async with client.stream("POST", "/v1/completions", json=payload) as r:
                if r.status_code != 200:
                    raise RuntimeError((await r.aread()).decode())
                async for line in r.aiter_lines():
                    if not line.startswith("data: {"):
                        continue
                    event = json.loads(line[6:])
                    now = time.perf_counter()
                    usage = event.get("usage") or usage
                    content = "".join(
                        c.get("text", "") for c in event.get("choices", [])
                    )
                    if not content:
                        continue
                    if first is None:
                        first = now
                        if ready is not None:
                            ready.set()
                    if last is not None:
                        max_gap = max(max_gap, now - last)
                    last = now
                    text.append(content)
                    if event.get("usage"):
                        events.append([now, event["usage"]["completion_tokens"]])
            if (
                first is None
                or usage is None
                or usage["completion_tokens"] != num_outputs
            ):
                raise RuntimeError(f"Incomplete response: {label}, {usage}")
            if usage.get("prompt_tokens_details", {}).get("cached_tokens", 0):
                raise RuntimeError("Unique cache salt unexpectedly matched a prefix")
            return {
                "label": label,
                "case": case["id"],
                "started": started,
                "first": first,
                "last": last,
                "usage": usage,
                "decode_tps": (num_outputs - 1) / (last - first),
                "max_token_gap_seconds": max_gap,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(case["prompt_token_ids"]).encode()
                ).hexdigest(),
                "text": "".join(text),
                "token_events": events,
            }

        with args.output.open("w") as output:

            def save(value):
                output.write(json.dumps(value, ensure_ascii=False) + "\n")
                output.flush()

            for iteration in range(args.warmup_rounds):
                for case in cases:
                    row = await consume(
                        case, f"warmup/{iteration}/{case['id']}", args.output_tokens
                    )
                    save({"kind": "warmup", "rows": [row]})
                    print("WARMUP", row["label"], row["decode_tps"], flush=True)

            for iteration in range(args.rounds):
                for concurrency in args.concurrency:
                    for pattern in args.patterns if concurrency > 1 else ["single"]:
                        label = f"{iteration}/{concurrency}/{pattern}"
                        ready = asyncio.Event()

                        async def submit(
                            index, pattern=pattern, ready=ready, label=label
                        ):
                            if index and pattern == "stagger":
                                await ready.wait()
                                await asyncio.sleep(args.arrival_delay * index)
                            case = cases[index % len(cases)]
                            return await consume(
                                case,
                                f"{label}/{index}",
                                args.output_tokens,
                                ready if index == 0 else None,
                            )

                        before = (await client.get("/metrics")).text
                        tasks = [
                            asyncio.create_task(submit(i)) for i in range(concurrency)
                        ]
                        try:
                            rows = await asyncio.gather(*tasks)
                        finally:
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                        after = (await client.get("/metrics")).text
                        result = {
                            "kind": "measure",
                            "label": label,
                            "summary": summarize(rows),
                            "rows": rows,
                            "metrics_before": before,
                            "metrics_after": after,
                        }
                        save(result)
                        print(label, json.dumps(result["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["code_python"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["burst", "stagger"],
        default=["burst", "stagger"],
    )
    parser.add_argument("--arrival-delay", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--wait-ready", type=float, default=0)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or args.output_tokens < 2 or args.arrival_delay < 0:
        parser.error(
            "Positive concurrency, at least two outputs and nonnegative delay required"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
