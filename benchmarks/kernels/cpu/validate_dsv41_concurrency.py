# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check concurrent text and image request isolation on an existing HTTP server."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import pybase64 as base64


async def validate(args):
    async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        image = (
            base64.b64encode(args.image.read_bytes()).decode() if args.image else None
        )

        async def ask(index):
            a, b = 17 + index, 31 + 2 * index
            expected = str(a * b)
            content = f"计算 {a} 乘以 {b}，只输出十进制整数，不要解释。"
            if image and index == 0:
                expected = "42"
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + image},
                    },
                    {
                        "type": "text",
                        "text": "图中两个矩形的数字之和是多少？只输出整数。",
                    },
                ]
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"thinking": False},
                "temperature": 0,
                "max_tokens": 64,
            }
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            actual = result["choices"][0]["message"]["content"].strip()
            return {
                "index": index,
                "expected": expected,
                "actual": actual,
                "pass": actual == expected,
                "response": result,
            }

        rows = await asyncio.gather(*(ask(index) for index in range(16)))
        health = (await client.get("/health")).status_code
        metrics = (await client.get("/metrics")).text
        result = {
            "time": time.time(),
            "requests": rows,
            "health": health,
            "metrics": metrics,
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "passes": sum(row["pass"] for row in rows),
                    "requests": len(rows),
                    "health": health,
                }
            ),
            flush=True,
        )
        assert health == 200 and all(row["pass"] for row in rows)


async def validate_capacity(args):
    """Keep 16 long requests alive until the requested capacity is resident."""
    cases = json.loads(args.capacity_suite.read_text())["cases"]
    cases = [
        c for c in cases if c["id"] in {"zh_analysis", "en_analysis", "code_python"}
    ]
    ready = asyncio.Event()
    states = [{"first": False, "outputs": 0} for _ in range(16)]
    snapshot = None
    async with httpx.AsyncClient(base_url=args.url, timeout=1800) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]

        async def consume(index):
            nonlocal snapshot
            base = cases[index % len(cases)]["prompt_token_ids"]
            head, body, tail = base[:96], base[96:-512], base[-512:]
            count = args.capacity_prompt_tokens - len(head) - len(tail)
            ids = head + (body * (count // len(body) + 1))[:count] + tail
            payload = {
                "model": model,
                "prompt": ids,
                "temperature": 0,
                "max_tokens": 8192,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                    "continuous_usage_stats": True,
                },
            }
            async with client.stream(
                "POST", "/v1/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: {"):
                        continue
                    event = json.loads(line[6:])
                    state = states[index]
                    if event.get("usage"):
                        state["outputs"] = event["usage"]["completion_tokens"]
                    if any(choice.get("text") for choice in event.get("choices", [])):
                        state["first"] = True
                    if snapshot is None and all(s["first"] for s in states):
                        snapshot = {
                            "time": time.time(),
                            "requests": [dict(s) for s in states],
                            "committed_token_positions": sum(
                                args.capacity_prompt_tokens + s["outputs"] - 1
                                for s in states
                            ),
                        }
                        for _ in range(20):
                            metrics = (await client.get("/metrics")).text
                            running_now = [
                                float(line.rsplit(" ", 1)[1])
                                for line in metrics.splitlines()
                                if line.startswith("vllm:num_requests_running{")
                            ]
                            if sum(running_now) == 16:
                                break
                            await asyncio.sleep(1)
                        snapshot["running_at_check"] = running_now
                        snapshot["preemptions_at_check"] = [
                            float(line.rsplit(" ", 1)[1])
                            for line in metrics.splitlines()
                            if line.startswith("vllm:num_preemptions_total{")
                        ]
                        ready.set()
                    if ready.is_set():
                        break
                else:
                    raise RuntimeError(
                        "A long request ended before all 16 contexts were ready"
                    )

        await asyncio.gather(*(consume(index) for index in range(16)))
        assert snapshot is not None
        snapshot["prompt_tokens_each"] = args.capacity_prompt_tokens
        snapshot["cancelled_after_capacity_check"] = True
        for _ in range(60):
            metrics = (await client.get("/metrics")).text
            running = [
                float(line.rsplit(" ", 1)[1])
                for line in metrics.splitlines()
                if line.startswith("vllm:num_requests_running{")
            ]
            if running and sum(running) == 0:
                break
            await asyncio.sleep(1)
        snapshot["running_after_cancel"] = running
        snapshot["health"] = (await client.get("/health")).status_code
        args.output.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(json.dumps(snapshot), flush=True)
        assert snapshot["committed_token_positions"] >= args.capacity_tokens
        assert sum(snapshot["running_at_check"]) == 16
        assert snapshot["preemptions_at_check"]
        assert sum(snapshot["preemptions_at_check"]) == 0
        assert running and sum(running) == 0 and snapshot["health"] == 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--capacity-suite", type=Path)
    parser.add_argument("--capacity-prompt-tokens", type=int, default=131060)
    parser.add_argument("--capacity-tokens", type=int, default=2097152)
    args = parser.parse_args()
    asyncio.run(validate_capacity(args) if args.capacity_suite else validate(args))


if __name__ == "__main__":
    main()
