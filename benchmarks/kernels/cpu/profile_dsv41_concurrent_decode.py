# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile a bounded all-decoding window, then cancel the diagnostic requests."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def run(args):
    cases = json.loads(args.suite.read_text())["cases"]
    cases = [
        c for c in cases if c["id"] in {"zh_analysis", "en_analysis", "code_python"}
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    first = asyncio.Event()
    stop = asyncio.Event()
    states = [{"first": False, "outputs": 0} for _ in range(16)]
    async with httpx.AsyncClient(base_url=args.url, timeout=1200) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]

        async def rpc(action, **kwargs):
            response = await client.post(
                "/collective_rpc",
                json={
                    "method": "dsv41_decode_control",
                    "args": [json.dumps({"action": action, **kwargs})],
                    "timeout": 180,
                },
            )
            response.raise_for_status()
            return response.json()["results"]

        def save(name, data):
            (args.output / name).write_text(json.dumps(data, indent=2) + "\n")

        async def consume(index):
            payload = {
                "model": model,
                "prompt": cases[index % len(cases)]["prompt_token_ids"],
                "temperature": 0,
                "seed": 20260915,
                "max_tokens": 1024,
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
                    if event.get("usage"):
                        states[index]["outputs"] = event["usage"]["completion_tokens"]
                    if any(c.get("text") for c in event.get("choices", [])):
                        states[index]["first"] = True
                    if all(state["first"] for state in states):
                        first.set()
                    if stop.is_set():
                        break
                else:
                    raise RuntimeError("A request ended before the diagnostic window")

        async def profile():
            await first.wait()
            print(
                "All 16 requests have output; starting decode diagnostics", flush=True
            )
            await rpc("timing_start")
            if args.torch:
                await rpc("torch_profile_start")
            if args.native:
                await rpc("profile_start", interval=16)
            save("before.json", await rpc("snapshot"))
            (args.output / "metrics-before.txt").write_text(
                (await client.get("/metrics")).text
            )
            begin = time.monotonic()
            before = [state["outputs"] for state in states]
            await asyncio.sleep(args.seconds)
            after = [state["outputs"] for state in states]
            end = time.monotonic()
            save("after.json", await rpc("snapshot"))
            (args.output / "metrics-after.txt").write_text(
                (await client.get("/metrics")).text
            )
            save("timing.json", await rpc("timing_finish"))
            if args.native:
                save(
                    "native.json",
                    await rpc("profile_finish", directory=str(args.output)),
                )
            if args.torch:
                save(
                    "torch.json",
                    await rpc("torch_profile_finish", directory=str(args.output)),
                )
            stop.set()
            result = {
                "seconds": end - begin,
                "tokens": sum(after) - sum(before),
                "output_tps": (sum(after) - sum(before)) / (end - begin),
                "before_outputs": before,
                "after_outputs": after,
                "timing_note": "Diagnostic overhead included; not formal throughput",
            }
            save("window.json", result)
            print(json.dumps(result), flush=True)

        async with asyncio.TaskGroup() as group:
            for index in range(16):
                group.create_task(consume(index))
            group.create_task(profile())
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
        health = (await client.get("/health")).status_code
        save(
            "after-cancel.json",
            {"running": running, "health": health, "metrics": metrics},
        )
        assert running and sum(running) == 0 and health == 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 30:
        parser.error("Use a bounded profiling window of at most 30 seconds")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
