# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure an already running hybrid HTTP server with text and image inputs."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pybase64 as base64


def request(client, endpoint, payload, label):
    started = time.perf_counter()
    first = last = None
    text_parts = []
    usage = None
    request_id = None
    events = []
    token_events = []
    with client.stream("POST", endpoint, json=payload) as response:
        if response.status_code != 200:
            raise RuntimeError(response.read().decode())
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            now = time.perf_counter()
            usage = event.get("usage") or usage
            request_id = event.get("id", request_id)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = (
                    choice.get("text") or delta.get("content") or delta.get("reasoning")
                )
                if content:
                    first = first or now
                    last = now
                    text_parts.append(content)
                    events.append([now - started, content])
                    if event.get("usage"):
                        token_events.append([now, event["usage"]["completion_tokens"]])
    if first is None or last is None or usage is None:
        raise RuntimeError(f"Incomplete stream for {label}: {usage}")
    outputs = usage["completion_tokens"]
    if payload.get("ignore_eos") and outputs != payload["max_tokens"]:
        raise RuntimeError(f"Expected {payload['max_tokens']} outputs; got {outputs}")
    row = {
        "label": label,
        "request_id": request_id,
        "time": time.time(),
        "started_monotonic": started,
        "first_token_monotonic": first,
        "last_token_monotonic": last,
        "prompt_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "usage": usage,
        "ttft_seconds": first - started,
        "request_seconds": last - started,
        "prefill_tps": usage["prompt_tokens"] / (first - started),
        "decode_tps": (outputs - 1) / (last - first) if outputs > 1 else None,
        "text": "".join(text_parts),
        "events": events,
        "token_events": token_events,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--wait-ready", type=float, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--long-prompt", type=int)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--vision-prompt-tokens", type=int, default=0)
    parser.add_argument("--vision-normal-eos", action="store_true")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    chosen = {"zh_analysis", "en_analysis", "code_python"}
    cases = [case for case in suite["cases"] if case["id"] in chosen]
    if args.prompt_tokens is not None:
        if args.prompt_tokens < 608:
            raise ValueError("Resized text prompts must contain at least 608 tokens")
        for case in cases:
            original = case["prompt_token_ids"]
            head, body, tail = original[:96], original[96:-512], original[-512:]
            length = args.prompt_tokens - len(head) - len(tail)
            case["prompt_token_ids"] = (
                head + (body * (length // len(body) + 1))[:length] + tail
            )
    client = httpx.Client(
        base_url=args.url,
        timeout=3600,
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    deadline = time.monotonic() + args.wait_ready
    next_notice = 0.0
    while True:
        try:
            response = client.get("/v1/models", timeout=3)
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
            break
        except (httpx.HTTPError, ValueError, KeyError):
            if time.monotonic() >= deadline:
                raise
            if time.monotonic() >= next_notice:
                print(
                    "Waiting for the existing HTTP server to become ready", flush=True
                )
                next_notice = time.monotonic() + 30
            time.sleep(2)

    common = {
        "model": model,
        "temperature": 0,
        "max_tokens": args.max_output_tokens,
        "ignore_eos": True,
        "seed": 20260915,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": args.concurrency > 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:

        def run(endpoint, payload, label):
            row = request(client, endpoint, common | payload, label)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("events", "text", "token_events")
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if not args.vision_only:
            for phase, rounds in (
                ("warmup", args.warmup_rounds),
                ("measure", args.rounds),
            ):
                for iteration in range(rounds):
                    if args.concurrency == 1:
                        for case in cases:
                            run(
                                "/v1/completions",
                                {"prompt": case["prompt_token_ids"]},
                                f"{phase}/{iteration}/{case['id']}",
                            )
                    else:
                        rows = []
                        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                            futures = []
                            for index in range(args.concurrency):
                                case = cases[index % len(cases)]
                                payload = common | {"prompt": case["prompt_token_ids"]}
                                futures.append(
                                    pool.submit(
                                        request,
                                        client,
                                        "/v1/completions",
                                        payload,
                                        f"{phase}/{iteration}/{case['id']}/{index}",
                                    )
                                )
                            for future in as_completed(futures):
                                row = future.result()
                                rows.append(row)
                                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                                output.flush()
                                print(
                                    json.dumps(
                                        {
                                            k: v
                                            for k, v in row.items()
                                            if k
                                            not in ("events", "text", "token_events")
                                        }
                                    ),
                                    flush=True,
                                )
                        started = min(row["started_monotonic"] for row in rows)
                        first = min(row["first_token_monotonic"] for row in rows)
                        last = max(row["last_token_monotonic"] for row in rows)
                        outputs = sum(row["usage"]["completion_tokens"] for row in rows)
                        summary = {
                            "label": f"batch/{phase}/{iteration}",
                            "concurrency": args.concurrency,
                            "elapsed_seconds": last - started,
                            "output_tps": outputs / (last - started),
                            "generation_window_tps": (outputs - len(rows))
                            / (last - first),
                            "mean_ttft_seconds": sum(
                                row["ttft_seconds"] for row in rows
                            )
                            / len(rows),
                        }
                        begin = max(row["first_token_monotonic"] for row in rows)
                        end = min(row["last_token_monotonic"] for row in rows)
                        if end > begin and all(row["token_events"] for row in rows):

                            def count_at(row, timestamp):
                                return max(
                                    (
                                        count
                                        for t, count in row["token_events"]
                                        if t <= timestamp
                                    ),
                                    default=0,
                                )

                            tokens = sum(
                                count_at(row, end) - count_at(row, begin)
                                for row in rows
                            )
                            summary["all_decoding_seconds"] = end - begin
                            summary["all_decoding_tokens"] = tokens
                            summary["all_decoding_tps"] = tokens / (end - begin)
                        output.write(json.dumps(summary) + "\n")
                        output.flush()
                        print(json.dumps(summary), flush=True)
        if args.image:
            encoded = base64.b64encode(args.image.read_bytes()).decode()
            content = [
                {"type": "text", "text": ""},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + encoded},
                },
                {
                    "type": "text",
                    "text": (
                        "请识别图中两个矩形的颜色和各自的数字，"
                        "计算两个数字的和。先给出答案，再用中文解释。"
                    ),
                },
            ]
            payload = {
                "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"thinking": False},
            }
            if args.vision_prompt_tokens:
                padding = 0
                for _ in range(8):
                    content[0]["text"] = "背景资料：" + " a" * padding + "\n"
                    response = client.post("/tokenize", json={"model": model} | payload)
                    response.raise_for_status()
                    delta = args.vision_prompt_tokens - response.json()["count"]
                    if delta == 0:
                        break
                    padding += delta
                    if padding < 0:
                        raise ValueError("Vision prompt budget is too small")
                else:
                    raise RuntimeError("Could not construct exact vision prompt length")
            else:
                content.pop(0)
            if args.vision_normal_eos:
                payload.update(ignore_eos=False, max_tokens=128)
            for iteration in range(3):
                run(
                    "/v1/chat/completions",
                    payload,
                    f"vision/{args.vision_prompt_tokens}/{iteration}",
                )
        if args.long_prompt:
            base = cases[0]["prompt_token_ids"]
            head, body, tail = base[:96], base[96:-512], base[-512:]
            length = args.long_prompt - len(head) - len(tail)
            ids = head + (body * (length // len(body) + 1))[:length] + tail
            assert len(ids) == args.long_prompt
            run("/v1/completions", {"prompt": ids}, f"long/{args.long_prompt}")


if __name__ == "__main__":
    main()
