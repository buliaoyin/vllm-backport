# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fixed-token Qwen4Exp workloads through an OpenAI endpoint.

Use identical tokenizers, context sizes, prompts, MTP budgets, and GPU settings
for each server. Aggregate throughput includes prefill; per-request decode
excludes the first content event and counts output_tokens - 1. Native MTP may
emit several tokens in that event, so this metric has a small delivery bias.
All repetitions and generated text are retained in the output JSON.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


async def request(session, args, prompt, output_tokens):
    started = time.perf_counter()
    first = None
    chunks = []
    usage = None
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with session.post(args.url + "/v1/completions", json=body) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("text", "")
                if content:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(content)
    finished = time.perf_counter()
    if usage is None or first is None:
        raise RuntimeError("Server did not supply completion usage/content")
    tokens = usage["completion_tokens"]
    if tokens != output_tokens:
        raise RuntimeError(f"Expected {output_tokens} output tokens, received {tokens}")
    return {
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": tokens,
        "ttft_s": first - started,
        "elapsed_s": finished - started,
        "post_first_token_tps": (tokens - 1) / (finished - first),
        "text": "".join(chunks),
    }


async def main(args):
    if min(args.contexts) < 64 or min(args.concurrency) < 1:
        raise ValueError("Contexts must be at least 64; concurrency must be positive")
    if args.repeats < 1 or args.output_tokens < 2:
        raise ValueError("Repeats must be positive; output tokens must be at least 2")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = []
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain how a computer works in detail."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        await request(session, args, prompt, 64)
        for context in args.contexts:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    filler = "The following notes describe computer architecture. " * (
                        context // 4
                    )
                    filler += (
                        " Explain how a computer works in detail, including its "
                        "processor, memory, operating system, and networking."
                    )
                    prompts = []
                    for request_id in range(concurrency):
                        # Change within the first cache block in every phase/wave.
                        content = (
                            f"{args.prompt_tag}-{context}-{concurrency}-"
                            f"{repeat}-{request_id} benchmark. " + filler
                        )
                        prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": content}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        prompt = tokenizer.encode(prompt, add_special_tokens=False)
                        if len(prompt) < context:
                            raise ValueError("Prompt filler is too short")
                        prompts.append(prompt[: context - 64] + prompt[-64:])
                    started = time.perf_counter()
                    samples = await asyncio.gather(
                        *[
                            request(session, args, prompt, args.output_tokens)
                            for prompt in prompts
                        ]
                    )
                    elapsed = time.perf_counter() - started
                    if any(r["input_tokens"] != context for r in samples):
                        raise RuntimeError(
                            "Server input token counts differ from prompt"
                        )
                    row = {
                        "context_target": context,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "prompt_tag": args.prompt_tag,
                        "makespan_s": elapsed,
                        "aggregate_tps": sum(r["output_tokens"] for r in samples)
                        / elapsed,
                        "median_decode_tps": statistics.median(
                            r["post_first_token_tps"] for r in samples
                        ),
                        "requests": samples,
                    }
                    rows.append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "requests"}),
                        flush=True,
                    )
                    Path(args.output).write_text(json.dumps(rows, indent=2))
        if args.profile:
            async with session.post(args.url + "/start_profile") as response:
                response.raise_for_status()
            await asyncio.gather(
                *[request(session, args, prompt, 64) for prompt in prompts]
            )
            async with session.post(args.url + "/stop_profile") as response:
                response.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--url", default="http://127.0.0.1:18938")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[136, 8192, 60000])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-tag", default="steady")
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--profile", action="store_true")
    asyncio.run(main(parser.parse_args()))
