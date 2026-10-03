# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate numeric answers on local GSM8K cases through an OpenAI endpoint.

The dataset is JSON with an ``evals`` array of ``id``, ``question`` and numeric
``answer`` fields. Thinking is disabled consistently with the serving benchmark.
Generated answers are retained for review alongside the score.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import regex as re
from transformers import AutoTokenizer


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cases = json.loads(Path(args.dataset).read_text())["evals"][: args.limit]
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def run(case):
            async with semaphore:
                prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": case["question"]
                            + "\nGive your final numeric answer on a line starting "
                            "with ####.",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                started = time.perf_counter()
                async with session.post(
                    args.url + "/v1/completions",
                    json={
                        "model": args.model,
                        "prompt": prompt,
                        "max_tokens": 2048,
                        "temperature": 0,
                    },
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                generated = result["choices"][0]
                text = generated["text"]
                matches = re.findall(
                    r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
                    text.rsplit("</think>", 1)[-1],
                )
                answer = matches[-1].replace(",", "") if matches else None
                row = {
                    "id": case["id"],
                    "expected": case["answer"],
                    "actual": answer,
                    "correct": answer is not None
                    and float(answer) == float(case["answer"]),
                    "elapsed_s": time.perf_counter() - started,
                    "usage": result.get("usage"),
                    "finish_reason": generated["finish_reason"],
                    "text": text,
                }
                print(json.dumps({k: v for k, v in row.items() if k != "text"}))
                return row

        rows = await asyncio.gather(*[run(case) for case in cases])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "correct": sum(r["correct"] for r in rows),
                "total": len(rows),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--url", default="http://127.0.0.1:18938")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
