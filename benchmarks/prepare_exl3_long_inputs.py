# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare shared, exact-length EXL3 prompts with a code at known token offsets."""

import argparse
import gzip
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[8192, 16384, 32768, 65536]
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--close-thinking", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    rng = random.Random(args.seed)
    words = [
        "cedar",
        "harbor",
        "meadow",
        "silver",
        "copper",
        "willow",
        "ridge",
        "maple",
    ]
    archive = "\n".join(
        f"Archive item {i}: location {rng.choice(words)}, team {rng.choice(words)}, "
        f"status reviewed, note {rng.choice(words)} {rng.choice(words)}."
        for i in range(max(args.lengths) // 8)
    )
    body = tokenizer.encode(archive, add_special_tokens=False)
    cases = []
    for length in args.lengths:
        inputs, expected, offsets, keys = [], [], [], []
        for request, fraction in enumerate([0.1, 0.35, 0.65, 0.9]):
            key = f"record_{length}_{request}_vault"
            code = str(rng.randrange(100000, 1000000))
            marker = "EXL3_ARCHIVE_PLACEHOLDER"
            content = (
                f"Request {length}/{request}. Read the archive and locate the exact "
                f"access code assigned to {key}. The code occurs once in the archive.\n"
                f"BEGIN ARCHIVE\n{marker}\nEND ARCHIVE\n"
                f"Return only the six-digit access code for {key}. Do not explain."
            )
            template = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if args.close_thinking:
                template += "</think>\n"
            before, after = template.split(marker)
            prefix = tokenizer.encode(before, add_special_tokens=False)
            suffix = tokenizer.encode(after, add_special_tokens=False)
            needle = tokenizer.encode(
                f"\nAuthoritative record: {key}. Access code: {code}.\n",
                add_special_tokens=False,
            )
            remaining = length - len(prefix) - len(suffix) - len(needle)
            assert 0 < remaining <= len(body)
            position = min(max(int(length * fraction) - len(prefix), 0), remaining)
            ids = prefix + body[:position] + needle + body[position:remaining] + suffix
            assert len(ids) == length
            decoded = tokenizer.decode(ids)
            assert decoded.count(code) == 1
            assert code not in before and code not in after
            inputs.append(ids)
            expected.append(code)
            offsets.append(len(prefix) + position)
            keys.append(key)
        for batch in [1, 4]:
            cases.append(
                {
                    "name": f"p{length}_b{batch}",
                    "inputs": inputs[:batch],
                    "expected_strings": expected[:batch],
                    "needle_offsets": offsets[:batch],
                    "target_keys": keys[:batch],
                }
            )
    generation = json.loads((args.checkpoint / "generation_config.json").read_text())
    result = {
        "model": str(args.checkpoint.resolve()),
        "eos_ids": generation["eos_token_id"],
        "cases": cases,
        "evals": [],
        "protocol": "Exact token lengths, four distinct keys and codes per length; "
        "needle offsets 10/35/65/90 percent. Same token IDs in both engines.",
        "seed": args.seed,
        "close_thinking": args.close_thinking,
    }
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
    if args.output.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print("Saved", len(cases), "cases to", args.output, "bytes", len(payload))


if __name__ == "__main__":
    main()
