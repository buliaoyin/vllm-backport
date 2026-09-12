# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-vocabulary logits along identical autoregressive token prefixes."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsm-limit", type=int, default=64)
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    inputs = read(args.artifacts / "eval-inputs.json.gz")
    native = read(args.artifacts / "native-confirm.json.gz")
    answers = {row["id"]: row for row in native["eval"]}
    records = [
        {
            "name": f"gsm_{row['id']}",
            "prompt": row["input_ids"],
            "targets": answers[row["id"]]["token_ids"],
            "answer": row["answer"],
        }
        for row in inputs["evals"][: args.gsm_limit]
    ]
    groups = [
        (f"gsm_{start // 4}", records[start : start + 4])
        for start in range(0, len(records), 4)
    ]
    if not args.skip_long:
        long_inputs = read(args.artifacts / "inputs.json.gz")
        long_native = read(args.artifacts / "native2048.json.gz")
        cases = {c["name"]: c for c in long_inputs["cases"]}
        results = {r["case"]: r for r in long_native["perf"] if r["repeat"] == 0}
        eos = inputs["eos_ids"]
        eos = {eos} if isinstance(eos, int) else set(eos)
        for name in ("8k", "16k", "32k", "64k", "8k_8k_8k_8k", "32k_32k"):
            rows = []
            for i, (prompt, output) in enumerate(
                zip(cases[name]["inputs"], results[name]["rows"])
            ):
                targets = output["token_ids"]
                stop = next(
                    (j + 1 for j, token in enumerate(targets) if token in eos),
                    len(targets),
                )
                rows.append(
                    {
                        "name": f"long_{name}_{i}",
                        "prompt": prompt,
                        "targets": targets[:stop],
                    }
                )
            groups.append((f"long_{name}", rows))
    manifest = {
        "model": inputs["model"],
        "groups": groups,
        "protocol": (
            "Raw full-vocabulary logits before forcing the same native token. "
            "Pure decode uses the same INT8 GEMV; mixed batches may use prefill. "
            "Native repeat is the noise control. Eager token bucket padding is off."
        ),
        "chunk": 6144,
        "pp": [16, 15, 14],
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).with_name("exl3_divergence_worker.py"),
            ]
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                ("EXL3_", "VLLM_EXL3_", "CUDA_", "NCCL_", "VLLM_PP_", "VLLM_TOKEN_")
            )
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False))
        temporary.replace(args.output)

    save()
    llm = LLM(
        model=inputs["model"],
        dtype="bfloat16",
        pipeline_parallel_size=3,
        max_model_len=66560,
        max_num_seqs=4,
        max_num_batched_tokens=6144,
        kv_cache_memory_bytes=2 * 1024**3,
        enable_prefix_caching=False,
        enforce_eager=True,
        compilation_config={"mode": 0},
        limit_mm_per_prompt={"image": 0, "video": 0},
        worker_extension_cls="exl3_divergence_worker.Exl3DivergenceWorker",
    )
    manifest["runtime"] = llm.collective_rpc("install_divergence_probe")
    assert [len(r["decoder_layers"]) for r in manifest["runtime"]] == [16, 15, 14]
    for name, rows in groups:
        phases = [
            ("reference", False, 9),
            ("native_repeat", False, 9),
            ("int8_forced", True, 9),
        ]
        if name.startswith("long_"):
            phases.append(("int8_4096", True, 4096))
        for phase, enabled, minimum in phases:
            llm.sleep(level=0)
            ids = llm.enqueue(
                [{"prompt_token_ids": r["prompt"]} for r in rows],
                [
                    SamplingParams(
                        temperature=0, max_tokens=len(r["targets"]), ignore_eos=True
                    )
                    for r in rows
                ],
                use_tqdm=False,
            )
            metadata = {
                req: {
                    "name": r["name"],
                    "prompt_length": len(r["prompt"]),
                    "targets": r["targets"],
                }
                for req, r in zip(ids, rows)
            }
            states = llm.llm_engine.output_processor.request_states
            external_ids = {req: states[req].external_req_id for req in ids}
            llm.collective_rpc(
                "begin_divergence", args=(phase, enabled, metadata, minimum)
            )
            start = time.monotonic()
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(use_tqdm=False)
            by_id = {o.request_id: o for o in outputs}
            for req, meta in metadata.items():
                output = by_id[external_ids[req]]
                assert list(output.outputs[0].token_ids) == meta["targets"], (
                    name,
                    phase,
                    req,
                )
                assert output.num_cached_tokens == 0
            result = llm.collective_rpc("finish_divergence")
            expected = sum(len(r["targets"]) for r in rows)
            if phase != "reference":
                assert len(result[-1]["metrics"]) == expected
            assert all((r["calls"].get("int8", 0) > 0) == enabled for r in result)
            assert all(r["calls"].get("decode_int8", 0) > 0 for r in result)
            manifest["results"].append(
                {
                    "group": name,
                    "phase": phase,
                    "elapsed_with_probe": time.monotonic() - start,
                    "ranks": result,
                }
            )
            save()
            metrics = result[-1]["metrics"]
            mean = (
                sum(r["kl_native_int8"] for r in metrics) / len(metrics)
                if metrics
                else 0
            )
            print(name, phase, expected, "mean_KL", mean, flush=True)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
