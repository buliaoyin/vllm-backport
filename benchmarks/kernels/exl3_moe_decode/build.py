# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build expert INT8 decode from a pinned, unchanged ExLlamaV3 checkout."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.exl3_m32.build import REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected ExLlamaV3 {REVISION}, got {revision}")
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "exllamav3/exllamav3_ext",
        ],
        check=True,
    )
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    upstream = source / "exllamav3/exllamav3_ext"
    hashes = {}
    for path in sorted(upstream.rglob("*")):
        if path.suffix not in (".h", ".cuh"):
            continue
        relative = path.relative_to(upstream)
        target = out / "upstream" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / "decode.cu", out / "decode.cu")
    shutil.copyfile(here / "LICENSE-exllamav3", out / "LICENSE")
    include = Path(torch.__file__).parent / "include"
    command = [
        args.nvcc,
        "--shared",
        "--cudart",
        "shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-Xptxas=-v",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_120,code=sm_120",
        "-I" + str(include),
        "-I" + str(include / "torch/csrc/api/include"),
        "-I" + str(out),
        str(out / "decode.cu"),
        "-o",
        str(out / "decode.so"),
    ]
    metadata = {
        "upstream_revision": revision,
        "header_sha256": hashes,
        "torch": str(torch.__version__),
        "nvcc": subprocess.check_output([args.nvcc, "--version"], text=True),
        "command": command,
        "experiment_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (here / "decode.cu", Path(__file__).resolve())
        },
    }
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    with (out / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["library_sha256"] = hashlib.sha256(
        (out / "decode.so").read_bytes()
    ).hexdigest()
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    print(out / "decode.so")


if __name__ == "__main__":
    main()
