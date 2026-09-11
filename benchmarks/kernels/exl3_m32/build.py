# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build SM80 row-reuse experiments from a pinned, unmodified ExLlamaV3 checkout."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import regex as re
import torch

REVISION = "6ff3a17ea7f3d0026b273d43239398d57f71b788"


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
    for original, target in (
        ("exl3_gemm_inner.cuh", "gemm_rows.cuh"),
        ("exl3_moe_kernel.cuh", "moe_rows.cuh"),
    ):
        text = (upstream / "quant" / original).read_text()
        text = re.sub(
            r'#include "([^"]+)"',
            lambda match: (
                '#include "upstream/'
                + str((upstream / "quant" / match[1]).resolve().relative_to(upstream))
                + '"'
            ),
            text,
        )
        (out / target).write_text(text)
    patch = Path(__file__).with_name("rows.patch")
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch.resolve())],
        cwd=out,
        check=True,
    )
    shutil.copyfile(Path(__file__).with_name("LICENSE-exllamav3"), out / "LICENSE")
    factories = {
        "m16": "4,256,2,2,16,32,false",
        "m32": "4,256,2,2,32,32,false",
        "m32_predicated": "4,256,2,2,32,32,true",
    }
    code = '#include "moe_rows.cuh"\n'
    for name, parameters in factories.items():
        code += (
            f'extern "C" void* exl3_rows_{name}() {{ '
            f"return (void*) exl3_moe_rows_v2_kernel<{parameters}>; }}\n"
        )
    (out / "rows.cu").write_text(code)
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
        "-I" + str(include),
        "-I" + str(include / "torch/csrc/api/include"),
        "-I" + str(out),
        str(out / "rows.cu"),
        "-o",
        str(out / "rows.so"),
    ]
    metadata = {
        "upstream_revision": revision,
        "header_sha256": hashes,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "nvcc": subprocess.check_output([args.nvcc, "--version"], text=True),
        "command": command,
    }
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    with (out / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["library_sha256"] = hashlib.sha256(
        (out / "rows.so").read_bytes()
    ).hexdigest()
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    print(out / "rows.so")


if __name__ == "__main__":
    main()
