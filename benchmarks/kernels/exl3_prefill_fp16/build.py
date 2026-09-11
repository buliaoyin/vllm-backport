# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build batched reconstruction and Hadamard helpers for grouped FP16 prefill."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/home/bul/dev/exllamav3"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = root / "csrc/libtorch_stable/quantization/exl3"
    shutil.copytree(source / "upstream", out / "upstream")
    shutil.copy2(source / "LICENSE", out / "LICENSE")
    original = args.source / "exllamav3/exllamav3_ext/quant/reconstruct.cu"
    text = original.read_text()
    text = text[
        text.index("template <int K, int cb>") : text.index("#define __(i, cb)")
    ]
    text = text.replace("reconstruct_kernel", "reconstruct_experts")
    text = text.replace(
        "const uint16_t* __restrict__ g_packed,",
        "const uint16_t* const* __restrict__ packed_tables,",
    )
    text = text.replace("int packed_n_offset", "int size_k")
    text = text.replace(
        "    constexpr int packed_size",
        """    const int packed_n_offset = 0;
    const uint16_t* g_packed = packed_tables[blockIdx.z];
    g_unpacked += size_t(blockIdx.z) * size_k * packed_blocks_n * 16;
    constexpr int packed_size""",
    )
    (out / "reconstruct.cuh").write_text(
        "// SPDX-License-Identifier: MIT\n"
        "// Copyright (c) 2025 Turboderp; batched pointer-table adaptation.\n"
        '#include "upstream/quant/exl3_dq.cuh"\n' + text
    )
    shutil.copy2(Path(__file__).with_name("helpers.cu"), out / "helpers.cu")
    inc = Path(torch.__file__).parent / "include"
    cmd = [
        "/usr/local/cuda/bin/nvcc",
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
        "-I" + str(inc),
        "-I" + str(inc / "torch/csrc/api/include"),
        "-I" + str(out),
        str(out / "helpers.cu"),
        "-o",
        str(out / "helpers.so"),
    ]
    metadata = {
        "command": cmd,
        "upstream_reconstruct_sha256": hashlib.sha256(
            original.read_bytes()
        ).hexdigest(),
        "source_sha256": {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*")
            if p.is_file()
        },
    }
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    with (out / "build.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["library_sha256"] = hashlib.sha256(
        (out / "helpers.so").read_bytes()
    ).hexdigest()
    (out / "build.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
