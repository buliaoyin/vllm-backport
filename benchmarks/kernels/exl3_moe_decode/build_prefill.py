# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the SM80 pipeline, synchronization and M64 ablation library."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = out / "fp16"
    here = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            str(here.parent / "exl3_m32/build.py"),
            "--source",
            str(args.source.resolve()),
            "--build-dir",
            str(base),
            "--nvcc",
            args.nvcc,
        ],
        check=True,
    )
    shutil.copytree(base / "upstream", out / "upstream")
    hadamard = out / "upstream/quant/hadamard_inner.cuh"
    text = hadamard.read_text()
    end = "    atomicAdd(output_ptr + 96 + t, sh[96 + t]);"
    if text.count(end) != 1:
        raise ValueError("Hadamard scatter call site changed")
    text = text.replace(end, end + "\n    __syncwarp();")
    hadamard.write_text(text)
    shutil.copyfile(here / "LICENSE-exllamav3", out / "LICENSE")
    gemm = (
        (base / "gemm_rows.cuh")
        .read_text()
        .replace(
            'TILESIZE_M == 32, "Row-reuse prototype requires M=32"',
            'TILESIZE_M == 32 || TILESIZE_M == 64, "Expected M32/M64"',
        )
    )
    moe = (
        (base / "moe_rows.cuh")
        .read_text()
        .replace("cb,32,ROW_TILE_K", "cb,ROW_TILE_M,ROW_TILE_K")
    )
    code, variants, paths = "", {}, []
    for family, nosync, nolock, half in (
        ("base", False, False, False),
        ("nobar", True, False, False),
        ("nolock", False, True, False),
        ("both", True, True, False),
        ("half", True, True, True),
    ):
        g = gemm.replace("exl3_gemm_rows_v2_inner", f"exl3_gemm_{family}_inner")
        if nosync:
            before = "        __syncthreads();\n        advance1();"
            if g.count(before) != 1:
                raise ValueError("Fragment barrier call site changed")
            g = g.replace(before, "        advance1();")
        if nolock:
            g = g.replace(
                "        barrier_acquire(lock, lock_i);",
                "        if (tiles_n % gridDim.x != 0) barrier_acquire(lock, lock_i);",
            )
            g = g.replace(
                "        barrier_release(lock, lock_d, last);",
                "        if (tiles_n % gridDim.x != 0) "
                "barrier_release(lock, lock_d, last);",
            )
        if half:
            g = g.replace(
                "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)",
                "#if defined(__CUDA_ARCH__)",
            )
        m = (
            moe.replace('"gemm_rows.cuh"', f'"gemm_{family}.cuh"')
            .replace("exl3_gemm_rows_v2_inner", f"exl3_gemm_{family}_inner")
            .replace("exl3_moe_rows_v2_kernel", f"exl3_moe_{family}_kernel")
        )
        for filename, content in ((f"gemm_{family}.cuh", g), (f"moe_{family}.cuh", m)):
            path = out / filename
            path.write_text(content)
            paths.append(path)
        code += f'#include "moe_{family}.cuh"\n'
        shapes = (
            [(32, 32, 256)]
            if not half
            else [(32, 32, 256), (64, 32, 256), (64, 16, 256), (64, 32, 128)]
        )
        for tm, tk, tn in shapes:
            name = f"{family}_m{tm}_k{tk}_n{tn}"
            variants[name] = [tm, tk, tn]
            code += (
                f'extern "C" void* exl3_rows_{name}() {{ return (void*)'
                f"exl3_moe_{family}_kernel<4,{tn},2,2,{tm},{tk},true>; }}\n"
            )
    (out / "pipeline.cu").write_text(code)
    metadata = json.loads((base / "build.json").read_text())
    metadata["warp_scatter_sync"] = True
    metadata["patched_header_sha256"] = {
        "quant/hadamard_inner.cuh": hashlib.sha256(hadamard.read_bytes()).hexdigest()
    }
    command = [
        str(out / "pipeline.cu")
        if value == str(base / "rows.cu")
        else str(out / "pipeline.so")
        if value == str(base / "rows.so")
        else "-I" + str(out)
        if value == "-I" + str(base)
        else value
        for value in metadata["command"]
    ]
    metadata.update(
        command=command,
        variants=variants,
        experiment_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*paths, out / "pipeline.cu", Path(__file__).resolve()]
        },
    )
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    with (out / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["library_sha256"] = hashlib.sha256(
        (out / "pipeline.so").read_bytes()
    ).hexdigest()
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    print(out / "pipeline.so")


if __name__ == "__main__":
    main()
