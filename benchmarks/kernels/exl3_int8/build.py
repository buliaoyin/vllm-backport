# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build opt-in SM80 INT8 experts and an M32 FP16 control from pinned sources."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import regex as re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = out / "fp16"
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
    source = (base / "moe_rows.cuh").read_text()
    source = source.replace('#include "gemm_rows.cuh"', '#include "gemm_int8.cuh"')
    source = source.replace(
        "bool PREDICATE_ROWS>", "bool PREDICATE_ROWS, bool RESIDUAL>"
    )
    source = source.replace("exl3_moe_rows_v2_kernel", "exl3_moe_int8_kernel")
    for inp, output in (
        ("hidden_dim", "intermediate_dim"),
        ("intermediate_dim", "hidden_dim"),
    ):
        pattern = (
            r"                if constexpr \(ROW_TILE_M == 16\).*?"
            + re.escape(
                f"(in_addr + offset * {inp},trellis,out_addr + offset * {output},"
                f"rows,{inp},{output},locks,nullptr);"
            )
            + r"\n            }"
        )
        source, count = re.subn(
            pattern,
            f"                exl3_gemm_int8_inner<RESIDUAL>(in_addr + offset * {inp},"
            f"trellis,out_addr + offset * {output},rows,{inp},{output});"
            "\n            }",
            source,
            count=1,
            flags=re.S,
        )
        if count != 1:
            raise ValueError("Pinned expert GEMM call site no longer matches")
    source = source.replace(
        "        had_gather_gu_in();",
        """        had_gather_gu_in();
        if(gated) quantize_rows(
            temp_state_g,token_count,hidden_dim,warp_idx0,warps_per_group);
        quantize_rows(temp_state_u,token_count,hidden_dim,warp_idx0,warps_per_group);
        group_barrier(group_idx,group_size,barrier_counters_sense);""",
    )
    source = source.replace(
        "        had_guad();",
        """        had_guad();
        quantize_rows(temp_intermediate_g,token_count,intermediate_dim,warp_idx0,warps_per_group);
        group_barrier(group_idx,group_size,barrier_counters_sense);""",
    )
    (out / "moe_int8.cuh").write_text(source)
    shutil.copyfile(here / "gemm.cuh", out / "gemm_int8.cuh")
    shutil.copyfile(here / "LICENSE-exllamav3", out / "LICENSE")
    code = '#include "fp16/moe_rows.cuh"\n#include "moe_int8.cuh"\n'
    for name, residual in (("int8", "false"), ("int8_residual", "true")):
        code += (
            f'extern "C" void* exl3_rows_{name}() {{ return (void*)'
            f"exl3_moe_int8_kernel<4,256,2,2,32,32,true,{residual}>; }}\n"
        )
    code += (
        'extern "C" void* exl3_rows_m32_predicated() { return (void*)'
        "exl3_moe_rows_v2_kernel<4,256,2,2,32,32,true>; }\n"
    )
    (out / "int8.cu").write_text(code)
    metadata = json.loads((base / "build.json").read_text())
    command = [
        str(out / "int8.cu")
        if value == str(base / "rows.cu")
        else str(out / "int8.so")
        if value == str(base / "rows.so")
        else value
        for value in metadata["command"]
    ]
    metadata["command"] = command
    metadata["experiment_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            here / "gemm.cuh",
            Path(__file__).resolve(),
            out / "moe_int8.cuh",
            out / "int8.cu",
        )
    }
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    with (out / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["library_sha256"] = hashlib.sha256(
        (out / "int8.so").read_bytes()
    ).hexdigest()
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    print(out / "int8.so")


if __name__ == "__main__":
    main()
