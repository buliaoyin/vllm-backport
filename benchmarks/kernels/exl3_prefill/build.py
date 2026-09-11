# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build EXL3 prefill residency, pipeline and exact-codebook experiments."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

# M, K, N, shared stages, fragment stages, launch-bound resident blocks,
# load-next-fragment before MMA, exact codebook lookup, dynamic shared bytes.
VARIANTS = {
    "control": (32, 32, 256, 3, 2, 1, False, False, 90 * 1024),
    "compact": (32, 32, 256, 3, 2, 1, False, False, 50 * 1024),
    "loadfirst": (32, 32, 256, 3, 2, 1, True, False, 50 * 1024),
    "loadfirst_s4": (32, 32, 256, 4, 2, 1, True, False, 56 * 1024),
    "singlefrag": (32, 32, 256, 2, 1, 1, False, False, 44 * 1024),
    "k16_resident": (32, 16, 256, 3, 2, 2, False, False, 12 * 1024),
    "k16_singlefrag": (32, 16, 256, 3, 1, 2, False, False, 12 * 1024),
    "k16_n128": (32, 16, 128, 3, 2, 2, False, False, 8 * 1024),
    "m64_k16_n128": (64, 16, 128, 3, 2, 2, False, False, 12 * 1024),
    "lookup": (32, 32, 256, 3, 2, 1, False, True, 50 * 1024),
    "lookup_k16": (32, 16, 256, 3, 2, 2, False, True, 12 * 1024),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root / "csrc/libtorch_stable/quantization/exl3"
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "upstream", out / "upstream")
    shutil.copy2(source / "LICENSE", out / "LICENSE")
    shutil.copy2(Path(__file__).with_name("lookup.cuh"), out / "lookup.cuh")
    gemm = (source / "gemm_nobar.cuh").read_text()
    moe = (source / "moe_nobar.cuh").read_text()
    code = '#include "lookup.cuh"\n'
    configs = {}
    for name, (
        tm,
        tk,
        tn,
        stages,
        frags,
        resident,
        first,
        lut,
        shared,
    ) in VARIANTS.items():
        g = gemm.replace("exl3_gemm_nobar_inner", f"exl3_gemm_{name}_inner")
        if tk == 16:
            # No threadblock K reduction or output Hadamard uses this region.
            start = g.index("  const int sh_c_size = MAX")
            end = g.index(";", start) + 1
            g = g[:start] + "  const int sh_c_size = 0;" + g[end:]
        if first:
            g = g.replace("FSTAGE(1, 0);", "FSTAGE_OLD(1, 0);")
            g = g.replace("FSTAGE(0, 1);", "FSTAGE_OLD(0, 1);")
        if lut:
            g = g.replace("dq_dispatch<bits, cb>", "dq_lookup<bits, cb>")
        m = moe.replace('"gemm_nobar.cuh"', f'"gemm_{name}.cuh"')
        m = m.replace("exl3_gemm_nobar_inner", f"exl3_gemm_{name}_inner")
        bound = "16) void exl3_moe_nobar_kernel"
        assert m.count(bound) == 1
        m = m.replace(bound, f"16, {resident}) void exl3_moe_{name}_kernel")
        m = m.replace("MOE_SH_STAGES", str(stages))
        (out / f"gemm_{name}.cuh").write_text(g)
        (out / f"moe_{name}.cuh").write_text(m)
        code += f'#include "moe_{name}.cuh"\n'
        code += (
            f'extern "C" void* exl3_prefill_{name}() {{ return (void*)'
            f"exl3_moe_{name}_kernel<4,{tn},2,{frags},{tm},{tk},true>; }}\n"
        )
        configs[name] = {
            "m": tm,
            "k": tk,
            "n": tn,
            "stages": stages,
            "fragment_stages": frags,
            "minimum_blocks": resident,
            "load_first": first,
            "lookup": lut,
            "threads": 256 * tk // 16,
            "shared_bytes": shared,
        }
    code += """
__global__ void fill_lookup(half* values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  values[i] = decode_3inst<2>(i);
}
extern "C" int exl3_initialize_lookup(half* values, cudaStream_t stream) {
  cudaError_t status = cudaMemcpyToSymbol(exl3_lookup_table, &values, sizeof(values));
  if (status != cudaSuccess) return status;
  fill_lookup<<<256, 256, 0, stream>>>(values);
  return cudaGetLastError();
}
"""
    (out / "prefill.cu").write_text(code)
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
        str(out / "prefill.cu"),
        "-o",
        str(out / "prefill.so"),
    ]
    metadata = {
        "command": command,
        "variants": configs,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "nvcc": subprocess.check_output([args.nvcc, "--version"], text=True),
        "torch": str(torch.__version__),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
        "generated_sha256": {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*")
            if p.is_file()
        },
    }
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    with (out / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["library_sha256"] = hashlib.sha256(
        (out / "prefill.so").read_bytes()
    ).hexdigest()
    (out / "build.json").write_text(json.dumps(metadata, indent=2))
    print(out / "prefill.so")


if __name__ == "__main__":
    main()
