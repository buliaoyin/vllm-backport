# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build batched INT8 reconstruction and Hadamard helpers for SM80 prefill."""

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
    dq = (source / "upstream/quant/exl3_dq.cuh").read_text()
    begin = dq.index(
        "template <int cb>\n__device__ __forceinline__ void dq8_aligned_4bits"
    )
    end = dq.index("template <int cb>", begin + 20)
    dq = dq[begin:end].replace("dq8_aligned_4bits", "dq_integer")
    dq = dq.replace("decode_3inst_2<cb>", "integer_pair")
    integer = """
__device__ __forceinline__ half2 integer_pair(uint32_t a, uint32_t b) {
  int x = (int)__dp4a(a * 0x83DCD12Du, 0x01010101u, (uint32_t)-508);
  int y = (int)__dp4a(b * 0x83DCD12Du, 0x01010101u, (uint32_t)-508);
  x = max(-128, min(127, x >> 2));
  y = max(-128, min(127, y >> 2));
  return __halves2half2(__int2half_rn(x), __int2half_rn(y));
}
"""
    text = text.replace(
        "half* __restrict__ g_unpacked", "int8_t* __restrict__ g_unpacked"
    )
    text = text.replace("dq_dispatch<K, cb>", "dq_integer<cb>")
    begin = text.index("    int4* tile_int4")
    end = text.index("    *out_int4 = tile_int4[t];", begin) + len(
        "    *out_int4 = tile_int4[t];"
    )
    text = (
        text[:begin]
        + """    const half* values = reinterpret_cast<const half*>(tile) + t * 8;
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int q = __half2int_rn(values[i]);
      packed[i / 4] |= (uint32_t)(uint8_t)(int8_t)q << (8 * (i % 4));
    }
    int8_t* output = g_unpacked + (k * 16 + r) * out_blocks_n * 16 + n * 16 + c * 8;
    *reinterpret_cast<uint2*>(output) = make_uint2(packed[0], packed[1]);"""
        + text[end:]
    )
    text = integer + dq + text
    (out / "reconstruct.cuh").write_text(
        "// SPDX-License-Identifier: MIT\n"
        "// Copyright (c) 2025 Turboderp; batched pointer-table adaptation.\n"
        '#include "upstream/quant/exl3_dq.cuh"\n' + text
    )
    shutil.copy2(
        Path(__file__).parents[1] / "exl3_prefill_fp16/helpers.cu", out / "helpers.cu"
    )
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
