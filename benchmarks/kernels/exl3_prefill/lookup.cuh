// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// EXL3 bit extraction follows the bundled MIT-licensed exl3_dq.cuh.
#pragma once
#include <cuda_fp16.h>
#include "upstream/util.h"
#include "upstream/util.cuh"
#include "upstream/ptx.cuh"
#include "upstream/quant/exl3_dq.cuh"

__device__ __constant__ half* exl3_lookup_table;

template <int bits, int cb>
__device__ __forceinline__ void dq_lookup(const uint32_t* ptr, int offset,
                                          FragB& frag0, FragB& frag1) {
  static_assert(bits == 4 && cb == 2);
  const uint32_t i1 = offset >> 3;
  const uint32_t a = ptr[(i1 + 31) & 31];
  const uint32_t b = ptr[i1];
  uint32_t s;
  FSHF_IMM(s, b, a, 20);
  const uint32_t words[8] = {(s >> 8) & 0xffff,  (s >> 4) & 0xffff,
                             s & 0xffff,         (b >> 16) & 0xffff,
                             (b >> 12) & 0xffff, (b >> 8) & 0xffff,
                             (b >> 4) & 0xffff,  b & 0xffff};
  frag0[0] = __halves2half2(__ldg(exl3_lookup_table + words[0]),
                            __ldg(exl3_lookup_table + words[1]));
  frag0[1] = __halves2half2(__ldg(exl3_lookup_table + words[2]),
                            __ldg(exl3_lookup_table + words[3]));
  frag1[0] = __halves2half2(__ldg(exl3_lookup_table + words[4]),
                            __ldg(exl3_lookup_table + words[5]));
  frag1[1] = __halves2half2(__ldg(exl3_lookup_table + words[6]),
                            __ldg(exl3_lookup_table + words[7]));
}
