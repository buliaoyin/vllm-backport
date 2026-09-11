// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "upstream/util.h"
#include "upstream/util.cuh"
#include "upstream/ptx.cuh"
#include "upstream/quant/hadamard_inner.cuh"
#include "upstream/quant/exl3_moe_common.cuh"
#include "reconstruct.cuh"

__global__ void gather_input(const half* x, const int64_t* ids, half* y,
                             const half* const* scale, int slots, int width,
                             int topk) {
  int warp = blockIdx.x * 8 + threadIdx.x / 32;
  int per = width / 128;
  if (warp >= slots * per) return;
  int slot = warp / per, tile = warp % per;
  int expert = ids[slot];
  had_hf_r_128_inner<true, false>(x + (slot / topk) * width + tile * 128,
                                  y + slot * width + tile * 128,
                                  scale[expert] + tile * 128, 0.088388347648f);
}

__global__ void activate(half* gate, const half* up, const int64_t* ids,
                         const half* const* sg, const half* const* su,
                         const half* const* sd, int slots, int width,
                         float limit) {
  int warp = blockIdx.x * 8 + threadIdx.x / 32;
  int per = width / 128;
  if (warp >= slots * per) return;
  int slot = warp / per, tile = warp % per;
  int expert = ids[slot];
  had_hf_r_128_guad_inner(gate + slot * width + tile * 128,
                          up + slot * width + tile * 128,
                          gate + slot * width + tile * 128,
                          sg[expert] + tile * 128, su[expert] + tile * 128,
                          sd[expert] + tile * 128, 0.088388347648f, limit, 0);
}

__global__ void scatter_output(const half* down, float* out, const int64_t* ids,
                               const half* weights, const half* const* scale,
                               int slots, int width, int topk) {
  int warp = blockIdx.x * 8 + threadIdx.x / 32;
  int per = width / 128;
  if (warp >= slots * per) return;
  int slot = warp / per, tile = warp % per;
  int expert = ids[slot];
  had_hf_r_128_d_inner(down + slot * width + tile * 128,
                       out + (slot / topk) * width + tile * 128,
                       scale[expert] + tile * 128,
                       0.088388347648f * __half2float(weights[slot]));
}

extern "C" void* fp16_reconstruct() { return (void*)reconstruct_experts<4, 2>; }
extern "C" void* fp16_gather() { return (void*)gather_input; }
extern "C" void* fp16_activate() { return (void*)activate; }
extern "C" void* fp16_scatter() { return (void*)scatter_output; }
