// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from ExLlamaV3 SQ GEMV; see LICENSE.
#pragma once
#include "upstream/quant/exl3_gemv_int8_kernel.cuh"

template <bool fp32, bool small>
__global__ __launch_bounds__(256) void exl3_expert_gemv_compact(
    const half* inputs, const uint16_t* const* trellis, void* outputs,
    const half* const* input_scales, const half* const* output_scales,
    const int64_t* ids, int* workspace, const int* tasks, const int* task_count,
    int input_group_shift, int64_t workspace_stride) {
  constexpr int size_k = fp32 ? 2048 : 4096;
  constexpr int size_n = fp32 ? 4096 : 2048;
  constexpr int rows_per = (fp32 || small) ? 128 : 256;
  constexpr int rows_total = size_k / 16;
  constexpr int nb256_total = size_n / 256;
  constexpr int ksplit = rows_total / rows_per;
  constexpr int units = nb256_total * ksplit;
  constexpr int slice_stride = rows_per * 16;
  static_assert(rows_total % rows_per == 0);
  static_assert(units == (fp32 || small ? 16 : 8));
  // The producer supplies unique valid slot indices in tasks[0:task_count].
  // Counters are initially zero and the final contributor resets each one.
  const unsigned int total_work = *task_count * units;
  extern __shared__ uint32_t shmem[];
  half* sh_ah = (half*)shmem;
  uint32_t* sh_as = shmem + rows_per * 8;
  float* sh_tmp = (float*)(sh_as + slice_stride);
  __shared__ float sh_red[33];
  __shared__ int sh_last;
  int t = threadIdx.x;
  for (unsigned int work = blockIdx.x; work < total_work; work += gridDim.x) {
    const int slot = tasks[work / units];
    const int expert = ids[slot];
    const unsigned int unit = work % units;
    const int slice = unit / nb256_total;
    const int nb256 = unit % nb256_total;
    const int kb0 = slice * rows_per;
    constexpr int nrows = rows_per;
    // Host validation guarantees a power-of-two input group; down uses one
    // input per slot. K-slice boundaries match the original filtered kernel.
    const half* A =
        inputs + (fp32 ? slot : (slot >> input_group_shift)) * size_k;
    const uint16_t* B = trellis[expert];
    const half* suh = input_scales[expert];
    const half* svh = output_scales[expert];
    void* C = fp32 ? (void*)((float*)outputs + slot * size_n)
                   : (void*)((half*)outputs + slot * size_n);
    // Use the allocation's stride, including padding for other shapes or
    // residual decode, so old/compact launches share counter locations on
    // replay.
    int* counters = workspace + slot * workspace_stride;
    float* qsums = (float*)(counters + SQ_COUNTERS_CAP);
    int* partials = counters + SQ_WS_RESERVED;
    gemv_int8_stage_slice<1, false>(A, 1, size_k, suh, qsums + 4 * slice, sh_ah,
                                    sh_as, slice_stride, sh_red, kb0, nrows);
    gemv_int8_unit_wide<1, false, false>(B, partials + (size_t)slice * size_n,
                                         size_n, sh_as, slice_stride, nb256,
                                         kb0, nrows, size_n);
    __threadfence();
    __syncthreads();
    if (t == 0) sh_last = atomicAdd(&counters[nb256], 1) == ksplit - 1;
    __syncthreads();
    if (sh_last) {
      gemv_int8_epilogue_group_sq<1, fp32, false>(
          partials, qsums, size_n, ksplit, 1, C, svh, sh_tmp, nb256, size_n);
      if (t == 0) counters[nb256] = 0;
    }
    __syncthreads();
  }
}
