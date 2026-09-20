// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from ExLlamaV3 SQ GEMV; see LICENSE.
#include "upstream/quant/exl3_gemv_int8_kernel.cuh"
template <bool c_fp32, bool residual, int M>
__global__ __launch_bounds__(NUM_THREADS) void exl3_grouped_gemv(
    const half* inputs, const uint16_t* const* trellis, void* outputs,
    const half* const* input_scales, const half* const* output_scales,
    const int64_t* ids, int* workspace, const int* group_sizes, int size_k,
    int size_n, int workspace_stride) {
  constexpr int bits = 4;
  const int slot = blockIdx.z;
  const int size_m = group_sizes[slot];
  if (size_m == 0) return;
  const int expert = ids[slot];
  const half* A = inputs + slot * M * size_k;
  const uint16_t* B = trellis[expert];
  const half* suh = input_scales[expert];
  const half* svh = output_scales[expert];
  void* C = c_fp32 ? (void*)((float*)outputs + slot * M * size_n)
                   : (void*)((half*)outputs + slot * M * size_n);
  int* locks = workspace + slot * workspace_stride;
  extern __shared__ uint32_t shmem[];

  // Work decomposition: rows_per multiple of 8 (whole 128-spans for the local
  // Hadamard); the host mirrors this for the shared memory size and workspace
  // bound. Swept on 3090: the wall time is ~one unit's duration as long as all
  // (slice x 256-column) units fit in one wave of the grid, so take the
  // smallest slice height with units <= gridDim.x - but at least half-wave
  // occupancy up to 32 rows, since very short slices inflate ksplit and the
  // epilogue's serial per-slice combine
  int rows_total = size_k >> 4;
  int nb256_total = size_n / 256;
  int r = CEIL_DIVIDE(rows_total * nb256_total, (int)gridDim.x);
  int rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
  rows_per = MAX(rows_per, SQ_MINROWS);
  rows_per = MIN(rows_per, gemv_int8_sq_rows_max(M, residual));
  rows_per = MIN(rows_per, (rows_total + 7) & ~7);
  int ksplit = CEIL_DIVIDE(rows_total, rows_per);
  int units = nb256_total * ksplit;
  int slice_stride = rows_per * 16;
  int pstride = size_n * (residual ? 2 : 1);

  int* counters = locks;
  float* qsums = (float*)(locks + SQ_COUNTERS_CAP);
  int* partials = locks + SQ_WS_RESERVED;

  // Shared layout: [slice FP16 values][M x splats (+ M x residual splats)][B
  // stage][epilogue tmp]
  half* sh_ah = (half*)shmem;
  uint32_t* sh_as = shmem + rows_per * 8;
  uint32_t* sh_b = sh_as + slice_stride * M * (residual ? 2 : 1);
  float* sh_tmp =
      (float*)(sh_b +
               (gemv_int8_stage_smem(bits) ? 8 * GEMV_STAGE_D * 16 * bits : 0));
  __shared__ float sh_red[33];
  __shared__ int sh_last;

  int t = threadIdx.x;
  int prev_slice = -1;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    int slice = unit / nb256_total;
    int nb256 = unit % nb256_total;
    int kb0 = slice * rows_per;
    int nrows = MIN(rows_per, rows_total - kb0);
    if (slice != prev_slice) {
      gemv_int8_stage_slice<M, residual>(A, size_m, size_k, suh,
                                         qsums + 4 * slice * M, sh_ah, sh_as,
                                         slice_stride, sh_red, kb0, nrows);
      prev_slice = slice;
    }
    int* pacc = partials + (size_t)slice * M * pstride;
    if constexpr (bits == 4)
      gemv_int8_unit_wide<M, residual, false>(
          B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
    else if constexpr (gemv_int8_stage_smem(bits))
      gemv_int8_unit_smem<bits, M, residual, false>(B, pacc, pstride, sh_as,
                                                    slice_stride, sh_b, nb256,
                                                    kb0, nrows, size_n);
    else
      gemv_int8_unit_narrow<bits, M, residual, false>(
          B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

    // Completion counter: the ksplit-th contributor runs the epilogue for this
    // 256-column group. (sh_last reuse across iterations is ordered by the next
    // iteration's __syncthreads.)
    __threadfence();
    __syncthreads();
    if (t == 0)
      sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
    __syncthreads();
    if (sh_last) {
      gemv_int8_epilogue_group_sq<M, c_fp32, residual>(partials, qsums, pstride,
                                                       ksplit, size_m, C, svh,
                                                       sh_tmp, nb256, size_n);
      if (t == 0) counters[nb256] = 0;
    }
  }
}

extern "C" void* exl3_grouped_plain_half_m2() {
  return (void*)exl3_grouped_gemv<false, false, 2>;
}
extern "C" void* exl3_grouped_plain_float_m2() {
  return (void*)exl3_grouped_gemv<true, false, 2>;
}
extern "C" void* exl3_grouped_plain_half_m4() {
  return (void*)exl3_grouped_gemv<false, false, 4>;
}
extern "C" void* exl3_grouped_plain_float_m4() {
  return (void*)exl3_grouped_gemv<true, false, 4>;
}
