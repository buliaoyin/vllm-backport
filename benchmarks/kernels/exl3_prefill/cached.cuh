// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Uses the bundled EXL3 bit/fragment layout and CUDA helpers under MIT terms.
#pragma once
#include "upstream/ptx.cuh"
#include "upstream/quant/exl3_dq.cuh"

#define EXL3_GEMM_BASE_THREADS 256

template <EXL3_GEMM_T_ARGS, bool shmem_out_had, bool PREDICATE_ROWS>
inline __device__ void exl3_gemm_nobar_inner(
    const half* __restrict__ A, const uint16_t* __restrict__ B,
    void* __restrict__ C, const int size_m, const int size_k, const int size_n,
    int* __restrict__ locks, const half* post_scale) {
  static_assert(bits == 4 && cb == 2 && !c_fp32 && !shmem_out_had);
  static_assert((TILESIZE_N == 128 || TILESIZE_N == 256) && TILESIZE_K == 16);
  constexpr int M_FRAGS = TILESIZE_M == 64 && TILESIZE_N == 256 ? 2 : 1;
  constexpr int WARP_N = TILESIZE_N / 32;
  static_assert(EXL3_GEMM_BASE_THREADS ==
                TILESIZE_M / 16 / M_FRAGS * WARP_N * 32);
  constexpr int CACHE_K = 64;
  constexpr int A_COLS = CACHE_K / 8;
  constexpr int N_BLOCKS = TILESIZE_N / 16;
  constexpr int K_BLOCKS = CACHE_K / 16;
  constexpr int A_WORDS = TILESIZE_M * CACHE_K / 8;
  constexpr int Q_WORDS = CACHE_K * TILESIZE_N / 32;
  extern __shared__ int4 cached_shared[];
  int4* shared_a = cached_shared;
  int4* shared_q = shared_a + 2 * A_WORDS;
  uint4* decoded = reinterpret_cast<uint4*>(shared_q + 2 * Q_WORDS);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_n = warp % WARP_N;
  const int warp_m = warp / WARP_N;
  const int matrix_blocks_n = size_n / 16;
  const int slices = size_k / CACHE_K;

  for (int tile_n = blockIdx.x; tile_n < size_n / TILESIZE_N;
       tile_n += gridDim.x) {
    FragC accumulator[M_FRAGS][4] = {};
    auto copy_stage = [&](int slice) {
      if (slice < slices) {
        const int stage = slice % 2;
        for (int i = threadIdx.x; i < A_WORDS; i += EXL3_GEMM_BASE_THREADS) {
          const int row = i / A_COLS;
          const int col = i % A_COLS;
          if (row < size_m) {
            const int swizzle = col ^ ((row >> 1) & (A_COLS - 1));
            cp_async(shared_a + stage * A_WORDS + row * A_COLS + swizzle,
                     reinterpret_cast<const int4*>(A) + row * size_k / 8 +
                         slice * A_COLS + col);
          }
        }
        for (int i = threadIdx.x; i < Q_WORDS; i += EXL3_GEMM_BASE_THREADS) {
          const int kb = i / (N_BLOCKS * 8);
          const int nb = (i / 8) % N_BLOCKS;
          const int block = (slice * K_BLOCKS + kb) * matrix_blocks_n +
                            tile_n * N_BLOCKS + nb;
          cp_async(shared_q + stage * Q_WORDS + i,
                   reinterpret_cast<const int4*>(B) + block * 8 + i % 8);
        }
      }
      cp_async_fence();
    };
    copy_stage(0);
    for (int slice = 0; slice < slices; ++slice) {
      copy_stage(slice + 1);
      if (slice + 1 < slices)
        cp_async_wait<1>();
      else
        cp_async_wait<0>();
      __syncthreads();
      const int stage = slice % 2;
      // Each quantized 16x16 block is decoded once, shared by all row warps.
#pragma unroll 1
      for (int block = warp; block < K_BLOCKS * N_BLOCKS;
           block += EXL3_GEMM_BASE_THREADS / 32) {
        FragB b0, b1;
        const uint32_t* q = reinterpret_cast<const uint32_t*>(
            shared_q + stage * Q_WORDS + block * 8);
        dq_dispatch<4, 2>(q, lane * 8, b0, b1);
        half2_uint32 p0(0u), p1(0u), p2(0u), p3(0u);
        p0.as_half2 = b0[0];
        p1.as_half2 = b0[1];
        p2.as_half2 = b1[0];
        p3.as_half2 = b1[1];
        decoded[block * 32 + lane] =
            make_uint4(p0.as_uint32, p1.as_uint32, p2.as_uint32, p3.as_uint32);
      }
      __syncthreads();
      if (warp_m * M_FRAGS * 16 < size_m) {
        FragA a[2][M_FRAGS];
        FragB b[2][4];
        auto fragments = [&](int buf, int kb) {
#pragma unroll
          for (int m = 0; m < M_FRAGS; ++m) {
            if ((warp_m * M_FRAGS + m) * 16 >= size_m) continue;
            const int row =
                (warp_m * M_FRAGS + m) * 16 + lane % 8 + 8 * ((lane / 8) % 2);
            const int col = (lane / 16 + kb * 2) ^ ((row >> 1) & (A_COLS - 1));
            ldsm4(a[buf][m], shared_a + stage * A_WORDS + row * A_COLS + col);
          }
#pragma unroll
          for (int nb = 0; nb < 2; ++nb) {
            const uint4 values =
                decoded[(kb * N_BLOCKS + warp_n * 2 + nb) * 32 + lane];
            b[buf][nb * 2][0] = half2_uint32(values.x).as_half2;
            b[buf][nb * 2][1] = half2_uint32(values.y).as_half2;
            b[buf][nb * 2 + 1][0] = half2_uint32(values.z).as_half2;
            b[buf][nb * 2 + 1][1] = half2_uint32(values.w).as_half2;
          }
        };
        auto multiply = [&](int buf) {
#pragma unroll
          for (int m = 0; m < M_FRAGS; ++m) {
            if ((warp_m * M_FRAGS + m) * 16 >= size_m) continue;
#pragma unroll
            for (int n = 0; n < 4; ++n)
              ptx_mma_m16n8k16(a[buf][m], b[buf][n], accumulator[m][n]);
          }
        };
        fragments(0, 0);
#pragma unroll 1
        for (int kb = 0; kb < K_BLOCKS; kb += 2) {
          fragments(1, kb + 1);
          multiply(0);
          if (kb + 2 < K_BLOCKS) fragments(0, kb + 2);
          multiply(1);
        }
      }
      // Retire all A/Q readers before reusing their stage; also protects B.
      __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < M_FRAGS; ++m) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int row = (warp_m * M_FRAGS + m) * 16 + lane / 4;
        const int col =
            tile_n * TILESIZE_N + warp_n * 32 + n * 8 + lane % 4 * 2;
        half* out = reinterpret_cast<half*>(C);
        if (row < size_m)
          *reinterpret_cast<half2*>(out + row * size_n + col) =
              __floats2half2_rn(accumulator[m][n][0], accumulator[m][n][1]);
        if (row + 8 < size_m)
          *reinterpret_cast<half2*>(out + (row + 8) * size_n + col) =
              __floats2half2_rn(accumulator[m][n][2], accumulator[m][n][3]);
      }
    }
  }
}
