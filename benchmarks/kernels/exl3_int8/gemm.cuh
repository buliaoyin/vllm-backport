// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Trellis indexing adapted from ExLlamaV3; see LICENSE-exllamav3.
#pragma once
#include "upstream/ptx.cuh"
#include "upstream/quant/exl3_dq.cuh"

struct I8A {
  uint32_t v[4];
};
struct I8B {
  uint32_t v[2];
};
struct I8C {
  int v[4];
};
__device__ __forceinline__ void imma(const I8A& a, const I8B& b, I8C& c) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c.v[0]), "+r"(c.v[1]), "+r"(c.v[2]), "+r"(c.v[3])
      : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]), "r"(b.v[0]),
        "r"(b.v[1]));
}
__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
  uint32_t x, y;
  asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;" : "=r"(x) : "r"(d), "r"(c));
  asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;"
      : "=r"(y)
      : "r"(b), "r"(a), "r"(x));
  return y;
}
template <bool RESIDUAL>
__device__ __forceinline__ void decode_i8(const uint32_t* ptr, int lane,
                                          uint32_t (&b)[2], uint32_t (&r)[2]) {
  int t = (lane & ~3) + 2 * (lane & 1);
  uint32_t prev = ptr[(t + 31) & 31], cur = ptr[t], next = ptr[t + 1];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    int shift = 24 - 4 * (lane & 2) - 16 * n;
    uint32_t pair0 = __funnelshift_r(cur, prev, shift),
             pair1 = __funnelshift_r(next, cur, shift);
    uint32_t w[4] = {(pair0 >> 4) & 0xffff, pair0 & 0xffff,
                     (pair1 >> 4) & 0xffff, pair1 & 0xffff};
    int q[4], rr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int v = (int)__dp4a(w[i] * 0x83DCD12Du, 0x01010101u, (uint32_t)-508);
      q[i] = v >> 2;
      if constexpr (RESIDUAL) {
        q[i] = min(q[i], 127);
        rr[i] = v - 2 - 4 * q[i];
      }
    }
    b[n] = pack4(q[0], q[1], q[2], q[3]);
    if constexpr (RESIDUAL) r[n] = pack4(rr[0], rr[1], rr[2], rr[3]);
  }
}
__device__ __forceinline__ void quantize_rows(half* A, int M, int K, int start,
                                              int stride) {
  extern __shared__ half shared[];
  float* maxima = (float*)shared;
  float* sums = maxima + 16;
  const int tid = threadIdx.x, lane = tid % 128, warp = tid / 32,
            local_row = tid / 128;
  for (int base = blockIdx.x * 4; base < M; base += gridDim.x * 4) {
    int row = base + local_row;
    half* input = A + (row < M ? row : 0) * K;
    float amax = 0, sum = 0;
    if (row < M)
      for (int k = lane * 4; k < K; k += 512) {
        float2 x = __half22float2(((const half2*)(input + k))[0]);
        float2 y = __half22float2(((const half2*)(input + k))[1]);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)),
                                 fmaxf(fabsf(y.x), fabsf(y.y))));
        sum += (x.x + x.y) + (y.x + y.y);
      }
    amax =
        __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
#pragma unroll
    for (int x = 16; x; x >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, x);
    if (!(tid & 31)) {
      maxima[warp] = amax;
      sums[warp] = sum;
    }
    __syncthreads();
    int r = local_row * 4;
    amax = fmaxf(fmaxf(maxima[r], maxima[r + 1]),
                 fmaxf(maxima[r + 2], maxima[r + 3]));
    sum = (sums[r] + sums[r + 1]) + (sums[r + 2] + sums[r + 3]);
    float scale = fmaxf(amax / 127.f, 1e-20f), inv = 1.f / scale;
    uint32_t* output = (uint32_t*)input;
    for (int base_k = 0; base_k < K; base_k += 512) {
      int k = base_k + lane * 4;
      uint32_t q = 0;
      if (row < M && k < K) {
        float2 x = __half22float2(((const half2*)(input + k))[0]);
        float2 y = __half22float2(((const half2*)(input + k))[1]);
        q = pack4(__float2int_rn(x.x * inv), __float2int_rn(x.y * inv),
                  __float2int_rn(y.x * inv), __float2int_rn(y.y * inv));
      }
      __syncthreads();
      if (row < M && k < K) output[k / 4] = q;
      __syncthreads();
    }
    if (!lane && row < M) {
      float* tail = (float*)(input + K - 4);
      tail[0] = scale;
      tail[1] = sum;
    }
    __syncthreads();
  }
}
template <bool RESIDUAL>
__device__ __forceinline__ void exl3_gemm_int8_inner(const half* A,
                                                     const uint16_t* B, half* C,
                                                     int M, int K, int N) {
  extern __shared__ half shared[];
  uint8_t* sha = (uint8_t*)shared;
  uint16_t* shb = (uint16_t*)(sha + 3 * 1024);
  int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
  float scale[4], sum[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    int row = lane / 4 + 8 * m;
    scale[m] = 1.f;
    sum[m] = 0.f;
    if (row < M) {
      const float* tail = (const float*)(A + (row + 1) * K - 4);
      scale[m] = tail[0];
      sum[m] = tail[1];
    }
  }
  for (int nt = blockIdx.x; nt < N / 256; nt += gridDim.x) {
    I8A fa[2][2];
    I8B fb[2][2], fr[2][2];
    I8C fc[2][2] = {}, rc[2][2] = {};
    const int steps = K / 32;
    auto load = [&](int k) {
      if (k < steps) {
        int slot = k % 3;
        if (tid < 64) {
          int row = tid / 2, col = tid % 2, sw = col ^ ((row >> 2) & 1);
          if (row < M)
            cp_async(
                (int4*)(sha + slot * 1024) + row * 2 + sw,
                (const int4*)((const uint8_t*)A + row * K * 2 + k * 32) + col);
          else
            ((int4*)(sha + slot * 1024))[row * 2 + sw] = make_int4(0, 0, 0, 0);
        }
        if (tid < 256) {
          int kb = tid / 128, nb = tid % 128;
          cp_async(
              (int4*)(shb + slot * 2048) + tid,
              (const int4*)(B + (k * 2 + kb) * (N / 16) * 64 + nt * 16 * 64) +
                  nb);
        }
      }
      cp_async_fence();
    };
    auto fragments = [&](int k, int buf) {
      const uint8_t* ap = sha + (k % 3) * 1024;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if (m == 1 && M <= 16) continue;
        int row = (lane % 8) + 8 * ((lane / 8) % 2) + m * 16, col = lane / 16;
        FragA v;
        ldsm4(v, (const int4*)ap + row * 2 + (col ^ ((row >> 2) & 1)));
#pragma unroll
        for (int j = 0; j < 4; ++j) fa[buf][m].v[j] = ((uint32_t*)&v)[j];
      }
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        uint32_t bv[2], rv[2];
        decode_i8<RESIDUAL>(
            (const uint32_t*)(shb + (k % 3) * 2048 + (kb * 16 + warp) * 64),
            lane, bv, rv);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          fb[buf][n].v[kb] = bv[n];
          if constexpr (RESIDUAL) fr[buf][n].v[kb] = rv[n];
        }
      }
    };
    load(0);
    load(1);
    cp_async_wait<1>();
    __syncthreads();
    fragments(0, 0);
#define INT8_STEP(BUF)                                                \
  load(k + 2);                                                        \
  cp_async_wait<1>();                                                 \
  __syncthreads();                                                    \
  if (k + 1 < steps) fragments(k + 1, 1 - BUF);                       \
  _Pragma("unroll") for (int m = 0; m < 2; ++m) {                     \
    if (m == 1 && M <= 16) continue;                                  \
    _Pragma("unroll") for (int n = 0; n < 2; ++n) {                   \
      imma(fa[BUF][m], fb[BUF][n], fc[m][n]);                         \
      if constexpr (RESIDUAL) imma(fa[BUF][m], fr[BUF][n], rc[m][n]); \
    }                                                                 \
  }                                                                   \
  ++k;
    for (int k = 0; k < steps;) {
      INT8_STEP(0);
      INT8_STEP(1);
    }
#undef INT8_STEP
    cp_async_wait<0>();
    __syncthreads();
    const float kinv = __half2float(__ushort_as_half(0x1eee));
    const float bias = 1534.f * kinv + __half2float(__ushort_as_half(0xc931));
#pragma unroll
    for (int m = 0; m < 2; ++m)
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          int row = lane / 4 + m * 16 + r * 8;
          if (row < M) {
            float v0 = 4.f * fc[m][n].v[r * 2],
                  v1 = 4.f * fc[m][n].v[r * 2 + 1];
            if constexpr (RESIDUAL) {
              v0 += rc[m][n].v[r * 2];
              v1 += rc[m][n].v[r * 2 + 1];
            }
            half2 v = __floats2half2_rn(
                v0 * (scale[2 * m + r] * kinv) + bias * sum[2 * m + r],
                v1 * (scale[2 * m + r] * kinv) + bias * sum[2 * m + r]);
            *((half2*)(C + row * N + nt * 256 + warp * 16 + n * 8 +
                       (lane % 4) * 2)) = v;
          }
        }
      }
    __syncthreads();
  }
}
