// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Turboderp; batched pointer-table adaptation.
#include "upstream/quant/exl3_dq.cuh"

__device__ __forceinline__ half2 integer_pair(uint32_t a, uint32_t b) {
  int x = (int)__dp4a(a * 0x83DCD12Du, 0x01010101u, (uint32_t)-508);
  int y = (int)__dp4a(b * 0x83DCD12Du, 0x01010101u, (uint32_t)-508);
  x = max(-128, min(127, x >> 2));
  y = max(-128, min(127, y >> 2));
  return __halves2half2(__int2half_rn(x), __int2half_rn(y));
}
template <int cb>
__device__ __forceinline__ void dq_integer(const uint32_t* ptr, int t_offset,
                                           FragB& frag0, FragB& frag1) {
  uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
  i1 = t_offset >> 3;
  i0 = (i1 + 31) & 31;
  a = ptr[i0];
  b = ptr[i1];
  FSHF_IMM(s, b, a, 20);
  w7 = b & 0xffff;
  BFE16_IMM(w6, b, 4);
  BFE16_IMM(w5, b, 8);
  BFE16_IMM(w4, b, 12);
  BFE16_IMM(w3, b, 16);
  w2 = s & 0xffff;
  BFE16_IMM(w1, s, 4);
  BFE16_IMM(w0, s, 8);
  frag0[0] = integer_pair(w0, w1);
  frag0[1] = integer_pair(w2, w3);
  frag1[0] = integer_pair(w4, w5);
  frag1[1] = integer_pair(w6, w7);
}

template <int K, int cb>
__global__ __launch_bounds__(256) void reconstruct_experts(
    int8_t* __restrict__ g_unpacked,
    const uint16_t* const* __restrict__ packed_tables, int packed_blocks_n,
    int size_k) {
  const int packed_n_offset = 0;
  const uint16_t* g_packed = packed_tables[blockIdx.z];
  g_unpacked += size_t(blockIdx.z) * size_k * packed_blocks_n * 16;
  constexpr int packed_size = 256 * K / 16;  // in uint16s

  int t = threadIdx.x;
  int lane_id = t % 32;
  int warp_id = t / 32;
  int k = blockIdx.y;
  int n = blockIdx.x * 8;
  int tiles_n = gridDim.x;
  int out_blocks_n = tiles_n * 8;

  // Load packed 16*128 tile
  __shared__ uint32_t s_packed[8][packed_size / 2];
  g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
  if (t < packed_size) ((int4*)s_packed)[t] = ((int4*)g_packed)[t];
  __syncthreads();

  // Dequant
  FragB frag[2];
  dq_integer<cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

  // Shuffle from tensor core layout to row major tile
  __shared__ half2 tile[16][8][8];

  half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
  half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
  half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
  half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

  if (!(lane_id & 4)) {
    half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
    half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
    half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
    half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
    half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
    half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
    half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
    half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
    int r0 = (lane_id % 4) * 2;
    int r1 = r0 + 1;
    int r2 = r0 + 8;
    int r3 = r0 + 9;
    int c0 = lane_id / 8;
    int c1 = c0 + 4;
    tile[r0][warp_id][c0] = m0;
    tile[r1][warp_id][c0] = m1;
    tile[r2][warp_id][c0] = m2;
    tile[r3][warp_id][c0] = m3;
    tile[r0][warp_id][c1] = m4;
    tile[r1][warp_id][c1] = m5;
    tile[r2][warp_id][c1] = m6;
    tile[r3][warp_id][c1] = m7;
  }
  __syncthreads();

  // Store unpacked tile
  int r = t / 16;
  int c = t % 16;
  const half* values = reinterpret_cast<const half*>(tile) + t * 8;
  uint32_t packed[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int q = __half2int_rn(values[i]);
    packed[i / 4] |= (uint32_t)(uint8_t)(int8_t)q << (8 * (i % 4));
  }
  int8_t* output =
      g_unpacked + (k * 16 + r) * out_blocks_n * 16 + n * 16 + c * 8;
  *reinterpret_cast<uint2*>(output) = make_uint2(packed[0], packed[1]);
}
