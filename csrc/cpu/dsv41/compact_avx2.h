// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "iqk/iqk_utils.h"

#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(HAVE_VNNI256)

namespace dsv41_avx2 {

struct InputBlock {
  uint16_t d[8];
  int8_t qs[128];
};
static_assert(sizeof(InputBlock) == 144);

inline __m256 scales(const uint8_t* source) {
  const auto e = _mm256_cvtepu8_epi32(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source)));
  const auto normal = _mm256_cmpgt_epi32(e, _mm256_set1_epi32(1));
  const auto a =
      _mm256_slli_epi32(_mm256_sub_epi32(e, _mm256_set1_epi32(1)), 23);
  const auto b =
      _mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(1)), 21);
  return _mm256_castsi256_ps(_mm256_blendv_epi8(b, a, normal));
}

inline __m256i dot(const uint8_t* packed, __m256i low, __m256i high) {
  // Each signed 16-bit lane accumulates eight products: at most 24 * 128 * 8.
  // This permits two widening operations instead of IQK's four on AVX2.
  const auto mask = _mm256_set1_epi8(15);
  const auto table = _mm256_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11,
                                      10, 9, 8, 6, 4, 0, 12, 13, 14, 15, 16, 18,
                                      20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
  const auto one = _mm256_set1_epi16(1);
  auto first = _mm256_setzero_si256();
  auto second = _mm256_setzero_si256();
  #define DSV41_DOT_PART(J, SUM)                                               \
    {                                                                          \
      const auto bits =                                                        \
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed) + J);    \
      const auto lo =                                                          \
          _mm256_shuffle_epi8(table, _mm256_and_si256(bits, mask));            \
      const auto hi = _mm256_shuffle_epi8(                                     \
          table, _mm256_and_si256(_mm256_srli_epi16(bits, 4), mask));          \
      SUM = _mm256_add_epi16(                                                  \
          SUM, _mm256_maddubs_epi16(lo, _mm256_shuffle_epi32(low, J * 0x55))); \
      SUM = _mm256_add_epi16(                                                  \
          SUM,                                                                 \
          _mm256_maddubs_epi16(hi, _mm256_shuffle_epi32(high, J * 0x55)));     \
    }
  DSV41_DOT_PART(0, first);
  DSV41_DOT_PART(1, first);
  DSV41_DOT_PART(2, second);
  DSV41_DOT_PART(3, second);
  #undef DSV41_DOT_PART
  return _mm256_add_epi32(_mm256_madd_epi16(first, one),
                          _mm256_madd_epi16(second, one));
}

// Keep IQK's per-block FP32 FMA order and its final offset correction.
// Two projections share the input block and scale conversion.
template <int Projections>
inline void project(int n, int rows, size_t weight_stride, const char* first,
                    const char* second, const uint8_t* input, float* output,
                    float limit) {
  const auto* q = reinterpret_cast<const InputBlock*>(input);
  for (int row = 0; row < rows; row += 8) {
    const char* weights[2] = {first + row * weight_stride,
                              second ? second + row * weight_stride : nullptr};
    __m256 accum[Projections][2] = {};
    for (int block = 0; block < n / 128; ++block) {
      const auto d = _mm_castsi128_ps(_mm_slli_epi32(
          _mm_cvtepu16_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q[block].d))),
          16));
      const auto sum = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q[block].d + 4))));
      float ds[8];
      _mm_storeu_ps(ds, d);
      _mm_storeu_ps(ds + 4, _mm_mul_ps(d, sum));
      for (int k = 0; k < 4; ++k) {
        const auto* y = reinterpret_cast<const __m128i*>(q[block].qs + 32 * k);
        const auto low = _mm256_broadcastsi128_si256(_mm_loadu_si128(y));
        const auto high = _mm256_broadcastsi128_si256(_mm_loadu_si128(y + 1));
        for (int p = 0; p < Projections; ++p) {
          const auto* weight = reinterpret_cast<const uint8_t*>(weights[p]) +
                               (4 * block + k) * 136;
          const auto scale = scales(weight);
          const auto value = dot(weight + 8, low, high);
          accum[p][0] =
              _mm256_fmadd_ps(_mm256_mul_ps(scale, _mm256_set1_ps(ds[k])),
                              _mm256_cvtepi32_ps(value), accum[p][0]);
          accum[p][1] =
              _mm256_fmadd_ps(scale, _mm256_set1_ps(ds[k + 4]), accum[p][1]);
        }
      }
    }
    auto result =
        _mm256_fmadd_ps(accum[0][1], _mm256_set1_ps(-12), accum[0][0]);
    if constexpr (Projections == 2) {
      auto up = _mm256_fmadd_ps(accum[1][1], _mm256_set1_ps(-12), accum[1][0]);
      if (limit > 1e-6f) {
        result = _mm256_min_ps(_mm256_set1_ps(limit), result);
        up = _mm256_max_ps(_mm256_set1_ps(-limit),
                           _mm256_min_ps(up, _mm256_set1_ps(limit)));
      }
      result = _mm256_mul_ps(up, v_silu(result));
    }
    _mm256_storeu_ps(output + row, result);
  }
}

}  // namespace dsv41_avx2
#endif
