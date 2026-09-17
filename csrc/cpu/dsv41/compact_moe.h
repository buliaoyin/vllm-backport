// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "compact_avx2.h"
#include "compact_profile.h"
#include "ggml.h"
#include "iqk/iqk_mul_mat.h"
#include "numa.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <omp.h>

// Reuse IQK's projection kernels while keeping only CPU route intermediates.
struct CompactMoE {
  struct Row {
    int32_t slot, token;
  };
  struct Group {
    int expert, first, count;
  };
  static constexpr int up_tile = 128;
  std::vector<Group> groups;
  std::vector<Row> up_rows, down_rows;
  std::vector<int> positions, slots;
  std::vector<uint8_t> input_q, activation_q;
  std::vector<float> activation, down;

  template <typename T>
  static void grow(std::vector<T>& buffer, size_t size) {
    if (buffer.size() < size) buffer.resize(size);
  }

  CompactProfile profile;
  // Bits: 1 serial input, 2 fixed down tiles, 4 one final barrier;
  // 8 AVX2 up/gate and 16 AVX2 down for single-token expert groups.
  // -1 selects the schedule validated for each batch size.
  int schedule = -1;

  void forward(const std::array<ggml_tensor*, 3>& weights, int tokens, int topk,
               int threads, float limit, const float* input, const int32_t* ids,
               const float* routes, float* output,
               const dsv41::NumaPlan& numa) {
    if (profile.record(tokens, weights[0]->ne[0], topk, input, ids, routes)) {
      forward_impl<true>(weights, tokens, topk, threads, limit, input, ids,
                         routes, output, numa);
    } else {
      forward_impl<false>(weights, tokens, topk, threads, limit, input, ids,
                          routes, output, numa);
    }
  }

  template <bool Profile>
  void forward_impl(const std::array<ggml_tensor*, 3>& weights, int tokens,
                    int topk, int threads, float limit, const float* input,
                    const int32_t* ids, const float* routes, float* output,
                    const dsv41::NumaPlan& numa) {
    uint64_t begin_ns = 0, parallel_ns = 0;
    if constexpr (Profile) {
      profile.start(threads);
      begin_ns = CompactProfile::now();
    }
    const int hidden = weights[0]->ne[0];
    const int intermediate = weights[0]->ne[1];
    const int capacity = tokens * topk;
    positions.clear();
    groups.clear();
    slots.assign(capacity, -1);
    for (int p = 0; p < capacity; ++p) {
      if (ids[p] >= 0) positions.push_back(p);
    }
    if (positions.empty()) {
      std::fill_n(output, size_t(tokens) * hidden, 0.0f);
      return;
    }
    std::sort(positions.begin(), positions.end(), [ids](int a, int b) {
      return ids[a] == ids[b] ? a < b : ids[a] < ids[b];
    });
    const int count = positions.size();
    up_rows.resize(count);
    down_rows.resize(count);
    for (int row = 0; row < count; ++row) {
      const int p = positions[row], expert = ids[p];
      if (groups.empty() || groups.back().expert != expert) {
        groups.push_back({expert, row, 0});
      }
      ++groups.back().count;
      slots[p] = row;
      up_rows[row] = {row, p / topk};
      down_rows[row] = {row, 0};
    }

    const auto type =
        ggml_internal_get_type_traits(weights[0]->type).vec_dot_type;
    const auto quantize = ggml_internal_get_type_traits(type).from_float;
    const size_t input_stride = ggml_row_size(type, hidden);
    const size_t activation_stride = ggml_row_size(type, intermediate);
    grow(input_q, size_t(tokens) * input_stride);
    grow(activation_q, size_t(count) * activation_stride);
    grow(activation, size_t(count) * intermediate);
    const int down_tile = numa.enabled()
                              ? dsv41::NumaPlan::tile_rows
                              : ((hidden / 8 + threads - 1) / threads) * 8;
    grow(down, size_t(threads) * count * down_tile);
    const int input_tiles = (hidden + up_tile - 1) / up_tile;
    const int up_tiles = (intermediate + up_tile - 1) / up_tile;
    const int down_tiles = threads;
    const size_t quant_tile_bytes = ggml_row_size(type, up_tile);
    const int group_count = groups.size();

    const int flags = schedule < 0 ? (tokens == 1 ? 15 : 8) : schedule;
    const bool serial_input = flags & 1;
    if (serial_input) {
      for (int token = 0; token < tokens; ++token) {
        quantize(input + size_t(token) * hidden,
                 input_q.data() + token * input_stride, hidden);
      }
    }
    dsv41::NumaPlan::Binding caller(numa);
    caller.bind_team();
    numa.check_affinity();
    if constexpr (Profile) parallel_ns = CompactProfile::now();
#pragma omp parallel num_threads(threads)
    {
      dsv41::NumaPlan::Binding binding(numa, true);
      if (numa.enabled())
        binding.bind(numa.compute_team[omp_get_thread_num()].cpu);
      CompactProfile::Worker* worker = nullptr;
      if constexpr (Profile) {
        worker = &profile.workers[omp_get_thread_num()];
        worker->marks[0] = CompactProfile::now();
      }
      if (!serial_input) {
#pragma omp for schedule(static) nowait
        for (int task = 0; task < tokens * input_tiles; ++task) {
          const int token = task / input_tiles;
          const int tile = task % input_tiles;
          const int begin = tile * up_tile;
          quantize(
              input + size_t(token) * hidden + begin,
              input_q.data() + token * input_stride + tile * quant_tile_bytes,
              std::min(up_tile, hidden - begin));
        }
        if constexpr (Profile) worker->marks[1] = CompactProfile::now();
#pragma omp barrier
        if constexpr (Profile) worker->marks[2] = CompactProfile::now();
      } else if constexpr (Profile) {
        worker->marks[1] = worker->marks[2] = CompactProfile::now();
      }

      const auto run_up = [&](int task) {
        const auto& group = groups[task / up_tiles];
        const int tile = task % up_tiles;
        const int begin = tile * up_tile;
        const int rows = std::min(up_tile, intermediate - begin);
        const auto address = [&](int projection) {
          const auto* weight = weights[projection];
          return static_cast<const char*>(weight->data) +
                 group.expert * weight->nb[2] + begin * weight->nb[1];
        };
        uint64_t up_begin = 0, quant_begin = 0;
        if constexpr (Profile) up_begin = CompactProfile::now();
#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(HAVE_VNNI256)
        if ((flags & 8) && group.count == 1 && hidden % 128 == 0) {
          const auto& row = up_rows[group.first];
          dsv41_avx2::project<2>(
              hidden, rows, weights[0]->nb[1], address(0), address(2),
              input_q.data() + row.token * input_stride,
              activation.data() + size_t(row.slot) * intermediate + begin,
              limit);
        } else
#endif
        {
          const bool ok = iqk_moe_fused_up_gate(
              rows, group.count, hidden, 1, GGML_UNARY_OP_SILU,
              weights[0]->type, address(2), address(0), weights[0]->nb[1], type,
              input_q.data(), input_stride, nullptr, nullptr,
              activation.data() + begin, intermediate * sizeof(float), 0,
              up_rows.data() + group.first, limit, 0, 1);
          GGML_ASSERT(ok);
        }
        if constexpr (Profile) {
          quant_begin = CompactProfile::now();
          worker->up += quant_begin - up_begin;
        }
        for (int row = group.first; row < group.first + group.count; ++row) {
          quantize(activation.data() + size_t(row) * intermediate + begin,
                   activation_q.data() + row * activation_stride +
                       tile * quant_tile_bytes,
                   rows);
        }
        if constexpr (Profile)
          worker->quant += CompactProfile::now() - quant_begin;
      };
      if (numa.enabled()) {
        for (int s = omp_get_thread_num(); s < int(numa.compute_team.size());
             s += omp_get_num_threads()) {
          const auto& slot = numa.compute_team[s];
          binding.bind(slot.cpu);
          const int first = numa.boundary(up_tiles, slot.node);
          const int width = numa.boundary(up_tiles, slot.node + 1) - first;
          for (int task = slot.rank; task < group_count * width;
               task += slot.workers)
            run_up((task / width) * up_tiles + first + task % width);
        }
      } else {
#pragma omp for schedule(dynamic, 1) nowait
        for (int task = 0; task < group_count * up_tiles; ++task) run_up(task);
      }
      if constexpr (Profile) worker->marks[3] = CompactProfile::now();
#pragma omp barrier
      if constexpr (Profile) worker->marks[4] = CompactProfile::now();

      auto* partial =
          down.data() + size_t(omp_get_thread_num()) * count * down_tile;
      auto run_down = [&](int begin, int end) {
        const int rows = end - begin;
        if (!rows) return;
        uint64_t down_begin = 0, reduce_begin = 0;
        if constexpr (Profile) down_begin = CompactProfile::now();
        const auto* weight = weights[1];
        for (const auto& group : groups) {
          const auto* data = static_cast<const char*>(weight->data) +
                             group.expert * weight->nb[2] +
                             begin * weight->nb[1];
#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(HAVE_VNNI256)
          if ((flags & 16) && group.count == 1 && intermediate % 128 == 0) {
            dsv41_avx2::project<1>(
                intermediate, rows, weight->nb[1], data, nullptr,
                activation_q.data() + group.first * activation_stride,
                partial + group.first * down_tile, limit);
          } else
#endif
          {
            const bool ok = iqk_mul_mat_moe(
                rows, group.count, intermediate, count, weight->type, data,
                weight->nb[1], type, activation_q.data(), activation_stride,
                partial, down_tile * sizeof(float), 0,
                down_rows.data() + group.first, 0, 1);
            GGML_ASSERT(ok);
          }
        }
        if constexpr (Profile) {
          reduce_begin = CompactProfile::now();
          worker->down += reduce_begin - down_begin;
        }
        for (int token = 0; token < tokens; ++token) {
          for (int col = 0; col < rows; ++col) {
            // GGML rounds each weighted product to FP32, then sums in FP64.
            double sum = 0;
            for (int k = 0; k < topk; ++k) {
              const int p = token * topk + k, row = slots[p];
              if (row >= 0) {
                const float weighted =
                    partial[row * down_tile + col] * routes[p];
                sum += double(weighted);
              }
            }
            output[size_t(token) * hidden + begin + col] = float(sum);
          }
        }
        if constexpr (Profile)
          worker->reduce += CompactProfile::now() - reduce_begin;
      };
      const auto run_tile = [&](int tile) {
        run_down((hidden / 8 * tile / down_tiles) * 8,
                 (hidden / 8 * (tile + 1) / down_tiles) * 8);
      };
      if (numa.enabled()) {
        const int tiles = (hidden + down_tile - 1) / down_tile;
        for (int s = omp_get_thread_num(); s < int(numa.compute_team.size());
             s += omp_get_num_threads()) {
          const auto& slot = numa.compute_team[s];
          binding.bind(slot.cpu);
          const int end = numa.boundary(tiles, slot.node + 1);
          for (int tile = numa.boundary(tiles, slot.node) + slot.rank;
               tile < end; tile += slot.workers)
            run_down(tile * down_tile,
                     std::min((tile + 1) * down_tile, hidden));
        }
      } else if (flags & 2) {
        for (int tile = omp_get_thread_num(); tile < down_tiles;
             tile += omp_get_num_threads())
          run_tile(tile);
      } else {
#pragma omp for schedule(dynamic, 1) nowait
        for (int tile = 0; tile < down_tiles; ++tile) run_tile(tile);
      }
      if constexpr (Profile) worker->marks[5] = CompactProfile::now();
      if (!(flags & 4)) {
#pragma omp barrier
      }
      if constexpr (Profile) worker->marks[6] = CompactProfile::now();
    }
    if (numa.enabled()) numa.check_affinity();
    if constexpr (Profile)
      profile.finish(begin_ns, parallel_ns, CompactProfile::now());
  }
};
