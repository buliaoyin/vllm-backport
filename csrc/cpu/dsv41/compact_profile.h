// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

struct CompactProfile {
  struct alignas(64) Worker {
    std::array<uint64_t, 7> marks{};
    uint64_t up = 0, quant = 0, down = 0, reduce = 0;
  };
  struct Frame {
    int tokens, hidden, topk;
    std::vector<float> input, routes;
    std::vector<int32_t> ids;
  };
  int interval = 0;
  std::array<uint64_t, 64> totals{};
  std::vector<Worker> workers;
  std::vector<Frame> frames;

  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void reset(int every) {
    interval = every;
    totals.fill(0);
    frames.clear();
  }

  bool record(int tokens, int hidden, int topk, const float* input,
              const int32_t* ids, const float* routes) {
    if (!interval) return false;
    ++totals[0];
    totals[1] += tokens;
    std::array<int, 384> counts{};
    for (int p = 0; p < tokens * topk; ++p) {
      if (ids[p] >= 0 && ids[p] < int(counts.size())) ++counts[ids[p]];
    }
    int groups = 0;
    for (const int count : counts) {
      if (!count) continue;
      ++groups;
      totals[2] += count;
      ++totals[32 + std::min(count, 16)];
    }
    totals[3] += groups;
    if (!groups) ++totals[4];
    if (frames.size() < 128) {
      frames.push_back({tokens,
                        hidden,
                        topk,
                        {input, input + size_t(tokens) * hidden},
                        {routes, routes + tokens * topk},
                        {ids, ids + tokens * topk}});
    }
    return groups && (totals[0] - 1) % interval == 0;
  }

  void start(int threads) { workers.assign(threads, Worker{}); }

  void finish(uint64_t begin, uint64_t parallel, uint64_t end) {
    ++totals[5];
    totals[7] += end - begin;
    totals[8] += parallel - begin;
    uint64_t first = std::numeric_limits<uint64_t>::max();
    std::array<uint64_t, 3> last{}, arrived{};
    for (const auto& w : workers) {
      if (!w.marks[0]) continue;
      ++totals[6];
      first = std::min(first, w.marks[0]);
      for (int stage = 0; stage < 3; ++stage) {
        totals[10 + 2 * stage] += w.marks[2 * stage + 1] - w.marks[2 * stage];
        totals[11 + 2 * stage] +=
            w.marks[2 * stage + 2] - w.marks[2 * stage + 1];
        last[stage] = std::max(last[stage], w.marks[2 * stage + 2]);
        arrived[stage] = std::max(arrived[stage], w.marks[2 * stage + 1]);
      }
      totals[16] += w.up;
      totals[17] += w.quant;
      totals[18] += w.down;
      totals[19] += w.reduce;
    }
    totals[9] += first - parallel + end - last[2];
    totals[20] += last[0] - first;
    totals[21] += last[1] - last[0];
    totals[22] += last[2] - last[1];
    totals[23] += arrived[0] - first;
    totals[24] += arrived[1] - arrived[0];
    totals[25] += arrived[2] - arrived[1];
    totals[26] += first - parallel + end - arrived[2];
  }

  void dump(const char* path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open CPU trace file");
    const char magic[] = "DSV41TR1";
    file.write(magic, 8);
    const uint32_t count = frames.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& frame : frames) {
      const uint32_t shape[] = {uint32_t(frame.tokens), uint32_t(frame.hidden),
                                uint32_t(frame.topk)};
      file.write(reinterpret_cast<const char*>(shape), sizeof(shape));
      file.write(reinterpret_cast<const char*>(frame.input.data()),
                 frame.input.size() * sizeof(float));
      file.write(reinterpret_cast<const char*>(frame.ids.data()),
                 frame.ids.size() * sizeof(int32_t));
      file.write(reinterpret_cast<const char*>(frame.routes.data()),
                 frame.routes.size() * sizeof(float));
    }
    if (!file) throw std::runtime_error("Cannot write CPU trace file");
  }
};
