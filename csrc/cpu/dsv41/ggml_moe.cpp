// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef DSV41_IK
  #include "iqk/iqk_quantize.h"
#else
  #include "ggml-cpu.h"
  #include "ggml-cpu/repack.h"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__SSE2__)
  #include <emmintrin.h>
#endif

#ifdef __linux__
  #include <sys/mman.h>
  #include <unistd.h>
#endif

#ifdef DSV41_IK
  #include "compact_moe.h"
#endif

namespace {
thread_local std::string last_error;

struct Graph {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_cgraph* graph = nullptr;
  ggml_tensor *input = nullptr, *ids = nullptr, *routes = nullptr;
  ggml_tensor* output = nullptr;
  int tokens = 0;

  ~Graph() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
  }
};

struct MoE {
  int experts, hidden, intermediate, topk, load_threads, num_threads;
  float limit;
  ggml_backend_t backend = nullptr;
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t weights_buffer = nullptr;
  std::array<ggml_tensor*, 3> weights{};
  std::vector<bool> loaded;
  std::unique_ptr<Graph> execution;
#ifdef DSV41_IK
  CompactMoE compact;
  dsv41::NumaPlan numa;
  bool use_compact = true;
#endif

  MoE(int e, int h, int i, int k, int threads, float clamp)
      : experts(e),
        hidden(h),
        intermediate(i),
        topk(k),
        load_threads(std::min(8, threads)),
        num_threads(threads),
        limit(clamp),
        loaded(3 * e, false) {
    if (e <= 0 || h <= 0 || i <= 0 || k <= 0 || k > e || threads <= 0 ||
        h % 32 || i % 32 || clamp < 0) {
      throw std::invalid_argument("Invalid MXFP4 MoE geometry or thread count");
    }
    backend = ggml_backend_cpu_init();
    if (!backend) throw std::runtime_error("Cannot create GGML CPU backend");
    ggml_backend_cpu_set_n_threads(backend, threads);
#ifdef DSV41_IK
    const char* executor = std::getenv("DSV41_CPU_EXECUTOR");
    if (executor && std::strcmp(executor, "graph") == 0) use_compact = false;
#endif
    ctx = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    if (!ctx) throw std::bad_alloc();
#ifdef DSV41_IK
    const auto type = GGML_TYPE_MXFP4_R8;
    const auto buft = ggml_backend_cpu_buffer_type();
#else
    const auto type = GGML_TYPE_MXFP4;
    const auto buft = ggml_backend_cpu_repack_buffer_type();
    if (!buft) throw std::runtime_error("GGML CPU repack support is required");
#endif
    weights[0] = ggml_new_tensor_3d(ctx, type, h, i, e);
    weights[1] = ggml_new_tensor_3d(ctx, type, i, h, e);
    weights[2] = ggml_new_tensor_3d(ctx, type, h, i, e);
    for (int p = 0; p < 3; ++p) {
      ggml_set_name(weights[p], ("experts.w" + std::to_string(p + 1)).c_str());
    }
    weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!weights_buffer) throw std::bad_alloc();
#ifdef __linux__
    const char* hugepages = std::getenv("DSV41_CPU_HUGEPAGES");
    if (hugepages && std::strcmp(hugepages, "1") == 0) {
      // Advise only full pages owned by the expert buffer, before first touch.
      const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
      const auto base = reinterpret_cast<uintptr_t>(
          ggml_backend_buffer_get_base(weights_buffer));
      const auto first = (base + page - 1) & ~(page - 1);
      const auto last =
          (base + ggml_backend_buffer_get_size(weights_buffer)) & ~(page - 1);
      if (last > first && last - first >= 2 * 1024 * 1024) {
        madvise(reinterpret_cast<void*>(first), last - first, MADV_HUGEPAGE);
      }
    }
#endif
  }

  ~MoE() {
    execution.reset();
    if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
    if (ctx) ggml_free(ctx);
    if (backend) ggml_backend_free(backend);
  }

  void load(int expert, int projection, const uint8_t* packed,
            const uint8_t* scales) {
    if (expert < 0 || expert >= experts || projection < 0 || projection > 2 ||
        !packed || !scales) {
      throw std::invalid_argument("Invalid expert weight slice");
    }
    auto* weight = weights[projection];
    auto* destination =
        static_cast<uint8_t*>(weight->data) + expert * weight->nb[2];
#ifdef DSV41_IK
    // Native R8 blocks interleave eight rows: eight E8M0 bytes followed by
    // four groups of four E2M1 bytes per row. Convert directly from HF, without
    // an intermediate tensor or the generic repacker's per-matrix thread pool.
    const int row_blocks = weight->ne[0] / 32;
    const auto convert_row = [&](int row) {
      for (int block = 0; block < row_blocks; ++block) {
        auto* dst = destination + (size_t(row / 8) * row_blocks + block) * 136;
        for (int lane = 0; lane < 8; ++lane) {
          const size_t source_block = size_t(row + lane) * row_blocks + block;
          const auto* src = packed + source_block * 16;
          dst[lane] = scales[source_block];
  #if defined(__SSE2__)
          const auto input =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
          const auto mask = _mm_set1_epi8(15);
          const auto low = _mm_and_si128(input, mask);
          const auto high = _mm_and_si128(_mm_srli_epi16(input, 4), mask);
          const auto even =
              _mm_or_si128(low, _mm_slli_epi16(_mm_srli_si128(low, 8), 4));
          const auto odd =
              _mm_or_si128(high, _mm_slli_epi16(_mm_srli_si128(high, 8), 4));
          auto output = _mm_unpacklo_epi8(even, odd);
          for (int j = 0; j < 4; ++j) {
            const uint32_t word = _mm_cvtsi128_si32(output);
            std::memcpy(dst + 8 + j * 32 + lane * 4, &word, sizeof(word));
            output = _mm_srli_si128(output, 4);
          }
  #else
          for (int j = 0; j < 16; ++j) {
            const int shift = (j & 1) * 4;
            dst[8 + (j / 4) * 32 + lane * 4 + j % 4] =
                ((src[j / 2] >> shift) & 15) |
                (((src[j / 2 + 8] >> shift) & 15) << 4);
          }
  #endif
        }
      }
    };
    if (numa.enabled()) {
      numa.weight_rows(weight->ne[1], load_threads, true, convert_row);
    } else {
  #pragma omp parallel for num_threads( \
          load_threads) if (weight->ne[0] * weight->ne[1] >= 1048576)
      for (int row = 0; row < weight->ne[1]; row += 8) convert_row(row);
    }
#else
    const size_t blocks = size_t(hidden) * intermediate / 32;
    std::vector<uint8_t> ggml_packed(blocks * 17);
    for (size_t b = 0; b < blocks; ++b) {
      auto* dst = ggml_packed.data() + b * 17;
      const auto* src = packed + b * 16;
      dst[0] = scales[b];
      // HF pairs adjacent values; GGML pairs the two 16-value halves.
      for (int j = 0; j < 16; ++j) {
        const int shift = (j & 1) * 4;
        dst[j + 1] = ((src[j / 2] >> shift) & 15) |
                     (((src[j / 2 + 8] >> shift) & 15) << 4);
      }
    }
    ggml_tensor slice = *weight;
    slice.ne[2] = 1;
    slice.nb[3] = slice.nb[2];
    slice.data = destination;
    ggml_backend_tensor_set(&slice, ggml_packed.data(), 0, ggml_packed.size());
#endif
    loaded[projection * experts + expert] = true;
  }

  void export_weight(int expert, int projection, uint8_t* packed,
                     uint8_t* scales) const {
#ifdef DSV41_IK
    if (expert < 0 || expert >= experts || projection < 0 || projection > 2 ||
        !packed || !scales || !loaded[projection * experts + expert]) {
      throw std::invalid_argument("Invalid or unloaded expert weight slice");
    }
    const auto* weight = weights[projection];
    const auto* source =
        static_cast<const uint8_t*>(weight->data) + expert * weight->nb[2];
    const int row_blocks = weight->ne[0] / 32;
    const auto convert_row = [&](int row) {
      for (int block = 0; block < row_blocks; ++block) {
        const auto* src = source + (size_t(row / 8) * row_blocks + block) * 136;
        for (int lane = 0; lane < 8; ++lane) {
          const size_t destination_block =
              size_t(row + lane) * row_blocks + block;
          auto* dst = packed + destination_block * 16;
          scales[destination_block] = src[lane];
  #if defined(__SSE2__)
          uint32_t words[4];
          for (int j = 0; j < 4; ++j) {
            std::memcpy(&words[j], src + 8 + j * 32 + lane * 4,
                        sizeof(words[j]));
          }
          const auto input =
              _mm_setr_epi32(words[0], words[1], words[2], words[3]);
          const auto mask = _mm_set1_epi16(15);
          const auto first = _mm_or_si128(
              _mm_and_si128(input, mask),
              _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(input, 8), mask), 4));
          const auto second = _mm_or_si128(
              _mm_and_si128(_mm_srli_epi16(input, 4), mask),
              _mm_and_si128(_mm_srli_epi16(input, 8), _mm_set1_epi16(240)));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                           _mm_packus_epi16(first, second));
  #else
          for (int j = 0; j < 8; ++j) {
            const int offset = 8 + (j / 2) * 32 + lane * 4 + (j % 2) * 2;
            const uint8_t first = src[offset], second = src[offset + 1];
            dst[j] = (first & 15) | ((second & 15) << 4);
            dst[j + 8] = (first >> 4) | (second & 240);
          }
  #endif
        }
      }
    };
    if (numa.enabled()) {
      numa.weight_rows(weight->ne[1], load_threads, false, convert_row);
    } else {
  #pragma omp parallel for num_threads( \
          load_threads) if (weight->ne[0] * weight->ne[1] >= 1048576)
      for (int row = 0; row < weight->ne[1]; row += 8) convert_row(row);
    }
#else
    throw std::runtime_error("Expert export requires the IK R8 backend");
#endif
  }

  void prepare(int tokens) {
    if (execution && execution->tokens == tokens) return;
    auto g = std::make_unique<Graph>();
    g->tokens = tokens;
    g->ctx = ggml_init(
        {ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false),
         nullptr, true});
    if (!g->ctx) throw std::bad_alloc();
    auto* c = g->ctx;
    g->input = ggml_new_tensor_3d(c, GGML_TYPE_F32, hidden, 1, tokens);
    g->ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, topk, tokens);
    g->routes = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, topk, tokens);
    ggml_tensor* activated;
#ifdef DSV41_IK
    activated = ggml_moe_up_gate(c, weights[2], weights[0], g->input, g->ids,
                                 GGML_UNARY_OP_SILU);
    // The pinned IQK fused operator reads its clamp from parameter slot 1.
    std::memcpy(activated->op_params + 1, &limit, sizeof(limit));
#else
    auto* gate = ggml_mul_mat_id(c, weights[0], g->input, g->ids);
    auto* up = ggml_mul_mat_id(c, weights[2], g->input, g->ids);
    if (limit > 0) {
      gate =
          ggml_clamp(c, gate, -std::numeric_limits<float>::infinity(), limit);
      up = ggml_clamp(c, up, -limit, limit);
    }
    activated = ggml_mul(c, ggml_silu(c, gate), up);
#endif
    auto* down = ggml_mul_mat_id(c, weights[1], activated, g->ids);
    auto* weighted = ggml_mul(c, down, g->routes);
    auto* transposed = ggml_cont(c, ggml_permute(c, weighted, 1, 0, 2, 3));
    g->output =
        ggml_reshape_2d(c, ggml_sum_rows(c, transposed), hidden, tokens);
    g->graph = ggml_new_graph_custom(c, 128, false);
    ggml_build_forward_expand(g->graph, g->output);
    g->buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    if (!g->buffer) throw std::bad_alloc();
    execution = std::move(g);
  }

  void forward(int tokens, const float* input, const int32_t* ids,
               const float* routes, float* output) {
    if (tokens < 0) throw std::invalid_argument("Negative token count");
    if (!tokens) return;
    if (!input || !ids || !routes || !output) {
      throw std::invalid_argument("Null MoE input");
    }
    for (int t = 0; t < tokens * topk; ++t) {
      const auto e = ids[t];
#ifdef DSV41_IK
      if (e == -2) continue;  // This route is computed by the GPU cache.
#endif
      if (e < 0 || e >= experts || !loaded[e] || !loaded[experts + e] ||
          !loaded[2 * experts + e]) {
        throw std::invalid_argument(
            "Route selects an invalid or unloaded expert");
      }
    }
#ifdef DSV41_IK
    if (use_compact && (tokens <= 64 || numa.enabled())) {
      // Bound workspace for replay/concurrent DSpark batches on NUMA hosts.
      if (execution && execution->tokens > 128) execution.reset();
      for (int first = 0; first < tokens; first += 128) {
        compact.forward(weights, std::min(128, tokens - first), topk,
                        num_threads, limit, input + size_t(first) * hidden,
                        ids + size_t(first) * topk,
                        routes + size_t(first) * topk,
                        output + size_t(first) * hidden, numa);
      }
      return;
    }
#endif
    prepare(tokens);
    auto& g = *execution;
    ggml_backend_tensor_set(g.input, input, 0, size_t(tokens) * hidden * 4);
    ggml_backend_tensor_set(g.ids, ids, 0, size_t(tokens) * topk * 4);
    ggml_backend_tensor_set(g.routes, routes, 0, size_t(tokens) * topk * 4);
    if (ggml_backend_graph_compute(backend, g.graph) != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("GGML MoE execution failed");
    }
    ggml_backend_tensor_get(g.output, output, 0, size_t(tokens) * hidden * 4);
  }
};
}  // namespace

extern "C" {
const char* dsv41_moe_error() { return last_error.c_str(); }

int dsv41_moe_supports_cached_routes() {
#ifdef DSV41_IK
  return 1;
#else
  return 0;
#endif
}

int dsv41_moe_supports_export() {
#ifdef DSV41_IK
  return 1;
#else
  return 0;
#endif
}

int dsv41_moe_export(void* handle, int expert, int projection, uint8_t* packed,
                     uint8_t* scales) {
  try {
    if (!handle) throw std::invalid_argument("Null MoE handle");
    static_cast<MoE*>(handle)->export_weight(expert, projection, packed,
                                             scales);
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

void* dsv41_moe_create(int experts, int hidden, int intermediate, int topk,
                       int threads, float limit) {
  try {
    return new MoE(experts, hidden, intermediate, topk, threads, limit);
  } catch (const std::exception& e) {
    last_error = e.what();
    return nullptr;
  }
}

void dsv41_moe_destroy(void* handle) { delete static_cast<MoE*>(handle); }

int dsv41_moe_configure_numa(void* handle, int count, const int* node_ids,
                             const int* offsets, const int* cpus) {
  try {
#ifdef DSV41_IK
    if (!handle || count < 1 || !node_ids || !offsets || !cpus || offsets[0])
      throw std::invalid_argument("Invalid CPU NUMA topology");
    auto* moe = static_cast<MoE*>(handle);
    if (std::any_of(moe->loaded.begin(), moe->loaded.end(),
                    [](bool b) { return b; }))
      throw std::invalid_argument(
          "NUMA topology must be set before loading weights");
    auto& plan = moe->numa;
    plan.nodes.clear();
    plan.cores = offsets[count];
    std::vector<int> unique;
    for (int n = 0; n < count; ++n) {
      if (node_ids[n] < 0 || offsets[n + 1] <= offsets[n])
        throw std::invalid_argument("Empty or invalid CPU NUMA node");
      plan.nodes.push_back({node_ids[n],
                            offsets[n],
                            {cpus + offsets[n], cpus + offsets[n + 1]}});
      unique.insert(unique.end(), cpus + offsets[n], cpus + offsets[n + 1]);
    }
    std::sort(unique.begin(), unique.end());
    if (unique.front() < 0 ||
        std::adjacent_find(unique.begin(), unique.end()) != unique.end())
      throw std::invalid_argument("Invalid or duplicate CPU in NUMA topology");
    plan.balance_rows(moe->intermediate);
    plan.load_team = plan.team(moe->load_threads);
    plan.compute_team = plan.team(moe->num_threads);
    if (plan.enabled()) {
      // Large pages can straddle several row owners. Keep first-touch granular.
      const auto page = uintptr_t(sysconf(_SC_PAGESIZE));
      const auto base = reinterpret_cast<uintptr_t>(
          ggml_backend_buffer_get_base(moe->weights_buffer));
      const auto first = (base + page - 1) & ~(page - 1);
      const auto last =
          (base + ggml_backend_buffer_get_size(moe->weights_buffer)) &
          ~(page - 1);
      if (last > first)
        madvise(reinterpret_cast<void*>(first), last - first, MADV_NOHUGEPAGE);
      dsv41::LocalMemoryPolicy probe(true);
      plan.local_policy = probe.changed;
      return probe.error;
    }
    return 0;
#else
    throw std::invalid_argument("NUMA sharding requires the IK CPU backend");
#endif
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

static int query_numa_pages(void* handle, int expert, uint64_t* counts) {
#ifdef DSV41_IK
  try {
    if (!handle || !counts) return EINVAL;
    auto* moe = static_cast<MoE*>(handle);
    if (expert < -1 || expert >= moe->experts) return EINVAL;
    const auto& plan = moe->numa;
    std::fill_n(counts, 4, 0);
    if (!plan.enabled()) return 0;
    std::vector<void*> pages;
    std::vector<int> expected;
    const size_t page_size = sysconf(_SC_PAGESIZE);
    for (const auto* weight : moe->weights) {
      const int tiles = (weight->ne[1] + plan.tile_rows - 1) / plan.tile_rows;
      const int first = expert < 0 ? 0 : expert;
      const int end_expert = expert < 0 ? moe->experts : expert + 1;
      const int stride = expert < 0 ? std::max(1, moe->experts / 8) : 1;
      for (int e = first; e < end_expert; e += stride) {
        const auto base =
            reinterpret_cast<uintptr_t>(weight->data) + e * weight->nb[2];
        for (int n = 0; n < int(plan.nodes.size()); ++n) {
          const auto begin =
              base + plan.boundary(tiles, n) * plan.tile_rows * weight->nb[1];
          const auto end =
              base + std::min(int(weight->ne[1]),
                              plan.boundary(tiles, n + 1) * plan.tile_rows) *
                         weight->nb[1];
          if (end <= begin) continue;
          const auto address =
              ((begin + (end - begin) / 2) / page_size) * page_size;
          if (address >= begin && address + page_size <= end) {
            pages.push_back(reinterpret_cast<void*>(address));
            expected.push_back(plan.nodes[n].id);
          }
        }
      }
    }
    if (pages.empty()) return 0;
    std::vector<int> status(pages.size());
    // Query only: do not migrate or replicate resident weights.
    if (syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr,
                status.data(), 0) < 0)
      return errno;
    counts[0] = pages.size();
    for (size_t i = 0; i < pages.size(); ++i)
      ++counts[status[i] < 0 ? 3 : (status[i] == expected[i] ? 1 : 2)];
    return 0;
  } catch (const std::bad_alloc&) {
    return ENOMEM;
  }
#else
  return ENOSYS;
#endif
}

int dsv41_moe_numa_pages(void* handle, uint64_t* counts) {
  return query_numa_pages(handle, -1, counts);
}

int dsv41_moe_numa_expert_pages(void* handle, int expert, uint64_t* counts) {
  if (expert < 0) return EINVAL;
  return query_numa_pages(handle, expert, counts);
}

int dsv41_moe_set_threads(void* handle, int threads) {
  try {
    if (!handle || threads <= 0) {
      throw std::invalid_argument("Invalid MoE handle or thread count");
    }
    auto* moe = static_cast<MoE*>(handle);
#ifdef DSV41_IK
    if (!moe->numa.nodes.empty())
      moe->numa.compute_team = moe->numa.team(threads);
#endif
    ggml_backend_cpu_set_n_threads(moe->backend, threads);
    moe->num_threads = threads;
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

int dsv41_moe_set_execution_mode(void* handle, int compact) {
#ifdef DSV41_IK
  if (handle && (compact == 0 || compact == 1)) {
    static_cast<MoE*>(handle)->use_compact = compact;
    return 0;
  }
#endif
  last_error = "Execution mode requires an IK backend and mode 0 or 1";
  return -1;
}

int dsv41_moe_set_schedule(void* handle, int schedule) {
#ifdef DSV41_IK
  if (handle && schedule >= -1 && schedule <= 31) {
    static_cast<MoE*>(handle)->compact.schedule = schedule;
    return 0;
  }
#endif
  last_error = "Invalid CPU schedule";
  return -1;
}

int dsv41_moe_set_profile(void* handle, int interval) {
#ifdef DSV41_IK
  if (handle && interval >= 0) {
    static_cast<MoE*>(handle)->compact.profile.reset(interval);
    return 0;
  }
#endif
  last_error = "Profiling requires IK and a nonnegative sampling interval";
  return -1;
}

int dsv41_moe_get_profile(void* handle, uint64_t* values, int capacity) {
#ifdef DSV41_IK
  if (handle && values && capacity >= 64) {
    const auto& totals = static_cast<MoE*>(handle)->compact.profile.totals;
    std::copy(totals.begin(), totals.end(), values);
    return 0;
  }
#endif
  last_error = "Invalid CPU profile buffer";
  return -1;
}

int dsv41_moe_dump_trace(void* handle, const char* path) {
  try {
#ifdef DSV41_IK
    if (handle && path) {
      static_cast<MoE*>(handle)->compact.profile.dump(path);
      return 0;
    }
#endif
    throw std::invalid_argument("Invalid CPU trace destination");
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

int dsv41_moe_load(void* handle, int expert, int projection,
                   const uint8_t* packed, const uint8_t* scales) {
  try {
    if (!handle) throw std::invalid_argument("Null MoE handle");
    static_cast<MoE*>(handle)->load(expert, projection, packed, scales);
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

int dsv41_moe_forward(void* handle, int tokens, const float* input,
                      const int32_t* ids, const float* routes, float* output) {
  try {
    if (!handle) throw std::invalid_argument("Null MoE handle");
    static_cast<MoE*>(handle)->forward(tokens, input, ids, routes, output);
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}
}
