// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime_api.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>

namespace {
using Forward = int (*)(void*, int, const float*, const int32_t*, const float*,
                        float*);
using Error = const char* (*)();

struct Task {
  void* moe;
  Forward forward;
  Error error;
  int tokens, hidden, topk;
  const float* input;
  const int32_t* ids;
  const float* routes;
  float* output;
  std::atomic<int> failed{0};
  char message[512]{};
  int device = 0;
  // Calls, tokens, native nanoseconds, out-of-range histogram IDs, 384 bins.
  std::array<uint64_t, 388> stats{};
  std::array<uint64_t, 4> thread_info{};
  uint64_t cached_routes = 0;

  void fail(const char* message_) {
    std::snprintf(message, sizeof(message), "%s", message_);
    failed.store(1, std::memory_order_release);
  }
};

// CUDA invokes this outside Python. All called functions are CPU-only: CUDA
// APIs are forbidden in a host callback. Stream ordering makes the preceding
// D2H copies visible here and delays the following H2D copy until completion.
void CUDART_CB run(void* ptr) {
  auto& task = *static_cast<Task*>(ptr);
  std::fill_n(task.output, size_t(task.tokens) * task.hidden, 0.0f);
  try {
    int actual = 0;
    while (actual < task.tokens && task.ids[actual * task.topk] != -1) ++actual;
    for (int t = actual; t < task.tokens; ++t) {
      if (task.ids[t * task.topk] != -1) {
        task.fail("CPU graph expects valid tokens followed by padding");
        break;
      }
    }
    if (!task.failed.load(std::memory_order_acquire) && actual) {
      const auto tid = uint64_t(syscall(SYS_gettid));
      if (!task.thread_info[0]) {
        task.thread_info[0] = tid;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
          task.thread_info[3] = CPU_COUNT(&mask);
        }
      } else if (task.thread_info[1] != tid) {
        ++task.thread_info[2];
      }
      task.thread_info[1] = tid;
      const auto start = std::chrono::steady_clock::now();
      const int status = task.forward(task.moe, actual, task.input, task.ids,
                                      task.routes, task.output);
      task.stats[0] += 1;
      task.stats[1] += actual;
      task.stats[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
      if (status) {
        task.fail(task.error());
      } else {
        for (int i = 0; i < actual * task.topk; ++i) {
          const int expert = task.ids[i];
          if (expert >= 0 && expert < 384)
            ++task.stats[4 + expert];
          else if (expert == -2)
            ++task.cached_routes;
          else
            ++task.stats[3];
        }
      }
    }
  } catch (const std::exception& error) {
    task.fail(error.what());
  } catch (...) {
    task.fail("Unknown failure in CPU MoE CUDA host callback");
  }
  if (task.failed.load(std::memory_order_acquire)) {
    std::fill_n(task.output, size_t(task.tokens) * task.hidden,
                std::numeric_limits<float>::quiet_NaN());
  }
}
}  // namespace

extern "C" {
void* dsv41_cuda_task_create(void* moe, void* forward, void* error, int tokens,
                             int hidden, int topk, const float* input,
                             const int32_t* ids, const float* routes,
                             float* output) {
  auto task = std::make_unique<Task>();
  task->moe = moe;
  task->forward = reinterpret_cast<Forward>(forward);
  task->error = reinterpret_cast<Error>(error);
  task->tokens = tokens;
  task->hidden = hidden;
  task->topk = topk;
  task->input = input;
  task->ids = ids;
  task->routes = routes;
  task->output = output;
  if (cudaGetDevice(&task->device) != cudaSuccess) return nullptr;
  return task.release();
}

int dsv41_cuda_task_enqueue(void* handle, void* stream) {
  return cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), run, handle);
}

const char* dsv41_cuda_task_error(void* handle) {
  auto& task = *static_cast<Task*>(handle);
  return task.failed.load(std::memory_order_acquire) ? task.message : nullptr;
}

// The owner must finish its CUDA work before reading these serial-use counters.
void dsv41_cuda_task_stats(void* handle, uint64_t* output) {
  auto& task = *static_cast<Task*>(handle);
  std::copy(task.stats.begin(), task.stats.end(), output);
}

uint64_t dsv41_cuda_task_cached_routes(void* handle) {
  return static_cast<Task*>(handle)->cached_routes;
}

// Like task_stats, read only after the owning stream has completed its work.
void dsv41_cuda_tasks_route_counts(void* const* handles, int count,
                                   uint64_t* output) {
  output[0] = output[1] = 0;
  for (int i = 0; i < count; ++i) {
    const auto& task = *static_cast<Task*>(handles[i]);
    output[0] += task.cached_routes;
    for (size_t expert = 4; expert < task.stats.size(); ++expert) {
      output[1] += task.stats[expert];
    }
  }
}

void dsv41_cuda_task_thread_info(void* handle, uint64_t* output) {
  auto& task = *static_cast<Task*>(handle);
  std::copy(task.thread_info.begin(), task.thread_info.end(), output);
}

void dsv41_cuda_task_destroy(void* handle) {
  auto* task = static_cast<Task*>(handle);
  if (!task) return;
  int previous = task->device;
  cudaGetDevice(&previous);
  cudaSetDevice(task->device);
  // A captured host node can replay on a different stream than capture used.
  cudaDeviceSynchronize();
  cudaSetDevice(previous);
  delete task;
}
}
