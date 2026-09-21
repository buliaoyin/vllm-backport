// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t page_size = 4096;
constexpr size_t batch_pages = 1024;
constexpr size_t queue_depth = 128;
using Clock = std::chrono::steady_clock;

uint64_t ns(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                              start)
      .count();
}

struct Source {
  int fd = -1;
  uint64_t offset = 0, size = 0;
  ~Source() {
    if (fd >= 0) close(fd);
  }
  void open_file(const char* path, uint64_t base, uint64_t bytes) {
    fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error("Engram open(O_DIRECT): " +
                               std::string(std::strerror(errno)));
    struct stat info{};
    if (fstat(fd, &info) || !S_ISREG(info.st_mode))
      throw std::runtime_error("Engram SSD requires a regular file");
    size = info.st_size;
    offset = base;
    if (base > size || bytes > size - base)
      throw std::runtime_error("Engram tensor extends beyond checkpoint file");
  }
};

struct Piece {
  uint64_t page;
  uint32_t output;
  uint16_t begin, length;
};

struct CacheEntry {
  uint64_t age = 0;
  uint32_t id = UINT32_MAX;
};

struct Reader {
  Source weights, scales;
  aio_context_t context = 0;
  void* pages = nullptr;
  int device = 0, dim, scale_dim, max_rows, vocab_start, vocab_end;
  const int32_t* ids;
  uint8_t* output;
  int32_t* status;
  std::atomic<bool> failed{false};
  char error[512]{};
  std::vector<Piece> pieces;
  std::vector<int> misses;
  std::vector<CacheEntry> cache;
  std::vector<uint8_t> cached_rows;
  size_t cache_sets = 0;
  uint64_t clock = 0;
  std::array<iocb, batch_pages> requests{};
  std::array<iocb*, batch_pages> request_ptrs{};
  std::array<uint64_t, batch_pages> offsets{};
  std::array<io_event, queue_depth> completions{};
  // calls, rows, valid rows, weight pages, scale pages, read bytes, read ops,
  // planning ns, IO ns, total callback ns, cached rows.
  std::array<uint64_t, 11> stats{};

  Reader(int dim_, int scale_dim_, int max_rows_, int start, int end,
         uint64_t cache_bytes, const int32_t* ids_, uint8_t* output_,
         int32_t* status_)
      : dim(dim_),
        scale_dim(scale_dim_),
        max_rows(max_rows_),
        vocab_start(start),
        vocab_end(end),
        ids(ids_),
        output(output_),
        status(status_) {
    if (dim <= 0 || dim > page_size || scale_dim <= 0 ||
        scale_dim > page_size || max_rows <= 0 || start < 0 || end <= start ||
        uint64_t(max_rows) * (dim + scale_dim) > UINT32_MAX)
      throw std::runtime_error("Invalid Engram SSD workspace geometry");
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("Cannot determine Engram SSD CUDA device");
    pieces.reserve(size_t(max_rows) * 2);
    misses.reserve(max_rows);
    cache_sets = cache_bytes / (4 * (dim + scale_dim + sizeof(CacheEntry)));
    cache.resize(cache_sets * 4);
    cached_rows.resize(cache.size() * (dim + scale_dim));
    if (posix_memalign(&pages, page_size, batch_pages * page_size))
      throw std::bad_alloc();
    if (syscall(SYS_io_setup, queue_depth, &context)) {
      free(pages);
      pages = nullptr;
      throw std::runtime_error("Engram io_setup: " +
                               std::string(std::strerror(errno)));
    }
  }
  ~Reader() {
    if (context) syscall(SYS_io_destroy, context);
    free(pages);
  }

  void fail(const char* message) {
    if (!failed.load(std::memory_order_relaxed)) {
      std::snprintf(error, sizeof(error), "%s", message);
      std::fprintf(stderr, "Engram SSD read failed: %s\n", error);
      failed.store(true, std::memory_order_release);
    }
    *status = 1;
  }

  size_t cache_set(uint32_t id) const {
    uint64_t hash = uint64_t(id) * 0x9e3779b97f4a7c15ULL;
    return ((hash >> 32) ^ hash) % cache_sets * 4;
  }

  void fill_cached(int rows) {
    misses.clear();
    const size_t width = dim + scale_dim;
    for (int row = 0; row < rows; ++row) {
      const int id = ids[row];
      if (id < vocab_start || id >= vocab_end) continue;
      ++stats[2];
      bool hit = false;
      if (cache_sets) {
        size_t set = cache_set(id);
        for (size_t slot = set; slot < set + 4; ++slot) {
          if (cache[slot].id != uint32_t(id)) continue;
          cache[slot].age = ++clock;
          std::memcpy(output + row * width, cached_rows.data() + slot * width,
                      width);
          ++stats[10];
          hit = true;
          break;
        }
      }
      if (!hit) misses.push_back(row);
    }
  }

  void update_cache() {
    if (!cache_sets) return;
    const size_t width = dim + scale_dim;
    for (int row : misses) {
      const uint32_t id = ids[row];
      size_t set = cache_set(id), victim = set;
      for (size_t slot = set; slot < set + 4; ++slot) {
        if (cache[slot].id == id) {
          victim = slot;
          break;
        }
        if (cache[slot].age < cache[victim].age) victim = slot;
      }
      cache[victim] = {++clock, id};
      std::memcpy(cached_rows.data() + victim * width, output + row * width,
                  width);
    }
  }

  void read_batch(Source& source, size_t count) {
    size_t submitted = 0, active = 0;
    std::string failure;
    while ((submitted < count && failure.empty()) || active) {
      if (submitted < count && failure.empty() && active < queue_depth) {
        size_t amount = std::min(count - submitted, queue_depth - active);
        long result = syscall(SYS_io_submit, context, amount,
                              request_ptrs.data() + submitted);
        if (result > 0) {
          submitted += result;
          active += result;
        } else if (result < 0 && errno == EINTR) {
          continue;
        } else if (result < 0 && errno == EAGAIN && active) {
          // Drain outstanding requests before retrying submission.
        } else {
          failure = "Engram io_submit: " + std::string(std::strerror(errno));
        }
      }
      if (!active) continue;
      long result = syscall(SYS_io_getevents, context, 1, active,
                            completions.data(), nullptr);
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) {
        // io_destroy drains the kernel before this staging buffer is released.
        syscall(SYS_io_destroy, context);
        context = 0;
        throw std::runtime_error("Engram io_getevents failed");
      }
      active -= result;
      for (int i = 0; i < result; ++i) {
        const auto& event = completions[i];
        const auto& req = requests[event.data];
        const uint64_t expected =
            std::min<uint64_t>(req.aio_nbytes, source.size - req.aio_offset);
        if (event.res < 0 || uint64_t(event.res) != expected || event.res2) {
          failure = "Engram direct read failed or checkpoint was truncated (" +
                    std::to_string(event.res) + ")";
        } else {
          stats[5] += event.res;
          ++stats[6];
        }
      }
    }
    if (!failure.empty()) throw std::runtime_error(failure);
  }

  void gather(Source& source, int width, int output_col, int stat) {
    auto start = Clock::now();
    pieces.clear();
    for (int row : misses) {
      const int64_t id = ids[row];
      uint64_t offset = source.offset + (id - vocab_start) * width;
      uint32_t dest = uint64_t(row) * (dim + scale_dim) + output_col;
      int remaining = width;
      while (remaining) {
        const int begin = offset % page_size;
        const int length = std::min<int>(remaining, page_size - begin);
        pieces.push_back(
            {offset / page_size, dest, uint16_t(begin), uint16_t(length)});
        dest += length;
        offset += length;
        remaining -= length;
      }
    }
    std::sort(pieces.begin(), pieces.end(),
              [](const Piece& a, const Piece& b) { return a.page < b.page; });
    stats[7] += ns(start);
    size_t begin = 0;
    while (begin < pieces.size()) {
      size_t end = begin, count = 0;
      while (end < pieces.size() && count < batch_pages) {
        uint64_t page = pieces[end].page;
        offsets[count++] = page;
        while (end < pieces.size() && pieces[end].page == page) ++end;
      }
      stats[stat] += count;
      size_t req_count = 0;
      for (size_t first = 0; first < count;) {
        size_t last = first + 1;
        while (last < count && offsets[last] == offsets[last - 1] + 1 &&
               last - first < 32)
          ++last;
        auto& req = requests[req_count];
        req = {};
        req.aio_data = req_count;
        req.aio_lio_opcode = IOCB_CMD_PREAD;
        req.aio_fildes = source.fd;
        req.aio_buf = reinterpret_cast<uint64_t>(pages) + first * page_size;
        req.aio_nbytes = (last - first) * page_size;
        req.aio_offset = offsets[first] * page_size;
        request_ptrs[req_count++] = &req;
        first = last;
      }
      start = Clock::now();
      read_batch(source, req_count);
      stats[8] += ns(start);
      size_t page_index = 0;
      for (size_t i = begin; i < end; ++i) {
        const auto& piece = pieces[i];
        while (offsets[page_index] < piece.page) ++page_index;
        std::memcpy(
            output + piece.output,
            static_cast<uint8_t*>(pages) + page_index * page_size + piece.begin,
            piece.length);
      }
      begin = end;
    }
  }

  void run(int rows) {
    auto start = Clock::now();
    std::memset(output, 0, size_t(rows) * (dim + scale_dim));
    if (failed.load(std::memory_order_acquire)) {
      *status = 1;
      return;
    }
    *status = 0;
    try {
      fill_cached(rows);
      gather(weights, dim, 0, 3);
      gather(scales, scale_dim, dim, 4);
      update_cache();
      ++stats[0];
      stats[1] += rows;
      stats[9] += ns(start);
    } catch (const std::exception& e) {
      fail(e.what());
    } catch (...) {
      fail("Unknown Engram SSD reader error");
    }
  }
};

struct Task {
  Reader* reader;
  int rows;
};
void CUDART_CB run_task(void* ptr) {
  auto* task = static_cast<Task*>(ptr);
  task->reader->run(task->rows);
}
thread_local std::string last_error;
}  // namespace

extern "C" {
int dsv41_engram_ssd_version() { return 1; }
void* dsv41_engram_ssd_create(const char* weights, uint64_t weight_offset,
                              const char* scales, uint64_t scale_offset,
                              int dim, int scale_dim, int max_rows, int start,
                              int end, uint64_t cache_bytes, const int32_t* ids,
                              uint8_t* output, int32_t* status) {
  try {
    auto reader = std::make_unique<Reader>(dim, scale_dim, max_rows, start, end,
                                           cache_bytes, ids, output, status);
    reader->weights.open_file(weights, weight_offset,
                              uint64_t(end - start) * dim);
    reader->scales.open_file(scales, scale_offset,
                             uint64_t(end - start) * scale_dim);
    return reader.release();
  } catch (const std::exception& e) {
    last_error = e.what();
    return nullptr;
  }
}
const char* dsv41_engram_ssd_error(void* handle) {
  if (!handle) return last_error.c_str();
  auto* reader = static_cast<Reader*>(handle);
  return reader->failed.load(std::memory_order_acquire) ? reader->error
                                                        : nullptr;
}
void* dsv41_engram_ssd_task(void* handle, int rows) {
  auto* reader = static_cast<Reader*>(handle);
  if (rows < 0 || rows > reader->max_rows) return nullptr;
  return new Task{reader, rows};
}
int dsv41_engram_ssd_enqueue(void* task, void* stream) {
  return cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), run_task, task);
}
// Caller synchronizes the owning CUDA stream before reading counters.
void dsv41_engram_ssd_stats(void* handle, uint64_t* result) {
  const auto& stats = static_cast<Reader*>(handle)->stats;
  std::copy(stats.begin(), stats.end(), result);
}
// Caller must finish all CUDA work before invalidating the bounded row cache.
void dsv41_engram_ssd_clear_cache(void* handle) {
  auto& reader = *static_cast<Reader*>(handle);
  std::fill(reader.cache.begin(), reader.cache.end(), CacheEntry{});
  reader.clock = 0;
}
void dsv41_engram_ssd_destroy(void* handle, void** tasks, int count) {
  auto* reader = static_cast<Reader*>(handle);
  if (!reader) return;
  int previous = reader->device;
  cudaGetDevice(&previous);
  cudaSetDevice(reader->device);
  cudaDeviceSynchronize();
  cudaSetDevice(previous);
  for (int i = 0; i < count; ++i) delete static_cast<Task*>(tasks[i]);
  delete reader;
}
}
