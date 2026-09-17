// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <utility>

#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <omp.h>

namespace dsv41 {

// Restore the caller's policy: OpenMP workers can also serve unrelated work.
struct LocalMemoryPolicy {
  int mode = 0, error = 0;
  bool changed = false;
  std::vector<unsigned long> mask;

  explicit LocalMemoryPolicy(bool enable) {
    if (!enable) return;
    mask.resize(16);
    while (syscall(SYS_get_mempolicy, &mode, mask.data(), mask.size() * 64 + 1,
                   nullptr, 0) != 0) {
      if (errno != EINVAL || mask.size() >= 1024) {
        error = errno;
        return;
      }
      mask.resize(mask.size() * 2);
    }
    // Only replace the default policy, not an explicit numactl policy.
    if (mode != MPOL_DEFAULT) return;
    changed = syscall(SYS_set_mempolicy, MPOL_LOCAL, nullptr, 0) == 0;
    if (!changed) error = errno;
  }

  ~LocalMemoryPolicy() {
    if (changed) {
      const bool empty = (mode & ~MPOL_MODE_FLAGS) == MPOL_DEFAULT ||
                         (mode & ~MPOL_MODE_FLAGS) == MPOL_LOCAL;
      syscall(SYS_set_mempolicy, mode, empty ? nullptr : mask.data(),
              empty ? 0 : mask.size() * 64 + 1);
    }
  }
};

struct NumaPlan {
  static constexpr int tile_rows = 128;
  struct Node {
    int id, first_core;
    std::vector<int> cpus;
  };
  struct Slot {
    int node, cpu, rank, workers;
  };
  std::vector<Node> nodes;
  std::vector<Slot> load_team, compute_team;
  int cores = 0;
  bool local_policy = false;
  mutable std::atomic<int> affinity_error{0};

  bool enabled() const { return nodes.size() > 1; }

  int boundary(int tiles, int node) const {
    const int prefix =
        node == int(nodes.size()) ? cores : nodes[node].first_core;
    return int(int64_t(tiles) * prefix / cores);
  }

  std::vector<Slot> team(int threads) const {
    if (threads > cores)
      throw std::invalid_argument(
          "CPU expert threads exceed available physical cores");
    // At least one slot per node. A smaller OpenMP team visits several nodes.
    std::vector<int> counts(nodes.size(), 1);
    for (int i = nodes.size(); i < threads; ++i) {
      int best = -1;
      for (int n = 0; n < int(nodes.size()); ++n) {
        if (counts[n] >= int(nodes[n].cpus.size())) continue;
        if (best < 0 || int64_t(counts[n]) * nodes[best].cpus.size() <
                            int64_t(counts[best]) * nodes[n].cpus.size())
          best = n;
      }
      ++counts[best];
    }
    std::vector<Slot> result;
    for (int n = 0; n < int(nodes.size()); ++n)
      for (int i = 0; i < counts[n]; ++i)
        result.push_back({n, nodes[n].cpus[i], i, counts[n]});
    return result;
  }

  struct Binding {
    const NumaPlan& plan;
    std::vector<unsigned long> previous, selected;
    bool changed = false;
    int current = -1;

    explicit Binding(const NumaPlan& p, bool worker = false) : plan(p) {
      if (!plan.enabled()) return;
      previous.resize(16);
      while (sched_getaffinity(0, previous.size() * sizeof(unsigned long),
                               reinterpret_cast<cpu_set_t*>(previous.data()))) {
        if (errno != EINVAL || previous.size() >= 16384) {
          plan.affinity_error.store(errno);
          previous.clear();
          return;
        }
        previous.resize(previous.size() * 2);
      }
      selected.resize(previous.size());
      // Reused OpenMP workers may have inherited a GPU-local mask. Release
      // them to the configured CPU set before the implicit join. The outer
      // guard separately restores the calling thread's original mask.
      if (worker && select_team()) previous = selected;
    }

    // Keep the caller on its current core when that node owns a work slot.
    std::pair<int, int> local_slot(const std::vector<Slot>& team) const {
      const int current_cpu = sched_getcpu();
      for (int s = 0; s < int(team.size()); ++s)
        if (team[s].cpu == current_cpu) return {s, current_cpu};
      for (int s = 0; s < int(team.size()); ++s) {
        const auto& cpus = plan.nodes[team[s].node].cpus;
        if (std::find(cpus.begin(), cpus.end(), current_cpu) != cpus.end())
          return {s, current_cpu};
      }
      for (int s = 0; s < int(team.size()); ++s) {
        const int cpu = team[s].cpu;
        if (cpu >= 0 && size_t(cpu / 64) < previous.size() &&
            (previous[cpu / 64] & (1UL << (cpu % 64))))
          return {s, cpu};
      }
      return {0, team.empty() ? -1 : team[0].cpu};
    }

    void bind(int cpu) {
      if (cpu == current || previous.empty()) return;
      std::fill(selected.begin(), selected.end(), 0UL);
      if (cpu < 0 || size_t(cpu / 64) >= selected.size()) {
        plan.affinity_error.store(EINVAL);
        return;
      }
      selected[cpu / 64] |= 1UL << (cpu % 64);
      if (sched_setaffinity(0, selected.size() * sizeof(unsigned long),
                            reinterpret_cast<cpu_set_t*>(selected.data()))) {
        plan.affinity_error.store(errno);
      } else {
        changed = true;
        current = cpu;
      }
    }

    bool select_team() {
      if (previous.empty()) return false;
      std::fill(selected.begin(), selected.end(), 0UL);
      for (const auto& node : plan.nodes) {
        for (int cpu : node.cpus) {
          if (cpu < 0 || size_t(cpu / 64) >= selected.size()) {
            plan.affinity_error.store(EINVAL);
            return false;
          }
          selected[cpu / 64] |= 1UL << (cpu % 64);
        }
      }
      return true;
    }

    void bind_team() {
      if (!select_team()) return;
      if (selected == previous) return;
      if (sched_setaffinity(0, selected.size() * sizeof(unsigned long),
                            reinterpret_cast<cpu_set_t*>(selected.data())))
        plan.affinity_error.store(errno);
      else
        changed = true;
    }

    ~Binding() {
      if (changed &&
          sched_setaffinity(0, previous.size() * sizeof(unsigned long),
                            reinterpret_cast<cpu_set_t*>(previous.data())))
        plan.affinity_error.store(errno);
    }
  };

  static Slot slot(const std::vector<Slot>& team, int index, int caller_slot,
                   int caller_cpu) {
    auto result = team[index == 0             ? caller_slot
                       : index == caller_slot ? 0
                                              : index];
    if (index == 0) result.cpu = caller_cpu;
    return result;
  }

  void check_affinity() const {
    if (int error = affinity_error.exchange(0))
      throw std::runtime_error(
          std::string("CPU expert NUMA affinity failed: ") +
          std::strerror(error));
  }

  // First-touch and execution use identical, thread-count-independent row
  // ranges.
  template <typename Work>
  void weight_rows(int rows, int threads, bool loading,
                   const Work& work) const {
    // New OpenMP workers inherit this mask, not a GPU-local callback mask.
    Binding caller(*this);
    const auto [caller_slot, caller_cpu] = caller.local_slot(load_team);
    caller.bind_team();
    check_affinity();
#pragma omp parallel num_threads(threads)
    {
      Binding binding(*this, true);
      LocalMemoryPolicy memory(loading && local_policy);
      const int tiles = (rows + tile_rows - 1) / tile_rows;
      for (int s = omp_get_thread_num(); s < int(load_team.size());
           s += omp_get_num_threads()) {
        const auto& slot =
            NumaPlan::slot(load_team, s, caller_slot, caller_cpu);
        binding.bind(slot.cpu);
        const int end = boundary(tiles, slot.node + 1);
        for (int tile = boundary(tiles, slot.node) + slot.rank; tile < end;
             tile += slot.workers)
          for (int row = tile * tile_rows;
               row < std::min((tile + 1) * tile_rows, rows); row += 8)
            work(row);
      }
    }
    check_affinity();
  }
};
}  // namespace dsv41
