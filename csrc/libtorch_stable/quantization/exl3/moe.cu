// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"
#include "moe_nobar.cuh"

#include <algorithm>
#include <map>
#include <mutex>
#include <type_traits>

namespace {
using torch::headeronly::ScalarType;
using torch::stable::Tensor;
constexpr int kThreads = 512;
constexpr int kSharedBytes = 90 * 1024;
constexpr int kLockInts = MOE_SCHED_OFFSET + MOE_SCHED_INTS;
template <int FETCH_K_FACTOR>
constexpr auto kernel =
    exl3_moe_nobar_kernel<4, 256, 2, 2, 32, 32, true, FETCH_K_FACTOR>;

void check_cuda(cudaError_t error) {
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

template <int FETCH_K_FACTOR>
int resident_groups(int device) {
  // Occupancy and function attributes belong to each kernel specialization.
  static std::mutex mutex;
  static std::map<int, int> cache;
  const std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find(device);
  if (it != cache.end()) return it->second;
  auto* props = get_device_prop();
  STD_TORCH_CHECK(props->major == 8 && props->minor == 0,
                  "EXL3 M32 requires SM80");
  check_cuda(cudaFuncSetAttribute(kernel<FETCH_K_FACTOR>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kSharedBytes));
  int active;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel<FETCH_K_FACTOR>, kThreads, kSharedBytes));
  const int groups = std::min(
      MOE_MAX_GROUPS, props->multiProcessorCount * active / MOE_SMS_PER_EXPERT);
  STD_TORCH_CHECK(groups > 0, "EXL3 M32 requires a resident expert group");
  cache.emplace(device, groups);
  return groups;
}

template <bool DECODE>
void moe_m32(const Tensor& hidden, const Tensor& output, const Tensor& counts,
             const Tensor& tokens, const Tensor& routing, const Tensor& state_g,
             const Tensor& state_u, const Tensor& intermediate_g,
             const Tensor& intermediate_u, const Tensor& gate_q,
             const Tensor& gate_su, const Tensor& gate_sv, const Tensor& up_q,
             const Tensor& up_su, const Tensor& up_sv, const Tensor& down_q,
             const Tensor& down_su, const Tensor& down_sv, const Tensor& locks,
             double limit) {
  STD_TORCH_CHECK(hidden.is_cuda(), "EXL3 M32 requires CUDA tensors");
  const auto device = hidden.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto check = [&](const Tensor& tensor, ScalarType dtype) {
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.get_device_index() == device &&
                        tensor.is_contiguous() && tensor.scalar_type() == dtype,
                    "EXL3 M32 tensor device, dtype or layout mismatch");
  };
  check(hidden, ScalarType::Half);
  check(output, ScalarType::Float);
  check(counts, ScalarType::Long);
  check(tokens, ScalarType::Long);
  check(routing, ScalarType::Half);
  check(locks, ScalarType::Int);
  STD_TORCH_CHECK(hidden.dim() == 2 && output.dim() == 2 &&
                      hidden.size(0) == output.size(0) &&
                      hidden.size(1) == output.size(1),
                  "EXL3 M32 output shape mismatch");
  STD_TORCH_CHECK(intermediate_g.dim() == 3, "EXL3 M32 requires 3D workspace");
  const int rows = hidden.size(0);
  const int hidden_dim = hidden.size(1);
  const int intermediate_dim = intermediate_g.size(2);
  STD_TORCH_CHECK(
      hidden_dim >= 256 && hidden_dim <= 8192 && intermediate_dim >= 256 &&
          intermediate_dim <= 8192 && hidden_dim % 256 == 0 &&
          intermediate_dim % 256 == 0,
      "EXL3 M32 dimensions must be multiples of 256 in [256, 8192]");
  STD_TORCH_CHECK(counts.dim() == 1 && counts.numel() >= 2 &&
                      tokens.dim() == 1 && routing.dim() == 1 &&
                      tokens.numel() == routing.numel(),
                  "EXL3 M32 routing shape mismatch");
  if (rows == 0) return;
  const int experts = counts.numel() - 1;
  STD_TORCH_CHECK(tokens.numel() % rows == 0 && tokens.numel() / rows > 0 &&
                      tokens.numel() / rows <= experts,
                  "EXL3 M32 invalid top-k size");
  for (auto* ptr : {&gate_q, &gate_su, &gate_sv, &up_q, &up_su, &up_sv, &down_q,
                    &down_su, &down_sv}) {
    check(*ptr, ScalarType::Long);
    STD_TORCH_CHECK(ptr->dim() == 1 && ptr->numel() == experts,
                    "EXL3 M32 expert pointer table size mismatch");
  }
  STD_TORCH_CHECK(state_g.dim() == 3, "EXL3 M32 requires 3D workspace");
  const int capacity = state_g.size(1);
  const int concurrency = state_g.size(0);
  // Eight N256 tiles for gate/up and sixteen for down give each of the
  // eight CTAs complete K columns, all divisible by 64. Keep other shapes
  // on the original kernel, including its partial-K reduction boundaries.
  const bool packed_k64 =
      DECODE && hidden_dim == 4096 && intermediate_dim == 2048;
  const int resident =
      packed_k64 ? resident_groups<2>(device) : resident_groups<1>(device);
  const int groups = std::min(concurrency, resident);
  STD_TORCH_CHECK(groups > 0 && capacity >= rows && locks.numel() >= kLockInts,
                  "EXL3 M32 workspace is too small");
  for (auto* ws : {&state_g, &state_u, &intermediate_g, &intermediate_u}) {
    check(*ws, ScalarType::Half);
    const int width =
        ws == &state_g || ws == &state_u ? hidden_dim : intermediate_dim;
    STD_TORCH_CHECK(ws->dim() == 3 && ws->size(0) == concurrency &&
                        ws->size(1) == capacity && ws->size(2) == width,
                    "EXL3 M32 workspace shape mismatch");
  }
  const auto stream = get_current_cuda_stream(device);
  auto launch = [&](auto fetch_k_factor) {
    constexpr int factor = decltype(fetch_k_factor)::value;
    exl3_moe_nobar_kernel<4, 256, 2, 2, 32, 32, true, factor>
        <<<dim3(MOE_SMS_PER_EXPERT, 1, groups), kThreads, kSharedBytes,
           stream>>>(static_cast<const half*>(hidden.data_ptr()),
                     static_cast<half*>(state_g.data_ptr()),
                     static_cast<half*>(state_u.data_ptr()),
                     static_cast<half*>(intermediate_g.data_ptr()),
                     static_cast<half*>(intermediate_u.data_ptr()),
                     static_cast<float*>(output.data_ptr()),
                     reinterpret_cast<const uint16_t**>(gate_q.data_ptr()),
                     reinterpret_cast<const half**>(gate_su.data_ptr()),
                     reinterpret_cast<const half**>(gate_sv.data_ptr()),
                     reinterpret_cast<const uint16_t**>(up_q.data_ptr()),
                     reinterpret_cast<const half**>(up_su.data_ptr()),
                     reinterpret_cast<const half**>(up_sv.data_ptr()),
                     reinterpret_cast<const uint16_t**>(down_q.data_ptr()),
                     reinterpret_cast<const half**>(down_su.data_ptr()),
                     reinterpret_cast<const half**>(down_sv.data_ptr()),
                     static_cast<const int64_t*>(counts.data_ptr()),
                     static_cast<const int64_t*>(tokens.data_ptr()),
                     static_cast<const half*>(routing.data_ptr()), hidden_dim,
                     intermediate_dim, experts, tokens.numel() / rows, capacity,
                     groups, static_cast<float>(limit), MOE_ACT_SILU, 4, 4, 4,
                     static_cast<int*>(locks.data_ptr()));
  };
  if (packed_k64)
    launch(std::integral_constant<int, 2>{});
  else
    launch(std::integral_constant<int, 1>{});
  check_cuda(cudaGetLastError());
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_exl3_C, m) {
  m.def(
      "moe_m32(Tensor hidden, Tensor! output, Tensor counts, Tensor tokens, "
      "Tensor routing, Tensor! state_g, Tensor! state_u, Tensor! "
      "intermediate_g, "
      "Tensor! intermediate_u, Tensor gate_q, Tensor gate_su, Tensor gate_sv, "
      "Tensor up_q, Tensor up_su, Tensor up_sv, Tensor down_q, Tensor down_su, "
      "Tensor down_sv, Tensor! locks, float limit) -> ()");
  m.def(
      "moe_m32_decode(Tensor hidden, Tensor! output, Tensor counts, Tensor "
      "tokens, "
      "Tensor routing, Tensor! state_g, Tensor! state_u, Tensor! "
      "intermediate_g, "
      "Tensor! intermediate_u, Tensor gate_q, Tensor gate_su, Tensor gate_sv, "
      "Tensor up_q, Tensor up_su, Tensor up_sv, Tensor down_q, Tensor down_su, "
      "Tensor down_sv, Tensor! locks, float limit) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("moe_m32", TORCH_BOX(&moe_m32<false>));
  m.impl("moe_m32_decode", TORCH_BOX(&moe_m32<true>));
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
