// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include "libtorch_stable/torch_utils.h"
#include "decode_kernel.cuh"
#include "decode_compact.cuh"

#include <algorithm>
#include <mutex>
#include <set>
#include <type_traits>

namespace {
using torch::headeronly::ScalarType;
using torch::stable::Tensor;

void check_cuda(cudaError_t status) {
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void check_tensor(const Tensor& t, int device, ScalarType dtype) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device &&
                      t.is_contiguous() && t.scalar_type() == dtype,
                  "EXL3 expert decode tensor device, dtype or layout mismatch");
}

void check_dimension(int64_t width) {
  STD_TORCH_CHECK(
      width >= 256 && width <= 8192 && width % 256 == 0,
      "EXL3 expert decode dimensions must be multiples of 256 in [256, 8192]");
}

template <bool fp32, bool residual, bool filtered>
void launch(const Tensor& input, const Tensor& trellis, const Tensor& su,
            const Tensor& sv, const Tensor& ids, const Tensor& output,
            const Tensor& scratch, int input_group, int grid, int shared_bytes,
            cudaStream_t stream, const Tensor* counts, int threshold) {
  static std::mutex mutex;
  static std::set<int> configured;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    if (configured.insert(input.get_device_index()).second) {
      check_cuda(cudaFuncSetAttribute(
          exl3_expert_gemv<fp32, residual, filtered>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
      check_cuda(cudaFuncSetAttribute(
          exl3_expert_gemv<fp32, residual, filtered>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
  }
  exl3_expert_gemv<fp32, residual, filtered>
      <<<dim3(grid, 1, ids.numel()), 256, shared_bytes, stream>>>(
          static_cast<const half*>(input.data_ptr()),
          reinterpret_cast<const uint16_t* const*>(trellis.data_ptr()),
          output.data_ptr(),
          reinterpret_cast<const half* const*>(su.data_ptr()),
          reinterpret_cast<const half* const*>(sv.data_ptr()),
          static_cast<const int64_t*>(ids.data_ptr()),
          static_cast<int*>(scratch.data_ptr()), input.size(1), output.size(1),
          8, input_group, scratch.size(1),
          counts ? static_cast<const int64_t*>(counts->data_ptr()) : nullptr,
          threshold);
}

template <bool fp32, bool small>
void launch_compact(const Tensor& input, const Tensor& trellis,
                    const Tensor& su, const Tensor& sv, const Tensor& ids,
                    const Tensor& output, const Tensor& scratch,
                    const Tensor& tasks, const Tensor& task_count,
                    int input_group, int sm_count, cudaStream_t stream) {
  static std::mutex mutex;
  static std::set<int> configured;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    const int device = input.get_device_index();
    if (!configured.count(device)) {
      check_cuda(cudaFuncSetAttribute(
          exl3_expert_gemv_compact<fp32, small>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
      check_cuda(cudaFuncSetAttribute(
          exl3_expert_gemv_compact<fp32, small>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      configured.insert(device);
    }
  }
  constexpr int per = fp32 || small ? 128 : 256;
  constexpr int units = fp32 || small ? 16 : 8;
  constexpr int shared = per * 96 + 1024;
  const int grid = std::min<int64_t>(ids.numel() * units, 32 * sm_count);
  const int group_shift = input_group == 8   ? 3
                          : input_group == 4 ? 2
                          : input_group == 2 ? 1
                                             : 0;
  exl3_expert_gemv_compact<fp32, small><<<grid, 256, shared, stream>>>(
      static_cast<const half*>(input.data_ptr()),
      reinterpret_cast<const uint16_t* const*>(trellis.data_ptr()),
      output.data_ptr(), reinterpret_cast<const half* const*>(su.data_ptr()),
      reinterpret_cast<const half* const*>(sv.data_ptr()),
      static_cast<const int64_t*>(ids.data_ptr()),
      static_cast<int*>(scratch.data_ptr()),
      static_cast<const int*>(tasks.data_ptr()),
      static_cast<const int*>(task_count.data_ptr()), group_shift,
      scratch.size(1));
}

template <bool filtered, bool compact = false>
void expert_gemv_impl(const Tensor& input, const Tensor& trellis,
                      const Tensor& su, const Tensor& sv, const Tensor& ids,
                      const Tensor& output, const Tensor& scratch, int64_t rows,
                      int64_t input_group, bool residual,
                      const Tensor* counts = nullptr, int threshold = 0,
                      const Tensor* tasks = nullptr,
                      const Tensor* task_count = nullptr) {
  static_assert(!compact || filtered);
  STD_TORCH_CHECK(input.is_cuda(), "EXL3 expert decode requires CUDA");
  const auto device = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const auto* props = get_device_prop();
  const bool sm80 = props->major == 8 && props->minor == 0;
  STD_TORCH_CHECK(sm80 || (props->major == 12 && props->minor == 0),
                  "EXL3 expert decode requires SM80 or SM120");
  check_tensor(input, device, ScalarType::Half);
  check_tensor(ids, device, ScalarType::Long);
  check_tensor(scratch, device, ScalarType::Int);
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Half ||
                      output.scalar_type() == ScalarType::Float,
                  "EXL3 expert decode output must be FP16 or FP32");
  check_tensor(output, device, output.scalar_type());
  STD_TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && ids.dim() == 1 &&
                      scratch.dim() == 2 && rows >= 1 &&
                      rows <= (filtered ? 128 : 8) && ids.numel() % rows == 0 &&
                      ids.numel() / rows >= 1 && ids.numel() / rows <= 8 &&
                      output.size(0) == ids.numel() &&
                      (input_group == 1 || input_group == ids.numel() / rows) &&
                      input.size(0) * input_group == ids.numel(),
                  "EXL3 expert decode shape or top-k mismatch");
  for (const auto* t : {&trellis, &su, &sv}) {
    check_tensor(*t, device, ScalarType::Long);
    STD_TORCH_CHECK(t->dim() == 1 && t->numel() == trellis.numel() &&
                        t->numel() >= ids.numel() / rows,
                    "EXL3 expert pointer table size mismatch");
  }
  if constexpr (filtered && !compact) {
    STD_TORCH_CHECK(
        sm80 && counts && threshold >= 1 && threshold <= 128,
        "EXL3 filtered expert decode requires SM80 and a valid threshold");
    check_tensor(*counts, device, ScalarType::Long);
    STD_TORCH_CHECK(counts->dim() == 1 && counts->numel() >= trellis.numel(),
                    "EXL3 filtered expert counts are too small");
  }
  const int k = input.size(1), n = output.size(1);
  check_dimension(k);
  check_dimension(n);
  if constexpr (compact) {
    const int top_k = ids.numel() / rows;
    const bool fp32 = output.scalar_type() == ScalarType::Float;
    STD_TORCH_CHECK(sm80 && !residual && rows >= 9 &&
                        (top_k & (top_k - 1)) == 0 && tasks && task_count,
                    "EXL3 compact decode requires plain SM80, 9..128 rows "
                    "and power-of-two top-k");
    STD_TORCH_CHECK(fp32 ? (k == 2048 && n == 4096 && input_group == 1)
                         : (k == 4096 && n == 2048),
                    "EXL3 compact decode requires GLM projection dimensions");
    check_tensor(*tasks, device, ScalarType::Int);
    check_tensor(*task_count, device, ScalarType::Int);
    STD_TORCH_CHECK(tasks->dim() == 1 && tasks->numel() == ids.numel() &&
                        task_count->dim() == 1 && task_count->numel() == 1,
                    "EXL3 compact decode task shape mismatch");
  }
  const int grid = sm80 ? (rows <= 4               ? 32
                           : filtered && rows > 64 ? 8
                                                   : 16)
                        : (rows == 1 || rows > 4 ? 64
                           : rows == 2           ? 32
                                                 : 16);
  const int total = k / 16;
  const int r = (total * (n / 256) + grid - 1) / grid;
  const int per = std::min(
      {std::max((std::max(r, std::min(2 * r, 32)) + 7) & ~7, SQ_MINROWS),
       gemv_int8_sq_rows_max(1, residual), (total + 7) & ~7});
  const int splits = (total + per - 1) / per;
  STD_TORCH_CHECK(
      scratch.size(0) >= ids.numel() &&
          scratch.size(1) >= SQ_WS_RESERVED + splits * n * (residual ? 2 : 1),
      "EXL3 expert decode scratch is too small");
  const int shared = per * (32 + 64 * (residual ? 2 : 1)) + 1024;
  const auto stream = get_current_cuda_stream(device);
  if constexpr (compact) {
    auto dispatch = [&](auto fp32, auto small) {
      launch_compact<decltype(fp32)::value, decltype(small)::value>(
          input, trellis, su, sv, ids, output, scratch, *tasks, *task_count,
          input_group, props->multiProcessorCount, stream);
    };
    if (output.scalar_type() == ScalarType::Float) {
      if (rows <= 64)
        dispatch(std::true_type{}, std::true_type{});
      else
        dispatch(std::true_type{}, std::false_type{});
    } else {
      if (rows <= 64)
        dispatch(std::false_type{}, std::true_type{});
      else
        dispatch(std::false_type{}, std::false_type{});
    }
    check_cuda(cudaGetLastError());
    return;
  }
  if (output.scalar_type() == ScalarType::Float) {
    if (residual)
      launch<true, true, filtered>(input, trellis, su, sv, ids, output, scratch,
                                   input_group, grid, shared, stream, counts,
                                   threshold);
    else
      launch<true, false, filtered>(input, trellis, su, sv, ids, output,
                                    scratch, input_group, grid, shared, stream,
                                    counts, threshold);
  } else {
    if (residual)
      launch<false, true, filtered>(input, trellis, su, sv, ids, output,
                                    scratch, input_group, grid, shared, stream,
                                    counts, threshold);
    else
      launch<false, false, filtered>(input, trellis, su, sv, ids, output,
                                     scratch, input_group, grid, shared, stream,
                                     counts, threshold);
  }
  check_cuda(cudaGetLastError());
}

void expert_gemv(const Tensor& input, const Tensor& trellis, const Tensor& su,
                 const Tensor& sv, const Tensor& ids, const Tensor& output,
                 const Tensor& scratch, int64_t rows, int64_t input_group,
                 bool residual) {
  expert_gemv_impl<false>(input, trellis, su, sv, ids, output, scratch, rows,
                          input_group, residual);
}

void expert_gemv_cold(const Tensor& input, const Tensor& trellis,
                      const Tensor& su, const Tensor& sv, const Tensor& ids,
                      const Tensor& output, const Tensor& scratch,
                      const Tensor& counts, int64_t rows, int64_t input_group,
                      int64_t threshold, bool residual) {
  expert_gemv_impl<true>(input, trellis, su, sv, ids, output, scratch, rows,
                         input_group, residual, &counts, threshold);
}

void expert_gemv_compact(const Tensor& input, const Tensor& trellis,
                         const Tensor& su, const Tensor& sv, const Tensor& ids,
                         const Tensor& output, const Tensor& scratch,
                         const Tensor& tasks, const Tensor& task_count,
                         int64_t rows, int64_t input_group, bool residual) {
  expert_gemv_impl<true, true>(input, trellis, su, sv, ids, output, scratch,
                               rows, input_group, residual, nullptr, 0, &tasks,
                               &task_count);
}

void expert_combine(const Tensor& input, const Tensor& routing,
                    const Tensor& output) {
  STD_TORCH_CHECK(output.is_cuda(), "EXL3 expert reduction requires CUDA");
  const auto device = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_tensor(input, device, ScalarType::Float);
  check_tensor(routing, device, ScalarType::Half);
  check_tensor(output, device, ScalarType::BFloat16);
  STD_TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && routing.dim() == 1 &&
                      output.size(0) >= 1 && output.size(0) <= 8 &&
                      input.size(0) % output.size(0) == 0 &&
                      input.size(0) / output.size(0) >= 1 &&
                      input.size(0) / output.size(0) <= 8 &&
                      input.size(1) == output.size(1) &&
                      routing.numel() == input.size(0),
                  "EXL3 expert reduction shape mismatch");
  check_dimension(output.size(1));
  exl3_expert_combine<<<(output.numel() + 255) / 256, 256, 0,
                        get_current_cuda_stream(device)>>>(
      static_cast<const float*>(input.data_ptr()),
      static_cast<const half*>(routing.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()), output.size(0),
      input.size(0) / output.size(0), output.size(1));
  check_cuda(cudaGetLastError());
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_exl3_C, m) {
  m.def(
      "expert_gemv(Tensor input, Tensor trellis, Tensor su, Tensor sv, Tensor "
      "ids, "
      "Tensor! output, Tensor! scratch, int rows, int input_group, bool "
      "residual) -> ()");
  m.def(
      "expert_gemv_cold(Tensor input, Tensor trellis, Tensor su, Tensor sv, "
      "Tensor ids, Tensor! output, Tensor! scratch, Tensor counts, int rows, "
      "int input_group, int threshold, bool residual=False) -> ()");
  m.def(
      "expert_gemv_compact(Tensor input, Tensor trellis, Tensor su, Tensor sv, "
      "Tensor ids, Tensor! output, Tensor! scratch, Tensor tasks, "
      "Tensor task_count, int rows, int input_group, bool residual=False) -> "
      "()");
  m.def("expert_combine(Tensor input, Tensor routing, Tensor! output) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("expert_gemv", TORCH_BOX(&expert_gemv));
  m.impl("expert_gemv_cold", TORCH_BOX(&expert_gemv_cold));
  m.impl("expert_gemv_compact", TORCH_BOX(&expert_gemv_compact));
  m.impl("expert_combine", TORCH_BOX(&expert_combine));
}
