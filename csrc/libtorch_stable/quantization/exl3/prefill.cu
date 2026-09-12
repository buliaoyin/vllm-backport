// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "upstream/util.h"
#include "upstream/util.cuh"
#include "upstream/ptx.cuh"
#include "upstream/quant/hadamard_inner.cuh"
#include "upstream/quant/exl3_moe_common.cuh"
#include "prefill_reconstruct.cuh"

__global__ void gather_input(const half* x, const int64_t* ids, half* y,
                             const half* const* scale, int slots, int width,
                             int topk) {
  int warp = blockIdx.x * 8 + threadIdx.x / 32;
  int per = width / 128;
  if (warp >= slots * per) return;
  int slot = warp / per, tile = warp % per;
  int expert = ids[slot];
  had_hf_r_128_inner<true, false>(x + (slot / topk) * width + tile * 128,
                                  y + slot * width + tile * 128,
                                  scale[expert] + tile * 128, 0.088388347648f);
}

__global__ void activate_kernel(half* gate, const half* up, const int64_t* ids,
                                const half* const* sg, const half* const* su,
                                const half* const* sd, int slots, int width,
                                float limit) {
  int warp = blockIdx.x * 8 + threadIdx.x / 32;
  int per = width / 128;
  if (warp >= slots * per) return;
  int slot = warp / per, tile = warp % per;
  int expert = ids[slot];
  had_hf_r_128_guad_inner(gate + slot * width + tile * 128,
                          up + slot * width + tile * 128,
                          gate + slot * width + tile * 128,
                          sg[expert] + tile * 128, su[expert] + tile * 128,
                          sd[expert] + tile * 128, 0.088388347648f, limit, 0);
}

__global__ void scatter_output(const half* down, float* out, const int64_t* ids,
                               const half* weights, const half* const* scale,
                               int slots, int width, int topk) {
  int warp = blockIdx.x * 8 + threadIdx.x / 32;
  int per = width / 128;
  if (warp >= slots * per) return;
  int slot = warp / per, tile = warp % per;
  int expert = ids[slot];
  had_hf_r_128_d_inner(down + slot * width + tile * 128,
                       out + (slot / topk) * width + tile * 128,
                       scale[expert] + tile * 128,
                       0.088388347648f * __half2float(weights[slot]));
}

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include "libtorch_stable/torch_utils.h"

namespace {
using torch::headeronly::ScalarType;
using torch::stable::Tensor;

void check_tensor(const Tensor& t, int device, ScalarType dtype) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device &&
                      t.is_contiguous() && t.scalar_type() == dtype,
                  "EXL3 INT8 prefill tensor device, dtype or layout mismatch");
}
void check_width(int64_t width) {
  STD_TORCH_CHECK(width >= 256 && width <= 8192 && width % 256 == 0,
                  "EXL3 INT8 prefill dimensions must be multiples of 256 "
                  "in [256, 8192]");
}
void check_table(const Tensor& table, int device) {
  check_tensor(table, device, ScalarType::Long);
  STD_TORCH_CHECK(
      table.dim() == 1 && table.numel() > 0 && table.numel() <= 65535,
      "Invalid EXL3 expert pointer table");
}
void check_slots(const Tensor& ids, int device) {
  check_tensor(ids, device, ScalarType::Long);
  STD_TORCH_CHECK(ids.dim() == 1 && ids.numel() > 0 && ids.numel() <= 1048576,
                  "Invalid EXL3 prefill slot count");
}
void check_sm80() {
  const auto* props = get_device_prop();
  STD_TORCH_CHECK(props->major == 8 && props->minor == 0,
                  "EXL3 INT8 prefill requires SM80");
}
void check_launch() {
  auto status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void prefill_reconstruct(const Tensor& table, const Tensor& weight) {
  STD_TORCH_CHECK(weight.is_cuda(), "EXL3 INT8 prefill requires CUDA");
  const int device = weight.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_sm80();
  check_tensor(weight, device, ScalarType::Char);
  check_table(table, device);
  STD_TORCH_CHECK(weight.dim() == 3 && weight.size(0) == table.numel(),
                  "EXL3 INT8 reconstructed weight shape mismatch");
  const int k = weight.size(1), n = weight.size(2);
  check_width(k);
  check_width(n);
  reconstruct_experts<4, 2><<<dim3(n / 128, k / 16, table.numel()), 256, 0,
                              get_current_cuda_stream(device)>>>(
      static_cast<int8_t*>(weight.data_ptr()),
      reinterpret_cast<const uint16_t* const*>(table.data_ptr()), n / 16, k);
  check_launch();
}

void prefill_gather(const Tensor& input, const Tensor& ids, const Tensor& scale,
                    const Tensor& output, int64_t topk) {
  STD_TORCH_CHECK(input.is_cuda(), "EXL3 INT8 prefill requires CUDA");
  const int device = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_sm80();
  check_tensor(input, device, ScalarType::Half);
  check_tensor(output, device, ScalarType::Half);
  check_table(scale, device);
  check_slots(ids, device);
  STD_TORCH_CHECK(
      input.dim() == 2 && output.dim() == 2 && topk > 0 &&
          topk <= scale.numel() && input.size(0) * topk == ids.numel() &&
          output.size(0) == ids.numel() && output.size(1) == input.size(1),
      "EXL3 prefill gather shape mismatch");
  const int width = input.size(1), slots = ids.numel();
  check_width(width);
  gather_input<<<(slots * (width / 128) + 7) / 8, 256, 0,
                 get_current_cuda_stream(device)>>>(
      static_cast<const half*>(input.data_ptr()),
      static_cast<const int64_t*>(ids.data_ptr()),
      static_cast<half*>(output.data_ptr()),
      reinterpret_cast<const half* const*>(scale.data_ptr()), slots, width,
      topk);
  check_launch();
}

void prefill_activate(const Tensor& gate, const Tensor& up, const Tensor& ids,
                      const Tensor& sg, const Tensor& su, const Tensor& sd,
                      double limit) {
  STD_TORCH_CHECK(gate.is_cuda(), "EXL3 INT8 prefill requires CUDA");
  const int device = gate.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_sm80();
  check_tensor(gate, device, ScalarType::Half);
  check_tensor(up, device, ScalarType::Half);
  check_slots(ids, device);
  for (const auto* t : {&sg, &su, &sd}) {
    check_table(*t, device);
    STD_TORCH_CHECK(t->numel() == sg.numel(),
                    "EXL3 prefill activation pointer table mismatch");
  }
  STD_TORCH_CHECK(gate.dim() == 2 && up.dim() == 2 &&
                      gate.size(0) == ids.numel() &&
                      up.size(0) == gate.size(0) && up.size(1) == gate.size(1),
                  "EXL3 prefill activation shape mismatch");
  const int width = gate.size(1), slots = ids.numel();
  check_width(width);
  activate_kernel<<<(slots * (width / 128) + 7) / 8, 256, 0,
                    get_current_cuda_stream(device)>>>(
      static_cast<half*>(gate.data_ptr()),
      static_cast<const half*>(up.data_ptr()),
      static_cast<const int64_t*>(ids.data_ptr()),
      reinterpret_cast<const half* const*>(sg.data_ptr()),
      reinterpret_cast<const half* const*>(su.data_ptr()),
      reinterpret_cast<const half* const*>(sd.data_ptr()), slots, width, limit);
  check_launch();
}

void prefill_scatter(const Tensor& input, const Tensor& ids,
                     const Tensor& routing, const Tensor& scale,
                     const Tensor& output, int64_t topk) {
  STD_TORCH_CHECK(input.is_cuda(), "EXL3 INT8 prefill requires CUDA");
  const int device = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_sm80();
  check_tensor(input, device, ScalarType::Half);
  check_tensor(output, device, ScalarType::Float);
  check_tensor(routing, device, ScalarType::Half);
  check_table(scale, device);
  check_slots(ids, device);
  STD_TORCH_CHECK(
      input.dim() == 2 && output.dim() == 2 && topk > 0 &&
          topk <= scale.numel() && output.size(0) * topk == ids.numel() &&
          input.size(0) == ids.numel() && output.size(1) == input.size(1) &&
          routing.dim() == 1 && routing.numel() == ids.numel(),
      "EXL3 prefill scatter shape mismatch");
  const int width = input.size(1), slots = ids.numel();
  check_width(width);
  scatter_output<<<(slots * (width / 128) + 7) / 8, 256, 4096,
                   get_current_cuda_stream(device)>>>(
      static_cast<const half*>(input.data_ptr()),
      static_cast<float*>(output.data_ptr()),
      static_cast<const int64_t*>(ids.data_ptr()),
      static_cast<const half*>(routing.data_ptr()),
      reinterpret_cast<const half* const*>(scale.data_ptr()), slots, width,
      topk);
  check_launch();
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_exl3_C, m) {
  m.def("prefill_reconstruct(Tensor table, Tensor! weight) -> ()");
  m.def(
      "prefill_gather(Tensor input, Tensor ids, Tensor scale, "
      "Tensor! output, int topk) -> ()");
  m.def(
      "prefill_activate(Tensor! gate, Tensor up, Tensor ids, Tensor sg, "
      "Tensor su, Tensor sd, float limit) -> ()");
  m.def(
      "prefill_scatter(Tensor input, Tensor ids, Tensor routing, Tensor scale, "
      "Tensor! output, int topk) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("prefill_reconstruct", TORCH_BOX(&prefill_reconstruct));
  m.impl("prefill_gather", TORCH_BOX(&prefill_gather));
  m.impl("prefill_activate", TORCH_BOX(&prefill_activate));
  m.impl("prefill_scatter", TORCH_BOX(&prefill_scatter));
}
