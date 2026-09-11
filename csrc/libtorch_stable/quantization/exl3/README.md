# EXL3 MoE kernels

The CUDA headers derive from ExLlamaV3 commit
`6ff3a17ea7f3d0026b273d43239398d57f71b788` under the included MIT license.
The M32 variant reuses decoded weights across two row fragments, predicates the
second fragment for short experts, retains FP32 accumulation, removes the
redundant fragment-load barrier, and orders Hadamard scatter staging reuse with
an explicit warp barrier. The source matches the validated `nobar_safe` variant.

`moe.cu` uses the PyTorch stable API and a caller-owned lock tensor. It does not
call ExLlamaV3's private `DevCtx` ABI. Dispatch is restricted to SM80, uniform
4-bit mul1 experts, and dimensions divisible by 256 between 256 and 8192.
The upstream extension handles other expert formats and GPU architectures.

`decode.cu` exposes the routed-expert DP4A kernels through the same stable API.
`decode_kernel.cuh` and `upstream/quant/exl3_gemv_int8_kernel.cuh` retain the
validated prototype's arithmetic. SM80 defaults to plain INT8 activations;
SM120 defaults to a second DP4A pass for the activation residual. This path
supports BF16 outputs, 1–8 input rows, 1–8 selected experts, and the same weight
format and dimension bounds as M32. Gate/up produce FP16 intermediates; down
produces FP32 intermediates before routing-weight reduction into BF16 output.
Unsupported configurations fall back to upstream decode.

The caller owns a shared INT32 scratch tensor with a fixed slot stride large
enough for both projection shapes and all supported batch sizes. Keeping that
stride fixed prevents partial sums from overlapping counters when reusing the
scratch. Each device uses a single stream, matching the backend's existing
restriction. Neither kernel accesses the private ExLlamaV3 `DevCtx` ABI.

For an incremental component build, use the same environment as vLLM:

```bash
TORCH_CUDA_ARCH_LIST='8.0;12.0' cmake \
  -S csrc/libtorch_stable/quantization/exl3 -B /tmp/vllm-exl3-build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DVLLM_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build /tmp/vllm-exl3-build --target _exl3_C
cmake --install /tmp/vllm-exl3-build --component _exl3_C
```

The component builds SM80 and, with CUDA 12.8 or newer, SM120. M32 runtime
dispatch remains restricted to SM80. Set `VLLM_EXL3_MOE_M_TILE=16` and
`VLLM_EXL3_MOE_DECODE=native` to select upstream kernels for both paths.
