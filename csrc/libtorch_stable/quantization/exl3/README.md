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

`expert_gemv_cold` adds a filtered SM80 launch for up to 128 rows. The caller
provides GPU expert counts; assignments whose count reaches the threshold are
skipped before reading weights. An optional second DP4A pass compensates the
activation quantization residual. The Python `exl3_decode.py` wrapper sends those
experts through M32 and combines both outputs in FP32. The validated threshold
is three rows per expert; grid X is 16 through 64 input rows and 8 above that.
The original 1–8-row operation keeps its arithmetic and dispatch.

Batched decode requires a 1024-slot shared scratch (128 rows × top-8), versus
64 slots for the original path; the stride also covers residual partial sums
when compensation is selected. Each projection resets its used counters before
reuse, including when an assignment changes between hot and cold on later calls.
The wrapper selects the input device and uses its current stream. Pure-decode
metadata is required; PIECEWISE graphs retain the existing path because their
captured operations can later serve prefill or mixed batches.

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

`prefill.cu` exposes INT8 reconstruction, input Hadamard gathering, fused
activation/Hadamard, and weighted output scattering through the stable API.
`prefill_reconstruct.cuh` freezes the validated batched 4-bit mul1 reconstruction
adaptation from the same ExLlamaV3 commit. The affine integer codebook matches the
`m64n128k64` prototype documented in `benchmarks/kernels/exl3_prefill_int8`.
The production caller is `vllm/model_executor/layers/quantization/utils/exl3_prefill.py`;
it uses grouped Triton IMMA and a caller-owned, bounded INT8 projection buffer.
No upstream checkout or runtime source extraction is needed to build or run it.

The prefill helpers validate tensor device, type, contiguity, dimensions and
workspace shapes. Pointer tables contain owned model tensors; routing IDs must
be valid expert indices. CUDA launches use the current stream for the tensor's
device. The Python caller also selects the tensor device for Triton launches.
SM80 is the only enabled INT8-prefill target; other devices retain native MoE.
To select only upstream kernels, set `VLLM_EXL3_MOE_PREFILL=native` in addition
to the M-tile and decode overrides above.
