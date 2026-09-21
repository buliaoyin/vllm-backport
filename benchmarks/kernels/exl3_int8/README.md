# EXL3 SM80 INT8 prefill experiment

This opt-in prototype uses `mma.m16n8k32.s32.s8.s8` for 32 expert rows,
reusing decoded weights across both M=16 fragments. It is a benchmark experiment,
not a serving default. Kernel gains are small and must be checked against model
quality and request latency; see
the validation report (local archive: `docs/validation/exl3-int8-20260911.md`).

Input rotations, gate/up activation, output rotations, expert scheduling and
routing remain in the fused expert kernel. Activations are quantized once per
expert and projection after rotation, with a separate absmax scale for each row.
The original FP16 workspace is reused: INT8 values occupy its first K bytes per
row, with the scale and row sum in the final eight bytes. Four warps cooperate on
each row, with vectorized reads, a shared absmax reduction and block barriers
protecting in-place conversion. The K loop has uniform barriers even for K=256.

For a trellis index, let `D = byte_sum(index * 0x83DCD12D) - 510`, with the
multiply wrapping at 32 bits. The codebook is the FP16 rounding of `D * alpha +
beta`, where alpha and beta follow the upstream FP16 constants. D needs more than
8 signed bits. The plain variant uses `q = clamp(floor((D + 2) / 4), -128, 127)`
and a scale of `4 * alpha`; the epilogue also applies beta times the original
activation row sum. This adds both activation and codebook error.

`int8_residual` additionally multiplies `D - 4*q` with a second IMMA operation.
It removes the integer codebook approximation, but retains activation rounding
and the difference from the upstream FP16-rounded codebook. This is a **weight
codebook residual**, distinct from the upstream `EXL3_INT8_GEMV=1` activation
residual. That environment variable does not select either prefill variant.

The launcher supports only SM80, ExLlamaV3 1.4.8, uniform 4-bit mul1 experts and
dimensions divisible by 256. Other GPUs retain native experts in model tests.
The source checkout and installed extension are not modified.

## Build and check

The source must be at `6ff3a17ea7f3d0026b273d43239398d57f71b788` with an
unmodified extension. Use a new build directory:

```bash
.venv/bin/python benchmarks/kernels/exl3_int8/build.py \
  --source /home/bul/dev/exllamav3 \
  --build-dir /tmp/exl3-int8-build \
  --nvcc /usr/local/cuda/bin/nvcc

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  VLLM_EXL3_TEST_ROWS_LIBRARY=/tmp/exl3-int8-build/fp16/rows.so \
  VLLM_EXL3_TEST_INT8_LIBRARY=/tmp/exl3-int8-build/int8.so \
  .venv/bin/python -m pytest tests/quantization/test_exl3.py -q
```

The builder first reconstructs the pinned M32 FP16 experiment, then replaces its
GEMM calls and inserts activation quantization. `build.json` records upstream and
experiment hashes, compiler invocation, and the library hash. `build.log` records
register and spill information. The resulting library exports `m32_predicated`,
`int8` and `int8_residual`.

The tests reuse independently reconstructed weights and explicit Hadamard
transforms. INT8 tests allow 2% relative L2 error; FP16 bounds are unchanged.
They cover short rows, hot experts, chunking and CUDA graph replay with changed
inputs and routes. These bounds are not a substitute for model evaluation.

## Measure full experts

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 \
  .venv/bin/python benchmarks/kernels/benchmark_exl3_moe.py \
  --checkpoint /home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.22.mlp.experts \
  --route-sample /tmp/exl3-opt-20260910/routes/layer-22.pt \
  --prefill-rows 512 1024 2048 --prefill-chunks 1024 \
  --row-kernel-library /tmp/exl3-int8-build/int8.so \
  --row-kernel-variants m32_predicated int8 int8_residual m32_predicated \
  --row-kernel-max-relative-error 0.02 \
  --output /tmp/exl3-int8-layer22.json
```

Timing includes activation quantization, routing, rotations and reductions in
one CUDA graph replay. A 256 MiB buffer is written before each timed interval to
evict L2. Keep the FP16 control before and after candidates. The explicit error
bound is required because INT8 intentionally changes the arithmetic.

For model tests use
`kernels.exl3_int8.worker.Exl3Int8WorkerExtension` and set `rows_library` and
`rows_variant` in the existing `--moe-variants` JSON interface. Keep the same
`11,11,11,12` partition, token IDs, 1024-token chunks and KV allocation. Decode
mode comparisons need separate engine processes so captured CUDA graphs reflect
the selected `EXL3_INT8_GEMV` mode.
