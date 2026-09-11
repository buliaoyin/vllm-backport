# EXL3 temporary FP16 prefill experiment

This optional SM80 prototype reconstructs one full expert projection at a time,
then invokes the existing grouped FP16 Triton GEMM with FP32 accumulation. It
reuses the same weight buffer for gate, up and down, and retains the checkpoint's
Hadamard transforms, scaling, activation and routing weights. It supports
uniform 4-bit `mul1` experts. Production dispatch does not enable this backend.

For GLM-5.3-Flash's 288 experts with hidden/intermediate dimensions 4096/2048,
the weight buffer alone occupies 4.5 GiB per layer invocation workspace.
Additional activation buffers scale with workspace capacity and top-k. The
benchmark records total workspace bytes. Full reconstruction includes inactive
experts, so the extra memory and reconstruction traffic can outweigh reuse.

Each backend owns scratch for one CUDA stream. Instantiate a separate backend
for independent streams. Compile, warm up and allocate scratch before capture.

## Build and correctness

The builder adapts the upstream reconstruction kernel and copies the bundled
MIT-licensed helpers into a separate directory. It does not install or replace
the production extension. The tested upstream checkout is ExLlamaV3 1.4.8,
commit `6ff3a17ea7f3d0026b273d43239398d57f71b788`; build metadata records source,
compiler options and library hashes.

```bash
.venv/bin/python benchmarks/kernels/exl3_prefill_fp16/build.py \
  --source /path/to/exllamav3 \
  --build-dir /tmp/exl3-prefill-fp16

CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 \
  VLLM_EXL3_TEST_FP16_PREFILL_LIBRARY=/tmp/exl3-prefill-fp16/helpers.so \
  .venv/bin/python -m pytest tests/quantization/test_exl3.py \
  -k experimental_batched_fp16_prefill -q
```

The tests compare independent rotated-weight reconstruction and exercise token
counts 9/65/513/2049, capacity chunking and changed-input CUDA Graph replay.

## Real-routing measurement

Use actual saved routes containing at least as many rows as the largest case.

```bash
CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 VLLM_EXL3_MOE_MAX_TOKENS=8192 \
  .venv/bin/python -m benchmarks.kernels.exl3_prefill_fp16.benchmark \
  --checkpoint /path/to/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.22.mlp.experts \
  --routes /path/to/routes/layer-22.pt \
  --library /tmp/exl3-prefill-fp16/helpers.so \
  --rows 2048 4096 8192 --repeats 15 \
  --output /tmp/exl3-prefill-fp16-results.json
```

Timing includes reconstruction, routing, all three GEMMs, transforms and output
reduction within the full wrapper's CUDA Graph. Each sample evicts 256 MiB
outside the CUDA-event interval, and native measurements bracket each sweep.
The benchmark checks finite outputs and relative L2 error below 1% before and
after graph replay. This is a kernel experiment; model output quality and
whole-request latency require separate validation before production use.
