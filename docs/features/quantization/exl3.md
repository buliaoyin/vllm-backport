# EXL3

EXL3 is an experimental weight-only quantization backend using the MIT-licensed
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) CUDA extension. Weights stay
in their original trellis representation. Each matrix retains its own bit width,
codebook and Hadamard sign/scale vectors, including separately quantized matrices
inside vLLM's fused projections.

## Installation

Install ExLlamaV3 1.4.8 or later into the same environment as vLLM. Its compiled
extension must match the installed PyTorch, CUDA toolkit and GPU architectures.
An ExLlamaV3 wheel built for a different PyTorch release is not interchangeable.
For a source build, for example:

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
TORCH_CUDA_ARCH_LIST='8.0;12.0' MAX_JOBS=8 \
uv pip install --no-build-isolation --no-deps /path/to/exllamav3
```

The source build requires PyTorch, setuptools, wheel and ninja in the environment.
vLLM imports `exllamav3_ext` directly; it does not run ExLlamaV3's model loader,
generator, attention implementation or allocator configuration.

## Usage

EXL3 is detected from the checkpoint's `quantization_config.quant_method`.
Point vLLM at a local model directory containing `config.json` and safetensors:

```bash
vllm serve /path/to/exl3-model \
    --dtype bfloat16 --tensor-parallel-size 1
```

For a larger model, use pipeline parallelism:

```bash
vllm serve /path/to/exl3-model \
    --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 4
```

The safetensors headers determine individual tensor shapes and bit widths.
The average `bits` value in `quantization_config` is not a storage-layout setting.
Full-precision projections remain full precision. Converted calibration
`input_ids` are excluded from model weights.

## Current support and limits

- NVIDIA GPUs with compute capability 8.0 or newer; FP16 and BF16 activations.
- Local checkpoints; tensor parallelism and expert parallelism must both be 1.
  Pipeline parallelism is supported.
- Linear projections, fused QKV/MLP projections, and the quantized output head.
- SiLU MoE with uniform expert shapes and per-projection bit widths across
  experts, and the same `mul1` or `mcg` codebook across gate/up/down.
- Modern `suh`/`svh` scales; packed `su`/`sv` signs are also accepted for linear
  layers. MoE requires `suh`/`svh`.
- Correctness coverage includes mixed bit widths, incomplete checkpoints,
  independently rotated fused projections, hot experts, and changing inputs and
  expert selections during CUDA Graph replay.

The initial model validation covers Qwen3.5 dense and GLM5Next MoE text inference.
[Long-context validation](../../validation/exl3-long-context-20260910.md)
extends the two tested checkpoints to 65536 input tokens with one or four
submitted requests. The report records hardware, cache capacity, timing and
basic retrieval checks; this does not establish a general context limit.
Multimodal generation, speculative decoding, LoRA, CPU offload, sleep mode,
expert load balancing, and distributed tensor/expert sharding are not validated.
Shared experts run serially because upstream GEMMs share a device-wide lock
workspace. Dual batch overlap is explicitly rejected.
Do not infer support for those features from model architecture support alone.

The prefill kernels compute EXL3 products with FP16 operands; BF16 inputs and
outputs are converted at the operation boundary. Large linear prefill batches temporarily
reconstruct bounded column slices, then use GEMM. The entire model is never
permanently expanded to FP16. MoE batches are divided by the allocated expert
workspace capacity, which defaults to the smaller of 2048 tokens and the
scheduler's `max_num_batched_tokens`. Each expert fits even when every token
selects it. Batches of at least 256 tokens with more experts than concurrent
execution groups schedule the largest experts first;
routes, counts and all nine weight pointer tables are reordered together.
Up to eight decode tokens use the expert decode path without sorting routes.
For BF16 activations, uniform 4-bit mul1 weights, dimensions divisible by 256 in
[256, 8192], and up to eight selected experts per token, the default is INT8
DP4A on SM80 and INT8 DP4A with activation residual compensation on SM120.
Other configurations use upstream batched expert GEMMs.

EXL3 defaults `max_num_batched_tokens` to 2048 when no explicit token budget is
provided. `--max-num-batched-tokens` overrides this scheduler setting. Other
quantization backends retain their existing defaults. The usual throughput-mode,
unchunked-prefill and multimodal budget adjustments still apply.

SM80 uses the bundled M=32, FP32-accumulating expert kernel for uniform 4-bit
mul1 weights with dimensions divisible by 256 in [256, 8192]. Short expert tails
retain the smaller row path. Other formats and GPU architectures use the upstream
kernel. `VLLM_EXL3_MOE_M_TILE=16` explicitly selects the upstream implementation.
The [component build instructions](../../../csrc/libtorch_stable/quantization/exl3/README.md)
allow updating `_exl3_C` without rebuilding unrelated extensions.

`VLLM_EXL3_MOE_DECODE=hybrid` is the default expert decode policy. Set `native`
to use upstream expert decode, or `plain` / `residual` to override the activation
mode on supported configurations. These policies are independent of the prefill
M tile and `EXL3_INT8_GEMV`. Set them before process startup because decode is
captured in CUDA Graphs. The bundled `_exl3_C` provides both M32 and expert INT8
kernels; no experimental worker or separately loaded kernel library is needed.

`VLLM_EXL3_MOE_MAX_TOKENS` sets a positive workspace/token-group limit.
`VLLM_EXL3_MOE_PRIORITY=0` disables largest-expert-first scheduling for ablation.
These settings do not change the scheduler's outer chunk size. In pipeline
parallelism, increasing the outer chunk can reduce overlap between stages, so
benchmark both settings together for the intended input length and concurrency.

Workspaces are shared across layers with the same expert dimensions on a device.
For hidden size 4096 and intermediate size 2048, the default 2048-row workspace
uses 384 MiB on the tested CMP 170HX and 1104 MiB on the tested RTX PRO 6000
Blackwell. M32 additionally shares a roughly 4 MiB lock buffer per device.
For these dimensions, hybrid decode adds a shared 5.25 MiB scratch on SM80
and 9.25 MiB on SM120.
Setting the scheduler token budget to 1024 also caps expert workspace capacity
at 1024, halving the main workspace. Workspace memory grows
linearly with capacity and with the extension's number of concurrent expert
groups. See the [optimization experiments](../../validation/exl3-optimization-20260910.md)
for kernel ablations and complete-model measurements.

ExLlamaV3 controls its optional INT8 GEMV path through `EXL3_INT8_GEMV`. Use
both `EXL3_INT8_GEMV=0` and `VLLM_EXL3_MOE_DECODE=native` to validate without
activation INT8. Use the same ordinary GEMV setting in both engines when comparing
speed or numerical accuracy, and record the expert decode policy separately.
Set these variables before starting the process. Mode 2 is
the upstream default and uses approximate INT8 activations; its numerical error
is evaluated separately from the stricter FP16 path. It affects eligible ordinary
GEMVs, not the routed-expert decode policy or M32 prefill. See the
[combined INT8 measurements](../../validation/exl3-int8-combination-20260911.md)
for ordinary GEMV comparisons and the
[default expert decode validation](../../validation/exl3-hybrid-default-20260911.md)
for the separate expert policy comparison, accuracy limits and runtime defaults.

## Reproducing checks

```bash
EXL3_INT8_GEMV=0 .venv/bin/python -m pytest tests/quantization/test_exl3.py -q

EXL3_INT8_GEMV=0 .venv/bin/python benchmarks/kernels/benchmark_exl3.py \
    --checkpoint /path/to/exl3-model \
    --matrix model.layers.0.mlp.gate_proj --output /tmp/exl3-kernels.json
```

`benchmarks/benchmark_exl3.py` measures both complete engines from a shared JSON
of tokenized inputs, records cache hits and actual output lengths, and optionally
runs GSM8K questions. Run its `exllamav3` backend in the original engine's own
environment. Distinguish GPU kernel time from complete-engine throughput and
record the actual device placement for multi-GPU comparisons.

`benchmarks/prepare_exl3_long_inputs.py` builds deterministic retrieval inputs
at 8K, 16K, 32K and 64K, including exact token counts and expected answers.
The benchmark accepts gzip JSON inputs and configurable context/cache sizes.
For fixed-length generation, retrieval checks decode only the response before
the first configured stop token; full generated tokens remain in the results.
