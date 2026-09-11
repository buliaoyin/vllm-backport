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

The CUDA kernels compute EXL3 products with FP16 operands; BF16 inputs and outputs
are converted at the operation boundary. Large linear prefill batches temporarily
reconstruct bounded column slices, then use GEMM. The entire model is never
permanently expanded to FP16. MoE batches are divided by the allocated expert
workspace capacity, which defaults to the smaller of 1024 tokens and the
scheduler's `max_num_batched_tokens`. Each expert fits even when every token
selects it. Batches of at least 256 tokens with more experts than concurrent
execution groups schedule the largest experts first;
routes, counts and all nine weight pointer tables are reordered together.
Up to eight decode tokens use upstream batched expert GEMMs without sorting the
routes.

`VLLM_EXL3_MOE_MAX_TOKENS` sets a positive workspace/token-group limit.
`VLLM_EXL3_MOE_PRIORITY=0` disables largest-expert-first scheduling for ablation.
These settings do not change the scheduler's outer chunk size. In pipeline
parallelism, increasing the outer chunk can reduce overlap between stages, so
benchmark both settings together for the intended input length and concurrency.

Workspaces are shared across layers with the same expert dimensions on a device.
For hidden size 4096 and intermediate size 2048, a 1024-row workspace uses
192 MiB on the tested CMP 170HX and 552 MiB on the tested RTX PRO 6000 Blackwell.
The corresponding 512-row sizes are 96 MiB and 276 MiB. Workspace memory grows
linearly with capacity and with the extension's number of concurrent expert
groups. See the [optimization experiments](../../validation/exl3-optimization-20260910.md)
for kernel ablations and complete-model measurements.

ExLlamaV3 controls its optional INT8 GEMV path through `EXL3_INT8_GEMV`. Use
`EXL3_INT8_GEMV=0` for strict FP16 validation, and use the same setting in both
engines when comparing speed or numerical accuracy. Set it before starting the process for reproducible comparisons. Mode 2 is
the upstream default and uses approximate INT8 activations; its numerical error
is evaluated separately from the stricter FP16 path.

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
