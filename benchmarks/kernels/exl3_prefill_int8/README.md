# EXL3 temporary INT8 prefill experiment

This opt-in SM80 prototype reconstructs one full expert projection at a time
into INT8, then runs grouped IMMA with INT32 accumulation. Gate/up/down reuse
the same weight buffer. Hadamard transforms, activation and routing follow the
FP16 experiment. Activations are quantized per row after each input transform.

This is a separate experiment from the fused M32 INT8 kernel and from the
ordinary `EXL3_INT8_GEMV` decode switch. It targets larger prefill chunks where
reconstruction can be reused across many rows. Production does not enable it.

## Arithmetic and memory

The weight representation uses the previously tested affine codebook formula:
`D = byte_sum(index * 0x83DCD12D) - 510`, with multiplication wrapping at 32 bits,
and `q = clamp(floor((D + 2) / 4), -128, 127)`. Reconstruction stores q, and the
GEMM epilogue applies `4 * alpha * activation_scale` plus `beta * row_sum`.
Alpha/beta are derived from the upstream FP16 constants. The implementation
adds both weight and activation rounding; it is not numerically lossless.

For 288 experts and dimensions 4096/2048, the weight buffer occupies 2.25 GiB.
At capacity 8192 and top-k 8, the FP16 activations use another 1 GiB, and INT8
activation scratch uses about 0.25 GiB. Metadata reports the activation
quantization allocation separately from `workspace_bytes`; add both for total
scratch. Quantized checkpoint weights and the original EXL3 workspace remain
resident. Each backend is intended for one CUDA stream.

## Build and correctness

Use the same pinned ExLlamaV3 1.4.8 checkout as the FP16 prototype, commit
`6ff3a17ea7f3d0026b273d43239398d57f71b788`. Build into a new directory. The builder
records source and library hashes, compiler options and register reports.

```bash
.venv/bin/python benchmarks/kernels/exl3_prefill_int8/build.py \
  --source /path/to/exllamav3 --build-dir /tmp/exl3-prefill-int8

CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 \
  VLLM_EXL3_TEST_BATCHED_INT8_LIBRARY=/tmp/exl3-prefill-int8/helpers.so \
  .venv/bin/python -m pytest tests/quantization/test_exl3.py \
  -k experimental_batched_int8_prefill -q
```

The tests use independent rotated-weight references and an explicit 2%
relative L2 limit, matching the earlier INT8 experiment. They cover tails,
workspace capacity overflow, and CUDA Graph replay after changing input/routes.
Model quality needs separate evaluation; these bounds alone are insufficient.

## Full-wrapper benchmark

```bash
CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 VLLM_EXL3_MOE_MAX_TOKENS=8192 \
  .venv/bin/python -m benchmarks.kernels.exl3_prefill_int8.benchmark \
  --checkpoint /path/to/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.22.mlp.experts \
  --routes /path/to/routes/layer-22.pt \
  --library /tmp/exl3-prefill-int8/helpers.so \
  --rows 2048 4096 8192 --repeats 15 --max-relative-error 0.02 \
  --output /tmp/exl3-prefill-int8-results.json
```

Timing includes reconstruction, activation quantization, routing, transforms,
all three grouped GEMMs and output reduction. Native CUDA Graph measurements
bracket each sweep, with 256 MiB L2 eviction outside every event interval.
Kernel metadata records registers, spills, shared memory and IMMA presence.

For phase attribution, use
`benchmarks.kernels.exl3_prefill_fp16.profile --backend int8` with the same
checkpoint/prefix/routes/library and an output JSON. It reports a clean timing
control alongside non-overlapping phase events so instrumentation overhead is
visible. GPU event intervals must not be treated as whole-model latency.

## Opt-in model comparison

Use worker extension
`benchmarks.kernels.exl3_prefill_int8.worker.WorkerExtension`. On supported SM80
ranks it selects this backend for at least 4096 input rows. Smaller inputs use
the original expert path; SM120 retains the original dispatch.

Use chunk 6144 with the ordinary projection FP16 cache disabled. The
cache ablation (local archive: `docs/validation/exl3-linear-cache-20260911.md`) measures
its small throughput contribution against 10.09 GiB of resident weight copies
across the three SM80 ranks. The original 4700-target confirmation used a
4 GiB cache budget per SM80; its recorded results retain that configuration.

The archived inputs refer to the local GLM checkpoint; change only the model
path when reproducing on another machine. Run from the repository root in a
shell without other EXL3 experiment overrides:

The command below uses inputs from the local validation archive, which is
not included in the repository. Set `--inputs` to your own input file when
running from a fresh checkout.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export PYTHONPATH="$PWD/benchmarks:$PWD"
export VLLM_TRITON_USE_TD=0
export EXL3_INT8_GEMV=2
export VLLM_EXL3_BATCHED_INT8_LIBRARY=/tmp/exl3-prefill-int8/helpers.so
export VLLM_EXL3_BATCHED_INT8_CONFIG=m64n128k64
export VLLM_EXL3_BATCHED_INT8_MIN_ROWS=4096
export VLLM_EXL3_MOE_MAX_TOKENS=6144
unset VLLM_EXL3_LINEAR_CACHE_GIB
export VLLM_PP_LAYER_PARTITION=11,11,11,12

.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm \
  --inputs docs/validation/exl3-target4700-20260911/confirmation-inputs.json.gz \
  --output /tmp/exl3-prefill-int8-model.json \
  --pp 4 --expected-layer-counts 11 11 11 12 \
  --max-model-len 66560 --kv-cache-gib 8 --batch-size 4 \
  --chunk-size 6144 --warmups 1 --repeats 5 --tokens 32 --skip-eval \
  --worker-extension-cls benchmarks.kernels.exl3_prefill_int8.worker.WorkerExtension
```

The input file contains one 64K request per sample (B1); `--batch-size 4` is
engine capacity. Add `--event-profile` for a separate diagnostic request after
clean performance sampling. Runtime metadata reports the active threshold,
IMMA kernels and scratch. To reproduce the cache-enabled ablation, explicitly
set `VLLM_EXL3_LINEAR_CACHE_GIB=4`; that budget is additional to expert scratch.
Use a separate engine for every configuration. For the default control, unset
the `VLLM_EXL3_BATCHED_INT8_*` and cache variables, set both chunk and
`VLLM_EXL3_MOE_MAX_TOKENS` to 2048, and select worker extension
`benchmarks.kernels.exl3_prefill.worker.WorkerExtension`.

For GLM on three CMP 170HX GPUs, the tested long-input configuration uses
`CUDA_VISIBLE_DEVICES=0,1,2`, PP `16,15,14`, chunk 6144, and 2 GiB KV cache per
rank with `PYTORCH_ALLOC_CONF=expandable_segments:True`. Keep the ordinary
projection cache disabled. The three-GPU validation report (local archive: `docs/validation/exl3-3x170hx-20260911.md`) includes the full
command, partition/chunk search, startup memory limits, and 8K–64K throughput.
The three-GPU B1 measurements favor native M32 / chunk 2048 at 8K–16K and
INT8 / chunk 6144 at 32K–64K. This deployment choice does not change production
defaults.

For the 64-question quality comparison, use `inputs.json.gz` from the same
archive, replace `--skip-eval` with `--skip-perf --eval-batch-size 1
--eval-tokens 768`, and set `VLLM_EXL3_BATCHED_INT8_MIN_ROWS=9` on the candidate.
This forces short prompts through the new arithmetic. It is a quality check
configuration, not the measured serving policy. Inspect actual backend calls
and restore the 4096 threshold for performance measurements.

The validation report (local archive: `docs/validation/exl3-target4700-20260911.md`)
includes all samples, shorter-input regressions, quality limits, memory costs,
source hashes and the exact measurement script (`benchmark_model.py.gz`). The
public benchmark supports the same explicit batching and runtime inspection;
its defaults are not a substitute for the recorded command/environment.

## Production integration

The validated arithmetic now lives in `_exl3_C` and
`vllm/model_executor/layers/quantization/utils/exl3_prefill.py`. Normal inference
uses `VLLM_EXL3_MOE_PREFILL=auto|native|int8` (default `native`) and does not use this worker or
`helpers.so`. See [the production guide](../../../docs/features/quantization/exl3.md)
and concurrent validation (local archive: `docs/validation/exl3-adaptive-prefill-20260912.md`).

For historical worker experiments, explicitly set `VLLM_EXL3_MOE_PREFILL=native`
so the production dispatcher and adaptive scheduler do not supersede the
experimental worker. Keep the original source/library hashes in experiment
records. The original `benchmark.py` still exercises the prototype;
`grouped_workspace.py` below exercises the production path.

## Bounded production workspace

`grouped_workspace.py` compares the production INT8 wrapper with all experts
resident (`--groups 0`), 64 experts per group, and 32 per group. It loads captured
routes and original checkpoint weights; no experimental library is required.
`--hot-routes` also checks a skewed route with all tokens assigned to the final
eight experts. The full wrapper, including routing and reconstruction, is timed.
