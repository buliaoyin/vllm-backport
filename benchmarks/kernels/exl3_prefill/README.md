# EXL3 prefill experiments

Opt-in SM80 experiments for the bundled 4-bit mul1 expert prefill kernel.
They cover shared-memory allocation, fragment staging, exact codebook lookup,
block residency, warp layout, compile-time matrix dimensions and shared decoded
weight tiles. Each launcher's
scratch and locks belong to one CUDA stream, matching the production EXL3
execution contract.

The `wide_*` variants distribute output columns across 16 warps without a
threadblock K reduction. This gives M64 the same accumulator count per thread
as the original M32/K32 layout. `fixed_*` variants specialize hidden/intermediate
dimensions to 4096/2048 and reject other dimensions. All variants use the
checkpoint's original weights and FP32 accumulation.

The `cached_*` variants decode K64/K128 weight tiles into shared memory and
reuse them across M32/M64 rows, with N128/N256 column tiles. `adaptive*` selects
M32 or M64 according to each expert's row count within the same kernel. These
are experiments, including candidates that measured slower than the default.

## Build and check

Build into a new directory. The output includes source/generated hashes,
compiler options, register/spill reports and the library hash.

```bash
.venv/bin/python benchmarks/kernels/exl3_prefill/build.py \
  --build-dir /tmp/exl3-prefill-experiments

CUDA_VISIBLE_DEVICES=0 \
  VLLM_EXL3_TEST_PREFILL_LIBRARY=/tmp/exl3-prefill-experiments/prefill.so \
  .venv/bin/python -m pytest tests/quantization/test_exl3.py \
  -k experimental_prefill_residency_and_codebook -q
```

`--variants` can limit compilation; select matching pytest parameter names with
`-k` when testing a partial library. Tests compare independent rotated weight
reconstructions and exercise hot experts, tails and changed-input CUDA Graph
replay. A smaller shared-memory allocation does not alone guarantee more
resident blocks: the launcher also reserves an adequate shared carveout and
records the CUDA occupancy result. The benchmark's `--carveout 0` reproduces
the initial L1-preferred residency ablation.

## Real-routing measurement

Capture routes with `benchmarks/benchmark_exl3.py --capture-routing-dir` and
provide a sample containing at least as many actual rows as the largest test.
The helper loads all 288 GLM-5.3-Flash experts in the selected checkpoint layer.

```bash
CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 VLLM_EXL3_MOE_MAX_TOKENS=8192 \
  .venv/bin/python -m benchmarks.kernels.exl3_prefill.benchmark \
  --checkpoint /path/to/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.22.mlp.experts \
  --routes /path/to/routes/layer-22.pt \
  --library /tmp/exl3-prefill-experiments/prefill.so \
  --rows 1024 2048 4096 8192 --repeats 15 \
  --output /tmp/exl3-prefill-results.json
```

The measured boundary is the full expert wrapper replayed as a CUDA Graph.
Each sample evicts 256 MiB outside the timing interval; native measurements
bracket each candidate sweep. CUDA events are used because CUPTI explicitly
rejects the CMP 170HX. JSON records eager/graph relative L2, raw timings,
useful FLOPs (`6 * routed_assignments * hidden * intermediate`), occupancy,
workspace bytes and artifact hashes. Kernel microbenchmarks do not directly
predict whole-request latency.

## Model and timeline experiments

Use worker extension
`benchmarks.kernels.exl3_prefill.worker.WorkerExtension`. Set both environment
variables below to install an experimental prefill kernel during model loading:

```bash
export VLLM_EXL3_EXPERIMENTAL_PREFILL=k16_resident
export VLLM_EXL3_PREFILL_LIBRARY=/tmp/exl3-prefill-experiments/prefill.so
export PYTHONPATH=/home/bul/dev/vllm-backport/benchmarks:/home/bul/dev/vllm-backport
```

The worker applies the override on supported SM80 ranks and records the actual
launcher state. Run candidates in separate processes. Use identical token IDs,
output lengths, KV budgets and PP `11/11/11/12` for model comparisons.

An independent non-expert projection cache is enabled by setting
`VLLM_EXL3_LINEAR_CACHE_GIB=4` with the same worker extension. This budget is per
SM80 device and is additional to the quantized weights and KV cache. It caches
reconstructed FP16 weights for eligible layer projections and uses them only
for at least 1024 input rows. Smaller inputs use the original quantized path.
Runtime metadata records cache allocations and Python warmup/capture calls;
these counters do not count CUDA Graph replays.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/quantization/test_exl3.py \
  -k prefill_projection_cache_preserves_quantized_outputs -q
```

For a separate full-expert projection reconstruction experiment, see
[the grouped FP16 prototype](../exl3_prefill_fp16/README.md).

Without the experiment variables, the extension only adds instrumentation when
`--event-profile` is requested. It calibrates a GPU event against the common
host monotonic clock, reports the calibration bracket, and records layer and
expert intervals. Nested intervals must not be summed. Cross-rank alignment is
approximate; the calibration bracket does not bound all clock drift. A critical
path reconstructed from observed intervals is an analysis model, and scheduling
gaps may change when kernel durations change.
