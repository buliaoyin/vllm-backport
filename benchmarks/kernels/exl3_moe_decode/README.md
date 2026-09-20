# EXL3 multi-expert INT8 decode experiment

This opt-in prototype extends ExLlamaV3's small-M SQ DP4A GEMV to GPU-selected
expert pointer tables. Grid Z indexes token/expert assignments. Gate and up read
the original input rows directly; down produces one FP32 output per assignment,
then a separate kernel applies routing weights and reduces into BF16 output.

Unlike the existing `EXL3_INT8_GEMV` switch, this changes routed MoE decode.
It preserves the mul1 codebook and supports ordinary INT8 activation rounding
(`plain`) or a second DP4A pass for the activation residual (`residual`). It is
separate from the experimental INT8 Tensor Core prefill kernel.

The experiment launcher requires ExLlamaV3 1.4.8, SM80 or SM120, BF16 activations,
1–128 rows (the original production path uses 1–8),
1–8 selected experts per row, uniform 4-bit mul1 weights, and dimensions divisible
by 256 in [256, 8192]. Routes must contain valid expert IDs. The input tensor's
CUDA device must be current. One stream owns each launcher's workspace, matching
the integration's existing single-stream restriction. Other formats and shapes
are rejected by the experiment launcher. The validated hybrid policy is now
bundled in production `_exl3_C` and defaults to `VLLM_EXL3_MOE_DECODE=hybrid`;
production falls back for unsupported configurations. See the
[default validation](../../../docs/validation/exl3-hybrid-default-20260911.md).

## Build and verify

```bash
PYTHONPATH=. .venv/bin/python benchmarks/kernels/exl3_moe_decode/build.py \
  --source /home/bul/dev/exllamav3 \
  --build-dir /tmp/exl3-expert-decode \
  --nvcc /usr/local/cuda/bin/nvcc

CUDA_VISIBLE_DEVICES=0 \
  VLLM_EXL3_TEST_DECODE_LIBRARY=/tmp/exl3-expert-decode/decode.so \
  .venv/bin/python -m pytest tests/quantization/test_exl3.py \
  -k experimental_expert_int8_decode -q
```

The build verifies the pinned upstream revision and extension cleanliness. It
records header/source hashes, compiler details and the binary hash in
`build.json`. Upstream-derived CUDA code retains MIT terms in
`LICENSE-exllamav3`. Repeat correctness checks on SM120 as well. Tests compare
independent rotated dense reconstructions and replay graphs after input and
expert ID changes, including zero activations and changed routing weights.

## Model runs

Use `benchmarks/benchmark_exl3.py` with the worker extension
`kernels.exl3_moe_decode.worker.Exl3DecodeWorkerExtension` and these environment
variables:

```bash
export VLLM_EXL3_EXPERIMENTAL_DECODE=residual
export VLLM_EXL3_DECODE_LIBRARY=/tmp/exl3-expert-decode/decode.so
export PYTHONPATH=/home/bul/dev/vllm-backport/benchmarks:/home/bul/dev/vllm-backport
```

Run `native`, `plain`, `residual` and `hybrid` in **separate processes**, since decode is
captured in CUDA Graphs during model initialization. The extension installs the
requested experiment before model warmup, fails on unsupported configurations,
and reports the selected grid and library hash in worker runtime state. Its
Python call counts include warmup/capture and do not count CUDA Graph replays.
`hybrid` uses plain INT8 on SM80 and residual compensation on SM120; runtime
state records the actual activation mode separately from the requested policy.
The experiment worker explicitly selects production `M_TILE=16` and
`MOE_DECODE=native` before installing its overrides, keeping historical controls
independent of the new production defaults.

`EXL3_INT8_GEMV=0` isolates expert decode changes from ordinary linear GEMVs.
Use identical prefill variants, token IDs, layer placement, KV cache budget,
output lengths and batch sizes. The documented comparison uses pipeline split
`11,11,11,12`, chunk 1024 and an 8 GiB KV budget per GPU.

The optional prefill factories `nobar_m32_k32_n256` and
`half_m32_k32_n256` are separate pipeline ablations. Their source, build metadata
and measured outcomes are preserved with the validation report; the latter also
changes accumulation precision and requires its own model quality evaluation.

Build the prefill ablations separately:

```bash
.venv/bin/python benchmarks/kernels/exl3_moe_decode/build_prefill.py \
  --source /home/bul/dev/exllamav3 \
  --build-dir /tmp/exl3-pipeline \
  --nvcc /usr/local/cuda/bin/nvcc
```

Pass `/tmp/exl3-pipeline/pipeline.so` as `rows_library`, with
`base_m32_k32_n256`, `nobar_m32_k32_n256` or `half_m32_k32_n256` as
`rows_variant`. This builder also exports the unsuccessful lock and M64 variants
for reproducing the ablations. Optional tests use
`VLLM_EXL3_TEST_PIPELINE_LIBRARY=/tmp/exl3-pipeline/pipeline.so` and
`-k experimental_moe_pipeline`.

## Route-based kernel sweep

Reuse the existing checkpoint loader and captured route file:

```bash
CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 \
  .venv/bin/python benchmarks/kernels/benchmark_exl3_moe.py \
  --checkpoint /home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.22.mlp.experts \
  --route-sample /tmp/exl3-opt-20260910/routes/layer-22.pt \
  --decode-kernel-library /tmp/exl3-expert-decode/decode.so \
  --decode-rows 1 2 4 8 --decode-grids 4 8 16 32 64 \
  --output /tmp/exl3-expert-decode.json
```

Both activation modes run between native controls. This times full-wrapper GPU
work, including input conversion, activation and weighted reduction; it excludes
Python dispatch. If the capture comes from prefill, these are small-M slices of
prefill data. Report the capture's origin, and use model runs to validate actual
decode performance. Every sample uses an explicit 256 MiB cache eviction outside
the CUDA Event interval; the raw 15 timings and correctness errors are saved.

The prefill builder also adds a warp barrier after Hadamard scatter consumes its
shared staging buffer. Both the original-fragment-barrier control and the
fragment-barrier ablation produced the same warp access warnings in an SM120
diagnostic racecheck build. The added barrier orders reads before the next
loop iteration overwrites that buffer. `build.json` records this derived header's
hash; the upstream checkout and installed extension are not edited.

## Validated configuration

The final candidate uses `hybrid` decode with the synchronization-fixed
`nobar_m32_k32_n256` prefill on SM80 and native prefill on SM120. At the fixed
11/11/11/12 split, chunk 1024 improves 8K–64K input throughput by 3.8%–4.1% and
output throughput by 12.3%–12.9% over the previous M32 baseline in this setup.
Chunk 2048 improves the final candidate's 64K input throughput by another 11.2%,
but reduces its 8K throughput by 5.9% and doubles the expert workspace. It is a
separate configuration experiment in that report. Later validation made chunk
2048, safe M32 on SM80, and hybrid expert decode the production defaults.

See the [full validation report](../../../docs/validation/exl3-moe-20260911.md)
for all positive and negative results, quality checks, numerical limits, and
matched-layer AWQ/NVFP4 comparisons.

## Batched decode and expert reuse

`batched.py` compares native M32, ordinary DP4A, small-row grouped DP4A,
and the production sparse-INT8/hot-FP16 wrapper. Use actual decode captures:
`benchmark_exl3.py --capture-routing-max-rows 128 --profile-case b16
--profile-tokens 24 --capture-routing-dir /tmp/exl3-decode-routes`.
The capture temporarily disables target-model CUDA Graph replay after performance
measurements and saves up to four batches with their original row counts.
Concatenated captures are a shape probe; they are not a single scheduler batch.

After building the production component, no experimental library is needed:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.kernels.exl3_moe_decode.batched \
  --checkpoint /home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw \
  --route-sample /tmp/exl3-decode-routes/layer-22.pt \
  --production --rows 12 16 32 64 128 \
  --output /tmp/exl3-batched-decode.json
```

The microbenchmark invokes the production wrapper directly so it can compare
identical saved routes without a scheduler. Model runs additionally validate
pure-decode dispatch, graph reuse and end-to-end throughput. Timings exclude
allocation and compilation, but include all GPU routing and reduction operations.
The cache-eviction protocol is the same as above; CUPTI is unsupported on CMP 170HX.

For rejected alternatives, `build.py --variant grouped` compiles M2/M4 weight
reuse, and `--variant cold` compiles a tunable sparse/hot threshold prototype.
Each variant also exports the ordinary DP4A factories and returns `decode.so`;
pass the corresponding libraries as `--library`, `--grouped-library` or
`--cold-library`. Use separate build directories. The sweep reverses candidate
order in the second timing round and records every sample and relative L2 error.

For a paired comparison of the hot-expert fetch pipeline, add
`--production --k32-control` to `batched.py`. The `production_*_k32` variants
keep the same cold DP4A path and use the original K32 fetch for hot experts.
The regular production variants use K64 fetch for SM80 decode with
hidden/intermediate dimensions 4096/2048. Compare the complete wrapper;
the Tensor Core computation and FP32 accumulation order are unchanged.

Add `--uncompacted-control` to retain the original cold-expert scheduling.
Together with `--k32-control`, `production_plain_uncompacted_k32` compares
against the complete original batched-decode path using the same inputs,
precision, stream and workspace. The two controls also separate cold scheduling
from hot-expert pipeline changes.
