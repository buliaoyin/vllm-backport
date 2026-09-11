# EXL3 SM80 row reuse experiment

This prototype decodes each B fragment once and consumes it with two M=16 A/C
fragments. The `m32_predicated` variant skips the second fragment for a tail of
16 rows or fewer. Gate, up and down keep the existing expert scheduler, rotations,
activation, output scatter and FP32 MMA accumulation. Register fragment staging
is reduced from three to two; shared-memory staging remains three.

The build also exports `m16` (two fragment stages, for the pipeline ablation) and
`m32` (unconditional second row fragment). The native extension supplies the
original M=16, three-stage control.

This is an opt-in benchmark implementation, using the private ExLlamaV3 1.4.8
kernel and DevCtx ABI. It supports only SM80, 4-bit mul1 experts and dimensions
divisible by 256. The launcher rejects unsupported configurations and grids that
cannot remain fully resident. It does not modify the installed extension or the
source checkout. The model worker leaves other architectures on the native path.
The production serving default is unchanged.

## Build

Use the repository's Python environment and a CUDA 13 compiler:

```bash
.venv/bin/python benchmarks/kernels/exl3_m32/build.py \
  --source /home/bul/dev/exllamav3 \
  --build-dir /tmp/exl3-rows-build \
  --nvcc /usr/local/cuda-13.0/bin/nvcc
```

The build directory must be new. The source checkout must be at
`6ff3a17ea7f3d0026b273d43239398d57f71b788` with no tracked extension changes.
Headers are copied before applying `rows.patch`. `build.json` records the source
hashes, compiler command/version and resulting library hash. `build.log` includes
ptxas register and spill statistics. The derivative headers retain ExLlamaV3's
MIT terms in `LICENSE-exllamav3`.

## Correctness

The extra cases reuse the existing dense reconstruction/Hadamard reference and
exercise short row tiles, a hot expert, workspace overflow chunking, and graph
replay with changed inputs and routes:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  VLLM_EXL3_TEST_ROWS_LIBRARY=/tmp/exl3-rows-build/rows.so \
  .venv/bin/python -m pytest tests/quantization/test_exl3.py -q
```

The optional cases skip when the library variable is absent or the GPU is not
SM80. This does not validate other quantization codebooks, bit widths or GPUs.

## Real routing benchmark

Capture routes with `benchmark_exl3.py --capture-routing-dir` or use a preserved
route sample. Pass the matching checkpoint layer and routing sample:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 \
  .venv/bin/python benchmarks/kernels/benchmark_exl3_moe.py \
  --checkpoint /home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.3.mlp.experts \
  --route-sample /tmp/exl3-opt-20260910/routes/layer-3.pt \
  --prefill-rows 512 1024 2048 --prefill-chunks 1024 \
  --row-kernel-library /tmp/exl3-rows-build/rows.so \
  --output /tmp/exl3-rows-layer3.json
```

Timing covers the complete expert wrapper's GPU work in one CUDA graph replay.
It excludes Python dispatch. A 256 MiB eviction buffer is written before each
sample, outside the measured CUDA events. The native control runs before and
after the candidates. Every candidate is checked against the native output.

## Same-engine model comparison

Use `benchmark_exl3.py` with:

- `--worker-extension-cls kernels.exl3_m32.worker.Exl3RowsWorkerExtension`
- `--moe-variants` pointing to the JSON below
- `VLLM_PP_LAYER_PARTITION=11,11,11,12`
- `--pp 4 --expected-layer-counts 11 11 11 12`

```json
[
  {"name": "native_before", "capacity": 1024, "priority": true},
  {
    "name": "m32_sm80", "capacity": 1024, "priority": true,
    "rows_library": "/tmp/exl3-rows-build/rows.so",
    "rows_variant": "m32_predicated"
  },
  {"name": "native_after", "capacity": 1024, "priority": true}
]
```

Use identical token IDs, chunk size, KV allocation and output length for all
formats. Runtime state records the actual decoder layer IDs and the library
installed on each rank. Rank-local operator times are diagnostic intervals;
their sum is not request TTFT.
