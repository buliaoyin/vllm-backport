# Qwen3.8-Flash-Next NVFP4 on RTX PRO 6000

The Qwen4Exp NVIDIA implementation has measured SM120 BF16 projection plans,
FP8 E4M3 main KV cache support, overlapping QSA projections, and exact metadata
updates inside fused multi-step MTP CUDA graphs. An optional token map reduces
the draft output head to frequently used target token IDs; the target head and
input embeddings retain the full vocabulary.

## Launch

From the repository root, with its editable installation:

```bash
CUDA_VISIBLE_DEVICES=1 bash examples/online_serving/qwen38_nvfp4_pro6000.sh \
  /path/to/Qwen3.8-Flash-Next-NVFP4 --host 127.0.0.1 --port 18938
```

The preset finds the FR-Spec map in the sibling `sglang-rtxpro6000` checkout;
set `MTP_TOKEN_MAP` to use another location. The measured map contains 65536
unique target IDs and has SHA256
`becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25`.
It comes from `configs/pennyroyal/frspec/flash-next-64k.pt` at SGLang remote
revision `12846e83153f` and is specific to this checkpoint/tokenizer.

The script uses BF16 model dtype and GDN states, FP8 E4M3 main KV storage, CPU
PLE embedding offload, three MTP draft tokens,
a 8192-token batch budget with 4096-token chunks per long prefill, and four
concurrent requests. Its default maximum
context is 65536 tokens;
`MAX_MODEL_LEN` overrides it. SM120 GEMM selection happens automatically for
the measured projection shapes and token counts. Other shapes retain their
existing implementation.
The preset leaves Torch CPU thread counts to vLLM: weight loading may use a
bounded worker pool, then serving switches to one thread. External
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` overrides are honored.

The preset opts into `sm120_rowwise_fp8_head`, `sm120_rowwise_fp8_hc`, and
`sm120_rowwise_fp8_output`
through `--hf-overrides`. The output head and main-model HC projections use
rowwise E4M3 weights with FP32 scales and BF16 activations. The four injection
rows in each merged HC down projection retain their original BF16 weights;
the draft backbone retains BF16 HC. Online MXFP8 is enabled only for GDN
`in_proj_qkvz` and QSA
`qkv_proj`. The main-model GDN `out_proj` and QSA `o_proj` use resident rowwise
FP8 weights: at most 32 rows use BF16 activations and FP32 split-K accumulation;
larger batches quantize activations per token and use the SM120 CUTLASS FP8
GEMM with FP32 activation/weight scales and BF16 output. This adds activation
quantization during prefill. The draft attention outputs stay BF16; remaining
dense projections retain their measured BF16 plans.
The reduced draft head selects both weight rows and their matching scales.
The flag requires exactly SM120, BF16 model/head computation, TP1/PP1, untied
embeddings, and no LoRA. Model construction otherwise retains the original
head implementation.

The reduced draft head is a snapshot of the serving checkpoint. Restart after
updating target weights to rebuild it and its scales. The full target head
supports standard layerwise weight reload; direct writes of BF16 checkpoint
weights into resident FP8 storage are rejected.

Optional adaptive verification calibrates draft budgets 1, 2, and 3 using
observed acceptance and complete step costs. Its selected budget changes both
draft generation and verification; it uses separately captured draft graphs
for each budget. When enabled, calibration happens during the priming requests. Periodic
probes now remain active until a stable same-budget timing sample completes;
one-step probes previously discarded both transitions and could not refresh
an already measured budget's cost. The CPU regression covers synchronous and
delayed feedback.

The measured configuration disables `index_share_for_mtp_iteration`. Draft
metadata updates rebuild sparse cache positions from live GPU sequence lengths,
request mappings, and block tables without allocating new metadata buffers.
The auxiliary indexer stream joins before attention reads its selection. An
online-quantized indexer uses the serial path because FlashInfer MXFP8 scratch
storage is shared across streams.

FP8 KV reads use the same per-tensor K/V scales as native cache writes. Queries
and attention dot products remain BF16. Per-head cache quantization is outside
this implementation's supported configurations.

## Measurement method

The local comparison uses one RTX PRO 6000 Blackwell Workstation Edition,
TP=1, a 500W power limit, and the same RadixArk NVFP4 checkpoint and tokenizer.
No other GPU workload runs during measurements. Model loading, JIT compilation,
autotuning, and CUDA graph capture are outside the timed requests.

The vLLM baseline starts at `6ea416cd8c` with BF16 KV storage and three MTP draft
tokens. The SGLang checkout was updated from `77890db7ae4` to merge commit
`eb06aa87ebf5f858d963dd0656bd6d8456919608`, incorporating remote branch
`pennyroyal-main-sm120-final` at
`12846e83153ff1778cb899f00d678beaac6ee53f` from
<https://github.com/jpezzulli/sglang-rtxpro6000>. The local request-abort fix was
retained and its 34 CPU regression tests passed. A backup branch,
`codex/before-remote-update-20261003`, preserves the original checkout.
The final remote check found `f9c4a89cd87b62751decfd732db7250f6315020b` and
merged it locally as `d3908187ff422c63c7699e9dfb102157f6072b6e`. That commit
changes only GitHub traffic reporting: the inference Python tree, kernels,
serving configurations, dependency file, and token map have identical Git
object hashes to the measured `eb06aa87` checkout. The measurements below
therefore cover the current inference code; the recorded launched commit
remains `eb06aa87`. The traffic-report suite also passed 15 tests.
The latest comparison uses native NEXTN index sharing, FP8 KV, FlashInfer linear
attention, BF16 Mamba states, CPU PLE offload, a 24-slot Mamba pool, and its
reference 0.981 static memory fraction, online MXFP8, and the 65536-entry
FR-Spec token map. The original vLLM baseline uses a 0.94 memory fraction;
the final preset also uses 0.94, leaving space for fresh FlashInfer autotuning.
Both engines use a 65536-token maximum context,
four request slots, and 4096-token chunks per long prefill. The final vLLM
batch budget allows two such chunks in one step. Hierarchical storage is
omitted from this GPU serving comparison.

| Software | vLLM | SGLang |
| --- | --- | --- |
| PyTorch | 2.13.0 + cu130 | 2.13.0 + cu130 |
| FlashInfer | 0.6.18.post1 | 0.6.17 |
| Transformers | 5.16.1 | 5.12.1 |

The serving benchmark sends exactly 136, 8192, or 60000 input tokens and requests
1024 output tokens at temperature zero with EOS ignored. Thinking is disabled.
Every request has a distinct prefix within the first cache block. Priming and
formal measurement use different prefix tags. After two priming repetitions,
each scenario is repeated three times; reported values are medians. HTTP
connections close after every request, avoiding idle keepalive expiry during
long concurrent groups. Earlier runs with shared prefixes are pilot artifacts
and are excluded from the final comparison.

Aggregate throughput divides all generated tokens by the complete group wall
time, including prefill. Per-request decode throughput starts at the first
content event and uses `output_tokens - 1`; MTP can deliver multiple tokens in
that first event, producing a small streaming bias. Concurrent decode includes
interruptions while other requests prefill. The raw JSON preserves
every request's token counts, timings, and generated text.

```bash
.venv/bin/python benchmarks/benchmark_qwen4_exp_serving.py \
  --tokenizer /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --url http://127.0.0.1:18938 --output /tmp/qwen38-serving.json
```

## Serving results

These are aggregate output tokens/second, including prefill, from three
independent-prefix repetitions per scenario. All six final medians are at
least as high as the latest SGLang inference-code medians. The short C4
case is effectively tied: its 0.35% median difference is small relative to
run-to-run variation. The result applies to the launch preset and measured
workloads, rather than guaranteeing parity for every workload or latency metric.

| Input tokens | Concurrent requests | Original vLLM | Latest SGLang | Final vLLM | vs original | vs SGLang |
| --- | --- | --- | --- | --- | --- | --- |
| 136 | 1 | 178.61 | 233.61 | 251.77 | +40.96% | +7.77% |
| 136 | 4 | 543.13 | 642.70 | 644.92 | +18.74% | +0.35% |
| 8192 | 1 | 163.79 | 208.13 | 223.98 | +36.75% | +7.62% |
| 8192 | 4 | 408.76 | 313.64 | 491.77 | +20.31% | +56.80% |
| 60000 | 1 | 98.69 | 117.03 | 121.38 | +22.99% | +3.71% |
| 60000 | 4 | 95.34 | 140.59 | 178.41 | +87.13% | +26.90% |

The final short C4 aggregate range is 644.54–655.87 tokens/s, versus
641.44–649.74 for SGLang. Full ranges for all scenarios are in `summary.json`.
Per-request decode and median first-content latency are shown separately:

| Input tokens | Concurrent requests | SGLang decode tok/s | Final decode tok/s | SGLang TTFT s | Final TTFT s |
| --- | --- | --- | --- | --- | --- |
| 136 | 1 | 247.42 | 265.41 | 0.246 | 0.219 |
| 136 | 4 | 176.56 | 174.77 | 0.463 | 0.412 |
| 8192 | 1 | 238.08 | 262.04 | 0.629 | 0.668 |
| 8192 | 4 | 148.43 | 164.04 | 1.678 | 2.048 |
| 60000 | 1 | 245.29 | 259.56 | 4.563 | 4.504 |
| 60000 | 4 | 92.93 | 119.40 | 11.657 | 13.031 |

Short C4 per-request decode is about 1% below SGLang while aggregate
throughput is tied; the streaming first-event bias described above also
affects this metric. First-content latency is higher for 8K/C1, 8K/C4 and 60K/C4,
even though their complete-group throughput is higher. Throughput parity
does not imply that every latency metric improves.

The final preset retains fixed three-step MTP. Adaptive verification,
different QSA split counts, alternate MoE backends, rowwise QKV, and shared
expert FP8 candidates were measured separately and remain unselected.
Native vLLM reduces Torch CPU threads from 32 at loading to one for serving;
the preset no longer forces four threads past that transition. Its final
FlashInfer cache has 126 configurations. A fresh-cache start also passed
at the selected 0.94 memory fraction; an earlier 0.97 fresh-autotune start
ran out of memory and is retained as a failed pilot.

## Kernel validation

The SM120 GEMM sweep checks candidate outputs against FP32 linear results and
times graph replay with both hot and flushed L2 caches using CUPTI. Only
candidates faster in both regimes are selected; the table has 48 points across
14 projection shapes. Selected plans are also checked after changing inputs
between CUDA graph replays.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/benchmark_qwen4_exp_skinny_gemm.py \
  --output /tmp/qwen4-exp-sm120-gemm.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/kernels/test_bf16_skinny_gemm.py -k qwen4_exp -v

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_qsa_reference.py -v
```

The QSA reference suite covers BF16 and native FP8 cache
writes with distinct non-unit K/V scales, empty selections, page boundaries,
MTP request compaction, padded requests, and live sequence lengths during
CUDA graph replay. The selected GEMM validation passed 54 tests, with 31
SM90-only cases skipped.

The rowwise FP8 head suite passed 15 CUDA graph cases, including empty inputs,
odd dimensions, complete and reduced vocabulary shapes, and the bounded
large-batch fallback. Six production head cases passed independent FP32
quantized oracles. Cold-L2 full-head latency falls from 796–823 to 414–434
microseconds; reduced draft-head latency falls from 229–235 to 112–118
microseconds for batch sizes 1, 4, and 16. These quantify projection latency;
the serving table reports the combined effect.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/benchmark_qwen4_exp_skinny_gemm.py \
  --rowwise-fp8-head --projections lm_head draft_lm_head --tokens 1 4 16 \
  --output /tmp/qwen38-head.json
```

The HC down kernels use per-output-row reduction for M1 and 16-way split-K
for M4/M16. All nine HC projection cases passed quantized FP32 oracles and
changed-input graph replay. Cold-L2 merged down latency improves from
7.18/8.02/10.35 to 5.25/6.88/6.85 microseconds at M1/M4/M16; up latency improves
from 6.38/8.11/7.90 to 5.20/5.49/5.70 microseconds. The original BF16 injection
rows participate in the same projection, with unit scales. Generated SM120
SIMT PTX has no local loads/stores, and the kernel uses 64 bytes of shared memory.
Above 32 rows, the measured HC up shape uses a two-dimensional Triton GEMM
with FP32 accumulation and a fused row-scale epilogue before the BF16 output
store. It avoids FP32 output temporaries and output-driven GEMM chunks.
High-K HC down projections, CPU execution, and the head retain FP32-output
GEMM before scaling. This avoids BF16 reduced-precision reduction errors
without changing process-wide PyTorch precision settings.
The selected HC up tile is M64/N128/K64, four warps and three stages. At
M4096/M8192 its cold latency is 102.64/197.41 microseconds, compared with
317.02/588.67 for the original FP32-output FP8 fallback and 95.46/179.31 for
the original BF16 weights. The final six prefill cases and 31 head/HC graph
regressions passed. Compiled-kernel metadata reports 188 registers, no spills,
32 KiB of shared memory, and no global scratch; PTX has no local loads/stores.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/benchmark_qwen4_exp_skinny_gemm.py \
  --rowwise-fp8-hc --projections hc_down_inject hc_final_down hc_up \
  --tokens 1 4 16 --output /tmp/qwen38-hc.json
```

The online MXFP8 sweep includes activation quantization, changed-input graph
replay, and FlashInfer autotuning. All 42 cases passed correctness checks.
The selected GDN projection improves from 55.7 to 32.1 microseconds at M1
and 841.6 to 521.1 microseconds at M4096; QSA QKV improves from 45.6 to 27.2
and 687.5 to 430.7 microseconds, respectively, with cold L2. Default untuned
tactics obscured these gains; the existing startup autotuning must stay enabled.
Indexer and shared-expert projections retain BF16 based on the sweep.
CUTLASS NVFP4 MoE is retained because Marlin improved small decode
shapes but made the 4096-token prefill case 1.62 times slower.

The attention output shape `(2560, 6144)` benefits from a separate rowwise
FP8 path. The cold-L2 decode sweep measured BF16 latencies of
21.94/22.77/30.90/31.12 microseconds at M1/M4/M12/M16 versus approximately
14 microseconds with eight-way FP32 split-K and the fused row-scale reduction.
Keeping BF16 activations for large batches made this projection slower, so
the prefill path instead uses per-token FP8 activation scales and the native
SM120 CUTLASS scaled GEMM. Including activation quantization, its cold latency
at M4096/M8192 is 238.85/432.22 microseconds versus 339.89/655.55 for BF16.
Both paths pass independent FP32 quantized oracles and changed-input graph
replay. The prefill projection's relative L2 difference from the original
BF16 weights/activations is approximately 3.74% on these random inputs; the
model regression below checks the combined quantization changes.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/benchmark_qwen4_exp_skinny_gemm.py \
  --rowwise-fp8-output --projections attn_out --tokens 1 4 12 16 33 4096 8192 \
  --output /tmp/qwen38-attention-output.json
```

SM120 FP8 attention uses one pipeline stage in both warmup and runtime.
The block-64 prefill profile requires 106496 bytes of shared memory with two
stages, exceeding the device's 101376-byte limit; one stage uses 73728 bytes.
Regression cases include production page size 1664 and selection widths 2051
and 2053, as well as full selections spanning multiple attention tiles.

A gather-plus-XQA candidate was evaluated and retained as a measurement
artifact. It lost to direct paged Triton attention in most tested scenarios,
so it is not selected in production.

## Model regression

The local GSM8K check uses the first 64 test questions, zero-shot chat with
thinking disabled, temperature zero, a 2048-token completion limit, and four
concurrent requests. Numeric answers are parsed from the final `####` line.
This is a small regression sample rather than a full model evaluation.

```bash
.venv/bin/python benchmarks/qwen4_exp_gsm8k_eval.py \
  --tokenizer /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --dataset /path/to/local-gsm8k-cases.json \
  --url http://127.0.0.1:18938 --output /tmp/qwen38-eval.json
```

The input JSON contains an `evals` list of objects with `id`, `question`, and
numeric `answer` fields. The saved `eval-inputs.json` has SHA256
`6b1f6f61b27ee430f28dceb4aeb1cd83399db50e2c88bb655222b2643e7915af`.
It also records the source dataset checksum and the checksum of the original
local input file. The original vLLM, latest SGLang, and final vLLM runs all
scored 63/64, missing the same question (ID 12).

## Additional checks and artifacts

The final configuration/reload suite passed 53 CPU tests with two CUDA-only
tests skipped. The FP8 head/HC graph suite passed 31 cases, and the new hybrid
output suite passed 10 cases covering empty inputs, noncontiguous inputs,
M32/M33 dispatch, changed per-token activation scales, and bias behavior.
The QSA suite passed 90 tests, and the adaptive-verification suite passed 79.
The changed files passed normal pre-commit hooks and the manual Python 3.12
mypy hook. No C++ or CUDA extension source changed.

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_config.py -q

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/kernels/test_bf16_skinny_gemm.py -k qwen4_exp_rowwise_fp8 -q

CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_adaptive_verification.py -q

.venv/bin/pre-commit run --files <changed-files>
.venv/bin/pre-commit run mypy-3.12 --hook-stage manual --files <changed-files>
```

Raw benchmark responses, per-scenario ranges, launches, source fingerprints,
kernel measurements, profiler traces, and test logs are retained locally in
`docs/validation/qwen38-pro6000-20261003/`, which is ignored by Git. Failed
pilots and slower candidates remain there with their original labels; only
the independent-prefix baseline, latest SGLang, and final serving runs enter
the comparison. The final metadata records source SHA256 values, checkpoint
configuration/index checksums, token-map checksum, software versions, and
the exact launch configuration.

These results cover TP1/PP1, four concurrent requests, 136/8192/60000-token
inputs and a 65536-token context limit on this checkpoint. They do not validate
the checkpoint's full advertised context length, other GPU architectures,
other tensor-parallel layouts, LoRA, or broader model quality.
