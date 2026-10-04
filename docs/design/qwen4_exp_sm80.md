# Qwen3.8-Flash-Next NVFP4 on two CMP 170HX GPUs

The local optimization uses the same RadixArk NVFP4 checkpoint as the
RTX PRO 6000 comparison. Public results were researched before starting the
dual-card measurements. Quantization, request count, MTP settings, generated
content, and timing definitions differ across those reports; their headline
rates are reference points rather than interchangeable benchmarks.
The selected local PP2/MTP3 preset improves aggregate throughput by
4.1–145.0% across six repeated serving scenarios. Baseline and final both
score 63/64 on the same GSM8K sample.

## Local measurement hardware and provenance

The local machine has an AMD EPYC 7532, approximately 503 GiB of host RAM,
and two selected 64 GiB CMP 170HX GPUs (physical GPU IDs 0 and 2, SM80).
Both selected cards negotiate PCIe 2.0 x16, with GPU peer access unavailable.
The power limit is 260 W per card and the NVIDIA driver is 610.57.04.

The optimization branch is `codex/qwen38-nvfp4-dual170hx`, based on
`ace010914a`, which contains the preceding PRO 6000 optimizations. Public
references below were checked on 2026-10-03. They do not establish local
throughput or qualify a local launch configuration.

## Public reference: the same RadixArk NVFP4 checkpoint

[Qwen-Flash-SM80-170HX](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX)
documents historical measurements with
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, two CMP 170HX cards, TEP2, MTP6,
GDS PLE reads, INT8 draft scoring, and BF16 candidate checks.
Its [performance record](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/11421471c050cccc036b004e91345583e33993b2/docs/%E6%80%A7%E8%83%BD%E8%AE%B0%E5%BD%95.md)
and [context results](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/11421471c050cccc036b004e91345583e33993b2/docs/context-benchmark-20260908.json)
report the following single-request runs on 2026-09-08:

| Actual input tokens | Output tokens | Decode tok/s | First output, s |
| ---: | ---: | ---: | ---: |
| 8,193 | 28,134 | 168.29 | 4.12 |
| 16,385 | 37,308 | 152.99 | 7.87 |
| 32,769 | 27,132 | 156.44 | 15.84 |
| 65,537 | 31,032 | 159.88 | 32.19 |
| 131,073 | 31,143 | 151.05 | 63.63 |

Each input length has one run, with independent cache salts. The task generates
a website after padded design notes, with thinking enabled at template-default
`xhigh`, temperature 1, top-p 0.95, top-k 20, presence penalty 1.5, and no fixed
seed. Outputs terminate naturally under a 50,000-token cap. Decode includes
reasoning and content tokens divided by first-to-last nonempty stream arrival;
the reported prefill rate is input tokens divided by first-output delay.

A separate [short-prompt record](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/11421471c050cccc036b004e91345583e33993b2/docs/benchmark-summary.json)
uses temperature 0 and seed 0: 29,010 output tokens at 170.826 tok/s, with
57.646% draft acceptance, in one run.
The [reference hardware](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/11421471c050cccc036b004e91345583e33993b2/docs/REFERENCE_HARDWARE.md)
has an EPYC 7532, 32 GiB host RAM, and an asymmetric PCIe 2.0 x16/x4 GPU
connection with P2P available. The performance record gives approximately
1.66 GB/s P2P bandwidth. Its GDS path reads PLE data from NVMe and broadcasts
the requested rows between ranks.

### Latest public code is separate from the historical speed results

The checked repository HEAD is
`11421471c050cccc036b004e91345583e33993b2`.
Version 0.2.0 uses NVIDIA's NVFP4 checkpoint and adds history speculation and
PCIe IPC communication optimizations. Its
[2026-10-02 release validation](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/11421471c050cccc036b004e91345583e33993b2/docs/RELEASE_VALIDATION.en.md)
covers functionality and long contexts, but publishes no new generation
throughput. The repository explicitly separates the historical RadixArk
measurements from v0.2.0. Copying already present history is also excluded from
generation-from-scratch benchmarks.

## Public references with different quantization

These results describe useful SM80 deployment choices, but use different
weights from the local RadixArk NVFP4 checkpoint.

| Checkpoint format | Parallelism / MTP | Reported performance | Primary source |
| --- | --- | --- | --- |
| W4A16-FP8PLE | PP2 / 3 draft tokens | 186.93 output tok/s at **8 concurrent requests** | [Author's 2026-09-22 post](https://x.com/MoonlitMaven/status/2102471011882975451) |
| FP6 experts plus INT8 dense weights | TP2 / 1 draft token | 60.9 tok/s single-stream median; approximately 61/85/101 aggregate at 1/2/3 streams | [Checkpoint card](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8) |
| W4A16 AutoRound | PP2, split 26/22 / 4 draft tokens | Single-stream medians 116.9/104.8/116.5 tok/s at 32K/128K/256K | [Deployment and measurements](https://github.com/gavinxym/170hx-2-qwen3.8-flash-next/blob/main/README.md) |

The W4A16-FP8PLE author uses two 64 GB cards over PCIe Gen2 x1, an i5-8500T,
38.04 GiB RAM, and vLLM 0.29.0. The same post reports 4,770.57 input tok/s for
cold 262,052-token prefill and 121,035.78 effective input tok/s for a cached
prefix. Those are distinct workloads: 186.93 is neither single-stream decode
nor a demonstrated NVFP4 rate. Direct X retrieval returned HTTP 403; the
author's public text, timestamp, and identity were retrieved through the
[public tweet mirror API](https://api.fxtwitter.com/MoonlitMaven/status/2102471011882975451).

The FP6-INT8 report uses a SGLang fork, PCIe Gen2 x4 without P2P, BF16 KV,
and approximately 50 GiB of pinned host memory for PLE. Its 53K-prompt
end-to-end prefill rate is 1,126 tok/s. The AutoRound report also has no P2P,
uses 128 GB RAM, and measures 7/6/6 single-stream trials at the three context
lengths. Its authors caution against quoting the 164 tok/s best individual
run; raw CSV and logs are not published.

## Comparison scope

Local comparisons must preserve the checkpoint and tokenizer, distinguish
per-request decode from aggregate throughput, and report first-token delay.
The existing serving harness provides repeated fixed-output measurements at
C1/C4 with short, 8K, and 60K inputs. A direct reproduction of the historical
RadixArk report additionally needs its webpage prompt, thinking and sampling
settings, long natural outputs, and MTP6 configuration. Changes in generated
content and acceptance rate can materially change speculative decode speed.

## Formal serving results

On the local pair without P2P, the selected PP2/MTP3 preset improves median
aggregate throughput in all six scenarios by 4.1–145.0% over the TP2/MTP3
baseline. Each scenario has three formal repetitions with identical inputs
between baseline and final, and exactly 1,024 output tokens per request.
All rates below are output tokens/s. Decode and first-token columns show
baseline / final values; first-token delays are in seconds.

| Input | Concurrency | Baseline aggregate | Final aggregate | Improvement | Decode, baseline / final | First token, baseline / final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 136 | 1 | 152.44 | 158.69 | 4.1% | 157.08 / 163.15 | 0.179 / 0.182 |
| 136 | 4 | 366.88 | 437.02 | 19.1% | 95.57 / 114.98 | 0.393 / 0.339 |
| 8,192 | 1 | 109.00 | 135.45 | 24.3% | 151.20 / 161.35 | 2.630 / 1.218 |
| 8,192 | 4 | 193.82 | 308.74 | 59.3% | 84.03 / 108.54 | 8.859 / 3.763 |
| 60,000 | 1 | 39.87 | 77.09 | 93.4% | 155.66 / 161.01 | 19.084 / 6.933 |
| 60,000 | 4 | 47.48 | 116.30 | 145.0% | 44.85 / 72.66 | 57.997 / 20.015 |

The larger long-input gains combine faster prefill with improved decode.
Short single-request first-token delay is approximately unchanged and
increases by about 4 ms in the median. The public historical NVFP4
single-stream decode rates are useful context, but use different generation
tasks, output lengths, sampling settings, and P2P-capable hardware. These
local numbers do not establish superiority under that public protocol.
Baseline and final GSM8K runs both score 63/64 (98.44%), with the same
incorrect case, ID 12. Full generated answers are retained. This is a
64-case quality check, not a claim of complete output equivalence.

## Implementation

SM80 keeps the checkpoint's NVFP4 experts on the existing Marlin W4A16 path.
Dense projections, KV, and Mamba states use BF16. A separate SM80 dispatch
table selects 25 measured BF16 GEMM configurations across 11 shapes, including
HC, GDN and QSA projections, and the reduced draft head. Other batch sizes
retain the existing linear implementation. SM90, SM103, and SM120 dispatch
tables are unchanged.

The CuTe skinny GEMM uses a warp leader predicate on SM80 instead of the
SM90+ warp-election instruction. The existing warp reduction, FP32 partial
storage, CTA barrier, and final reduction remain in place. Offline SM80
PTX/SASS inspection found no warp-election instructions or local-memory
spills in the inspected HC configurations.

FR-Spec now supports TP2/PP1 and TP1/PP2 in addition to TP1/PP1. TP2 gathers
the original BF16 head once during loading, selects the global token map,
then distributes the reduced head evenly across ranks. This avoids severely
unbalanced rank-local slices: the measured 65,536-token map selects 65,432
IDs from the first original vocabulary shard and only 104 from the second.
The temporary full head is released before KV allocation. The target head
is unchanged, and the existing distributed logits and local-argmax paths
operate on the reduced head with a shared global draft-to-target mapping.

The TP2 path requires a standard contiguous, unquantized BF16 vocabulary
head with no added tokens or alternative TP group. Unsupported head layouts
fail during loading. The PP2 path installs the reduced head on the last
pipeline stage. Target weight reloads require restarting the server to
rebuild the draft-head snapshot.

## Kernel measurements and correctness

The extended existing kernel harness covers TP2 and PP2 geometry. It uses
CUPTI measurements for both repeatedly reused tensors and rotated cold
tensors, checks candidates against an independent FP32 reference, and
checks selected configurations with changed inputs under CUDA graphs.
Of 48 measured cases, 25 configurations improved both timing modes by more
than 5%. Representative cold-tensor timings on CMP 170HX are:

| Shape N × K | Rows | Existing, µs | Selected CuTe, µs |
| --- | ---: | ---: | ---: |
| HC 336 × 10240 | 1 | 19.89 | 9.81 |
| GDN 8192 × 2560 | 1 | 44.43 | 33.87 |
| QSA 6656 × 2560 | 1 | 38.94 | 28.80 |
| TP2 reduced head 32768 × 2560 | 4 | 151.79 | 128.98 |
| PP2 reduced head 65536 × 2560 | 4 | 278.32 | 238.85 |

The BF16 kernel suite passed 41 SM80 cases, with 95 architecture skips and
390 unrelated cases deselected. The SM120 regression run passed 48 cases,
with 56 architecture skips and 422 unrelated cases deselected. Native TP2
NCCL checks on physical GPUs 0 and 2 passed 24 CUDA-graph replays with three
token maps, padding, ties, and changed inputs. Both ranks matched independent
full/reduced BF16 head references; target weights and pointers stayed intact.

The CPU model configuration suite passed 64 cases with two CUDA-only skips.
The reduced-head installation and mapping subset passed 22 cases after the
final installation change. Normal pre-commit hooks and manual Python 3.12
mypy checks passed on the production and test changes.

GPU suites can be reproduced with:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest \
  tests/kernels/test_bf16_skinny_gemm.py \
  -k 'qwen4_exp_selected_shapes or cute_residual_epilogue_all_supported_token_counts' -q
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/kernels/test_bf16_skinny_gemm.py -k qwen4_exp_selected_shapes -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_config.py -q
```

MXFP8 dense projections were also measured through Marlin W8A16. They
improved small-row decode projections by approximately 16–32%, but made
4,096-row prefill projections 1.57–1.66 times slower. They are not part of
the serving configuration.

## Serving protocol and candidate selection

The baseline runs `ace010914a` with TP2, MTP3, BF16 KV and Mamba states,
CPU PLE offload, and no reduced draft head or local-argmax flag. Its workers
imported source before the SM80 edits. The final configuration uses PP2,
MTP3, the same checkpoint and state types, and the reduced BF16 draft head.
Both use a 65,536-token limit, four request slots, an 8,192-token batched
budget, 4,096-token prefill chunks, and 0.94 GPU memory utilization.

The existing serving harness measures exact inputs of 136/8,192/60,000 tokens
at concurrency 1/4, with 1,024 output tokens, temperature 0, ignored EOS, and
thinking disabled. Each request has an independent prefix within the first
cache block. Two priming repetitions precede three formal repetitions;
priming and formal runs use separate salts. Tables report the median of
the three formal repetitions. Baseline and final formal runs use exactly
the same salted inputs (`baseline-steady`). No other GPU workloads run
during measurements.

Aggregate throughput divides all output tokens by complete group wall time,
including prefill. Per-request decode uses output tokens after the first
token divided by time from the first nonempty stream chunk to completion.
Speculative multi-token delivery in the first chunk can slightly bias this
stream-based decode estimate. First-token delay is measured at the client.
Quality uses the same first 64 GSM8K cases as the PRO 6000 comparison, with
temperature 0, thinking disabled, four concurrent requests, and a 2,048-token
output cap.

Initial two-repetition pilots compared the following configurations:

| Configuration | Short C1 aggregate | Short C4 aggregate | 8K C4 aggregate |
| --- | ---: | ---: | ---: |
| TP2 + EP, MTP6, reduced head, 25 SM80 plans | 140.76 | 353.78 | 187.00 |
| TP2, MTP3, reduced head, 25 SM80 plans | 159.04 | 383.28 | 197.69 |
| PP2, MTP3, reduced head, 25 SM80 plans | 162.69 | 427.32 | 307.04 |

The TP2+EP MTP6 candidate lost to the baseline. Its benchmark-window
position acceptance rates were approximately
`[0.784, 0.587, 0.419, 0.292, 0.212, 0.149]`.
The final three draft steps add only 0.653 expected accepted tokens, while
requiring three extra serial draft passes. PP2/MTP3 was selected for formal
measurement after improving all four short/8K pilot scenarios.

A preceding PP2/MTP6 diagnostic imported only the first HC SM80 plan,
before the full dispatch table was installed. It is not a comparison of
the fully tuned PP2 implementation. Its separate 8K/C4 profile with 64
output tokens placed all draft work on the last stage; NCCL durations on
the first stage included waits for that work. Optional profiler-stop
acknowledgement did not complete before server shutdown, so its client
exited nonzero after all eight timed pilot groups and trace files had been
saved. Formal runs do not invoke profiling.

## Launch and reproduce

The measured preset is
`examples/online_serving/qwen38_nvfp4_dual170hx.sh`:

```bash
CUDA_VISIBLE_DEVICES=0,2 bash examples/online_serving/qwen38_nvfp4_dual170hx.sh \
  /home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --host 127.0.0.1 --port 18940
```

Set `MTP_TOKEN_MAP` if the SG repository is not in the default sibling
directory. Set `CUDA_VISIBLE_DEVICES` to the two CMP 170HX cards on another
host. The script defaults to local physical IDs 0 and 2 and rejects lists
that do not contain exactly two devices. The measured map SHA256 is
`becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25`.

The formal measurement command is:

```bash
.venv/bin/python benchmarks/benchmark_qwen4_exp_serving.py \
  --tokenizer /home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --url http://127.0.0.1:18940 --contexts 136 8192 60000 --concurrency 1 4 \
  --repeats 3 --prompt-tag baseline-steady --output /tmp/final-steady.json
```

Before it, run the same command with `--repeats 2`,
`--prompt-tag baseline-prime`, and a separate output path.
To reproduce quality scoring:

```bash
.venv/bin/python benchmarks/qwen4_exp_gsm8k_eval.py \
  --tokenizer /home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --dataset docs/validation/qwen38-dual170hx-20261003/eval-inputs.json \
  --url http://127.0.0.1:18940 --output /tmp/final-eval.json
```

Raw measurements, generated answers, timing ranges, source hashes, launch
commands, hardware logs, primary-source snapshots, compiler inspection,
and test logs are preserved locally under
`docs/validation/qwen38-dual170hx-20261003/`. This directory is ignored by
Git. The GSM8K input SHA256 is
`6b1f6f61b27ee430f28dceb4aeb1cd83399db50e2c88bb655222b2643e7915af`.
Source hashes captured before the final preset starts must match the final
committed Python files and preset. Microbenchmarks use physical GPU 3 in
isolation; native TP2 checks use GPUs 0 and 2 outside serving measurements.
The SM120 regression run uses physical GPU 1.

## Adaptive draft validation (2026-10-04)

Scheduler-owned adaptive MTP was measured on the same physical CMP 170HX
GPUs 0 and 2 with TP1/PP2 and native MRv2. Three fresh servers used the
runtime source at `23520ce082`: fixed MTP3, adaptive maximum 3, and adaptive
maximum 6. The model, FR-Spec map, BF16 state/KV types, CPU PLE offload,
memory utilization and serving limits were identical. The maximum-6 run
used capture sizes `[1,2,3,4,6,7,8,12,14,16,21,28]`; maximum 3 retained
`[1,2,3,4,6,8,12,16]`. A CPU dispatch check covered all 48 target shapes
(two stages, C1–C4, K1–K6) and eight draft decode shapes.

Native scheduler logs confirm completed verification rounds for every K
from 1 to 3 in the maximum-3 run and from 1 to 6 in the maximum-6 run,
without forced budgets. These counters accumulate over the whole process,
including priming, performance, quality and diagnostics; they are not
per-scenario frequencies. The focused CPU regression selection passed
103 tests, covering budget/cost feedback, changing draft work, graph
routing, pipeline output fences and cancellation bookkeeping.

All three servers scored 63/64 on the same GSM8K subset, with the same
incorrect case 12 and identical extracted numeric values. Maximum 6
formatted case 58 as `57.00` instead of `57`. Each server also passed
12/12 serving diagnostics: exact copying, arithmetic, typed JSON, mixed
prompt lengths, concurrent and staggered arrivals, one seeded
nonzero-temperature case, and a fresh request after a streaming client
disconnected. All 12 diagnostic outputs matched between configurations.
The disconnect/recovery check does not independently prove internal abort
completion; one seeded case does not establish sampling-distribution
equivalence.

All 45 long-form greedy benchmark outputs differed from fresh fixed 3.
The preceding fixed-3 run also differed from fresh fixed 3 on all 45
matched prompts. GSM8K explanations differed on 41/64 cases for adaptive
maximum 3 and 40/64 for maximum 6. Generated text and output-prefix
comparisons are retained. The quality result is bounded evidence for this
subset, not a claim of byte-identical output or complete model equivalence.

Each server ran two priming repetitions and three formal repetitions of
all six scenarios, with 1,024 output tokens per request. Prompt tags and
tokenized inputs matched across servers. Every timed request changed its
first cache block within and across phases. The table reports aggregate
output throughput, including prefill, against the fresh fixed-MTP3
reference from this session; the earlier measurement above is preserved.

| Input tokens | Concurrency | Fresh fixed 3 | Adaptive max 3 | Change | Adaptive max 6 | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 136 | 1 | 164.30 | 159.41 | -2.98% | 157.45 | -4.17% |
| 136 | 4 | 428.44 | 432.50 | +0.95% | 426.34 | -0.49% |
| 8,192 | 1 | 132.58 | 133.98 | +1.06% | 131.01 | -1.18% |
| 8,192 | 4 | 308.13 | 304.17 | -1.29% | 295.35 | -4.15% |
| 60,000 | 1 | 76.14 | 76.13 | -0.02% | 77.22 | +1.42% |
| 60,000 | 4 | 116.86 | 117.57 | +0.61% | 117.30 | +0.38% |

Values are tok/s and changes are relative to fresh fixed 3. These medians
and retained min/max ranges describe three repetitions per scenario;
they are not significance tests. The results do not establish a general
speed improvement from enabling adaptive MTP on this configuration.
Generated text can change acceptance and routing workloads, so these
rates are end-to-end observations, not isolated controller-overhead
measurements.
The launch preset therefore retains fixed MTP3.

To reproduce adaptive maximum 3, append a complete speculative config to
the preset command. Preserve the map and reduction settings:

```bash
CUDA_VISIBLE_DEVICES=0,2 bash examples/online_serving/qwen38_nvfp4_dual170hx.sh \
  /home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --host 127.0.0.1 --port 18940 \
  --speculative-config '{
    "method":"mtp", "num_speculative_tokens":3,
    "enable_adaptive_verification":true,
    "index_share_for_mtp_iteration":false,
    "mtp_token_map":"/home/bul/dev/sglang-rtxpro6000/configs/pennyroyal/frspec/flash-next-64k.pt",
    "use_local_argmax_reduction":true
  }'
```

For adaptive maximum 6, change `num_speculative_tokens` to 6 and append:

```bash
--compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,6,7,8,12,14,16,21,28]}'
```

Use the priming, formal and quality commands from the preceding section
for each fresh server. Raw results, all generated answers, native budget
logs, source/checkpoint/input hashes, exact launch commands, hardware
samples, CPU test selection and serving diagnostic fixtures are retained
under `docs/validation/qwen38-dual170hx-adaptive-20261004/` (ignored by
Git). Its `comparison.json` and `comparison.md` include full repetition
ranges, decode rates and first-token delays. All benchmark and diagnostic
clients exited zero; all three serving processes shut down normally, and
the measurement GPUs were released.
