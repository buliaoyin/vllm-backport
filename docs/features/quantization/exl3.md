# EXL3

EXL3 is a weight-only quantization backend using the MIT-licensed
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

GLM5Next MTP can load quantized draft weights from `mtp.safetensors` alongside
an EXL3 checkpoint, including when the main shard index omits that file. The
checkpoint directory does not need modification. MTP runs on the final pipeline
rank and shares the target output head; under PP it loads an additional embedding
on that rank.

Qwen3.5 MTP also loads EXL3 `mtp.*` weights from the main shards, including the
Qwen3.8-27B dense checkpoint. The draft reads its own quantization metadata so
the target's multimodal weight-name mapping cannot discard its MTP projections
or rename its output head. It then shares the loaded target head.

Qwen3.8-Flash-Next checkpoints using `Qwen4ExpForConditionalGeneration` load
compressed PLE n-gram rows from `ngram_embedding.safetensors`, even when the
main shard index omits that file. Rows retain their packed EXL3 representation;
lookup decodes only the selected 160-dimensional rows, including the exported
head bias and hash layout. Set `--engram-config '{"cpu_offload": true}'` to
keep this table in pinned host memory and gather it through CUDA UVA. This
offloads the PLE table only. The validated 6-bit table requires about 36.36 GiB
of host memory. This format currently requires embedding tensor parallelism 1.
Its recursive MTP layer loads independent EXL3 metadata and shares target/draft
expert workspaces and the loaded output head.

The Qwen3.8-Flash-Next checkpoint can run on three CMP 170HX GPUs with 16
layers per pipeline stage, or on one 96 GiB GPU with the PLE table offloaded.
For the tested three-GPU configuration:

```bash
NCCL_P2P_DISABLE=1 vllm serve /path/to/qwen3.8-flash-next-exl3 \
    --pipeline-parallel-size 3 \
    --engram-config '{"cpu_offload": true}' \
    --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 512 \
    --kv-cache-memory-bytes 2147483648 --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5, "enable_adaptive_verification": true}'
```

Select the intended devices with `CUDA_VISIBLE_DEVICES`. Use
`--language-model-only` instead of the modality limit for text-only serving.

For the three-CMP-170HX GLM checkpoint used in validation, start with one draft
token and an explicit KV budget:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 \
VLLM_PP_LAYER_PARTITION=16,15,14 VLLM_EXL3_MOE_PREFILL=native \
vllm serve /path/to/glm5next-exl3 \
    --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 3 \
    --max-model-len 66560 --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --kv-cache-memory-bytes 2147483648 --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

MTP remains opt-in. Draft acceptance and the size of the target verification
batch both affect throughput; increasing the number of draft tokens can reduce
performance.

With the V2 model runner and pipeline parallelism, GLM5Next and Qwen4Exp EXL3 join decode
requests into a common pipeline phase once their preceding outputs are available.
This prevents chunked prefill from leaving concurrent requests in permanently
separate small decode batches. It applies with or without MTP. Set
`VLLM_EXL3_PP_DECODE_BATCHING=0` before startup to restore independent phases.

### Adaptive GLM5Next and Qwen MTP

Single-layer GLM5Next, Qwen3.5 and Qwen4Exp MTP support a scheduler-selected
draft length.
This includes Qwen3.8-27B checkpoints with the `Qwen3_5ForConditionalGeneration`
architecture, and Qwen3.8-Flash-Next with `Qwen4ExpForConditionalGeneration`,
when they have one `text_config.mtp_num_hidden_layers` layer:

```bash
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "enable_adaptive_verification": true}'
```

The configured token count is the maximum. The scheduler chooses a common
length from 1 through that maximum for each pipeline batch, using observed
acceptance and elapsed step costs. These models do not need a confidence head: this
path learns from verified drafts. Single-request decoding smooths rejection
bursts and requires stronger evidence to shorten a budget than to lengthen it.
Concurrent batches track acceptance faster, require a larger predicted gain to
grow than to shrink, and start by measuring budgets 1 through 3 (up to the
configured maximum).

Qwen4Exp EXL3 starts with K=3 (or the configured maximum if lower). Each
request supplies 32 verification observations before exploratory changes;
prefill overlap also keeps this default. A shorter unmeasured shape requires
an estimated 8% gain and three matching timings before selection. Scores
within 5% of K=3 prefer K=3. After shortening, six K=3 rounds refresh censored
positions every 32 decisions while the first draft's acceptance remains
healthy. This allows recovery without repeatedly probing fully rejecting
requests. Each budget retains its last 32 fully verified prefix yields;
shorter proposals do not overwrite longer-prefix observations. Core and
longer-budget comparisons use these measured yields, avoiding compounded
conditional rate estimates during rejection bursts.

With a maximum above 3, ordinary decoding uses the same three-draft policy.
Longer budgets require observed acceptance of the preceding prefix and enough
predicted marginal benefit to justify a bounded trial. Each trial collects
three matching proposal/verification timings, with at most 12 decisions.
Trial cost predictions use measured longer-budget timings when available.
A promising completed K=4 trial can immediately test K=5 with a 1% predicted
gain; promoting the longer budget still requires a measured 3% gain.
Unprofitable trials back off from 128 to at most 1024 decisions; improving
prefix acceptance can reopen exploration. Longer budgets require a 3% gain to
grow, but drop when the shorter policy predicts a 1% advantage, with a two-step
hold. Recent excess trial and transition costs also penalize subsequent trials.
Longer shapes never supply timing samples to the three-draft policy.
For single Qwen4Exp EXL3 requests, initial longer-budget trials and promotions
require an 8% advantage over the core policy. Longer budgets return to the
core when their measured advantage falls below 5%. Rare nonlinear trials wait
until decision 256. The measured K=4-to-K=5 follow-up keeps its 1% trial gate;
concurrent batches keep the general longer-budget thresholds above.
Each prefix position retains its own last 32 observations, so shorter rounds
cannot evict evidence needed for the next longer trial. Suffix observations
estimate acceptance conditioned on the preceding position being accepted.
The scheduler multiplies these rates by the current request's prefix probability
before averaging across requests, so a declining prefix reduces the predicted
suffix benefit even while older suffix observations remain available. Suffix
observations expire after 1024 decisions involving their request, matching the
maximum retry interval; other requests do not age them. A recovery from low
prefix acceptance also clears stale suffix observations before probing again.
If stale K=4 acceptance blocks a K=5 trial, the next retry remeasures K=4
instead of repeatedly applying the same stale estimate. A substantial rise in
prefix acceptance can reopen this retry after 64 decisions without K=4
observations, when the predicted trial cost is still plausible. A profitable
selected K=5 does not trigger these refresh trials.
The periodic scheduler log reports prefix acceptance estimates, matching
per-budget median step costs, trial budget, cooldown and amortized excess cost.
Unknown prefix estimates appear as `None`. The general speculative decoding
log divides each position's accepted count by all draft rounds, including rounds
that never proposed that position; it is not that position's conditional rate.

Acceptance and selection state are independent for each single request. Timing
samples are reused within the engine for the same batch size, context-length
bucket and prefill/decode mix, so new requests do not repeat every measurement.
The selected length controls actual draft model calls as well as verification.
An in-flight proposal retains its original length when the next budget changes.
The policy uses the requests eligible for the current pipeline batch, rather
than the total number of submitted or queued requests. Merely raising the
configured maximum from 3 to 5 does not change the policy within budgets 1–3.

This mode requires the V2 model runner, async scheduling, CUDA Graphs, and data
parallel size 1. Pipeline parallelism is supported. Do not combine it with
`num_speculative_tokens_per_batch_size`; LoRA and eager mode are rejected.
CUDA Graphs, recurrent state slots and cache reservations still cover the
configured maximum. Selecting fewer drafts saves compute, not reserved memory.

It remains opt-in. Acceptance varies with the task and batching, and collecting
cost samples takes real decode steps, so it need not beat a well-chosen fixed
length on every workload. Compare fixed and adaptive budgets using the same
reasoning effort, input lengths, pipeline partition, and cache budget. Count
reasoning tokens in total decode throughput and report end-to-end latency as
well. A larger maximum also reserves more recurrent state and can reduce the
number of requests that fit in the cache.

### Image input

GLM5Next image input is supported with EXL3 vision weights. For the validated
checkpoint, replace the text-only modality limit with the following serving
options:

```bash
--limit-mm-per-prompt '{"image": 2, "video": 0}' \
--mm-processor-kwargs '{"max_image_tokens": 1024}'
```

The original EXL3 checkpoint template replaces OpenAI-style media content with
a text-only reminder. Replace its `chat_template.jinja` with the corresponding
GLM-5.3-Flash AWQ or original model template to preserve image placeholders with
the default `auto` content format. The local checkpoint used here now contains
the AWQ template. With the original EXL3 template, use
`--chat-template-content-format string` instead; for `LLM.chat`, set
`chat_template_content_format="string"`. The loader preserves separate EXL3
Q/K/V rotations and ignores the original fused FP16 QKV when both copies are
present.
The three-CMP-170HX capacity run uses PP `17/15/13`, a 5.5 GiB KV budget per
rank, and the native 1,048,576-token context limit. It completes 1,048,448-token
input with MTP1 enabled.

Qwen4Exp loads its separately exported `vision.safetensors` or
`vision_k*.safetensors` files, including independently rotated Q/K/V and their
biases. Quantized copies take precedence over the retained fused FP16 QKV.
Its vision MLP preserves the export's padded intermediate width. The
Qwen3.8-Flash-Next image smoke test works with fixed MTP3 and adaptive MTP5;
the draft consumes text-token embeddings and target hidden states, while the
target owns the image encoder.

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
Long-context validation (local archive: `docs/validation/exl3-long-context-20260910.md`)
extends the two tested checkpoints to 65536 input tokens with one or four
submitted requests. The report records hardware, cache capacity, timing and
basic retrieval checks; this does not establish a general context limit.
Multimodal generation outside the validated GLM5Next/Qwen4Exp image paths (including
video), speculative decoding outside GLM5Next and single-layer Qwen MTP,
LoRA, general model-weight CPU offload, sleep mode,
expert load balancing, and distributed
tensor/expert sharding are not validated.
Shared experts run serially because upstream GEMMs share a device-wide lock
workspace. Dual batch overlap is explicitly rejected.
Do not infer support for those features from model architecture support alone.

Qwen3.8-27B EXL3 MTP text validation covers fixed K=3 and adaptive maximum
K=3/5 on one GPU, plus adaptive maximum K=5 with three CMP 170HX GPUs in PP.
The scheduler changes actual draft calls and preserves accepted GDN states when
the next verification is shorter. These Qwen3.8-27B MTP checks use prefix
caching disabled and do not establish multimodal MTP support or a general
throughput advantage over fixed drafting.

Qwen4Exp compressed PLE tables support GPU-resident and pinned-host lookup,
including CUDA Graph replay. QSA cache pages align both the attention backend
and the ring capacity reserved for the configured maximum draft length; MTP5
does not require a manual block-size override. The 512-expert, top-10,
640-intermediate Qwen3.8-Flash-Next checkpoint uses native EXL3 expert kernels;
the GLM-specific INT8/M32 kernels require other expert shapes and top-k limits.
Model validation covers Chinese and English writing, Python code, four
submitted requests, a 4K-token retrieval prompt, and an image with colored
shapes. These checks do not establish the checkpoint's full context capacity,
video support, or a throughput advantage of adaptive MTP5 over fixed MTP3.

Both model runners carry draft-layer ownership in their cache specifications.
The cache planner marks groups containing draft layers while keeping target
GDN and PLE state groups outside draft-cache handling. Ownership does not change
cache geometry or split otherwise compatible target and draft layers, so this
identification does not increase the advertised KV cache capacity.
Qwen3.8-Flash-Next prefix-reuse validation with adaptive MTP5 covers Chinese
and English retrieval, a Python function, and four requests sharing a roughly
6.5K-token prefix on three CMP 170HX GPUs in PP. Repeated requests reuse 4896
tokens; all 14 final answers match the expected result. Thinking sequences
are not generally token-identical between cold and cached runs. These checks
do not establish general output equivalence or external KV offload support.

Native prefill computes EXL3 products with FP16 operands; BF16 inputs and
outputs are converted at the operation boundary. Supported large SM80 MoE
prefills can instead reconstruct one INT8 expert projection and use grouped
INT8 Tensor Core GEMMs, with INT32 accumulation and affine codebook scaling. Large linear prefill batches temporarily
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
Other configurations follow the native expert GEMM policy described below.

`VLLM_EXL3_MOE_BATCHED_DECODE=1` (the default) enables expert reuse for larger
pure decode batches on SM80. Experts receiving one or two rows use DP4A; experts receiving
at least three rows use M32 FP16 Tensor Cores. This path supports 9–128 rows,
32–1024 experts, hidden × intermediate size of at least 4096 × 2048, and
the same BF16/4-bit mul1/top-k restrictions. It requires M32 and respects the
allocated expert workspace capacity. The default `hybrid` policy uses plain INT8
on SM80 for both small and batched decode; `residual` explicitly enables
activation residual compensation. The SM120 policy remains compensated INT8.
For hidden/intermediate dimensions 4096/2048, the batched hot-expert kernel
fetches 64 K elements per shared-memory stage and processes two K32 steps.
This reduces synchronization while preserving FP32 accumulation order and the
existing workspace size. Native prefill and other dimensions retain K32 fetch.
Plain INT8 batched decode also compacts cold assignments into persistent tasks
for these dimensions and top-k values 1, 2, 4 and 8. The kernels specialize
projection geometry while retaining the original activation quantization and
K splits. Task indices are produced by the existing routing kernel; there is no
additional weight cache. Other dimensions and residual INT8 retain the original
cold-expert path.
Mixed/prefill batches and PIECEWISE CUDA Graphs retain the existing paths.
Set `VLLM_EXL3_MOE_BATCHED_DECODE=0` before startup to restore the original
larger-batch path and its smaller scratch pool. Restart after changing the flag;
FULL decode graphs capture the selected policy.
In the tested B16/MTP1 64-question GSM8K subset, the original path scored 64/64;
both plain and compensated batched INT8 scored 63/64, with different wrong answers.
Compensation reduces kernel error but has not established accuracy neutrality.
Plain INT8 provided higher throughput in the tested SM80 concurrent workload.

The default `VLLM_EXL3_MOE_PREFILL=native` keeps native prefill and a 2048-token
scheduler budget. A concurrent 64-question GSM8K comparison scored 64/64 with
native prefill and 61/64 with forced INT8 prefill, so the added quantization
remains opt-in despite its long-input speedup.

Set `VLLM_EXL3_MOE_PREFILL=auto` to enable adaptive prefill. On SM80 MoE models
with a context limit of at least 10240 tokens, the implicit scheduler capacity is 6144.
After prefix-cache lookup, a request with at least 10240 uncomputed prompt tokens
selects chunks of up to 6144; that plan remains active until the small tail.
Short prompts and steps with active decode requests use at most 2048 tokens.
Several short prompts do not become a long prompt by being submitted together.
Other EXL3 models keep the 2048 default. Explicit token budgets and existing
scheduler alignment limits remain upper bounds. When a hybrid cache block is
larger than the adaptive short-prompt budget, prefill advances in smaller chunks
and stops at each cache boundary.

`VLLM_EXL3_MOE_PREFILL=native` disables INT8 prefill and its workspace.
Set `int8` to use the configured scheduler budget without the adaptive latency
limits. This can improve bulk throughput but increase short-request TTFT or
interrupt ongoing output. The worker requires at least 4096 GEMM rows by default;
smaller batches and tails use native kernels. INT8 prefill supports at most eight
selected experts per token and 65535 total experts. Unsupported devices and formats
also use native kernels. FULL CUDA Graphs never select the INT8 prefill kernel,
independently of their padding; the expert decode policy remains separate. `VLLM_EXL3_MOE_INT8_MIN_TOKENS` overrides the row threshold
(range 9–6144); lowering it is useful for numerical evaluations, not a default
performance recommendation.

The auto threshold is 10240 tokens, inclusive. This threshold is a policy
setting, not a universal performance crossover. See the
concurrent validation (local archive: `docs/validation/exl3-adaptive-prefill-20260912.md`)
for the original 32768-token policy's TTFT, throughput, output gaps and accuracy, and the
[design](../../design/exl3-adaptive-prefill.md) for cache and lifecycle behavior.

SM80 uses the bundled M=32, FP32-accumulating expert kernel for uniform 4-bit
mul1 weights with dimensions divisible by 256 in [256, 8192]. Short expert tails
retain the smaller row path. Other formats and GPU architectures use the upstream
kernel. `VLLM_EXL3_MOE_M_TILE=16` selects upstream for native prefill; use
`VLLM_EXL3_MOE_PREFILL=native` as well to disable the new INT8 path. INT8 uses
a separate M64 / N128 / K64 grouped GEMM.
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

INT8 prefill supports uniform 4-bit mul1 experts and the same SM80 dimension
bounds as M32. Its CUDA helpers are included in `_exl3_C`, and its Triton kernels
are part of vLLM. It needs neither an experimental worker nor an external helper
library. It introduces activation/codebook rounding and is not lossless.

Workspaces are shared across layers with the same expert dimensions on a device.
For hidden size 4096 and intermediate size 2048, the default 2048-row workspace
uses 384 MiB on the tested CMP 170HX and 1104 MiB on the tested RTX PRO 6000
Blackwell. M32 additionally shares a roughly 4 MiB lock buffer per device.
For these dimensions, hybrid decode adds a shared 5.25 MiB scratch on SM80
and 9.25 MiB on SM120. Enabling batched expert decode increases the SM80 scratch
to 84 MiB with the default `hybrid` or explicit `plain` policy, or 148 MiB with
`residual`, shared across matching layers and allocated before KV profiling.
Relative to the original SM80 path, the increments are 78.75 and 142.75 MiB
per rank, respectively.
INT8 prefill reconstructs one expert group at a time. When
`VLLM_EXL3_PREFILL_EXPERTS_PER_GROUP` is unset, the tested GLM shape (288 experts,
hidden size 4096, intermediate size 2048, top-8, 6144-row INT8 capacity) uses
48 experts per group; other shapes use 64. Set a positive integer to override
the group size, or `0` to reconstruct all experts at once. Set this variable
before process startup. Grouping preserves the token chunk and INT8 arithmetic.
Activations are quantized once per projection, and GPU routing bounds keep each
GEMM within its expert group, including hot groups and tails.

INT8 temporaries share one arena, reusing storage after each projection consumes
its inputs. The allocator compares phase reuse with stage/up reuse and selects
the smaller layout. Native MoE also uses views into this arena when its scratch
fits; otherwise its separate allocation is reserved before the optional INT8
pool, preserving native fallback under memory pressure. The paths run serially
on the caller's stream. Decode scratch and synchronization locks remain independent.
For the GLM shape above on CMP 170HX, native and INT8 together use about
0.938 GiB per rank with the default 48-expert groups, saving 896 MiB per rank
compared with separate native and INT8 allocations. The 384 MiB native scratch
is already included in this total. Explicit groups of 64, 32 and 0 use about
1.063, 0.938 and 2.813 GiB, respectively. Count unique backing storage when
measuring these pools; summing overlapping tensor views overstates allocation.

The arena is allocated before KV cache profiling and CUDA Graph capture;
switching input lengths does not create additional pools or change their
addresses. Auto falls back to native if that allocation fails, while explicit
`int8` reports the allocation failure. Model and KV budgets must still fit the
device. An auto engine configured for a context shorter than 10240 tokens does
not reserve the INT8 pool; native-only configurations retain their native scratch.

Setting the scheduler token budget to 1024 also caps expert workspace capacity
at 1024, halving the main workspace. Workspace memory grows
linearly with capacity and with the extension's number of concurrent expert
groups. See the optimization experiments (local archive: `docs/validation/exl3-optimization-20260910.md`)
for kernel ablations and complete-model measurements.

ExLlamaV3 controls its optional INT8 GEMV path through `EXL3_INT8_GEMV`. Use
`EXL3_INT8_GEMV=0`, `VLLM_EXL3_MOE_DECODE=native` and
`VLLM_EXL3_MOE_PREFILL=native` to validate without activation INT8. Use the same ordinary GEMV setting in both engines when comparing
speed or numerical accuracy, and record the expert decode policy separately.
Set these variables before starting the process. Mode 2 is
the upstream default and uses approximate INT8 activations; its numerical error
is evaluated separately from the stricter FP16 path. It affects eligible ordinary
GEMVs, not the routed-expert decode policy or M32 prefill. See the
combined INT8 measurements (local archive: `docs/validation/exl3-int8-combination-20260911.md`)
for ordinary GEMV comparisons and the
default expert decode validation (local archive: `docs/validation/exl3-hybrid-default-20260911.md`)
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
runs GSM8K questions. Its vLLM backend accepts `--mtp N` and records draft and
accepted-token counters for each measured case. Run its `exllamav3` backend in
the original engine's own environment. Distinguish GPU kernel time from
complete-engine throughput and record the actual device placement for multi-GPU
comparisons.

For a fixed-batch comparison with the multiprocessing vLLM engine, add
`--synchronize-inputs` to queue the entire batch before resuming the scheduler.
Without it, requests may start before the remaining inputs arrive and produce
different effective batch sizes. Record which admission mode was used; this
benchmark option does not change normal serving or staggered-arrival cases.

`benchmarks/prepare_exl3_long_inputs.py` builds deterministic retrieval inputs
at 8K, 16K, 32K and 64K, including exact token counts and expected answers.
The benchmark accepts gzip JSON inputs and configurable context/cache sizes.
For fixed-length generation, retrieval checks decode only the response before
the first configured stop token; full generated tokens remain in the results.

For the earlier three-CMP-170HX GLM throughput comparisons through 64K, use
PP `16/15/14`, `max_num_seqs=4` and an explicit 2 GiB KV cache budget per rank.
The vision and 1M-context capacity configuration above uses a different partition
and budget. For the mixed
four-GPU comparison use PP `11/11/11/12`. These are deployment measurements,
not hard-coded layer assignments. Adaptive inference requires no benchmark
worker; the optional profile worker only collects diagnostic intervals.

The complete-engine benchmark also accepts `after_tokens` per input in a case.
For example, `[0, 4]` admits the second request after the first has produced four
tokens, and records each output arrival. Warmups use the measured output length
so the active-decode pattern is the same. Report maximum output gaps as well as
percentiles: a single multi-second interruption can be invisible at p95.
