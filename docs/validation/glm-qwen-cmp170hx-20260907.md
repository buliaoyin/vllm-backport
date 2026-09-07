# GLM / Qwen correctness and performance validation

Status: planned implementation, validation and the subsequent four-case Qwen Flash NVFP4 full-test supplement completed on 2026-09-07. Confirmed code defects are fixed; documented model-output failures remain unresolved.

## Environment and protocol

- Repository baseline: `a191da902315aa21e853958ab94c49bdb0f1ad62` (includes the existing GLM AWQ and PP MTP embedding fixes).
- Python: conda `vllm-backport`, accessed through `.venv/bin/python`.
- GPU 0–2: CMP 170HX, 64 GiB, SM80. GPU 3: RTX PRO 6000 Blackwell, 96 GiB, SM120. Host RAM: 503 GiB.
- GLM: PP4 / TP1, layer partition `11,11,11,12`. Qwen Flash: PP4 / TP1, partition `12,12,12,12`, PLE CPU offload. Qwen 27B: PP1 on GPU 3.
- Validation context: 32,768 tokens; maximum 16 sequences; maximum 8,192 batched tokens. `NCCL_P2P_DISABLE=1`.
- GSM8K: official **test** split; 32-question initial compatibility screens for Qwen 27B and the first Radix baseline, 128-question GLM/Flash screens, and 1,319-question full comparisons for GLM AWQ, GLM NVFP4 and Qwen Flash FP8, plus both Flash NVFP4 checkpoints (RadixArk and Inferact). Five fixed training examples provide the prompt, using the repository's `_build_gsm8k_prompts` helper. Temperature 0, seed 42, request concurrency 4 for screening and 16 for full evaluation (the same for each full MTP0/3 pair). GLM uses its template with reasoning effort `max`, 4,096 output tokens; Qwen uses thinking enabled, 8,192 output tokens.
- Test data SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- Timing probes: Chinese explanation, Python implementation, and English design, each with 512 actual output tokens. TTFT and throughput use response usage and wall time, not the number of SSE events.
- State probes: a 16–18K-token shared prefix, distinct request keys, concurrent requests of different lengths, cancellation, and subsequent slot reuse. Prefix-cache counters must demonstrate actual hits. Request-key correctness and exact output formatting are recorded separately. The strengthened protocol used from the AWQ screening onward waits for generated text before cancelling and adds four longer Chinese continuations with distinct keys; speculative cases must show both accepted and rejected drafts. Earlier Qwen short key-only probes accepted all drafts; full pairs repeat the strengthened checks. Full pairs additionally prime two different private keys followed by long neutral tails, reuse the prefixes in reverse order, and require per-request cache-hit boundaries to lie beyond the key positions before accepting the retrieval result.

Commands, checkpoint audits, responses, metrics, traces and per-case commit/diff records are saved in `/tmp/vllm-code-fixes-20260907/`. Model checkpoints are unchanged. Startup durations are recorded for reproducibility, but page-cache state is not controlled and they are not used as a loading-speed comparison. The checkpoints reside on a local ext4 RAID; process I/O confirms substantial cold reads in some PP ranks while others load from cache. Full-set runs use an identical concurrency-16 short warmup with distinct cache salts before timed evaluation.

## Implemented changes

| Commit | Change | Validation |
| --- | --- | --- |
| `033d90297` | Reconcile unequal NVFP4 gate/up global scales before fused MoE repacking, for ModelOpt, Quark and compressed-tensors. Preserve Humming's separate conversion and equal-scale no-op. Bound FP32 temporaries by expert chunks. | Numerical/frontend regressions; actual Marlin repack and matmul on SM80 and SM120. |
| `dc2d6a86a` | Use persistent request-slot block tables and `req_idx` in both Mamba state-copy kernels. | All four new permuted-row cases fail before the fix and pass after it. |
| `c7fb9e3d8` | Seed resumed recurrent-state positions using the Mamba group's block size. | Boundary and slot-reuse cases, including the pre-discovery fallback. |
| `10a73fd1c` | Reserve a Mamba state interval for speculative prefix replay. | Full-block and fine-grained lookup cases. |
| `577be2dd4` | Give the hybrid coordinator a matching Mamba replay margin. Its old assumption that Mamba never dropped a block became invalid after the replay change. | Two integration regressions fail before the fix; all 132 prefix/manager tests pass after it. Real GLM MTP cache hits restored. |
| `10596ecf5` | Clear the inherited GPU PP layer partition in the isolated PLE CPU worker. | Full-layer CPU ownership and existing PLE behavior tests. |
| `139777a13` | Preserve per-batch PLE readiness under asynchronous PP; queue pending forwards and snapshot MRV1 CPU inputs before their reuse. | Both original regressions fail; retaining a shared event with an unbounded queue still deadlocks in the gated GPU test. Fixed: 5 PLE tests pass on SM80, 22 connector/worker tests on SM120; lint/mypy pass. |
| `94d6a06de` | Preserve signed integers in GSM8K label parsing. The old helper read the two negative official test labels as positive. | Two old failures; four signed/positive regression cases pass. All 1,319 labels audited. The first 128 are unaffected. |
| `6d03d7ae7` | Honor the explicit Qwen PLE storage dtype independently of the main quantizer. This supports retained FP8 PLE tables inside an NVFP4 checkpoint. | Two constructor/loading regressions fail before the fix; 65 PLE/offload/loading tests, pre-commit and mypy pass. RadixArk MTP0 rerun completes: 31/32 GSM8K, all functional and strengthened state checks pass. |
| `3c9efa290` | Update stale Qwen proposer fixtures to the current QSA state-sizing fields. | Six proposer tests pass; no proposer implementation change. |

Relevant pre-commit checks and mypy passed. State-copy tests also passed on both GPU architectures. Existing GLM and Qwen weight-mapping, quantization and PLE tests were exercised; obsolete fixtures were updated to their current implementation contracts.

For the GLM-sized NVFP4 scale tensor `(288, 4096, 256)`, reconciliation's extra peak GPU allocation is **352 MiB**, compared with **1,440 MiB** for the unchunked reference, with numerically identical output. The retained output is 288 MiB in both correct implementations. This compares chunked and unchunked reconciliation; the original incorrect gate-only path did not allocate this reconciliation output. A single cold timing observation is not used as a loading-speed claim.

## Serving settings to reproduce the validated paths

All cases use `/home/bul/miniconda3/envs/vllm-backport/bin/vllm` from the existing environment. The exact per-case command and environment are in `command-inventory.json`.

| Model path | GPUs / partition | Additional settings |
| --- | --- | --- |
| `/home/bul/dev/models1/zai/LibertAIDAI/GLM-5.3-Flash-NVFP4` | `0,1,2,3`, PP4, `11,11,11,12` | Memory utilization 0.95; `glm47` reasoning and tool parsers. |
| `/home/bul/dev/models1/zai/cyankiwi/GLM-5.3-Flash-AWQ-INT4` | Same GLM layout | Same GLM settings. |
| `/home/bul/dev/models1/Qwen/Qwen3.8-Flash-Next-FP8` | `0,1,2,3`, PP4, `12,12,12,12` | `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `--moe-backend marlin`. |
| `/home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4` | Same Flash layout | Same PLE/FP8 environment; keep automatic MoE backend selection so the BF16 draft experts can use an unquantized backend. |
| `/home/bul/dev/models1/Qwen/Inferact/Qwen3.8-Flash-Next-NVFP4` | Same Flash layout | Same PLE/FP8 environment; `--moe-backend marlin`. |
| `/home/bul/dev/models1/Qwen/Qwen3.8-27B` and its `-FP8` sibling | GPU `3`, PP1 | No PLE offload or PP partition override. |

Qwen uses memory utilization 0.90, reasoning parser `qwen3`, and tool parser `qwen3_xml`. All models enable automatic tool choice. MTP3 adds `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; MTP0 omits this argument. These results cover 32K context and 16 maximum sequences. They do not validate the earlier `--max-model-len auto --max-num-seqs 64` configuration or multimodal requests.

## Targeted regression commands

The following use the existing conda environment through `.venv/bin/python`. GPU selections are isolated; each GPU suite was repeated with `CUDA_VISIBLE_DEVICES=0` (SM80) and `CUDA_VISIBLE_DEVICES=3` (SM120) where noted above.

```bash
.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py tests/v1/core/test_single_type_kv_cache_manager.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/kernels/mamba/test_precopy_mamba_align.py tests/v1/worker/test_mamba_hybrid_model_state.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/kernels/quantization/test_marlin_gemm.py -k reconciled_gate_up -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/v1/worker/test_gpu_model_runner.py -k ple_offload -q
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/v1/worker/test_gpu_model_runner.py tests/v1/worker/test_ple_offload_worker.py -k ple_offload -q
.venv/bin/python -m pytest tests/evals/gsm8k/test_gsm8k_correctness.py -k answer_keeps_negative_sign -q
```

The raw `*-red.log` and `*-green*.log` files retain pre-fix failures and post-fix results. Some early combined logs include stale fixture failures despite their `green` filename; `test-results-summary.json` identifies these intermediate runs and the later passing PLE/config/proposer regressions that resolve them. Counts across these overlapping suites are not summed as unique tests. Separate model-loading and quantization suites, pre-commit and mypy logs are in the artifact directory. The report's short/full model commands are recorded exactly per case in `command-inventory.json`; these are custom diagnostics using the existing GSM8K prompt helper, not the B200/H200 EvalScope protocol or its thresholds.

## Checkpoint audit

Both Qwen Flash NVFP4 checkpoints have equal gate/up global scales: 24,576 main-model pairs for RadixArk, and 25,088 main-plus-draft pairs for Inferact. Reconciliation is therefore a no-op for these pairs.

The three Flash checkpoints have identical chat-template files and generation configurations (template SHA256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`). Their other layouts differ: RadixArk has fused BF16 MTP experts and FP8 PLE tables; Inferact has NVFP4 MTP experts and BF16 PLE tables. The latter's PLE weights are approximately 95.4 GiB, versus 47.7 GiB for the FP8 tables. Each layout is validated separately.

The sampled rows of GLM's draft embedding match the checkpoint. After the serving loader installs the shared output head, the draft head is the target head and its sampled BF16 checkpoint rows match. The temporary head before this sharing step is not a valid runtime comparison point.

## Initial GLM NVFP4 results

| Case | GSM8K test 128 | Previous train-distance outlier | State keys | Exact key-only format | Prefix hit tokens |
| --- | --- | --- | --- | --- | --- |
| MTP0, corrected scales | 126/128 | Correct in screening | 14/14 | 14/14 | 182,784 |
| MTP3, corrected scales, coordinator margin fixed | 126/128 before the coordinator-only change | 3/3 diagnostic repeats correct | 14/14, reproduced | 13/14 | 125,440 |
| MTP3, prefix caching disabled | State control only | — | 14/14 | 13/14 | 0 |

The initial MTP0 repeat-control list had an indexing error and repeated unrelated items; those repetitions are not counted as train-distance repetitions. The screening itself included the correct question. Subsequent repetitions use zero-based index 16.

The two screening errors are the same with MTP0 and MTP3: zero-based index 12 (the model gives the 12-year break-even point; the label requires positive profit in year 13) and 119 (reasoning repetition hitting the 4,096-token limit). The official solution for item 119 also treats “A is 30% higher than B” as “B is 30% lower than A”; all reported scores still use the official labels. Screening elapsed time was 299.5 s and 197.3 s respectively. MTP3 accepted 80.4% of proposed GSM8K tokens. These are screening observations, not full-set scores or an attribution of all improvement to the scale change.

The early `glm-nvfp4-mtp3-state-margin` artifact records `correct=13` using the original exact-string check: its exceptional output starts with the right `583004` key, then adds reasoning and repeats that same key. Later cases record key correctness and exact formatting separately; the original failed artifact is unchanged. The key-format anomaly includes a generated extra assistant role after the correct answer. It occurs with prefix caching both enabled and disabled, without another request's key appearing. Raw token evidence and forced-prefix next-token comparisons are preserved. This is not scored as perfect formatting. The completed full NVFP4 MTP0 run also reproduces the natural format anomaly (14 correct keys, 13 exact outputs). In the fixed 16,149-token context, this corrected target selects `<|assistant|>` with log probability −0.064216, approximately 93.8%. Together with the old-scale target control, this establishes that neither MTP nor reconciliation is required to predict the extra role in that context; it does not establish a parser fix.

## GLM AWQ screening

The AWQ MTP0 baseline scores **126/128**, with the same failed indices 12 and 119 as corrected NVFP4. All five functional probes pass. Both the 14 short-key and four longer continuation checks preserve the correct request-specific keys; the short keys also meet exact formatting. These phases record 182,784 and 52,224 prefix-hit tokens respectively, and cancellation was observed after actual generated text.

AWQ MTP3 also scores **126/128**, with 24,057 output tokens in 160.1 s and 81.9% draft acceptance. All functional probes pass. It returns all 14 short keys with exact formatting (125,440 prefix-hit tokens). The four longer continuations also retain their correct keys, record 35,840 prefix-hit tokens and accept 310 of 723 proposed tokens (42.9%), exercising substantial rejection and continued generation. Unlike MTP0, item 119 finishes with 99,076.92, consistent with the stated salary relationship but still incorrect against the official label.

For the fixed 16,149-token context ending immediately before the previously observed extra assistant token, AWQ MTP0 also chooses `<|assistant|>` with log probability −0.11683 (approximately 88.97%). This shows that the continuation can be predicted by a target model without MTP. The corrected NVFP4 MTP0 full run now provides the matching target control described above. These are conditional-context comparisons, not evidence that AWQ and NVFP4 naturally generate identical preceding contexts.

## Qwen integration finding

The first FP8 PP4 run loaded its PLE weights correctly and answered six GSM8K questions, then failed with `queue.Full` in the PLE connector. PP host execution can enqueue several forwards before the background notifier runs. Re-recording a shared readiness event can also make an earlier request wait behind a later forward's unsatisfied PLE semaphore. This was reproduced separately from queue capacity.

Commit `139777a13` fixes both lifecycle issues. The successful FP8 MTP0 rerun scores **125/128**, with 526.6 s for 68,159 generated tokens. Its three errors are indices 12, 100 and 119; the last reaches the 8,192-token limit. All five functional probes pass. The 18,084-token state probe returns all 14 keys with exact formatting and records **241,472 prefix-hit tokens**. Chinese/code/English decode rates are 52.2/52.5/52.2 tokens/s. The failed attempt remains in `ple-queue-full-attempt/`; its six answers are not a completed evaluation.

FP8 MTP1 also completes: **126/128**, 405.1 s, 61,716 output tokens, 86.0% draft acceptance. Its errors are indices 12 and 119. All functional probes pass; all 14 state keys have exact formatting, with 221,760 prefix-hit tokens. The target-shared output head and sampled draft embedding rows match the BF16 checkpoint values. Chinese/code/English decode rates are 57.6/67.7/66.0 tokens/s. These fixed-length probes are more suitable for comparing decoding rates than total GSM8K duration, since the generated reasoning lengths differ.

FP8 MTP3 completes with **125/128**, the same failed indices as MTP0, in 285.9 s for 64,470 tokens. Draft acceptance is 71.6% overall (mean accepted length 3.147). All functional probes and all 14 exact-format state checks pass, with 224,000 prefix-hit tokens.

| Qwen Flash FP8 | GSM8K 128 | Chinese tok/s | Code tok/s | English tok/s | State keys / exact format |
| --- | --- | --- | --- | --- | --- |
| MTP0 | 125/128 | 52.17 | 52.52 | 52.16 | 14/14, 14/14 |
| MTP1 | 126/128 | 57.64 | 67.66 | 66.01 | 14/14, 14/14 |
| MTP3 | 125/128 | 67.51 | 107.62 | 90.18 | 14/14, 14/14 |

Torch profiling returned `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED` on this mixed CMP configuration. The generated traces contain CPU activity only; they are not evidence of GPU kernel costs. Separate CUDA-event stage timing and CPU PLE timing were successfully collected in all four GPU workers and the CPU worker. In the initial MTP3 decode sample, draft proposal averages 3.48 ms per step and CPU PLE computation averages 0.78 ms per step. A narrower graph-replay interval is included in the full-pair profile to distinguish execution from preceding PP readiness waits. Stage timings include stream waits and host launch gaps and are not isolated kernel timings.

## Mixed Qwen NVFP4 / FP8 PLE finding

RadixArk's first startup failed when the PLE worker loaded `ngram_embedding.weight_scale`: the selector only recognized a main-model `Fp8Config`, so its NVFP4 main configuration caused the explicitly FP8 PLE table to be allocated as BF16 without a scale parameter. The checkpoint declares `ple_embedding_dtype="float8_e4m3fn"` while excluding PLE from its NVFP4 conversion.

Commit `6d03d7ae7` gives that explicit storage declaration priority, preserving the previous quantizer-based selection when no PLE dtype is declared. CPU tests construct the real embedding, load small checkpoint shards and scales, and compare lookup/dequantization values. They cover NVFP4 with FP8 PLE, NVFP4 with implicit BF16 PLE, FP8 with explicitly BF16 PLE, and legacy FP8 PLE. The GPU placeholder already retains the checkpoint scale and selects FP8 IPC output accordingly. Failed startup artifacts are preserved in `ple-storage-mismatch-attempt/`. The rerun successfully loads 132 checkpoint tensors, verifies both PLE parameters, and scores **31/32** on GSM8K with no truncations. Its only failed label is index 12. All five functional probes and strengthened state checks pass. RadixArk MTP3 starts with its BF16 fused draft experts and verified embedding/head rows. It scores **125/128**, with failed indices 12, 87 and 119 (the latter two reach 8,192 tokens). Its first 32 match the MTP0 score of 31/32. All functional and strengthened state checks pass: 14/14 exact short keys with 224,000 prefix-hit tokens, and 4/4 long keys with 64,000 hit tokens and 44.1% draft acceptance. The matching MTP0 128-question run scores **124/128**, with failed indices 12, 85, 100 and 119, one truncation, and 70,169 output tokens in 537.8 s. Item 87 is correct both in this screen and its sequential repeat. MTP3 therefore introduces one observed failure while correcting two baseline failures; its overall score alone cannot establish equivalence. The final serial MTP3 diagnostic repeats item 87 twice: both reach 8,192 tokens with identical complete token sequences and no final answer. Their metrics record zero prefix hits. The following item-16 control answers 230 correctly in 321 tokens. Item 87 repeatedly debates whether the employee receives three or four raises. These serial MTP3 truncations remain reproducible observations for that run. The subsequent full comparison below also observes an MTP0 truncation on item 87, followed by correct MTP0 repeats, while full MTP3 answers it correctly. The combined evidence does not support an MTP-exclusive failure or isolate a checkpoint/inference root cause; original scores and serial reproductions remain unchanged.

## Inferact NVFP4 baseline and long-context key replacement

Inferact MTP0 loads successfully, including 131 PLE checkpoint tensors and its single BF16 table parameter. It scores **125/128** in 532.6 s for 69,639 output tokens; failed indices are 12, 100 and 119, with the last truncated at 8,192 tokens. All five functional probes pass. The short state probe returns all 14 keys with exact formatting and records 241,472 prefix-hit tokens.

The longer continuation probe fails one of four key checks: the final request outputs the shared archive's old key `582731` instead of its requested replacement `684004`, followed by an otherwise coherent explanation. This is retained as a failed model-output check; the original case ended at that assertion and is not reported as a fully passing run.

A separate MTP0 server with prefix caching disabled reproduces the same failure. More decisively, its **first user request**, sent alone with the exact failing long prompt, generates the same token sequence and text as the cache-on failure, with zero prefix hits. Its first-token probabilities favor the old key's initial `5` (60.4%) over the new key's `6` (25.2%). Reducing the neutral archive from 2,000 lines to four produces the correct new key. The normal cache-off state suite again gets 14/14 short keys and 3/4 long keys, with zero hits.

This failure therefore does not require MTP, prefix reuse, prior user requests or concurrency. It remains a long-context instruction-following failure under this checkpoint and inference configuration; this control does not prove that every underlying kernel or quantization operation is correct. No state-code change is justified by this finding alone. The original failing artifacts and `inferact-state-failure-assessment.json` preserve the evidence. The matching fresh-prompt FP8 MTP0 control now passes both the 2,000-line prompt and its four-line control, with zero prefix hits. On the long prompt, its first-token distribution prefers the new key's `6` (log probability −0.64528) over the old key's `5` (−0.89528); Inferact prefers the old key as described above. The long prompt therefore exposes checkpoint-dependent output differences, but this comparison alone does not isolate which quantized component is responsible. The matching FP8 MTP3 control is also included in the full pair.

Inferact MTP3 scores **126/128** with no truncations, correcting baseline item 100 and introducing no new failed indices. It produces 69,573 tokens in 260.2 s with 71.3% draft acceptance. All functional probes pass; Chinese/code/English decoding reaches 68.7/112.5/88.1 tokens/s. Short state checks remain 14/14 exact with 224,000 prefix-hit tokens. The long continuation again fails only the fourth key replacement, while recording 64,000 hit tokens and 268/639 accepted draft tokens (41.9%). The same failure is retained for MTP0 and MTP3. Runtime audit confirms both target and draft use NVFP4 Marlin, with the correct shared BF16 output head.

## Qwen 27B BF16 and FP8 compatibility regression

On GPU 3 with PP1, Qwen 27B BF16 MTP0 and MTP3 both score **31/32**, with only index 12 failing the official label. MTP0 produces 14,722 tokens in 149.1 s; MTP3 produces 15,119 in 59.4 s and accepts 74.6% of proposed tokens. This is a short compatibility screen, not a full accuracy evaluation of the 27B model.

| BF16 configuration | Chinese tok/s | Code tok/s | English tok/s | Short keys / exact format | Long keys |
| --- | --- | --- | --- | --- | --- |
| MTP0 | 28.78 | 28.78 | 28.78 | 14/14, 14/14 | 4/4 |
| MTP3 | 52.16 | 83.89 | 66.66 | 14/14, 14/14 | 4/4 |

All five functional probes pass in both cases. Short-key prefix hits are 233,632 and 235,200 tokens respectively; the long MTP3 probe records 67,200 hit tokens and accepts 246/531 proposed tokens (46.3%). Cancellation follows generated text and is followed by slot reuse. The actual draft is `Qwen3_5MTP`; its sampled BF16 embedding and head rows match the checkpoint, and runtime head sharing is verified. No Qwen3_5 model-loading implementation changes were needed.

Qwen 27B FP8 MTP0/3 also both score **31/32**, again failing only index 12, with no truncations. The baseline generates 14,116 tokens in 85.5 s; MTP3 generates 14,070 in 35.6 s with 74.7% draft acceptance. All functional probes, all 14 exact short keys and all four long keys pass. MTP3's long probe records 67,200 prefix-hit tokens and accepts 258/642 proposed tokens (40.2%).

| FP8 configuration | Chinese tok/s | Code tok/s | English tok/s |
| --- | --- | --- | --- |
| MTP0 | 50.21 | 50.21 | 50.21 |
| MTP3 | 83.36 | 125.85 | 110.34 |

The FP8 draft audit confirms its MLP is stored in E4M3 FP8 with FP32 block scales, while the embedding, FC, normalization and shared head retain BF16. Sampled embedding/head values and runtime sharing match the checkpoint. These four 27B cases cover text-only PP1 compatibility; they do not establish multimodal or full-test-set accuracy for 27B.

## Same-checkpoint GLM NVFP4 scale control

The diagnostic process restores only the old gate-only fused global-scale selection; the source tree retains the fix and all other current changes. A fresh-interpreter binding check verifies that ModelOpt, Quark and compressed-tensors import this override. The control script and its check log are preserved in the artifact directory.

| Scale handling | MTP | GSM8K 128 | Failed indices | Truncated | Draft acceptance |
| --- | --- | --- | --- | --- | --- |
| Old gate-only global scale | 0 | 125/128 | 12, 107, 119 | 2 | — |
| Reconciled scales | 0 | 126/128 | 12, 119 | 1 | — |
| Old gate-only global scale | 3 | 126/128 | 12, 119 | 1 | 79.3% |
| Reconciled scales | 3 | 126/128 | 12, 119 | 1 | 80.4% |

Old MTP0 generates 30,139 tokens in 289.1 s; old MTP3 generates 26,261 in 196.5 s. Both old-scale controls pass all five functional probes, all 14 exact short keys and all four long keys, with actual cache hits. Their fixed-output rates are 45.4/45.3/45.3 tokens/s for MTP0 and 55.2/69.2/57.5 for MTP3 (Chinese/code/English).

Index 107 reaches the limit only in the old MTP0 screen; old MTP3 answers it correctly both in screening and a repeat, and both corrected screens answer it correctly. The old MTP0 failure was observed once and was not repeated in that already-running case. Index 16 is correct in both old screens and all six old-control repetitions. Thus these model screens do not establish that scale handling alone caused the earlier MTP outlier or imply a universal throughput gain from reconciliation. The numerical regression establishes the scale error independently; reconciliation preserves effective scales within E4M3 rounding, with bounded temporary memory.

Both old NVFP4 controls also select the extra assistant token in the fixed role-boundary context without requiring speculative generation for the MTP0 control. The completed corrected-NVFP4 target comparison also chooses that token, as described above.

## Full GSM8K test-set evaluation

The following use all 1,319 official test questions at concurrency 16. Scores retain the official labels, including ambiguous or inconsistent items; diagnostic repetitions do not replace first-pass answers. All ten full GSM8K evaluations are complete. The requested four-case Flash NVFP4 supplement uses the same scoring, sampling, warmup, concurrency and output limits as the FP8 full pair. Source HEAD for all four supplemental cases is `7c8b65d60f58ceb460089b59dfc3deeeec9de0d8`; no inference code was changed for the supplement. FP8 and Inferact full cases retain failed long-key checks separately from their completed accuracy and functional results.

| Model | MTP | Correct / total | Accuracy | Output-limit cases | GSM8K seconds | Generated tokens |
| --- | --- | --- | --- | --- | --- | --- |
| GLM NVFP4 | 0 | 1,282/1,319 | 97.19% | 5 | 1,498.2 | 289,161 |
| GLM NVFP4 | 3 | 1,280/1,319 | 97.04% | 1 | 1,034.0 | 278,323 |
| GLM AWQ | 0 | 1,282/1,319 | 97.19% | 1 | 1,309.3 | 260,185 |
| GLM AWQ | 3 | 1,283/1,319 | 97.27% | 2 | 1,049.8 | 266,335 |
| Qwen Flash FP8 | 0 | 1,291/1,319 | 97.88% | 3 | 1,766.8 | 680,826 |
| Qwen Flash FP8 | 3 | 1,290/1,319 | 97.80% | 3 | 1,140.0 | 665,329 |
| Qwen Flash RadixArk NVFP4 | 0 | 1,289/1,319 | 97.73% | 6 | 1,735.4 | 695,844 |
| Qwen Flash RadixArk NVFP4 | 3 | 1,291/1,319 | 97.88% | 4 | 1,079.1 | 676,789 |
| Qwen Flash Inferact NVFP4 | 0 | 1,294/1,319 | 98.10% | 2 | 1,577.6 | 681,721 |
| Qwen Flash Inferact NVFP4 | 3 | 1,290/1,319 | 97.80% | 3 | 1,058.4 | 686,213 |

GLM NVFP4 MTP0 completes all five functional probes, all 14 request-key checks (13 exact formats), four longer continuations, and the private-prefix check. The two private keys occur before token 16,032; reverse-order reuse records 21,760 prefix-hit tokens per request and retrieves the correct key, exercising a cache boundary beyond the distinguishing information. The first-pass five truncations are indices 119, 450, 943, 1176 and 1265. Repeating failed cases recovers 252, 267, 450, 806 and 1265; the score remains the original 1,282. The signed labels at indices 489 and 1113 are both handled correctly.

Inspection confirms the completed numerical failures use the explicit final-answer marker rather than a parser fallback. Some are ordinary reasoning errors, while some reflect inconsistent labels: item 403 asks for energy saved by reducing a 900 W air conditioner from eight to three hours daily for 30 days; the model returns 135 kWh, while the label is 81 kWh (remaining usage). This example is retained as incorrect under official scoring.

GLM NVFP4 MTP3 introduces nine failed indices (255, 322, 340, 406, 425, 439, 583, 901, 967) and corrects seven baseline failures (252, 357, 450, 806, 1059, 1176, 1265), a net difference of two questions or −0.15 percentage points. Each new failure is repeated twice. Seven answer correctly at least once; 255 and 967 remain incorrect in both repetitions. Item 255 has conflicting ten/twenty-stall wording; item 967 changes the referent used for the sister's age. These are retained as observed failures, without claiming that MTP and baseline outputs are equivalent. Only 326/1,319 complete token sequences match exactly. Baseline repetitions also changed some answers, so a one-pass disagreement alone does not identify an implementation defect.

The MTP3 truncation is item 814, which loops while questioning `20/2 = 10`. Its repeat finishes with 17, matching the baseline answer but differing from the official label 11 (the label omits one of the two worse players). The repeated output remains a failed first-pass case. MTP3 accepts 82.07% of draft tokens; the fractions of draft iterations accepting through positions 1/2/3 are 93.99%/82.94%/69.28% (not conditional probabilities). Aggregate generated output rises from 193.00 to 269.16 tokens/s at concurrency 16, while fixed 512-token Chinese/code/English decoding rises from 45.36/45.29/45.29 to 53.08/74.74/65.08 tokens/s (1.17×/1.65×/1.44×).

All five MTP3 functional probes pass. The full run returns 14/14 exact short keys and 4/4 long keys, with 125,440 and 35,840 prefix-hit tokens respectively. The long probe accepts 358/867 draft tokens (41.29%), exercising rejection. Both reverse-order private-key reuses are correct and record 17,920 hit tokens each, beyond the key positions at 16,032. Earlier runs' occasional extra-role formatting remains a documented limitation despite this particular MTP3 run's exact outputs.

AWQ MTP0 scores 1,282/1,319 with one truncated response (901), which answers correctly on repeat. Other recovered first-pass failures are 252, 255, 439, 587, 1124 and 1265; all original answers remain scored. All five functional probes, all 14 exact short keys, four long continuations and two private-key reuses pass. Short/long phases record 182,784/52,224 cache-hit tokens; each private-key reuse hits 21,760 tokens, beyond the key at 16,032. Chinese/code/English fixed-length decoding is 45.83/45.75/45.75 tokens/s. The AWQ baseline also changes its interpretation of the conflicting ten/twenty-stall question between the first pass and its repeat; this observation does not require MTP.

AWQ MTP3 scores 1,283/1,319, introducing six failures (340, 357, 406, 425, 640, 1176) while correcting seven baseline failures (252, 255, 439, 587, 901, 1198, 1265). Five of the six new failures answer correctly at least once in two repeats; 406 remains at 240 rather than 200 because it uses the total cannoli count instead of the newly purchased count as the comparison quantity. Truncated indices are 943 and 1176; both repetitions of 1176 finish, with one correct and one incorrect answer. The net improvement is one question (+0.08 percentage points), not evidence of a general accuracy improvement. Complete token sequences match for 393/1,319 questions.

MTP3 accepts 81.94% of proposed GSM8K tokens; acceptance-through-position fractions are 94.22%/82.75%/68.85%. Chinese/code/English fixed-length decoding reaches 51.09/68.49/56.79 tokens/s, or 1.11×/1.50×/1.24× the matching baseline. All five functional probes, all 14 exact short keys, all four long keys and both private-cache reuses pass. Short/long phases record 125,440/35,840 hit tokens, with 303/678 accepted drafts (44.69%) in the long continuations. Private-key reuses each hit 17,920 tokens, beyond the distinguishing key position.

Qwen Flash FP8 MTP0 scores 1,291/1,319 with truncations at 119, 835 and 858. Repetitions recover 45, 675, 830, 835, 858 and 988, while 119 still truncates. The fixed control at 87 is correct on the first pass but returns 10,080 on repetition after 4,922 tokens, choosing four raises instead of three; this interpretation change occurs without MTP. All five functional probes pass. Fixed-length Chinese/code/English decoding is 52.20/52.19/52.19 tokens/s.

Its later long-key state check fails: 14/14 short keys have exact formatting and 258,720 prefix-hit tokens, but two of four longer answers return the public prefix's old `582731` instead of `684001` or `684004`. The long phase records 68,992 hit tokens. The isolated first-user-request `684004` control had passed on this same server with zero hits, which motivated the cache/batch controls below; FP8 must not be described as universally passing this prompt. The original run stops at this assertion and retains its complete GSM8K/functional/benchmark results, but has no passing completion marker. Separate cache-off/cache-on state diagnostics collect independent private-key and CUDA-event checks even when a model-output assertion fails. Such diagnostic runs retain a failure status rather than converting an assertion into a pass.

The FP8 MTP0 cache-off control also fails the *first single long request* (zero hits): the old key's initial `5` has log probability −0.50048, versus −1.12548 for the new `6`. Its four-line control passes; all 14 short keys pass, while the four mixed long answers pass 3/4 (key 684001 fails). This eliminates MTP, prefix reuse and concurrent requests as requirements for the long-key failure, but does not prove all single-request kernels or numerical paths correct. Cache-off and align mode also change the recurrent-state execution configuration, so a second control keeps align mode fixed and bypasses reuse using unique cache salts.

| FP8 MTP0, same align-mode server | Correct long keys | Prefix-hit tokens |
| --- | --- | --- |
| Serial, unique salts (bypass) | 3/4 | 0 |
| Concurrent four, unique salts | 3/4 | 0 |
| Serial, reused prefix | 3/4 | 68,992 |
| Concurrent four, reused prefix | 2/4 | 68,992 |

All four serial bypass/reuse pairs produce **identical complete token sequences** and identical first-token log probabilities. Key 684001 favors the old key in all four modes. Key 684004 flips only in the concurrent reused-prefix phase: bypass favors `6` over `5` by 0.625 logit units; reuse favors `5` over `6` by 0.125. This is evidence of sensitivity to batched execution/cache layout, not proof of a particular quantization or state-corruption cause. The reported public-key failures remain failed checks. Separately, both private keys are recovered in reverse order with 26,656 hit tokens each, beyond their position at 18,074. The MTP3 full run also successfully checks simultaneous restoration of these distinct private prefixes.

The aligned MTP0 diagnostic collects 127 decode steps from all four GPU workers and the CPU PLE worker despite the retained output failure. CPU PLE forward averages 0.787 ms. Target graph intervals on GPU 0/1/2/3 average 7.08/3.97/3.99/2.90 ms; outer target-execute intervals are 18.28–19.44 ms and include PP waits. These intervals are not isolated kernel costs and must not be added as independent rank work. The cache-off profile is retained as a separate configuration, not substituted for the aligned MTP0/MTP3 comparison.

Qwen Flash FP8 MTP3 scores 1,290/1,319, introducing six failed indices (100, 549, 752, 782, 1019, 1176) while correcting five baseline failures (45, 590, 830, 835, 988). Each new failure is repeated twice. Five answer correctly at least once; 100 remains incorrect in both repetitions (155 and 685 versus the official 175). That same question had already failed in the MTP0 128-question screen, so it is not an MTP-only failure. Both repetitions of 752 and 782 pass; 549, 1019 and 1176 each pass once. The 1176 chalk question reaches the output limit in its first pass and one repetition. Other first-pass truncations are 119 and 675. These original failures remain scored.

Only 45/1,319 complete Qwen token sequences match across MTP0/3. Together with baseline repeat variability, this prevents treating the similar scores as proof of identical output distributions. MTP3 accepts 71.60% of proposed GSM8K tokens, with acceptance-through-position fractions of 83.33%/71.33%/60.13%. It generates 665,329 tokens in 1,140.0 s (583.62 aggregate output tokens/s versus 385.35 for MTP0). All five functional probes pass. Fixed-length Chinese/code/English decoding reaches 67.28/106.91/89.87 tokens/s, or 1.29×/2.05×/1.72× the baseline. The long-key check again passes 2/4, while all 14 short keys have exact formatting. Short/long phases record 240,000/64,000 hit tokens; long continuations accept 254/642 proposed tokens (39.56%), exercising substantial rejection. The failed long-key assertion remains a failure while independent diagnostics continue.

The MTP3 same-align-mode matrix reproduces the baseline pattern: serial bypass 3/4, concurrent bypass 3/4, serial reuse 3/4, and concurrent reuse 2/4. Bypass phases have zero hits; reuse phases have 64,000. All four serial bypass/reuse pairs again produce identical complete token sequences. Both private keys are recovered in reverse order, with 25,600 hits per request beyond their token-18,074 positions. Concurrent private reuse also returns both correct keys and records 51,200 hits; even the conservative per-request lower bound (51,200 minus the other request's 28,051 prompt tokens) exceeds the key position. These passes do not erase the separate public-key instruction failures.

The MTP3 cache-off control also fails its first isolated long request, with the same first-token log probabilities as MTP0 cache-off (`5`: −0.50048; `6`: −1.12548), and passes the four-line control. The identical first-token distribution does not imply identical full continuations. All 14 short keys are exact with zero hits; the mixed long probe passes 2/4, versus 3/4 for MTP0 cache-off, again returning the old key for requests 1 and 4. It accepts 37.64% of proposed continuation tokens. The failed assertion and all five event files are retained. This completes the MTP/cache-on/off controls without establishing universal numerical equivalence or a checkpoint-level root cause.

### Supplemental Qwen Flash NVFP4 full comparisons

The supplemental repetitions retain the request settings and concurrency limit of 16, while their request set and resulting batch composition differ from the full first pass. They diagnose repeat variability rather than establish a deterministic equivalence test.

RadixArk MTP0 scores 1,289/1,319, with output-limit cases 87, 119, 675, 796, 984 and 1176. Repetitions recover first-pass failures 87, 406, 590, 752, 835 and 1071; the original score is unchanged. MTP3 scores 1,291/1,319, introducing five failures (255, 782, 1019, 1038, 1288) and correcting seven (87, 182, 234, 406, 590, 752, 1071). Each new failure is repeated twice: all five answer correctly at least once, but one repetition of 782 still reaches 8,192 tokens. MTP3 truncates on 119, 675, 1038 and 1176; 1038 contains unfinished answer text, whose last-number fallback parses the trailing `200` as incorrect. Thus four responses reach the limit, while three have empty final-content fields (the harness's `missing_final` counter); the fourth is also incomplete. Only 56/1,319 complete output token sequences are identical across MTP0/3. The net two-question gain (+0.15 percentage points) does not establish a general accuracy gain or output equivalence.

Item 87 revises the earlier RadixArk assessment. MTP0's full concurrency-16 first pass reaches 8,192 tokens without a final answer, while both subsequent repeats return the official 9,360 in 1,237 and 985 tokens. Full MTP3 returns 9,360 in 3,027 tokens and again on repetition in 2,150 tokens. Earlier MTP3 serial repetitions still contain identical 8,192-token loops. Truncation therefore occurs without MTP too, and changes with repeat/batch conditions; these observations do not isolate its numerical or checkpoint-level cause.

Both RadixArk full cases pass all five functional probes, all 14 exact short keys, all four long continuations, and every mode of the serial/concurrent cache-bypass/reuse matrix. MTP0/3 short phases record 241,472/224,000 prefix-hit tokens and long phases 68,992/64,000. MTP3 long continuations accept 267/606 proposed tokens (44.06%), exercising rejection. Both private keys are recovered in reverse order and concurrently: MTP0 hits 26,656 tokens per serial request and 53,312 combined, versus 25,600 and 51,200 for MTP3. All per-request cache boundaries are beyond the key positions at token 18,074. MTP3's GSM8K draft acceptance is 71.85%; fixed-output Chinese/code/English speed changes from 52.18/52.19/52.19 to 67.80/108.66/102.02 tokens/s (1.30×/2.08×/1.95×).

Inferact MTP0 scores 1,294/1,319, with truncations at 119 and 984. Repetitions recover 100, 234, 782 and 830 without replacing first-pass scores. All five functional probes and all 14 exact short keys pass, with 241,472 short-phase prefix-hit tokens. The long continuation check passes 3/4, returning the old public key instead of 684004; the failure remains recorded and the process exits with a failed assertion after independent diagnostics finish. In the same align-mode matrix, serial bypass, concurrent bypass and serial reuse all pass 4/4, while concurrent reuse passes 3/4. Bypass phases record zero hits and reuse phases 68,992. Both private keys are nevertheless recovered in reverse order (26,656 hits each) and concurrently (53,312 hits combined), with boundaries beyond the keys. This new matrix supplements the earlier fresh cache-off single-request failure; it does not erase it or prove a cache-only cause. Fixed-output Chinese/code/English rates are 52.19/52.18/52.23 tokens/s.

Inferact MTP3 scores 1,290/1,319, introducing seven failures (182, 406, 675, 796, 988, 1019, 1288) and correcting three baseline failures (119, 590, 830), a net difference of −4 questions (−0.30 percentage points). Only 61/1,319 complete token sequences match MTP0. The three truncated first-pass responses are 406, 675 and 796. Each new failure is repeated twice: five answer correctly at least once; 675 returns 21 in both repeats versus the official 33, and 796 returns 2,640,000 in both versus 2,880,000. Item 675's official solution changes from 24 dogs to 36 toys before subtracting three; the repeated model answer uses 24 − 3. Item 796's repeats apply the first new hires after the first month's salary, whereas the official answer includes them in that month. Both remain incorrect under unchanged official scoring. Similar aggregate accuracy and partial recovery on repetition do not establish output equivalence.

Inferact MTP3 accepts 72.06% of GSM8K draft tokens, with mean accepted length 3.162. Fixed-output Chinese/code/English decoding reaches 68.09/112.42/87.87 tokens/s, or 1.30×/2.15×/1.68× its baseline. All five functional probes and all 14 exact short keys pass (224,000 hits). The long continuation again passes 3/4, failing key 684004 with 64,000 hits and 41.94% draft acceptance. The same-align-mode matrix matches full MTP0: 4/4 serial bypass, 4/4 concurrent bypass, 4/4 serial reuse and 3/4 concurrent reuse, with zero bypass hits and 64,000 reuse hits. Both private keys are recovered in reverse order (25,600 hits each) and concurrently (51,200 combined), with all cache boundaries beyond the keys at token 18,074. The original failed state assertion remains a failure after these independent diagnostics complete. All four NVFP4 full cases retain event records from the four GPU workers and the CPU PLE worker in `cuda-event-summary.json`; those measurements have the same serving-interval limitations as the FP8 profiles below.

## Serving-stage measurements and optimization decision

For Qwen Flash FP8, the same isolated 128-output-token code prompt supplies the aligned MTP0/MTP3 event profiles below. MTP0 uses 127 recorded decode steps and MTP3 uses 41; per-step costs therefore are not costs per output token.

| Mean interval per decode step | MTP0 | MTP3 |
| --- | --- | --- |
| CPU PLE forward | 0.787 ms | 0.784 ms |
| GPU 0 target graph | 7.077 ms | 7.861 ms |
| GPU 1 target graph | 3.972 ms | 4.882 ms |
| GPU 2 target graph | 3.991 ms | 4.950 ms |
| GPU 3 target graph | 2.900 ms | 3.741 ms |
| GPU 3 draft proposal | — | 3.480 ms |
| GPU 3 sample/postprocess, including draft if enabled | 0.877 ms | 4.342 ms |

GPU 0's outer target interval is 19.44 ms for MTP0 and 33.11 ms for MTP3; its narrower graph interval is much smaller because the outer interval includes PP readiness waits and host scheduling. CPU PLE timing excludes transfer and waiting, so it cannot establish that the whole offload path is negligible. CMP does not support the attempted CUPTI collection; these are CUDA-event serving intervals, not a kernel-level attribution. The demonstrated performance change is bounded-memory scale reconciliation plus measured MTP acceleration. No additional PLE/QSA kernel rewrite is justified by this profile while output-quality differences remain unresolved.

## Observed full-run GPU memory

Values are `nvidia-smi` total device-used MiB sampled once per second, in GPU 0/1/2/3 order. Startup includes weight loading, profiling, graph capture and cache allocation before API readiness. Evaluation includes GSM8K and subsequent probes. These sampled peaks can miss brief allocations; they are distinct from the allocator-measured reconciliation microtest.

| Full case | Startup peak MiB, GPU 0/1/2/3 | Evaluation peak MiB, GPU 0/1/2/3 |
| --- | --- | --- |
| GLM NVFP4 MTP0 | 46044 / 57022 / 57034 / 62959 | 53388 / 64628 / 63964 / 72089 |
| GLM NVFP4 MTP3 | 45892 / 57402 / 57396 / 72978 | 52146 / 63410 / 64386 / 79138 |
| GLM AWQ MTP0 | 60402 / 64724 / 64720 / 72875 | 51218 / 62942 / 63402 / 79801 |
| GLM AWQ MTP3 | 60402 / 64724 / 64720 / 88678 | 52244 / 62808 / 63670 / 97284 |
| Qwen Flash FP8 MTP0 | 64895 / 64892 / 64892 / 68805 | 61775 / 60500 / 60500 / 60996 |
| Qwen Flash FP8 MTP3 | 64807 / 64892 / 64892 / 76776 | 61805 / 60830 / 60830 / 72668 |
| Qwen Flash RadixArk NVFP4 MTP0 | 58139 / 57830 / 57830 / 58330 | 61857 / 60268 / 60268 / 60738 |
| Qwen Flash RadixArk NVFP4 MTP3 | 58329 / 57998 / 57998 / 76736 | 61681 / 60610 / 60610 / 79054 |
| Qwen Flash Inferact NVFP4 MTP0 | 58139 / 57810 / 57810 / 58310 | 61857 / 60248 / 60248 / 60718 |
| Qwen Flash Inferact NVFP4 MTP3 | 58331 / 57980 / 57980 / 73274 | 61683 / 60590 / 60590 / 75590 |

AWQ MTP3 reaches about 95 GiB on GPU 3 at the validated 32K/16-sequence setting. Larger context or sequence limits need their own memory and state validation. Startup times and cold/warm file-cache differences are recorded in the artifact inventory, without claiming a controlled loading-speed improvement.

## Outcome and limits

The implementation defects demonstrated by failing regressions are fixed: NVFP4 fused gate/up scales, recurrent-state request-slot ownership and replay boundaries, PLE PP initialization, queued readiness/CPU input lifetime, and mixed PLE storage selection. Existing GLM draft-loading and quantization exclusions are preserved. Qwen draft audits do not show a missing embedding or shared-head load in the tested checkpoints. No shared draft/model loader change was needed, so the conditional DeepSeek regression was not triggered.

Across the full GSM8K pairs, MTP3 changes the number of correct answers by −2 for GLM NVFP4, +1 for GLM AWQ, −1 for Qwen Flash FP8, +2 for RadixArk NVFP4 and −4 for Inferact NVFP4. Fixed-length decoding benefits depend on the prompt: 1.11×–1.65× across the two GLM formats, 1.29×–2.05× for Qwen FP8, and 1.30×–2.15× across the two Qwen NVFP4 layouts. These are MTP-on/off measurements of the corrected repository, not a throughput A/B against the pre-fix source. Both Flash NVFP4 layouts now have full official test-set coverage; Qwen 27B BF16/FP8 retain their 32-question compatibility screens.

There are still observed output limitations: occasional GLM extra-role formatting, individual reasoning loops or interpretation changes (including Radix item-87 truncations observed with both MTP settings under different run conditions), and the Qwen long public-key instruction failure. The Qwen control can fail without MTP, cache reuse or preceding user requests, and its behavior changes with batch/cache layout. This narrows the conditions but does not isolate all numerical paths or establish that a quantized checkpoint is intrinsically defective. No full BF16 GLM/Flash reference was run. The recorded wrong answers and assertion failures remain intact.

Validated serving limits are 32K context and 16 maximum sequences, with text-only requests. The original auto-context/64-sequence configuration, Qwen 27B full-test accuracy, multimodal requests, and a complete cross-product of hardware/quantization settings are outside the completed coverage. Exact commands, per-case source hashes, original model responses, repetition outcomes, memory samples and profiling data are retained with the artifact inventory.
