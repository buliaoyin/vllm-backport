# GLM-5.3-Flash 对比优化验证（2026-09-09）

本机 PP4 对照显示，这组优化主要改善 MTP 解码：NVFP4 三道代码题提升 31.2%–37.5%，AWQ 提升 34.1%–37.8%；关闭 MTP 时提升约 1%。NVFP4 的 2K、8K、16K 输入解码提升 24.7%–32.9%，首段响应时间接近持平。

验证完成：八组主评测、NVFP4 长上下文对照及四组 32K 私有缓存补测已完成；源代码、脚本与样本完整性核对通过。

## 基线与实现

基线为 `9d12faf89bfbcd7280e966b4f1479b001bc85864`。基线和候选分别冻结在 `/tmp/glm53-opt-baseline-20260909`、`/tmp/glm53-opt-candidate-20260909`。候选源文件 SHA-256 和完整参数保存在[数据摘要](glm53-flash-optimization-20260909.json)。源码清单对应随本报告提交的实现。

| 改动 | 解决的问题 | 参考与本地适配 |
| --- | --- | --- |
| kpool tail 缓存为投机 lookahead 留空间 | 被拒绝的草稿覆盖已接受 token 的原始 K/gate，回滚后错误重建压缩池 | 参考 [PR #55219](https://github.com/vllm-project/vllm/pull/55219)，沿用当前缓存管理器，仅扩大池对齐的环并分开逻辑池宽与物理环宽；同步 AMD 算法 |
| PP 草稿回传使用 Triton scatter | GPU 布尔索引触发动态索引和主机等待 | 参考 [PP8 分支](https://github.com/promisezackr/glm53-flash-170hx-pp8)，保留已有流依赖与接收缓冲保护，支持行 stride 和索引 stride |
| GLM router 只算一次 | GLM MoE 与持有 gate 的 MoERunner 重复执行 router GEMM | 参考 [PR #55736](https://github.com/vllm-project/vllm/pull/55736) |
| KDA 直接读取投影视图 | q/k/v/beta 每次 `.contiguous()` 复制 | 参考同一 PR，给共享 recurrent 内核增加 token stride；保留本地状态索引边界与 int64 修复，不支持的布局继续复制 |
| NoPE query 写为 token 连续布局 | 投影后转置，再拼接宽度为零的 RoPE 部分 | 同时适配本机真正使用的 Triton/XPU MLA 实现及 FlashInfer 实现；RoPE 与 padding head 路径保留 |
| indexer 工作区按 kpool 压缩率分配 | 已压缩索引仍按原始 token 数预留 | 参考 [PR #55222](https://github.com/vllm-project/vllm/pull/55222) 的独立容量修正 |
| NoPE sparse MLA 自适应跳过空 tile | 短历史的固定 top-k 大量为 padding，仍运行 QK/PV | 本轮本地实验；GPU 索引探测选择两个计算循环，任意 padding 排列仍正确，CUDA graph 重放可动态切换 |

kpool=4 时，环容量按 `ceil((4 + num_speculative_tokens) / 4) * 4` 计算，MTP0/3/7 分别为 4/8/12。例：已接受位置 0、1 后验证位置 2–5，旧四槽环会被位置 4、5 覆盖 0、1；若草稿在位置 3 被拒绝，下一轮重建位置 0–3 的池就会读到错误内容。扩大物理环保留了回滚所需原始值。该定向复现说明代码存在问题，但不能据此把所有历史答错题归因于它。

不采用外部分支中已撤回的融合元数据实验。此次没有量化 BF16 草稿输出头，也没有启用新的推测解码算法。

## 对照协议

- 硬件：GPU 0–2 为 CMP 170HX（SM80），GPU 3 为 RTX PRO 6000 Blackwell（SM120）。使用 conda `vllm-backport`。
- 模型：`/home/bul/dev/models1/zai/cyankiwi/GLM-5.3-Flash-AWQ-INT4` 与 `/home/bul/dev/models1/zai/LibertAIDAI/GLM-5.3-Flash-NVFP4`。
- 两侧统一 PP4/TP1、分层 `11,11,11,12`、32K 上下文、`max-num-seqs=16`、`max-num-batched-tokens=8192`、显存利用率 0.95、prefix cache 开启、Mamba cache `align`。
- 统一 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=0,1,2,3`、`NCCL_P2P_DISABLE=1`、`VLLM_DETERMINISTIC_MOE_ALIGN=0`。分别测试 MTP 关闭和 `num_speculative_tokens=3`。
- 每项服务预热后先测速：中文、英文、异步 LRU、表达式解析器、SQLite 队列，共五道题，每道五次，强制生成 512 token，temperature=0、seed=42、reasoning_effort=low。报告五次中位数；速度为 `(completion_tokens-1)/(结束时间-首段非空文本时间)`。保留原始 SSE 与每次采样数值。 JSON 另附按首段实际累计 token 数修正的 `median_tps_excluding_first_chunk`，便于检查 MTP 多 token 首段及中文可见文本延迟的计量影响；本报告主表仍统一使用既定历史口径。
- GSM8K 使用官方 **test 全部 1319 题**，5-shot、temperature=0、seed=42、reasoning_effort=max、max_tokens=4096、并发16。数据文件与官方缓存 SHA-256 一致。超长截断单列，不调整主评分；额外重测失败题和固定对照题，重测结果不覆盖第一次分数。
- 额外验证 JSON、中文/Unicode、可执行 Fibonacci、算术和工具调用，以及取消、16K 长上下文、请求槽复用、共享前缀、私有缓存中的请求键隔离。
- 每种模型按 baseline MTP0 → candidate MTP0 → baseline MTP3 → candidate MTP3 串行交错。模型测试期间不并发运行 GPU microbenchmark。

## 整模型结果

<!-- MODEL_RESULTS_START -->
| 模型 | MTP | 版本 | 中文 TPS | LRU 代码 TPS | 解析器 TPS | 队列 TPS | 英文 TPS | GSM8K | 截断 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 0 | 基线 | 52.28 | 52.18 | 52.17 | 52.17 | 52.18 | 1279/1319（96.97%） | 4 |
| NVFP4 | 0 | 候选 | 52.95 | 52.85 | 52.84 | 52.84 | 52.85 | 1281/1319（97.12%） | 2 |
| NVFP4 | 3 | 基线 | 53.51 | 75.56 | 77.22 | 68.66 | 62.67 | 1280/1319（97.04%） | 2 |
| NVFP4 | 3 | 候选 | 77.26 | 99.10 | 104.76 | 94.42 | 82.29 | 1278/1319（96.89%） | 5 |
| AWQ | 0 | 基线 | 52.79 | 52.69 | 52.69 | 52.68 | 52.70 | 1286/1319（97.50%） | 1 |
| AWQ | 0 | 候选 | 53.37 | 53.27 | 53.22 | 53.22 | 53.27 | 1283/1319（97.27%） | 3 |
| AWQ | 3 | 基线 | 53.95 | 70.00 | 72.88 | 67.97 | 60.31 | 1288/1319（97.65%） | 0 |
| AWQ | 3 | 候选 | 71.98 | 96.05 | 97.73 | 93.64 | 83.46 | 1287/1319（97.57%） | 1 |

所有速度均为五次中位数，单请求、固定 512 输出 token。

| 模型 | MTP | 中文变化 | 三道代码变化范围 | 英文变化 | GSM8K 正确数变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 0 | +1.28% | +1.27%～+1.29% | +1.28% | 2 |
| NVFP4 | 3 | +44.38% | +31.16%～+37.52% | +31.30% | -2 |
| AWQ | 0 | +1.09% | +1.02%～+1.09% | +1.09% | -3 |
| AWQ | 3 | +33.41% | +34.11%～+37.78% | +38.40% | -1 |
<!-- MODEL_RESULTS_END -->

三道代码题的每轮估算耗时，NVFP4 约从 41.6 ms 降至 30.7–31.1 ms，AWQ 约从 43.5–43.6 ms 降至 31.6–31.8 ms；接受率没有系统提高，说明主要收益来自执行成本下降。本轮比较的是整组改动，未做逐项整模型消融，不能将全部收益归于某一个补丁。

首次 GSM8K 正确数变化分别为 NVFP4 MTP0 **+2**、MTP3 **−2**，AWQ MTP0 **−3**、MTP3 **−1**。例如 AWQ MTP0 有 8 题由对变错、5 题由错变对；新增错题中 5 题重测答对，说明存在答题波动，但不能据此证明质量等价或替换首次分数。

JSON 中 MTP 的 `approx_cycle_ms_median` 是“首段之后的解码时间 / 草稿组次数”的五次中位数，用于辅助区分每轮计算成本与接受率变化。它包含统计区间边界误差，并非单个 GPU 内核耗时；主结果仍采用完整请求的实测速度。

9 月 7 日约 45 token/s 的结果来自更早代码，只作历史参照，不能作为本轮优化前基线。GSM8K 不是原版 BF16 的基准复现；这里比较同一量化模型在两套代码下的结果。

<!-- CHECK_RESULTS_START -->
| 模型/版本/MTP | 基础探针 | 请求键 | 严格六位格式 | 长响应请求键 | 私有缓存单条/并发 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4/基线/0 | 5/5 | 14/14 | 12/14 | 4/4 | 2/2；2/2 |
| NVFP4/候选/0 | 5/5 | 14/14 | 13/14 | 4/4 | 2/2；2/2 |
| NVFP4/基线/3 | 5/5 | 14/14 | 14/14 | 4/4 | 2/2；2/2（边界未证明） |
| NVFP4/候选/3 | 5/5 | 14/14 | 14/14 | 4/4 | 2/2；2/2（边界未证明） |
| AWQ/基线/0 | 5/5 | 14/14 | 14/14 | 4/4 | 2/2；2/2 |
| AWQ/候选/0 | 5/5 | 14/14 | 14/14 | 4/4 | 2/2；2/2 |
| AWQ/基线/3 | 5/5 | 14/14 | 14/14 | 4/4 | 2/2；2/2（边界未证明） |
| AWQ/候选/3 | 5/5 | 14/14 | 14/14 | 4/4 | 2/2；2/2（边界未证明） |

请求键正确与严格格式分别计数；出现额外引号或说明文字时，可能键值正确但严格格式未通过。表中“边界未证明”表示原始并发用例返回的键均正确，但缓存计数不足以证明两个请求都恢复了包含私有键的状态；补测结果见下文。
<!-- CHECK_RESULTS_END -->

下面单独比较优化版本开、关 MTP。它与上面的“同一 MTP 配置优化前后”是不同的对照。单次全量得分的微小变化不构成质量等价证明；所有首次失败题、截断题及重测结果均保留，不能用挑选的重测结果抬高主分数。

<!-- MTP_RESULTS_START -->
| 优化后模型 | 关闭 MTP 的 GSM8K | MTP3 的 GSM8K | 正确数变化 | MTP3 相比普通解码的代码速度变化 |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 | 1281/1319 | 1278/1319 | -3 | +78.68%～+98.26% |
| AWQ | 1283/1319 | 1287/1319 | +4 | +75.94%～+83.62% |
<!-- MTP_RESULTS_END -->

## 并发吞吐与长上下文

GSM8K 的并发为 16，下表为完整评测区间“总生成 token / 总耗时”。两侧答案长度不同，且包含 prefill、排队及 HTTP 开销，因此这是该评测工作负载的观测吞吐，不能替代固定输出长度的单请求对照。

<!-- BATCH_RESULTS_START -->
| 模型 | MTP | 基线总输出 token/s | 候选总输出 token/s | 变化 | 基线/候选耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 0 | 208.16 | 219.07 | +5.24% | 1393.1 / 1297.3 |
| NVFP4 | 3 | 283.45 | 299.57 | +5.69% | 998.9 / 964.7 |
| AWQ | 0 | 208.85 | 219.20 | +4.95% | 1245.2 / 1223.3 |
| AWQ | 3 | 265.66 | 278.64 | +4.89% | 969.2 / 942.6 |
<!-- BATCH_RESULTS_END -->

额外对 NVFP4 MTP3 做独立服务配对补测：约 2K、8K、16K 输入，各四次、固定 512 输出 token；使用相同提示词和 token 数，并以唯一 `cache_salt` 确认每次计时请求的 prefix hits 为零。各项报告中位数，保存原始 SSE、提示词 SHA-256 及实际 token 数。

<!-- LONG_RESULTS_START -->
| 输入长度 | 解码 TPS 基线 → 候选 | 变化 | TTFT 秒基线 → 候选 | 每轮估算耗时 ms 基线 → 候选 | 接受率基线 → 候选 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 81.11 → 107.76 | +32.87% | 1.066 → 1.032 | 41.57 → 31.46 | 79.76% → 80.18% |
| 8192 | 77.88 → 98.23 | +26.12% | 2.467 → 2.431 | 41.61 → 31.34 | 74.82% → 69.70% |
| 16384 | 78.50 → 97.88 | +24.68% | 4.359 → 4.319 | 41.58 → 31.27 | 75.53% → 68.70% |
<!-- LONG_RESULTS_END -->

长上下文接受率同样取四次中位数。8K、16K 的接受率分别从 74.82% 降至 69.70%、75.53% 降至 68.70%；每轮耗时降低约 25%，因此最终解码仍提升约 25%–26%。这些接受率变化也保留在结果中，不将长输入收益解释为草稿更准确。

## 私有缓存覆盖补测

原始并发用例把两个不同私有键放在相同公共前缀的 16,032 token 处，查询长度为 25,048。AWQ 和 NVFP4 的 MTP3 两侧都返回正确的两个键，但两请求合计 prefix hits 为 35,840。只凭合计指标，要保证每个请求都命中私有键之后的缓存，保守下界必须满足 `total_hits - max_prompt_tokens > key_end_tokens`；原用例不满足。因此原测试进程以诊断错误退出，原始错误与退出码均保留在 JSON，不能把它描述为答错或跨请求串键。

源码中 MTP 的 Mamba 缓存查找会为重放保留一整块，不是只减去草稿 token 数。18K 查询的中间校准仍不足以覆盖 16K 私有键，实际返回仍正确。最终补测将两条分支的 prime 和 query 都延长至接近 32K，保留同一公共前缀和同一 cache salt；连续四轮并发检查两个键，并继续要求上述严格下界成立。没有修改生产缓存安全余量或放宽断言。

<!-- CACHE_RESULTS_START -->
| MTP3 模型/版本 | 补测 | 每轮两请求的最小缓存命中下界 | 私有键结束位置 |
| --- | --- | ---: | ---: |
| NVFP4/基线 | 8/8 个键正确，边界证明通过 | 21962.0 | 16032 |
| NVFP4/候选 | 8/8 个键正确，边界证明通过 | 21962.0 | 16032 |
| AWQ/基线 | 8/8 个键正确，边界证明通过 | 21962.0 | 16032 |
| AWQ/候选 | 8/8 个键正确，边界证明通过 | 21962.0 | 16032 |
<!-- CACHE_RESULTS_END -->

AWQ 优化版首次补测的等待脚本读取了正在重写的主结果 JSON，解析失败后提前释放服务；当时尚未发出补测请求。保留该脚本错误日志，并在独立的同版本 AWQ MTP3 服务上重跑完全相同的 32K 缓存检查，避免等待过程与结果写入重叠。该问题不影响八组主测速和 GSM8K 分数。

前期长上下文校准另保留：基线首次有 11 条有效样本，之后重复提示词触发 prefix-cache 零命中断言而中止；候选有 12 条计时样本，但当时 18K 缓存用例仍未证明覆盖。这些记录不混入最终成对补测。

## 内核测量与限制

两类 GPU 均使用 FlashInfer CUDA graph + cold L2，先验证数值，再排除编译、预热和输入构造计时。CMP 170HX 不支持当前 CUPTI 计时路径，因而两卡统一采用 graph 计时。PP scatter 另测包含 CPU 等待的 wall time。独立脚本位于 `benchmarks/kernels/benchmark_glm53_decode.py` 和 `benchmarks/kernels/benchmark_glm53_sparse_padding.py`。

<!-- KERNEL_RESULTS_START -->
| 测量 | CMP 170HX 优化前 → 后（µs） | RTX PRO 6000 优化前 → 后（µs） |
| --- | ---: | ---: |
| KDA，单请求验证4 token | 36.97 → 23.14 | 21.06 → 14.10 |
| KDA，16请求各1 token | 116.94 → 100.66 | 97.35 → 90.79 |
| NoPE 投影，4 token | 21.91 → 18.43 | 14.51 → 12.76 |
| NoPE 投影，64 token | 41.78 → 22.22 | 22.29 → 16.50 |
| PP scatter，16请求，含CPU等待 | 115.42 → 17.75 | 118.86 → 17.75 |

Sparse MLA 的两次完整 CMP 测量：16-token、256 条有效历史为 263.37→104.04 / 263.37→103.32µs；64-token、2048 条有效历史在首轮有回退，复测接近基线，完整数值与配置见 JSON。
<!-- KERNEL_RESULTS_END -->

KDA microbenchmark 在相同新内核上比较“先复制成连续张量”与“直接读取 stride”，隔离的是复制开销；整模型 A/B 才包含内核地址计算等全部差异。PP scatter 的微秒数不能直接换算成整模型 TPS 增幅。

空 tile 实验中，无条件加入分支虽加速短历史，但使部分长历史内核耗时增加约 40%，因此未采用。最终自适应版本在 CMP 的首轮长历史测量仍出现 8%–11% 回退，两次后续复测接近持平；保留全部测量，并在可用记录中保存自动调优配置。该项长历史收益不应声称稳定为正。

indexer 名义工作区在 32K 下从 165 MiB 降为 41.25 MiB。这是源码容量计算；共享工作区、峰值和 KV 分配会影响实际回收量，不能按层数累计或直接声称释放了等量显存。NVFP4 MTP0 的实际启动日志中，两侧各 rank 的峰值激活与已分配 KV（保留两位小数）一致，尚未观测到可增加 KV 容量的效果。

## 正确性与代码检查

- 冻结基线上的定向 kpool 拒绝重放用例确实失败，能够复现已接受 token 被拒绝草稿覆盖。修复后的测试覆盖 spec 1/3/7、不同池边界、prefill 种子、物理环重复绕回及 padding/stride。
- CPU 定向回归：33 passed。CMP 170HX 与 RTX PRO 6000 各核心 72 passed、1 skipped；共享 recurrent 的 GDN/KDA、真实 KDA 形状和独立参考均覆盖。
- PP 与 NoPE：两卡各 17 passed。旧 PP 测试夹具仍传入已删除参数，已按当前生产 API 修正。
- sparse MLA：两卡各 57 passed，覆盖 NoPE/RoPE、split、全空索引、正数越界、不同 padding 排列与 CUDA graph 重放。
- 扩展后的 pre-commit 与 Python 3.12 mypy 均通过。仅修改 Python/Triton，无需重新构建 C++/CUDA 扩展。
- 产物核对通过：21 个冻结源码/测试/基准文件哈希一致，八组各 1319 题和 25 个测速请求完整；最终长上下文两侧脚本与提示词哈希一致、各 12 个请求且 prefix hits 均为零；四组私有缓存补测共 32/32 个键正确且边界证明通过。
- AMD kpool 算法同步修正，但本机没有 ROCm 硬件，未实测 AMD 路径。

可用以下命令重跑定向检查（先停止占用这些 GPU 的测试服务）：

```bash
GLM_TEST_PYTHON=/home/bul/miniconda3/envs/vllm-backport/bin/python
"$GLM_TEST_PYTHON" -m pytest -q \
  tests/v1/attention/test_kpool_tail_slot_mapping.py \
  tests/models/test_glm5next_pipeline_parallel.py

for glm_test_gpu in 0 3; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$glm_test_gpu" \
  "$GLM_TEST_PYTHON" -m pytest -q \
    tests/kernels/test_kpool_decode_update_batched.py \
    tests/kernels/test_fused_recurrent_packed_decode.py \
    tests/v1/worker/test_pp_utils.py \
    tests/v1/attention/test_sparse_mla_backends.py \
    tests/kernels/attention/test_triton_mla_sparse_kernel.py

done

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
"$GLM_TEST_PYTHON" benchmarks/kernels/benchmark_glm53_decode.py \
  --timer graph --output /tmp/glm53-decode-gpu0.json

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
"$GLM_TEST_PYTHON" benchmarks/kernels/benchmark_glm53_sparse_padding.py \
  --baseline-kernel /tmp/glm53-opt-baseline-20260909/vllm/v1/attention/ops/triton_mla_sparse_kernel.py \
  --output /tmp/glm53-sparse-gpu0.json
```

完整日志、启动命令、源 diff、逐题响应和 SSE 保存在 `/tmp/glm53-opt-20260909`；便携数值摘要及本次实际使用的评测脚本源码、哈希随 JSON 入库；`validation_harnesses` 字段保留客户端与服务管理脚本，便于审计协议和恢复运行。首次隔离工作树因缺少安装生成的 flash-attn Python 文件启动失败，补齐同一环境构建产物后重启；该失败不计入模型结果，日志留存在 `failed-setup/`。前期来源筛选见[调研报告](glm53-flash-performance-research-20260909.md)。
