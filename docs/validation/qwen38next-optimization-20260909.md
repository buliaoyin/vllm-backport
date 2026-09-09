# Qwen3.8 参考分支优化与性能验证（2026-09-09）

## 结论

所有计划内模型评测已结束，结果以表格中的实际通过数为准。

本轮移植 Mamba speculative 边界防护、重复图像缓存引用、UniProc/dummy PLE 初始化、CPU PLE 异步输入顺序和 NVFP4 gate/up padding 修复；另实现融合 GPU PLE 查表及可选 UVA 后端。UVA 默认关闭，原来的 CPU offload 启动方式继续可用。

RadixArk NVFP4 的全量 GSM8K，基线与融合 UVA 均为 **1292/1319（97.95%）**。三类单请求解码中，代码中位数从 112.43 到 114.70 token/s，但 MTP 接受率和生成内容同时发生变化；按草稿轮数归一化后改善不到 1%，目前不能据此宣称 MTP 路径有明显的端到端加速。FP8 的英文 TPS 从 106.38 降到 98.00，同期接受率中位数从约 71.8% 到 62.6%；报告保留这个负向结果，不只展示提速样本。

无 MTP 时的 PLE 收益更明显：最终 CPU offload 的代码生成中位数为 67.39 token/s，UVA 为 79.44 token/s（约 +17.9%）；中文、英文变化幅度接近。两种路径的 GSM8K 前 32 题均为 31/32。这里比较的是已修复输入顺序的 CPU 路径与 UVA；旧 CPU 基线在并发请求中超时，没有可用速度，不能据此计算提升比例。本次机器上可按报告命令显式选择 UVA，默认配置仍保持关闭。

独立工作树还试过提前一层发起 CUDA stream 预取：相对基线每轮耗时约低 1%，相对同步 UVA 不到 1%。考虑收益与持久缓冲区、跨 stream 同步的额外复杂度，本轮没有合入预取实现。预取实验视觉为 23/24，表格算术题答 177 而正确值为 167；这项失败保留，不能写成视觉全通过。

## 版本与环境

- 基线：`a3c9ee61cc1ef9b66ebbb08bd8e6fc201ed465d8`，隔离目录 `/tmp/qwen38next-opt-baseline-20260909`。
- 参考：`/home/bul/dev/vllm-qwen38next`，`bul/qwen38next_offload_latest`，`a34e0656bc543b959338eb5e2b55606c87279701`。
- UVA 参考分支：`origin/hhy/ple_offload_uva`，`cc4e4f0911aa8b6b8887ea2e6ac598372ba0a608`。
- 候选：当前仓库，最终保留同步融合 UVA；预取实验目录 `/tmp/qwen38next-opt-prefetch-20260909` 单独存档。UVA 路径的数学代码在最终 CPU 输入顺序修复前后相同；后者只影响启用 CPU offload 时的 connector。
- 全部 Python 和服务使用 conda `/home/bul/miniconda3/envs/vllm-backport`；PyTorch 2.13.0+cu130、CUDA 13、NCCL 2.29.7。
- GPU 0–2：CMP 170HX / SM80 / 约 63.4 GiB；GPU 3：RTX PRO 6000 / SM120 / 约 95 GiB；单 NUMA 节点、503 GiB 系统内存。
- AI 辅助实现、复核和验证；没有修改模型权重。
- 性能基线是当前仓库修改前的版本；本轮没有把参考仓库整体启动作为另一组服务对照。

每个 case 的精确命令、环境变量、源码 diff 摘要、新 kernel SHA256、逐轮结果保存在 [结构化结果](qwen38next-optimization-20260909.json)。原始 diff 和完整输出在文末的测试目录中。

## 实现及适配

| 改动 | 参考来源 | 本地实现与适用范围 |
| --- | --- | --- |
| Mamba speculative 状态访问边界 | `a34e0656b` | 限制 accepted-token 和 block-table 列范围，跳过空 block，非法状态输出清零。保留本地持久 request slot 映射、int64 地址偏移和融合 gate 路径。 |
| 重复图像 encoder cache | `3958a420f0` | 同一请求中最后一次使用该 hash 后才释放请求引用，避免分块 prefill 中提前驱逐。 |
| UniProc PLE 初始化 | `4c10e7e63` | 在模型加载前启动 CPU worker，加载后等待其就绪；原有 Multiproc 路径保留。 |
| CPU PLE 的异步调度输入顺序 | `64806ec945` | 在模型流先提交 D2H，再提交 PLE 等待与 PP 通信；后台只等拷贝完成后发请求。保留本地每请求独立事件、MRV1 CPU 输入快照。 |
| PLE 与运行时 CUDA 模块加载 | 本次真实测试定位 | 在非 FULL 图的 forward 前确保 PLE 输出完成，避免新 QSA kernel 加载与跨进程 GPU 等待形成停顿；完整图重放维持异步。 |
| dummy PLE scale | `710a42b4f` | 补齐 FP8 GPU 占位层 scale，兼容已有独立 PLE dtype 选择和 CPU worker 输入快照。 |
| NVFP4 gate/up padding | `947755647e` | gate/up 分别补齐，weight 与 scale 同步布局。当前 PP4/TP1 Marlin 不触发此辅助函数，不把分数或速度变化归因于它。 |
| GPU PLE 请求布局和融合查表 | `0e0802f463` 的切图思路 | 单个 Triton kernel 完成请求边界查找、EOS/n-gram hash、TP 分片过滤、查表和 FP8 反量化；专用 split op 读取当前请求布局。 |
| 可选 UVA 表存储 | `cc4e4f0911` | 直接在 pinned RAM 分配 FP8/BF16 原格式表，用 CUDA 映射访问，避免加载过程中将整个 PLE 表搬到显存。 |

FP8 融合转换覆盖 E4M3FN 所有编码，包括 subnormal、正负零和 NaN，兼容没有原生 FP8 转换的 SM80。显式 `ple_embedding_dtype` 始终优先于主模型量化格式。

本次长 prefill 的堆栈停在新 QSA kernel 的 `load_binary`。NVIDIA 文档说明，延迟加载可能需要 context 同步；若已有 GPU 工作依赖其他并发工作完成，可能发生死锁。这个说明与本次观察相符，因此增加了非 FULL 图路径的 PLE 完成保护。参考：[CUDA Lazy Loading / Concurrent Execution](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/lazy-loading.html#concurrent-execution)。

## 质量与功能

| 模型 / 实现 | GSM8K | 输出截断数 | 视觉及文本对照 | 状态检查 |
| --- | --- | --- | --- | --- |
| RadixArk / 基线 CPU | 1292/1319（97.95%） | 3 | 未跑 | 短 14/14；长 4/4；矩阵 16/16 |
| RadixArk / 融合 UVA | 1292/1319（97.95%） | 2 | 未跑 | 短 14/14；长 4/4；矩阵 16/16 |
| RadixArk / 修复后 CPU | 未跑 | — | 24/24 | 未跑 |
| RadixArk / 预取实验（未合入） | 124/128（96.88%） | 2 | 23/24 | 短 14/14；长 4/4；矩阵 16/16 |
| FP8 / 基线 CPU | 125/128（97.66%） | 1 | 23/24 | 短 14/14；长 2/4；矩阵 11/16 |
| FP8 / 融合 UVA | 125/128（97.66%） | 1 | 23/24 | 短 14/14；长 2/4；矩阵 8/16 |
| Inferact / 基线 CPU | 126/128（98.44%） | 1 | 23/24 | 短 14/14；长 3/4；矩阵 12/16 |
| Inferact / 融合 UVA | 125/128（97.66%） | 1 | 24/24 | 短 14/14；长 3/4；矩阵 13/16 |
| RadixArk / 基线 CPU、MTP 0 | 未完成 | — | 未跑 | 执行超时 |
| RadixArk / 最终 UVA、MTP 0 | 31/32（96.88%） | 0 | 23/24 | 短 14/14；长 4/4；矩阵 16/16 |
| RadixArk / 最终 UVA、MTP 3 补测 | 31/32（96.88%） | 0 | 24/24 | 短 14/14；长 4/4；矩阵 16/16 |
| RadixArk / CPU 输入顺序修复、MTP 0（后续阻塞） | 31/32（96.88%） | 0 | 未跑 | 四路长 prefill 阻塞 |
| RadixArk / 最终 CPU、MTP 0 | 31/32（96.88%） | 0 | 23/24 | 短 14/14；长 4/4；矩阵 16/16 |
| RadixArk / 最终 CPU、MTP 3 | 124/128（96.88%） | 2 | 23/24 | 短 14/14；长 4/4；矩阵 15/16 |

FP8 两组均错在 GSM8K 索引 12、100、119（0-based），没有新增错题。两组短编号均为 14/14，长编号续写均为 2/4；视觉同一道表格算术题答 163，正确值为 167。四种长编号对照（串行绕过、并发绕过、串行复用、并发复用）的基线为 3/4、3/4、3/4、2/4，UVA 为 2/4、2/4、2/4、2/4，保留这一差异，不声称这些探针等价或已修复。UVA 额外运行了两个私有前缀的逆序恢复和并发恢复，均成功；并发恢复上下文最长 28,051 token，两个私有编号均位于缓存边界之前。

旧版本此前在关闭 prefix cache 和 MTP 的情况下也出现过长编号失败，详见 [既有 GLM/Qwen 报告](glm-qwen-cmp170hx-20260907.md)。这些结果说明问题并非本轮首次出现，但不足以判定全部差异的量化或实现根因。本轮没有修复这一模型输出限制。

Inferact 的首次成绩从 126/128 到 125/128，多错了索引 100。该题写“10 times more”，标准解答却按“增加 10 次”计算；UVA 首次回答 685，标准答案为 175。UVA 在同一服务上完成独立复测，串行 3 次、并发 4 次，共 2/7 得到 175，其他回答为 155 或 685。这说明单题输出不稳定；复测不替换首次成绩。旧代码同 checkpoint 的独立复测为 3/7：三次串行均为 175，四次并发均为 685。两组均有变化，但这不足以证明数值完全等价；保留原始差异。

Inferact 两组短编号均为 14/14、长编号续写均为 3/4。四种长编号矩阵基线为 2/4、3/4、3/4、4/4，UVA 为 4/4、3/4、3/4、3/4。视觉基线为 23/24，表格算术答 177；UVA 为 24/24。这些是本次样本结果，不据此宣称普遍改善准确率。

GSM8K 使用官方 test split、5-shot、temperature 0、seed 42、开启 thinking、最多输出 8192 token、并发 16。1319 是全量；128 和 32 均为该测试集前 N 题的兼容性抽测，不能等同于新一轮全量结果。首次回答计分，复测不替换原成绩。

最终 CPU 路径的 MTP 3 补测为 124/128，与旧版全量运行中前 128 题的 124/128 相同；错题索引也相同，均为 12、45、100、119。它是最后两项 CPU 同步修复后的抽测，不是又一次全量评测。

最终 CPU 的 MTP 0：短编号 14/14、长编号 4/4，四组长编号矩阵均 4/4。MTP 3：短编号 14/14、长编号 4/4，矩阵分别为 3/4、4/4、4/4、4/4；串行绕过缓存的一次请求把新编号 684004 回答为旧编号 582731，该组前缀缓存命中为 0。两组服务均完整结束，视觉均为 23/24，表格算术答 177 而正确值为 167。运行阻塞已消除，但这些模型答错仍保留，不能写成所有状态或视觉答案均正确。

RadixArk 两次全量总分相同，但有 5 题从对变错、另 5 题从错变对；没有要求逐 token 相同。保持 `VLLM_DETERMINISTIC_MOE_ALIGN=0`，同题的输出和 MTP 接受率存在波动。截断或无最终答案计入错误。

功能探针覆盖 JSON、Python 函数执行、多语言、算术和工具调用。状态检查包含约 18k token 长前缀、取消后恢复、MTP 拒绝后的缓存复用、不同请求私有 key、缓存命中和批次变化。全量 RadixArk 还检查 28k 上下文中缓存边界前的私有 key。

视觉组为 **23 个视觉探针 + 1 个文本对照**，包括真实照片、颜色、OCR、表格、空间关系、多图顺序及 4 个约 11k–16k token 的重复图像分块请求。具体未通过项见结构化结果；服务正常返回不能代替答案正确。

## 性能

| 模型 / 实现 | 中文 TPS | 代码 TPS | 英文 TPS | MTP 每轮耗时 ms（中/代码/英） |
| --- | --- | --- | --- | --- |
| RadixArk / 基线 CPU | 71.35 | 112.43 | 100.11 | 29.177/29.228/29.210 |
| RadixArk / 融合 UVA | 71.23 | 114.70 | 102.75 | 28.973/29.099/29.194 |
| RadixArk / 修复后 CPU | 71.62 | 113.36 | 105.22 | 29.123/29.148/29.255 |
| RadixArk / 预取实验（未合入） | 72.69 | 114.03 | 105.84 | 28.898/28.911/28.883 |
| FP8 / 基线 CPU | 70.53 | 109.42 | 106.38 | 29.429/29.462/29.469 |
| FP8 / 融合 UVA | 72.69 | 111.67 | 98.00 | 29.056/29.173/29.167 |
| Inferact / 基线 CPU | 71.69 | 115.75 | 95.93 | 28.884/28.938/28.938 |
| Inferact / 融合 UVA | 74.04 | 118.05 | 96.49 | 28.642/28.666/28.690 |
| RadixArk / 最终 UVA、MTP 0 | 79.45 | 79.44 | 79.45 | 不适用 |
| RadixArk / 最终 UVA、MTP 3 补测 | 71.99 | 113.34 | 106.16 | 28.853/28.930/28.952 |
| RadixArk / CPU 输入顺序修复、MTP 0（后续阻塞） | 67.43 | 67.42 | 67.43 | 不适用 |
| RadixArk / 最终 CPU、MTP 0 | 67.36 | 67.39 | 67.37 | 不适用 |
| RadixArk / 最终 CPU、MTP 3 | 71.41 | 112.93 | 98.46 | 29.150/29.194/29.168 |

表格为三个固定提示词各 **5 轮、512 个 completion token** 的中位数，启用 ignore EOS、关闭 thinking、单请求串行发送。统计服务器报告的 completion token，而非 SSE 分块数。模型启动参数在同 checkpoint 的两组间保持一致，主要对照均为 MTP 3。

解码 TPS 采用 `(completion_tokens - 1) / (总耗时 - TTFT)`。MTP 每轮耗时采用 `(总耗时 - TTFT) / num_drafts`，是帮助识别接受率变化的近似归一化指标，包含目标、草稿、调度和通信，**不是 PLE kernel 的单独耗时**。更精确的首个 SSE 分块 token 数、排除首块后的 TPS、TTFT 和每轮接受率同时保留在 JSON 中。

同题也可能生成不同 token；不要用单个峰值或不同试验的最大值证明优化。RadixArk 预热样本没有混入最终 5 轮。全量 GSM8K 阶段曾与独立 GPU 单元测试重叠，因此不把该阶段总耗时作为吞吐对照；表内测速期间没有并行 GPU 测试。加载顺序和文件缓存未控制，ready 时间仅作记录，不能用来比较冷启动快慢。

## 复现启动

示例以 RadixArk 为例；同一 checkpoint 的基线采用旧代码和 CPU offload，候选采用当前代码和 UVA，其他参数相同。

```bash
NCCL_P2P_DISABLE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_PP_LAYER_PARTITION=12,12,12,12 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_PLE_CPU_OFFLOAD=0 \
VLLM_PLE_USE_UVA=1 \
/home/bul/miniconda3/envs/vllm-backport/bin/vllm serve \
  /home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name local \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 127.0.0.1 --port 8001 --disable-uvicorn-access-log
```

CPU offload 对照将两项设置为 `VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_USE_UVA=0`。两者同时启用会报错。MTP 0 删除 `--speculative-config`。FP8 和 Inferact 除替换模型路径外均增加 `--moe-backend marlin`；RadixArk 的 BF16 草稿使用 auto。

模型路径：

- `/home/bul/dev/models1/Qwen/Qwen3.8-Flash-Next-FP8`：FP8 PLE。
- `/home/bul/dev/models1/Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4`：主模型 NVFP4，FP8 PLE，BF16 MTP。
- `/home/bul/dev/models1/Qwen/Inferact/Qwen3.8-Flash-Next-NVFP4`：主模型 NVFP4，BF16 PLE，NVFP4 MTP。

## 代码验证

- CPU encoder cache、NVFP4 padding、executor 定向：30 passed，5 deselected。
- SM80 PLE、causal conv、Mamba precopy、fused recurrent/sigmoid gating、hybrid state：318 passed。
- SM120 对应 PLE/Mamba kernel 组：302 passed。
- PLE 加载及 worker 集成：46 passed。
- 原 GPU model runner 的 PLE offload 回归：5 passed。输入顺序修复后加强同一套件，SM80、SM120 各 5 passed；新断言在旧代码明确失败。
- 加入运行时 kernel 加载保护后，该组回归在 SM80、SM120 各 6 passed；最终改用统一 accelerator API 后，在空闲 GPU 再跑新增用例，1 passed。
- 最终融合反量化与负零修复后的完整 PLE 套件：SM80、SM120 各 30 passed。
- 完整 executor 套件在空闲 GPU 3、spawn 模式下复测：12 passed。
- 最终源码 pre-commit（包含 mypy 3.10）及单独 mypy 3.12 检查通过，日志为 `final-runtime-source-checks.log`；中文文档和 JSON 的 pre-commit 另行通过。

测试组存在重叠，不相加作为独立用例数。覆盖 CPU/GPU 参考结果、FP8/BF16、TP 分片掩码、EOS、空请求、padding、图重放时边界变化、不连续 request slot、无效 accepted count 和 block lookup。真实多 rank TP 性能未测试；模型对照使用固定 MTP 3 和 MTP 0，未验证自适应草稿宽度。

以下命令使用本次 conda 环境，可在 GPU 空闲时复核最终 PLE 和相关 kernel：

```bash
CUDA_VISIBLE_DEVICES=0 /home/bul/miniconda3/envs/vllm-backport/bin/python -m pytest \
  tests/models/qwen4_exp/test_ple.py \
  tests/kernels/mamba/test_causal_conv1d.py \
  tests/kernels/mamba/test_precopy_mamba_align.py \
  tests/kernels/test_fused_sigmoid_gating_delta_rule.py \
  tests/v1/worker/test_mamba_hybrid_model_state.py -q

CUDA_VISIBLE_DEVICES=3 /home/bul/miniconda3/envs/vllm-backport/bin/python -m pytest \
  tests/models/qwen4_exp/test_ple.py -q

CUDA_VISIBLE_DEVICES=0 /home/bul/miniconda3/envs/vllm-backport/bin/python -m pytest \
  tests/v1/worker/test_gpu_model_runner.py -k ple_offload -q
```

## 限制与异常记录

- 最初隔离基线未包含生成的 FlashAttention Python 构建文件，启动失败；补齐同环境构建产物后重跑，此失败没有性能样本。
- 同步 UVA 全量结束后，视觉评测脚本把 HTTPX 的 `.text` 属性误写为 `.text()`，在发出任何视觉模型请求前退出；全量 GSM8K、测速和状态检查已完成。脚本已修复，视觉以随后独立补测为准。
- 最初误启动完整 executor 集成测试时与大模型服务竞争 GPU：1 个自定义 executor 用例启动超时，另 3 个显存预留不足。定向用例正常；补齐测试小模型缓存后，以空闲 GPU 3 和 spawn 模式完整复跑 executor，12 passed。初次失败还留下一个 300 MiB 的空闲 GPU 0 context，各组正式测速期间均存在且没有 GPU 计算；全部测速结束后已清理；记录在 `cleanup-status.json`。
- 旧代码 CPU PLE 的 MTP 0 对照在首批并发请求中停顿，随后 `sample_tokens` RPC 超时；这次运行没有 GSM8K 或有效测速成绩。主工作树保持旧 CPU 输入传输顺序时也复现；栈显示后台 notifier 等 D2H 事件，CPU worker 等请求。参考 `64806ec945` 适配了模型流输入拷贝顺序。该版完成 GSM8K、测速及串行状态检查后，在四路长 prefill 上又停顿；独立复现的栈定位到 Triton `load_binary` 加载新 QSA kernel。因此又补入非 FULL 图 forward 前等待 PLE 结果的保护，“最终 CPU”两组均已完整结束。首次 debug 服务抓栈后主动停止，第二次长 prefill 复现随后 RPC 超时退出；两次 debug 没有测速。中间版本已完成的测速单列，并标明后续阻塞；没有把旧版失败按 0 TPS 计算收益。
- UVA 锁定大块主机内存，FP8 PLE 表约 47.7 GiB，BF16 约 95.4 GiB。它没有压缩表，也没有证明普遍优于现有 CPU worker。
- 未加入真正 packed NVFP4 PLE；现有 NVFP4 主模型的 PLE 为 FP8 或 BF16，不能用它们验证 packed NVFP4 表。
- 未移植 V2 hybrid attention packing、大并发图扩容、NCCL 升级适配及进程生命周期清理；不把这些仍待验证的候选混入当前改动。

完整测试脚本、精确源码 diff、命令、日志、逐题响应和视觉素材引用：`/tmp/qwen38next-opt-20260909/`。`*-process.json` 中的 source diff 与新增 kernel SHA256 对应各次实际运行版本。仓库内保留精简 JSON 便于长期核对。
