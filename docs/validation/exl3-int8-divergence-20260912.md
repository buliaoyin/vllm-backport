# EXL3 强制 INT8 全模型 logits 散度（2026-09-12）

## 结果

三张 CMP 170HX、固定 PP **16/15/14**。64 道 GSM8K 共 **10,672 个输出预测位置**，
逐 token 固定相同历史，比较全词表分布。参考是原生 EXL3 prefill，不是未量化 BF16 模型。

| 对照 | 平均 KL | P95 KL | P99 KL | 最大 KL | 平均 JS | 平均 TV | Top-1 一致率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原生重复 / 原生参考 | 0.001093 | 0.005656 | 0.022373 | 0.181711 | 0.000270 | 0.572% | 99.447% |
| 强制 INT8 / 原生参考 | 0.002804 | 0.015972 | 0.051198 | 0.503459 | 0.000687 | 0.992% | 99.063% |

KL 为 **KL(原生参考 || 候选)**，KL/JS 单位 nats。TV 表示概率质量差异；
top-1 一致率是固定历史下下一 token 的一致率，不是题目正确率。

INT8 平均 KL 约为原生重复的 **2.57 倍**，
观察到 100 个 top-1 变化；原生自身重复为 59 个。
INT8 中，原生 top-1 概率 ≥90% 的位置发生 0 次翻转。
按题目分别求均值，59/64 题的 INT8 KL 高于原生重复。
低于 90% 原生置信度的 1648 个位置中，INT8 top-1 一致率为 93.932%；
其余 9024 个高置信度位置一致率为 100%。这仍只描述本次固定历史样本。
重复运行并非逐位一致，因此不能把 INT8 对照的全部散度都归于新增近似算术；
同时，较小的平均散度也不能证明自由生成答案无退化。

## 首 token 与后续 decode

| 阶段 | 位置数 | 原生重复平均 KL | INT8 平均 KL | INT8 P95 KL | INT8 Top-1 一致率 |
| --- | --- | --- | --- | --- | --- |
| Prefill 首 token | 64 | 0.004393 | 0.010867 | 0.031476 | 100.000% |
| 后续纯 decode | 10608 | 0.001073 | 0.002756 | 0.015734 | 99.057% |

这 64 题的 prefill 在各自 B4 的首批完成，后续为 1–4 行纯 decode。算子计数确认，
两边后续使用相同的专家 plain INT8 GEMV；散度可以随不同数值的 KV/状态延续到后续步骤。
这里的“后续 decode 散度”不是把 decode 改成新 INT8 prefill 内核后测得的。

## 上轮三道新增错题

| 题号 | 位置数 | 原生重复平均 KL | INT8 平均 KL | INT8 最大 KL | 原生 / INT8 Top-1 变化数 |
| --- | --- | --- | --- | --- | --- |
| 9 | 214 | 0.000739 | 0.001117 | 0.030301 | 2 / 2 |
| 12 | 288 | 0.002695 | 0.004802 | 0.137984 | 2 / 4 |
| 21 | 101 | 0.000999 | 0.006478 | 0.268917 | 0 / 1 |

在固定原生正确历史下，三题最后数字的 top-1 仍分别为 460、13、14，没有翻转。
第 21 题在输出 step 61（从 0 计数）的 KL 为 0.268917，
相同前缀 `Step 2: Find` 后的 top-1 从 `the` 变为 `when`；最终 `14` 位置
step 99 的 KL 约为 5.3e-7。这样的措辞分支本身不等于算术错误。
本轮没有自由生成到错误轨迹，不能据此宣称复现、定位或修复了上轮三题的失败。

## 长输入与并发

这一部分是访问码检索，**10 个请求仅有 49 个输出预测位置**；每请求为 4–5 个 token。
输入长度到 64K，但没有对全部输入 token 采集 logits。样本很少，仅作为长上下文数值检查。

| 输入组合 | 输出位置数 | 原生重复平均 KL | 全 prefill 强制 INT8 平均 KL | ≥4096 行 INT8 平均 KL | 强制 / ≥4096 Top-1 一致率 |
| --- | --- | --- | --- | --- | --- |
| 8K × 1 | 5 | 0.00447736 | 0.00163362 | 0.00225172 | 100.0% / 100.0% |
| 16K × 1 | 5 | 0.00171674 | 0.00470093 | 0.00262285 | 100.0% / 100.0% |
| 32K × 1 | 5 | 0.00111534 | 0.00125554 | 0.00170614 | 100.0% / 100.0% |
| 64K × 1 | 5 | 0.00152417 | 0.00393281 | 0.00158891 | 100.0% / 100.0% |
| 8K × 4 | 20 | 0.00163132 | 0.00104907 | 0.00176319 | 100.0% / 100.0% |
| 32K × 2 | 9 | 0.0028756 | 0.00176311 | 0.00131414 | 100.0% / 100.0% |

其中 3 个位置是 decode 与其他请求 prefill 合批，可能一并经过新 INT8 内核。
4096 行对照保持相同的 6144 调度 chunk，只让小尾块回到原生；它不是完整的 auto 调度运行，
也不是量化误差补偿。16K、64K 的两种 INT8 门槛实际算子/行数计数完全相同，
因为其所有 prefill chunk 均 ≥4096；它们的散度仍有变化，不能解读为 4096 门槛改善精度。
逐请求及纯/混合 decode 的细分保存在 summary.json。

## 判断

本轮测得强制 INT8 相对于原生重复有额外分布偏移，但 top-1 大部分仍相同。
散度无法替代题目正确率，尤其不能从 49 个检索输出位置外推完整长上下文能力。
上轮自由生成的原生 64/64、强制 INT8 61/64 仍是独立的质量记录；本轮不改写它，
也不证明其中全部差异来自 INT8。默认继续保持 native，本次没有修改生产内核或默认参数。

## 测量范围与口径

模型为 GLM-5.3-Flash EXL3 4.05bpw，三张 CMP 170HX / SM80，固定 PP 16/15/14。
42 个 MoE 层分配为 13/15/14；TP=1、BF16、每 rank KV 2 GiB、prefix cache 关闭。
所有对照固定 scheduler chunk 6144，原生专家 workspace 2048、M=32；
普通投影 EXL3_INT8_GEMV=2，专家纯 decode 使用 plain INT8 GEMV，没有普通投影 FP16 缓存。
本次仅测散度，探针的同步和数据复制会改变耗时，不作为吞吐成绩。

64 道 GSM8K 使用上轮原生正确输出的全部 token 作为共同历史，分成 16 组、每组 4 题。
长输入使用上轮精确长度的访问码检索输入，覆盖 B1 8K/16K/32K/64K、B4 4×8K、B2 2×32K。
每个长请求仅回放到原生首次 EOS（包含 EOS），不把 EOS 后的无效继续生成算进来。
因此，长输入的长度指上下文长度，散度采样位置是其输出 token，并非全部输入位置。

每组在同一个已加载的模型上依次运行：原生参考、原生重复、强制 INT8。
长输入再运行 INT8 最小行数 4096 的对照。参考 logits 只在内存中保留一组，
比较完即释放；原始逐位置统计、输入/目标 token IDs、批次调度及各 rank 算子计数存档。

- 强制 INT8 把实际行数门槛设为 9，让短 GSM8K prefill 真正经过新算术；
  纯 decode 的 1–4 行仍经过原有专家 GEMV。4096 对照只用于长输入。
- 诊断使用 eager、compile mode 0、VLLM_TOKEN_BUCKET_PAD=0，方便请求结束后切换算术。
  关闭 bucket padding 是必要的：否则小 decode 会补成 16 行，越过人为降低的 9 行门槛。
  这与默认 CUDA Graph 服务配置不同，不应把本轮统计当作默认服务的逐位复现。
- 所有请求先在暂停的 scheduler 中排队，再一起唤醒；跨阶段逐项核对请求名、位置、
  该请求本步 token 数及批次请求数。prefix cache 命中为零，输出 IDs 必须与目标逐一相同。
- 在最后 PP rank 的原始 compute_logits 后采集全词表 logits，然后才把采样结果改成
  共同的下一 token。下一步实际经过正常 attention、KV cache 和 decode 内核。
  两边比较时看到相同的 token 历史，但隐藏状态/KV 中的数值差异保留。
- 长输入并发的混合批次可能把已有请求的 decode token 一并送进 prefill 内核。
  记录实际批次 token 数，并将其与纯 decode 分开；未宣称混合批次的 decode 算术不变。
- 原始 logits 是 BF16、词表 154880；在温度 1 下转 FP64 做完整 softmax 和归约。
  greedy 的 temperature=0 只控制输出采样，不把散度计算退化为 one-hot 分布。

令 P 为本组原生参考、Q 为当前阶段候选：KL(P||Q) = Σ P log(P/Q)，
JS = [KL(P||M)+KL(Q||M)]/2，M=(P+Q)/2，TV=Σ|P−Q|/2。
KL 和 JS 使用自然对数，单位 nats；TV 位于 [0,1]。
原始 JSON 的 kl_native_int8、kl_int8_native 分别对应正向/反向 KL；
在 native_repeat 阶段，候选 Q 是另一次原生运行。
Top-1 用 argmax，平局选择最小 token ID；统计实现用可解析的三项概率分布校验。

“沿原生输出轨迹”是固定历史的条件分布诊断，不是重新自由生成的 GSM8K 正确率，
也不是独立语料困惑度。少数检索题的长上下文散度不能代表完整长上下文能力评测。

## 校验、文件与复现

所有 72 个阶段完成，22 组调度逐项一致；三卡算子计数、纯 decode 行数分布、
全量位置/目标 token 覆盖、数值范围、执行源码和依赖哈希核验通过。
新增诊断代码通过 pre-commit；可解析分布的 KL/TV、自比较、argmax 平局校验通过。
日志在 DONE 之后出现 vLLM 收尾清理超时及 shared_memory 提示；退出码为 0，
确认无残留 GPU 进程，采样阶段无异常。

- [逐组原始结果](exl3-int8-divergence-20260912/results.json.gz)：输入/目标 IDs、逐位置指标、调度和各 rank 计数。
- [统计汇总](exl3-int8-divergence-20260912/summary.json)：分组、首 token、decode、置信度及每请求统计。
- [逐位置 CSV](exl3-int8-divergence-20260912/positions.csv.gz)：可直接用于额外分析。
- [上下文样本](exl3-int8-divergence-20260912/contexts.json)：大散度位置及三道旧错题的 token 文本。
- [完整核验](exl3-int8-divergence-20260912/audit.json) 与 [文件清单](exl3-int8-divergence-20260912/manifest.json)。
- [运行日志](exl3-int8-divergence-20260912/run.log.gz) 和冻结的 runner/worker/分析脚本也在同一目录。

在已配置的仓库环境中运行，先清除其他 EXL3 环境变量覆盖；存档 run.py.gz 含完整的环境清理逻辑：

```bash
env CUDA_VISIBLE_DEVICES=0,1,2 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=16,15,14 \
  VLLM_TOKEN_BUCKET_PAD=0 VLLM_EXL3_MOE_PREFILL=int8 \
  VLLM_EXL3_MOE_INT8_MIN_TOKENS=9 VLLM_EXL3_MOE_MAX_TOKENS=2048 \
  EXL3_INT8_GEMV=2 VLLM_TRITON_USE_TD=0 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD/benchmarks:$PWD" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python benchmarks/benchmark_exl3_divergence.py \
  --artifacts docs/validation/exl3-adaptive-prefill-20260912 \
  --output /tmp/exl3-divergence.json
```
