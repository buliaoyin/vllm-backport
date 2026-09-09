# DeepSeek V4：A100 仓库与当前仓库优化差异

日期：2026-09-08。

后续已实施本文第 1–8 项中的适用部分，详见[三卡优化与验证](dsv4-optimization-20260908.md)。下文保留实施前的对比结论，避免混淆基线与优化后的状态。

## 对比范围和结论

| 仓库 | 当前分支 | 提交 |
| --- | --- | --- |
| `/home/bul/dev/vllm-backport` | `codex/deepseek-v4-vision-tested` | `9201d9936cf7c90be518d3a5fc880c32c32202c4` |
| `/home/bul/dev/dsv4/vllm-dsv4-a100` | `170hx-dsv4f-pp3-1m` | `a65c660da` |

两个仓库的共同祖先为 `62195e9784ebec1ece42b88a861734e0702cc2d5`。按对方相对共同祖先修改的文件建立差异清单，再核对当前源码的调用路径、同等实现和默认配置。不能只凭提交号、环境变量名或函数名判断功能缺失。

**当前最值得吸收的是 PP／DSpark 显存优化、结构化输出正确性修复、SM80 注意力执行路径的优化。** 对方的大量基础支持已经存在于当前仓库，计算图调度问题还被当前更晚的修复覆盖。

这是源码审查，没有移植运行代码，也没有在两个仓库之间进行模型性能 A/B。文中“预计收益”来自代码行为分析；对方注释中的性能数字属于对方的特定 A100 测量，不能当成本机实测收益。已有三卡视觉、MTP 和 GSM8K 抽测记录保持原样。

## 确认缺失或尚未接通的优化

### 1. PP 接收缓冲区直接复用

对方的 `f73035cd5` 和 `94ad349dd` 让 PP 通信直接接收至 V2 执行器的预分配输入缓冲区。`PPRecvBufferGuard` 跟踪接收、主模型使用、DSpark 使用和释放状态，并校验异步完成的代次；可额外校验 CUDA stream。开关 `VLLM_PP_REUSE_RECV_BUFFER` 默认开启，当前实现限制在 V2、TP1。

当前仓库仍先分配接收张量，再拷贝进执行器的固定缓冲区，没有上述接收接口及生命周期保护。这项直接适用于刚测的 PP3／TP1，预计减少临时显存和设备内拷贝；实际省多少、是否加速尚未测量。

证据：[对方接收缓冲区保护](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu/pp_utils.py:25)、[对方启用条件](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu_worker.py:180)、[当前接收接口](/home/bul/dev/vllm-backport/vllm/distributed/parallel_state.py:1114)。

### 2. DSpark 复用目标模型的工作区

对方在 DSpark 下不分配普通 MTP 使用的 `_mtp_hidden_buffer`；DSpark 自身不再常驻一份额外的 `hidden_states`，并在不与辅助隐藏状态别名重叠时，复用已经消费完的 PP 输入空间拼接辅助隐藏状态。

当前 `use_eagle()` 也包含 `dspark`，因此仍分配上述 MTP 缓冲区；草稿执行器还分配 `hidden_states`，沿用 `torch.cat` 加后续复制。对方的优化同时削减固定分配与临时分配，尤其适合三卡方案的最后一张卡。

证据：[对方 MTP 缓冲区条件](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/models/deepseek_v4/nvidia/model.py:91)、[对方 DSpark 工作区](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:83)、[当前 MTP 分配](/home/bul/dev/vllm-backport/vllm/models/deepseek_v4/nvidia/model.py:1394)、[当前 DSpark 分配](/home/bul/dev/vllm-backport/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:63)。

### 3. C128 元数据缓冲区合并

对方把 C128 层的 decode top-k 和 prefill top-k 放在一个缓冲区的互不重叠区域；两部分行数之和受本轮 token 数限制。当前分别按完整批次上限分配两份。

这是减少元数据显存，不是跨层共享所有临时张量。可节省一份 `max_num_batched_tokens × c128a_max_compressed × sizeof(int32)` 的分配，每个相应 metadata builder 分别计算；不能把它直接等同于整机可增加多少上下文。

证据：[对方单缓冲区](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/models/deepseek_v4/sparse_mla.py:183)、[当前双缓冲区](/home/bul/dev/vllm-backport/vllm/models/deepseek_v4/sparse_mla.py:164)。Ampere 的 metadata builder 继承这套基础实现，所以并非只对 Blackwell 有效。

### 4. V2 固页内存暂存池

对方新增 `PinnedStagingPool`，分别用于请求索引、logit 累计偏移、query 起始位置、grammar 索引和 grammar bitmask；启动时预留容量，扩容时保留旧缓冲区，避免尚未完成的异步复制读到已释放内存。

当前 `async_copy_to_gpu()` 仍调用 `x.pin_memory()`，没有这些按调用位置分开的暂存池。预计收益主要在批次增长、混合请求和 JSON Schema 约束输出的延迟抖动上，不能仅用稳定单并发的 token/s 评价。

证据：[对方暂存池](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu/buffer_utils.py:44)、[对方三个输入池](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu/model_runner.py:173)、[当前 pin_memory 调用](/home/bul/dev/vllm-backport/vllm/v1/worker/gpu/buffer_utils.py:40)。

### 5. PP／DSpark 结构化输出的批次对应关系

这是正确性修复，不应只按速度排序：

- `a6d6ab46f`：把草稿 token 随产生它的模型输出返回，携带 `producer_step_id`；调度器按请求及生成步骤匹配草稿，避免多个在途批次覆盖同一份草稿交接状态。
- `6e959b2ea`：V1 的 PP 结构化输出在计算下一份 grammar mask 前，等待相关请求的在途结果推进语法状态。
- 对方 `broadcast_draft()` 强制 `copy=True` 生成广播快照，并检查草稿形状；当前只做 dtype／contiguous 转换，当输入已经符合类型和布局时不保证产生独立副本。

当前没有步骤标识和对应的调度器缓存，也没有上述 V1 检查。此次未复现当前 JSON／Schema 请求的实际故障；此前通过的自然语言、OCR 和算术探针不能替代这些场景的回归。

`9dd006256` 是先禁用 PP 结构化输出推测解码的临时方案，后来被 `a6d6ab46f` 的完整修复替代，**不应把禁用补丁再单独移植回来**。其配套的 XGrammar 终止状态保护，当前已另有实现，也不属于缺失项。

证据：[对方异步草稿交接](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu/async_utils.py:68)、[对方步骤匹配](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/core/sched/scheduler.py:2186)、[对方 V1 在途检查](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/core/sched/scheduler.py:2580)、[对方广播快照](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/worker/gpu/pp_utils.py:358)、[当前草稿广播](/home/bul/dev/vllm-backport/vllm/v1/worker/gpu/pp_utils.py:239)。

### 6. C128 预填充 query 分块：辅助代码已有，主模型调用未接通

当前有 `build_query_blocks()`、`prefill_query_block_size()` 和分块内核，但 DeepSeek V4 的 Ampere／ROCm `_forward_prefill()` 不调用该分支，仍走组合索引和普通稀疏 prefill 路径。

对方在 C128 层按 query 块共享 KV 读取，并缓存块映射，跳过该路径不需要的索引组合与打包。`2fa9bbe26` 还修正 TP1 的共享内存问题：使用 `auto_block_m = 64 // block_h`，TP1 的 `block_h=16` 对应 `block_m=4`，不是 TP8 使用的 8。

这是值得验证的长输入性能优化。**移植到视觉模型时必须保留图片双向注意力区间**：对方是文本模型路径，没有当前 `left_visible`／`right_visible`／图片 sentinel 的完整处理。不能直接让视觉请求绕过现有图片区间逻辑。

证据：[对方 prefill 分支](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/models/deepseek_v4/amd/rocm.py:861)、[对方 TP1 tile 选择](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:2718)、[当前 prefill 路径](/home/bul/dev/vllm-backport/vllm/models/deepseek_v4/amd/rocm.py:990)、[当前独立 helper](/home/bul/dev/vllm-backport/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:3214)。

### 7. SM80 decode 按层类型选择 tile，并接通固定 split 开关

对方对 C4 层采用 `BLOCK_K=64`，C128／纯 SWA 保持 32；decode 的 head tile 也按实际头数选择。当前实际 decode 路径仍固定 `block_h=16`、`block_k=32`，即使存在相关辅助函数，也没有接入这条调用链。

对方注释记录 C4 内核在其 A100 场景改善约 6.4%–7.9%，但明确指出会改变 softmax 累加顺序。此数字不是全模型加速率，也不是本机实测。TP1 的 head 数本来就超过 8，动态 head tile 的那部分收益主要属于更大的 TP。

还有一个配置与代码不一致之处：当前注册了 `VLLM_DSV4_FIXED_DECODE_SPLITS`，默认 16，但全仓库运行源码中除环境变量注册外没有读取它；对方在 decode 执行函数中实际使用该开关。因此不能将当前设置了 16 当作已经固定了 split 数。

证据：[对方层类型 tile 选择](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:2903)、[对方实际 decode 调用](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:3121)、[当前实际 decode 参数](/home/bul/dev/vllm-backport/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:3675)、[当前开关注册](/home/bul/dev/vllm-backport/vllm/envs.py:2273)。

### 8. 压缩 KV 融合内核的 warp 调整

对方根据压缩比及 overlap 设置 warp 数：512 维压缩器的 C4 情形为 4，C128 为 16，避免较大的归约块在 SM80 上使用过少 warp。当前对应路径统一为 4。

这项改动较小，但需要 C4／C128 的压缩结果对照和实际内核测量；移植时应按设备能力选用，保留当前 ROCm 的 NaN 处理等较新代码。

证据：[对方 warp 选择](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:33)、[当前固定值](/home/bul/dev/vllm-backport/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:70)。

### 9. FP8 反量化为 BF16 时的按层排除

当前已经有 `VLLM_MARLIN_FP8_DEQUANT_BF16`，对方额外提供 `VLLM_MARLIN_FP8_DEQUANT_EXCLUDE`，可按层名前缀让指定层继续使用 Marlin；其余层反量化后走 BF16 线性运算。

这是精细控制速度和显存交换的工具，不是无代价的普遍提速。三卡显存紧张，优先级低于不增加权重占用的缓冲区优化。还必须保留当前 `_block_scale_name()` 对不同量化模型 scale 名称的兼容，否则可能影响现有 GLM／Qwen 支持。

证据：[对方按层排除逻辑](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/model_executor/kernels/linear/scaled_mm/marlin.py:34)。

## 存在额外实现，但当前 PP3／TP1 优先级低

| 项目 | 当前状态 | 适用条件和判断 |
| --- | --- | --- |
| INT8 压缩 all-reduce 与 mHC 融合消费 | 有 `ar_int8.py` 和开关，但没有完整 CUDA 算子注册、模型调用与 mHC INT8 消费链 | 对方完整实现面向 TP 通信；TP1 不起作用，且有量化误差，需要单独质量验证 |
| `VLLM_LOCAL_ARGMAX_ALLREDUCE` | 当前已有局部 argmax 后交换候选，使用 all-gather；没有对方的可选 all-reduce 交换方式 | 只影响 TP>1；对方默认关闭，注释明确记录 graph warmup 未初始化数据可导致非法 token ID |
| Marlin MoE 的 48／64 block 自适应选择 | 当前没有 `select_block_size_m()`／`moe_padded_rows()` 这套候选接口 | 对方正式调用也不传 `padded_rows`，仍退回原有阶梯规则；额外同步开销抵消内核收益，不应宣传为默认已生效的优化 |

INT8 链路证据：[对方 CUDA 算子](/home/bul/dev/dsv4/vllm-dsv4-a100/csrc/libtorch_stable/custom_all_reduce.cu:128)、[对方注册](/home/bul/dev/dsv4/vllm-dsv4-a100/csrc/libtorch_stable/torch_bindings.cpp:971)、[当前仅保留的 Python 调用](/home/bul/dev/vllm-backport/vllm/model_executor/kernels/mhc/ar_int8.py:172)。Marlin 候选接口的限制写在[对方函数说明](/home/bul/dev/dsv4/vllm-dsv4-a100/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:103)。

## 已有、等效实现或默认配置差异

- **SM80 基础适配已存在。** Ampere backend、FP8 转换支持及多项 indexer／Marlin 基础优化已经移植；Ampere backend 文件本身两边完全一致。
- **长上下文 logits 按行分块已存在。** 当前 `VLLM_DSV4_LOGITS_ROW_CHUNK` 默认 128，并兼容 `DSV4_LOGITS_ROW_CHUNK` 别名；对方默认关闭，README 建议显式设为 64。对方 `a65c660da` 不能作为当前缺失此功能的依据。
- **确定性 top-k 已存在。** `topk.cu`、`topk_histogram_4096.cuh` 等关键文件完全一致；不能按另一仓库的汇总提交重复记为新增。
- **DSpark 基本 PP 能力、词表分片和融合 Markov 已存在。** 本机 PP3／PP4 已实际跑通。应吸收本报告列出的内存及批次对应修复，而不是回退整个 DSpark 实现。
- **计算图缺口已有更新的修复。** 当前 `c74dd2d20` 分别为 FULL 和 PIECEWISE 构造候选范围，覆盖对方 `_pad_up_candidate()` 要解决的模式遮蔽问题；同时保留当前动态推测长度、视觉输入和图显存估算适配。此次补跑相应 5 项现有测试，全部通过。
- **共享 eager scratch 的撤销不是缺失。** 当前已有 `f1178f3a0` 撤销共享 scratch，不应恢复对方较早版本已撤销的方案。
- **XGrammar 的终止状态保护已有。** 当前 `validate_tokens()` 已检查终止状态并在接受终止 token 后停止。

计算图证据：[当前候选范围构造](/home/bul/dev/vllm-backport/vllm/v1/worker/gpu/cudagraph_utils.py:295)、[对应混合批次回归](/home/bul/dev/vllm-backport/tests/v1/cudagraph/test_cudagraph_manager.py:285)。分块配置证据：[当前环境变量及别名](/home/bul/dev/vllm-backport/vllm/envs.py:2276)。

下面这些同名开关两边都注册了，不属于“缺少代码”。是否实际执行仍须检查上层开关和调用条件：

| 开关 | 当前默认 | 对方默认 | 对三卡 TP1 的意义 |
| --- | --- | --- | --- |
| `VLLM_DETERMINISTIC_MOE_ALIGN` | 1 | 0 | 有速度与数值稳定性取舍，不建议照抄关闭 |
| `VLLM_MHC_PRENORM_SHARD` | 0 | 1 | TP 分工；另外受融合平方和开关限制 |
| `VLLM_UNREPLICATE_ATTN_GEMMS` | 0 | 1 | TP 分工，TP1 没有去重收益 |
| `VLLM_INDEXER_QUERY_SHARD` | 0 | 1 | TP 分工，TP1 不生效 |
| `VLLM_SPARSE_PREFILL_EXACT_TILE` | 0 | 1 | 取决于实际 head tile 是否精确匹配 |
| `VLLM_SPARSE_RAGGED_FAST_SCAN` | 0 | 1 | 特定 prefill 长度下可测，需确认进入相应扫描分支 |

## 不建议移植的实验和能力边界

对方保留了 `VLLM_MARLIN_DENSE_OCCUPANCY`、`VLLM_MARLIN_RIGHTSIZE_SMEM`、`VLLM_MARLIN_SPIKE_WARPS` 等实验代码，但其[实验记录](/home/bul/dev/dsv4/vllm-dsv4-a100/benchmarks/kernels/dsv4_sm80_refutations.md:61)明确报告负收益。它们是研究留档，不能归入建议默认开启的优化。

对方 [README](/home/bul/dev/dsv4/vllm-dsv4-a100/README.md:3) 的“170HX 三卡、1M 上下文”针对 `DeepSeek-V4-Flash-0731` 文本模型，还注明 GPU 解锁依赖。本次没有核实其 1M 服务和质量结果；当前测试的是 `DeepSeek-V4-Flash-Vision-Exp`，不能直接据此扩大已经验证的上下文或并发范围。

对方分支较旧，不包含本地完整视觉实现。整体覆盖文件会丢失视觉路由所需的 PP 原始 token ID、图片区间处理、部分更新的量化兼容和计算图逻辑。应按功能移植，保留当前代码中的这些能力。

## 建议顺序与验证范围

1. **先做 PP／DSpark 正确性和显存：** 第 1–5 项。PP3／TP1，MTP 开关对照；JSON object／严格 Schema、普通请求混跑、请求取消与结束、连续多轮；核对草稿接受率、输出合法性和峰值显存。
2. **再做 SM80 注意力：** 第 6–8 项分别 A/B。覆盖 C4／C128、短输入与较长输入、单请求和混合 prefill／decode；视觉用例必须包含图片双向区间、多图、长前缀及 chunk 边界。固定 split 的开关要先证明真正生效。
3. **最后考虑按层 BF16 与 TP 专用优化。** 第 9 项先测新增显存是否值得；INT8 all-reduce 等留给未来确实需要 TP>1 的配置。

每个影响模型数值或异步状态的组合，沿用现有视觉探针与 GSM8K 质量门槛；正式采纳时应补全量 GSM8K，而不能只依赖此前 32 题抽测。当前 GLM／Qwen 的 PP、MTP、量化兼容也需要回归。

## 本次验证和证据

执行现有测试：

```bash
PYTHONPATH=/home/bul/dev/vllm-backport \
/home/bul/miniconda3/envs/vllm-backport/bin/python -m pytest \
  tests/v1/cudagraph/test_cudagraph_manager.py \
  -k 'uniform_decode or mixed_batch' -q
```

结果：`5 passed, 6 deselected`。未新增测试，未启动模型服务，未修改对方仓库。

文件差异清单、逐文件 diff、环境变量清单与测试日志保存在 `/tmp/dsv4-a100-comparison-20260908/`。本次只新增此中文对比报告；此前尚未提交的三卡验证报告修改保持不变。
