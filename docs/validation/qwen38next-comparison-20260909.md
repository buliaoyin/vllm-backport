# qwen38next 与当前仓库的实用改动对比

日期：2026-09-09。本文记录移植前的源码比较和最小 CPU 复核，当时未修改运行代码或启动模型测速；由 AI 辅助完成。

本文保留移植前的比较结论。后续实现与真实模型 A/B 结果见
[优化与性能验证](qwen38next-optimization-20260909.md)。

## 比较范围

- 当前仓库：`/home/bul/dev/vllm-backport`，`codex/deepseek-v4-vision-tested`，`a3c9ee61cc1ef9b66ebbb08bd8e6fc201ed465d8`。
- 参考仓库当前检出：`/home/bul/dev/vllm-qwen38next`，`bul/qwen38next_offload_latest`，`a34e0656bc543b959338eb5e2b55606c87279701`。
- 共同祖先：`c01b50e390e6d3d0019aa53f41ff1198c8105e5a`。两侧分别有 147、83 个祖先不重合的提交；排除相同补丁和 merge 后，参考侧剩余 76 个提交。这不等于缺少 76 项功能：当前有不少等效或更完整的实现。
- 额外检查了参考仓库已有的 UVA PLE、hybrid attention packing、reasoning/grammar 等分支。`origin/*` 指本机已有的远端跟踪快照，本轮没有联网 fetch，也不声称它们代表远端最新状态。
- 实机环境为三张 CMP 170HX 和一张 RTX PRO 6000 Blackwell，使用 `vllm-backport` conda。读取到 PyTorch `2.13.0+cu130`、NCCL `2.29.7`。

结论：值得选择性补入正确性修复和 PLE 新能力；性能方面，UVA offload 值得单独实验。整体合并会带入大量不适用的上游更新，并可能覆盖当前已有的 PP/MTP 修复。

## 优先考虑的缺口

### 1. Mamba 推测解码的索引边界防护

来源：`a34e0656b`。

参考增加了以下防护，当前没有：

- causal convolution 在 `num_accepted_tokens < 1` 或超过本次序列长度时，不再据此访问卷积状态。
- 两个 recurrent/gating Triton 内核在读取 `ssm_state_indices` 前，检查 `num_accepted_tokens - 1` 是否在行范围内；无效状态路径清零输出。
- `_copy_mamba_state_block()` 对源/目标 block-table 列、偏移后的 temporal 列及非正 block ID 做保护。

这是 Qwen/Mamba 的稳定性补强，适用于 MTP、padding、恢复请求等路径。本轮源码确认缺失，没有通过故意非法 GPU 访问来复现崩溃，也不能据此断定先前的某个模型错答由它引起。

当前位置：[causal_conv1d.py](/home/bul/dev/vllm-backport/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:874)、[fused_sigmoid_gating.py](/home/bul/dev/vllm-backport/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:106)、[mamba_utils.py](/home/bul/dev/vllm-backport/vllm/v1/worker/mamba_utils.py:192)。

**移植边界**：只取防护逻辑。参考当前版本仍在部分状态拷贝处使用 batch 行号，当前已通过 `dc2d6a86a` 修正为持久 request slot；还要保留当前 recurrent 内核的 gate/sigmoid 融合参数。不能整文件覆盖。

验证：有效输入逐元素/容差一致、0/负数/过大接受数、无效 block ID、padding、非连续 request slot、抢占恢复、MTP 开关和 prefix cache 复用。

### 2. 同一请求重复图像的 encoder cache 引用生命周期

来源：`3958a420f0`（源提交标注 #54284）。

当前缓存以 request ID 记录引用，一个请求中重复使用同一图像时，释放首次出现的位置会提前移除整个请求的引用。参考在同一请求仍有其他位置使用相同图像 hash 时保留引用，避免该图像仍待使用就被驱逐或重算。

当前位置：[encoder_cache_manager.py](/home/bul/dev/vllm-backport/vllm/v1/core/encoder_cache_manager.py:224)。

从参考测试提取的两个最小 CPU 用例，在当前代码上均失败：提前进入可释放集合，并能被另一个请求挤出。这是已复核的行为缺口，与 DeepSeek Vision 的多图、多轮图片复用有关；之前普通视觉功能通过不代表覆盖了这个缓存压力场景。

验证：这两个用例、同图多次出现、图文分块边界、缓存压力、请求取消后再复用。

### 3. UniProc PLE offload 初始化及 dummy 加载补全

来源：`4c10e7e63`、`710a42b4f`。

当前 `MultiprocExecutor` 会启动并等待 PLE CPU worker，`UniProcExecutor` 缺少相应步骤。参考增加 `spawn_ple_offload → load_model → wait_ple_offload_ready` 的顺序。最小模拟 worker 测试在当前代码上失败，实际只看到 `init_worker → init_device → load_model`。

当前位置：[uniproc_executor.py](/home/bul/dev/vllm-backport/vllm/v1/executor/uniproc_executor.py:52)。

参考另有 dummy PLE 元数据初始化，补齐 FP8 scale、NVFP4 scale/LUT 和 offload 输出尺寸的准备。它用于 dummy 加载、profiling 与基准启动，不能当成真实模型质量验证。

对当前 PP4/MultiProc 启动没有直接提速收益；对 PP1/UniProc 加 CPU PLE offload 的完整性有用。移植时保留当前 CPU worker 的 PP 环境隔离。

### 4. PLE 的请求布局依赖与计算图边界

来源：`0e0802f463`；另有 `f93804f24` 中的 embedding 自定义算子。

参考把 n-gram ID 生成拆为 `qwen4_exp_compute_ple_ngram_ids`，并将其加入 piecewise graph 的切分操作。目的在于避免“补齐后的 token 总数相同，但请求数量/边界不同”时复用错误的 n-gram 布局。当前 GPU PLE 查表路径缺少这项切分。

参考位置：[ID 计算操作](/home/bul/dev/vllm-qwen38next/vllm/models/qwen4_exp/nvidia/ple_layer.py:1689)。参考还有 `qwen4_exp_ple_embed`，将数据相关索引与查表封装成自定义操作，减少 Inductor 对这段计算的介入。

主要适用 GPU 驻留 PLE 和编译路径；目前使用 CPU worker 查表的 Flash 配置不应据此推断会提速。源提交的 CPU mock 测试不等于完整 CUDA Graph 验证，需要同时检查两层自定义操作的实际切图效果。

验证：eager、piecewise、full graph 对照；相同 token 数下切换 1/2/多请求布局；MTP 开关、不同前缀和跨请求隔离。

### 5. MRV2 异步调度的 PLE 输入顺序（实施阶段补充确认）

来源：`64806ec945`（源提交 `4e8b849b8d97`）。

参考在模型流提交 PLE 的 GPU→CPU 输入拷贝，再由后台线程等待完成事件并发布请求；当前比较基线是在后台流中提交拷贝。后续真实 PP4、无 MTP、16 并发测试复现了基线阻塞：后台等待 D2H 完成、CPU worker 等待请求，其余 PP rank 等待第 0 卡。

这项应列入正确性修复。实现时保留本地每请求独立事件及 MRV1 CPU 输入快照，避免照搬后丢失已有的跨批次保护。修复和同参数复测记录见 [优化报告](qwen38next-optimization-20260909.md)。

## 有价值但取决于配置的改动

| 改动 | 来源 | 当前缺口与实际价值 | 后续验证重点 |
| --- | --- | --- | --- |
| PLE 表本身的 packed NVFP4 存储、查表和 CPU offload | `f93804f24`，配合 `710a42b4f` | 当前有“主模型 NVFP4 + PLE FP8/BF16”，没有 NVFP4 PLE 表方法。参考保留 packed codes、FP8 block scale 和 FP32 global scale，仅解量化命中的行。 | 真正采用 NVFP4 PLE 的 checkpoint；packed/sharded 加载、global scale 一致性、CPU/GPU 查表、PP、图捕获、全量质量。 |
| NVFP4 gate/up 分别 padding | `947755647e` | 当前 `align_fp4_moe_weights_for_fi()` 将 gate/up 连续复制到补齐后的融合张量，会把 up 的开头放入 gate 的 padding。CPU 回归已失败。 | 640/TP4=160 补齐到 192、scale 同步布局、目标后端实际 GEMM；保留已有 gate/up scale 协调。 |
| MTP uniform decode 图覆盖 | `d3d79ffc1e`，#50488 | 按请求数和草稿宽度补齐捕获尺寸，并覆盖动态草稿宽度。当前默认 token 网格可能遗漏较大或特殊宽度批次。 | 显存与启动耗时、固定/动态 MTP、显式 capture 配置、图命中率。 |
| 视觉路径 pinned H2D | `6c18a54648`、`46a83642f6` | 当前 V2 encoder 的 `is_mm_embed` 及部分 MM attention 元数据仍来自 pageable CPU 张量。源提交补 pinned 内存/异步上传帮助函数。 | 视觉 TTFT、prefill、并发图像请求、CPU 内存生命周期；不要把收益算到纯文本稳态 decode。 |
| 清理模型和层级 KV cache 的强引用 | `96242aa50d`、`fb68025138` | 当前缺少统一 `clear_layer_kv_caches()`，部分 finalizer/语言模型缓存持有强引用。适用于同进程重复创建/关闭模型和 profiling 清理。 | 对象存活引用、草稿/目标层缓存、CUDA Graph teardown、同进程加载两轮后的显存。 |
| NCCL communicator properties ABI 修复 | `026d5af7f4`，#53008 | 当前结构体 56 字节，参考补至对应新版布局并限制声明版本。当前 NCCL 为 2.29.7；在升级到 2.31+ 且调用该查询时才涉及源提交所述风险。 | 结构体大小、版本钳制、实际启用该查询的通信路径；不是当前 PP 性能差距。 |
| fused SiLU block quant 的 int64 token offset | `06cccf8730`，#53409 | 当前 kernel 的 `token_idx` 仍为 int。源提交避免超大元素偏移在 int32 乘法中溢出。 | 边界索引/大张量、重新编译对应 CUDA 扩展；当前小 prefill 分块不能说明会触发。 |

NVFP4 padding 当前位置：[flashinfer_utils.py](/home/bul/dev/vllm-backport/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py:198)。该缺陷与我们先前的 gate/up **全局 scale 协调**是两件事。它针对调用此辅助函数且需要补齐的 FlashInfer/TRTLLM 路径；当前 PP4/TP1、Marlin 的 Qwen 启动不能直接归因于该缺陷。

图覆盖方面，当前常用 `max-num-seqs=64`、DSpark 7 的最大 uniform batch 为 `64 × 8 = 512`，已处于当前捕获上限。源提交主要改善更大并发、其他草稿宽度和动态分档；没有证据表明直接移植会提高现有单请求测速。扩大图覆盖还会改变启动时间和显存占用。

### 现有 Qwen 权重与 NVFP4 PLE 的关系

| 已测试 checkpoint | PLE 格式 | packed NVFP4 PLE 能否直接带来压缩收益 |
| --- | --- | --- |
| `Qwen/Qwen3.8-Flash-Next-FP8` | FP8 | 不能；需有相应量化后的 PLE 权重。 |
| `Qwen/RadixArk/Qwen3.8-Flash-Next-NVFP4` | FP8，配置明确 `ple_embedding_dtype=float8_e4m3fn` | 不能；主模型 NVFP4 不代表 PLE 也是 NVFP4。 |
| `Qwen/Inferact/Qwen3.8-Flash-Next-NVFP4` | BF16，见已有权重检查记录 | 不能；同样需要真正的 NVFP4 PLE 权重。 |

参考 packed 表每 16 个值占用 8 字节 codes + 1 字节 block scale，另有 global scale。仅按表存储计算，约为 FP8 的 56.25%、BF16 的 28.125%；这是布局估算，不是已测总内存或吞吐提升。CPU offload 场景主要减少 CPU 表内存及查表结果传输量。参考还要求各 PLE shard 的 global scale 一致，不能假定任意 NVFP4 checkpoint 都可直接加载。

## 额外分支中的性能候选

### UVA PLE offload：最值得单独做性能实验

分支：`origin/hhy/ple_offload_uva`，`cc4e4f0911aa8b6b8887ea2e6ac598372ba0a608`。

实现将 PLE 表放到 pinned CPU 内存，使用 CUDA 映射视图和 Triton 直接查表；独立 CUDA stream 预取，并提前启动下一层的查表。它尝试减少现有 CPU worker 的请求通知、CPU gather 和结果返回链路。

当前没有该实现。源码中的 pinned 表支持保留 FP8 存储，GPU gather 输出为 BF16；此分支不是上面 packed NVFP4 PLE 能力的完整合集。它还复用 `VLLM_PLE_CPU_OFFLOAD` 的名称，但实际后端已不同，不能直接覆盖当前 CPU worker 方案。

建议作为可切换的新后端实验，保留现有实现用于 A/B。需要验证 CMP 170HX 的 UVA 路径、PP/MTP/图捕获、CPU NUMA、表页固定内存、并发随机查表和完整质量。PCIe 随机访问仍有成本，未测速前不能宣称一定更快。

### Hybrid attention packing：缓存粒度实验

分支：`origin/feat/multi_attn2mamba`，`d2c1b7d201d81ca27de38afa338569c49dfb8026`。

新增 `attn_pack_size`，把多个 attention 层的页组合起来，使 attention/Mamba 的共同 block 粒度可以缩小。可能有利于短请求尾部利用率、prefix cache 命中粒度和重放成本，但更密的 Mamba 状态保存也有成本；源代码对较大的 pack size 本身就提示内存浪费。

当前缺少此接口及布局实现。分支主要改动 V1 `gpu_model_runner.py` 的布局代码，没有把同等改造落实到当前使用的 V2 runner，因此不适合直接开启。需要先补 V2、PP 分组、草稿状态和 prefix replay 的集成，再做 1/2/4 的容量与延迟对照。

## 已有或不宜直接搬入的部分

- **Qwen3.8/PLE CPU worker 基础能力**：当前已有，不是新功能；但不包含上面单列的 UniProc 和 packed NVFP4 PLE 补全。
- **主模型 NVFP4 + FP8 PLE dispatch、分批加载 FP8 scale**：当前 `6d03d7ae7` 等实现已覆盖现有权重布局。应保留显式 PLE 存储 dtype 优先于主量化器的规则。
- **PLE 异步输入就绪**：当前 `139777a13` 还有逐请求 readiness event、CPU 输入快照和 PP 排队处理。不要用参考 `64806ec94` 整体替换 connector。
- **Mamba prefix replay 和 request slot**：当前 `dc2d6a86a`、`c7fb9e3d8`、`10a73fd1c`、`577be2dd4` 等修复必须保留；参考部分旧分支还通过禁用 async/spec decode 绕开问题，不宜照搬为默认策略。
- **思考结束后进入结构化输出**：额外分支 `origin/bugfix/specdecode-grammar-reasoning-new` 的核心问题已有等效实现。当前 bitmask 循环识别草稿内部 reasoning end，并约束 bonus 行；scheduler 还会裁掉思考前缀。相关位置：[structured_output](/home/bul/dev/vllm-backport/vllm/v1/structured_output/__init__.py:289)、[scheduler](/home/bul/dev/vllm-backport/vllm/v1/core/sched/scheduler.py:1981)。函数名不同不构成缺失。
- **FlashInfer FP8/MXFP8 权重重排**：当前已有批量向量化实现；参考仍有逐专家 Python 循环，整体覆盖会退回该旧实现。NVFP4 padding 修复应只取对应函数。
- **ROCm/XPU/TPU、Mooncake/PD、Ray、多节点 MNNVL**：参考确有额外改动，但不是当前 NVIDIA 单机 PP/MTP 路径的优先项。Hy4-preview、PLaMo3 MTP 等模型扩展也应按实际模型需求另行评估。

## 本轮复核与后续顺序

本轮只启动 CPU 小测试，使用当前 conda/current repo 代码，提取参考已存在的测试断言：

| 最小测试 | 当前结果 |
| --- | --- |
| 重复图像在最后一次使用前保持引用 | 失败 |
| 重复图像在最后一次使用前不被驱逐 | 失败 |
| NVFP4 gate/up 独立 padding | 失败；示例 up 输出 512/2560 元素不符 |
| UniProc PLE worker 在模型加载前启动、加载后等待 | 失败；缺少两个调用 |

这 4 个失败用于确认 3 类现存缺口，不是本轮引入的回归。第一次临时测试收集曾因独立导入 FlashInfer helper 触发循环导入；补齐正常 MoE 包导入顺序后，以上均执行到具体行为断言。未运行参考整套 GPU 测试或模型 A/B。

建议顺序：

1. 补 Mamba 边界防护、重复图像缓存、UniProc/dummy PLE 初始化；同时纳入已复现的 NVFP4 padding 修复。
2. 补 GPU PLE 计算图边界，验证当前 FP8、RadixArk、Inferact 的正确性，保留已有 PP/MTP 修复。
3. 按目标权重补 packed NVFP4 PLE；以独立开关试验 UVA offload。收益用同题、同并发、同图模式 A/B 测量。
4. 根据实际并发和生命周期需求选择扩大图覆盖、pinned H2D、显存清理；hybrid attention packing 放到 V2 适配之后。

原始比较列表、环境信息、提取测试及 JUnit/日志保存在 `/tmp/qwen38next-comparison-20260909/`。现有模型评测背景见 [GLM/Qwen 验证记录](glm-qwen-cmp170hx-20260907.md)。
