# EXL3 集成可行性评估，2026-09-10

结论：**技术上可行，适合以低码率、显存容量和减少 CPU offload 为目标立项；当前证据不足以承诺比已有 AWQ/NVFP4 路径更快。** 本仓库已有可用扩展点，EXL3 上游也提供可复用的 MIT 内核。真正的工作集中在原生检查点识别、融合层/专家加载、Hadamard 分片、预填充工作区及 vLLM 的编译和并发执行契约。

建议先建立通用格式解析和线性层数值基准，再交付一个 **GLM routed experts + PP4/TP1** 的有限范围实现。将 **Qwen3.8-Flash-Next 原生 EXL3 包在单张 96GB Blackwell 上运行**作为另一个容量验证目标。前者贴近当前已验证部署，后者潜在收益较大，但需要额外完成 PLE、模型头、MTP 和视觉权重的适配。通用 TP/EP、高并发优化及所有历史 EXL3 变体应分阶段验收。

本报告是源码、公开接口和检查点元数据评估。本次没有安装 ExLlamaV3、编译 EXL3 内核、下载完整模型权重或运行 EXL3 模型评测。文中的性能判断、开发工期和方案优先级均为工程推断；既有仓库/社区测量不视作本次验证结果。

**评估固定在以下版本。** 当前目录初始工作区干净，分支为 `sm80-optimize`，不是未经修改的上游 vLLM。

| 对象 | 评估基线 |
| --- | --- |
| 当前仓库 | `5fc5fa7c8cfd67640f83ff98215ae6e797b20426` |
| ExLlamaV3 | `6ff3a17ea7f3d0026b273d43239398d57f71b788`，源码版本 `1.4.8` |
| 第三方 vllm-exl3 | `4f50a69c5ff7c1f8687c9e4cc91cf539ae067ed9`，开发版本 `0.4.2` |
| Qwen 原生样本 | `turboderp/Qwen3.8-Flash-Next-exl3`，分支 `3.05bpw_h5_ng5`，revision `69e33439ae950f17bcbe95c98f117d80f759ab6d` |
| 本地 Python / Torch | `.venv/bin/python`：Python 3.12.14、PyTorch `2.13.0+cu130`、Torch CUDA 13.0 |
| GPU 0–2 | 3 × CMP 170HX，SM80，每张报告 65,344 MiB 显存 |
| GPU 3 | RTX PRO 6000 Blackwell Workstation Edition，SM120，报告 97,887 MiB 显存 |

上游包声明 Python ≥3.10.11、Torch ≥2.6.0；当前环境满足声明下限，但这不证明扩展 ABI 和编译组合已经验证。源码下载只放在临时目录。版本证据见 [ExLlamaV3 提交](https://github.com/turboderp-org/exllamav3/commit/6ff3a17ea7f3d0026b273d43239398d57f71b788)及其 [pyproject.toml](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/pyproject.toml)。

**EXL3 需要独立量化后端。** 当前 [ExllamaLinearKernel](/home/bul/dev/vllm-backport/vllm/model_executor/kernels/linear/mixed_precision/exllama.py:18)调用的是 GPTQ GEMM，使用标量整数量化、分组 scales、zero points 和可选重排。EXL3 使用 QTIP 衍生的 trellis 编码、过程式码本和块 Hadamard 变换。名称相似不代表文件格式或 GEMM 可以互换；现有 Marlin/Machete/GPTQ 解包器也不能直接处理 EXL3 bitstream。格式背景见 [EXL3 官方说明](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/doc/exl3.md)。

对常见的线性权重，令输入宽度为 d_in、输出宽度为 d_out、该张量位宽为 B，原生存储的核心约定为：

| 字段 | 含义及集成要求 |
| --- | --- |
| `trellis` | int16 存储的 bitstream，形状 `[d_in/16, d_out/16, 16*B]`；每个 tile 对应 16×16 权重，不能逐整数直接转 FP16 |
| `suh` / `svh` | 输入/输出通道的 FP16 符号与缩放向量，长度分别为 d_in / d_out；不能按 GPTQ group scales 解释 |
| `mcg` / `mul1` | 码本标记；必须区分并校验支持的标记值，不能仅看位宽 |
| `bias` | 可选，保持原模型的加法位置及 TP 语义 |
| 历史变体 | 有 packed `su`/`sv`、旧码本等情况；需要版本化支持或明确拒绝 |

在与上游相同的归一化约定下，线性变换可理解为：输入通道缩放 → 128 通道块 Hadamard → trellis 权重乘法 → 输出块 Hadamard → 输出通道缩放 → bias。上游 GEMM 分派包含 B=1…8 的实例，但具体快路径只覆盖其中部分位宽/码本/形状。证据见 [量化与打包实现](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/modules/quant/exl3_lib/quantize.py)和 [内核映射](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/exllamav3_ext/quant/exl3_kernel_map.cu)。

**全局 bits 是目标/平均码率，不能代替逐张量位宽。** 已实际读取 Qwen 样本的 config 和 sidecar：全局 `bits=3.05`、`codebook=mul1`、`head_bits=5`、`vision_bits=5`、`mtp_bits=3`。其 `tensor_storage` 有 74,398 个模块条目，其中 74,041 个标为普通 EXL3：73,740 个是 3-bit，301 个是 5-bit。这是模块条目数量，不是参数数量或加权平均码率。

例如 `model.language_model.layers.0.linear_attn.in_proj_qkv.trellis` 的形状为 `[160, 640, 80]`，对应输入 2560、输出 10240、B=5。将 `3.05` 截断成 3 后统一建立 `[160,640,48]` 参数会直接出错。sidecar 还记录存储 dtype、形状及码本 multiplier。解析器应综合 sidecar 与 safetensors header，验证两者一致，并在创建权重前完成名称映射。证据为固定 revision 的 [config.json](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/69e33439ae950f17bcbe95c98f117d80f759ab6d/config.json)和 [quantization_config.json](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/69e33439ae950f17bcbe95c98f117d80f759ab6d/quantization_config.json)。

**当前仓库的扩展点足够，但至少有三个确定的加载障碍。**

| 位置 | 已有能力 | EXL3 所需工作 |
| --- | --- | --- |
| [量化注册](/home/bul/dev/vllm-backport/vllm/model_executor/layers/quantization/__init__.py:60) | 支持自定义 QuantizationConfig，注册时扩充方法列表 | 新增 EXL3 config；声明设备、dtype、格式范围，支持可选插件加载 |
| [量化配置读取](/home/bul/dev/vllm-backport/vllm/model_executor/model_loader/weight_utils.py:241) | 读取 HF quantization_config | HF config 有量化字段时会提前返回；只实现 `get_config_filenames()` 不会自动合并 EXL3 sidecar |
| [权重文件筛选](/home/bul/dev/vllm-backport/vllm/model_executor/model_loader/default_loader.py:221) | 按 safetensors index 过滤分片 | 原生 Qwen 的独立 n-gram 文件不在主索引，标准路径会过滤掉；需要显式辅助权重加载或规范化索引 |
| [配置更新钩子](/home/bul/dev/vllm-backport/vllm/config/vllm.py:783) | 模型构建前传入 model、HF config、revision | 可用于补齐 sidecar/header 信息，保持 revision 一致；避免在此实例化整套 ExLlama 模型 |
| [线性层](/home/bul/dev/vllm-backport/vllm/model_executor/layers/linear.py:652) | MergedColumn、QKV、Row/Column parallel | 定制 packed 参数与 loader，按逻辑投影保留独立 B、suh、svh、码本及 padding |
| [RoutedExperts](/home/bul/dev/vllm-backport/vllm/model_executor/layers/fused_moe/routed_experts.py:892) | 专家映射、TP/EP、量化方法分派 | 现有 loader 把任意 3D checkpoint tensor 当作 fused experts；单专家 EXL3 trellis 恰好也是 3D，必须消除该歧义 |
| [MoE 方法接口](/home/bul/dev/vllm-backport/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py:28) | create_weights / apply / post-load | 新增 EXL3 MoE 方法，接收已有 router 的 top-k 输出，匹配 shared experts 和归并语义 |
| [CMake 构建](/home/bul/dev/vllm-backport/CMakeLists.txt:1441) | 普通扩展与 stable libtorch 扩展 | 上游使用 ATen/pybind，不能未经适配就声称符合本仓库 stable ABI；首期可用独立扩展，后续整理绑定 |

这三个确定障碍分别是 **sidecar 没有自然合并**、**辅助权重文件被主索引过滤**和 **3D trellis 被误判成堆叠专家**。因此增加一个 `--quantization exl3` 名称远不足以完成集成。规范化应写出可复核的独立产物或显式 loader 逻辑，不能原地盲改原生包、让 patch/重复权重的覆盖顺序变得不明确。

QKV、gate/up 合并还存在语义限制：只有输入变换相容时，才可共享输入 Hadamard 或直接拼接 packed 矩阵。B、码本或 suh 不同的投影，首期应独立执行后按原顺序拼接输出。这样会增加 kernel launch，但更容易建立正确性。GQA 的 K/V 复制、偏置只在正确的 rank 累加，以及未量化投影混装，都必须单独处理。

MoE 不能仅把 dense EXL3 循环 E 次当作生产实现。需要保留每个 expert 的 packed 权重，以 GPU 上的路由和批量执行处理实际激活专家，并正确处理 gate/up/down 顺序、top-k 权重、SwiGLU/clipping、零 token 专家、shared experts 和跨 rank 输出归并。任意“每个专家不同 B/码本”的格式还需要分桶或分开的指针/元数据表，不能强行塞进同一固定最后维度的矩形 tensor。

**第三方工作证明有可行路径，尚不能当作通用即装即用支持。**

| 已有工作 | 本次核查结果 | 对决策的意义 |
| --- | --- | --- |
| vLLM issue #19896 | 已关闭，评论显示 2026-04-27 因无活动自动关闭 | 不能解释为维护者正式否决 EXL3 的技术方案 |
| vLLM 开放 PR 检索 | `19896 in:body` 和 `exllamav3` 为 0；`exl3` 命中 3 个其他主题 PR | 本次未发现开放的完整 EXL3 后端 PR；检索不是不存在任何外部分支的证明 |
| Aphrodite / 现 dphnAI/sonar PR #1398 | GitHub API 显示 closed、未合并；作者说明权重可加载但内核产生 NaN | 可参考加载思路，不能作为已正确支持的证据 |
| vcruz305/vllm-exl3 | 已有 RoutedExperts、dense、原生包整理及 n-gram 路径；0.4.2 的完整 GPU 资格测试仍待完成 | 值得参考适配边界；本仓库有相近接口，但模型文件和加载路径需要实质对齐 |

上表依据 [vLLM issue 与评论](https://github.com/vllm-project/vllm/issues/19896)、[Aphrodite/sonar PR](https://github.com/dphnAI/sonar/pull/1398)及 [插件固定版本 README](https://github.com/vcruz305/vllm-exl3/blob/4f50a69c5ff7c1f8687c9e4cc91cf539ae067ed9/README.md)。本机没有 `gh`，已尝试规定的只读命令后，改用 GitHub 公开 API 完成等价查询；没有提交 issue、评论或 PR。

插件的附加 native MoE 快路径存在 `hidden=4096`、local intermediate 为 1024/2048 等几何条件；Qwen 样本的 2560/640 不符合该条件。部分 fat-prefill 快路径还要求 K4/MCG 和相容输入变换；原生 Qwen 的 mul1、3/5-bit 不能套用其 K4 微基准。插件也不是完整任意逐专家位宽实现。相关证据见 [插件实现](https://github.com/vcruz305/vllm-exl3/blob/4f50a69c5ff7c1f8687c9e4cc91cf539ae067ed9/src/vllm_exl3/exl3.py)；这些限制属于该适配器，不能据此推断 EXL3 格式本身不可支持其他形状。

**许可证会影响选用哪条实现路线。** ExLlamaV3 评估版本为 MIT，可以按其条款复用并保留版权/许可声明。第三方插件评估版本的项目整体许可为 AGPL-3.0-only，保留的 Apache 文本属于历史版本；其部分上游材料另有 MIT/Apache 声明。不能把最新版整体当作宽松许可代码复制进仍按 Apache-2.0 发布的本仓库。独立插件部署也不能仅凭进程/包边界就假定许可义务消失。若选择历史宽松许可快照，应逐文件核实具体 revision、来源和后来加入的代码；后续许可改变也不应被简单理解为抹去已授予的旧版本许可。证据见 [ExLlamaV3 LICENSE](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/LICENSE)及 [插件第三方声明](https://github.com/vcruz305/vllm-exl3/blob/4f50a69c5ff7c1f8687c9e4cc91cf539ae067ed9/THIRD_PARTY_NOTICES.md)。

因此优先方案是：**复用明确许可的 ExLlamaV3 packed 推理内核，围绕当前 vLLM 接口独立完成适配层**。第三方插件可用于识别已知问题和对照验证；是否使用其代码应单独确定许可基线。无需引入 ExLlama 的 scheduler、attention、KV cache 或服务 API 来支持权重量化格式。

**多卡中 PP 比通用 TP 更适合作为首期目标。**

| 能力 | 可行性判断 | 主要条件 |
| --- | --- | --- |
| 单 GPU dense | 高 | 正确加载各投影与 head，控制 BF16↔FP16 转换和工作区 |
| 单 GPU MoE | 高，但性能待测 | 批量专家内核、路由元数据、实际形状和码本覆盖 |
| PP，TP=1 | 较高，推荐优先 | 每个 stage 持有完整矩阵；正确跳过其他 stage 权重，按新权重/工作区大小重新分层 |
| 对齐良好的 dense TP | 中 | packed tile 与 128 通道 Hadamard 块都不能被错误切开；处理 K/V 复制 |
| 任意模型/任意 TP 度数 | 中低 | vLLM 等宽切片不一定与 128 块对齐；需限制组合或新增变换/分片方案 |
| EP | 中 | 专家整体分片有利于保留格式，但仍要适配 dispatch/combine、专家编号和通信 |
| EPLB / 动态专家迁移 | 后置 | 权重搬移后重建或更新所有指针表、元数据及捕获的执行状态 |
| DP | 相对容易 | 每个独立副本完成正确性和内存验证；不能忽略进程内共享缓存与多 stream 并发 |

一个具体 TP 反例是 Qwen 的 expert intermediate=640：TP2 等分得到 320，虽然能整除 16，却不能整除 128。直接把已有 trellis 切成两半后在各 rank 做本地 Hadamard，会改变跨边界的变换语义；给局部分片补零也不自动修复这个问题。上游自己的分配器以 128 通道为单位，不能据此假设可原样替换 vLLM 的等宽分片。证据见 [上游 Linear 分配器](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/modules/linear.py)。这是由变换布局推导出的限制，不是一次 TP2 运行失败的实测报告。

当前 GLM 验证使用 PP4/TP1，stage 分层为 `11,11,11,12`，正好绕开矩阵内部 TP 分片。EXL3 后的 stage 最优分配仍需按实际执行成本重测；3 张 SM80 与 1 张 SM120 不能只按显存容量均分工作。现有对照协议见 [GLM 本地验证](/home/bul/dev/vllm-backport/docs/validation/glm53-flash-optimization-20260909.md:25)。

**容量收益可以估算，实际运行峰值必须测量。** 对被量化的权重集合，理论 bitstream 大小为 `Σ(N_i * B_i / 8)`，还应计入 suh/svh、padding、非量化层、模型头及运行时结构。每 1000 亿个量化参数，4→3 bit 理论减少 12.5 GB（11.64 GiB），4→2 bit 减少 25 GB（23.28 GiB）。这些是权重数据的算术差额，不能直接宣称整模型显存下降 25%/50%。

Qwen 样本按 sidecar 中 tensor 名去重后，296,941 个张量声明共 83,400,539,644 bytes，即 **77.67 GiB**；其中 n-gram 名称下的张量为 32,640,162,072 bytes，即 **30.40 GiB**。但交叉核查发现 sidecar 与主索引不是同一份完整账本：主索引有 304,105 个名字，其中 7,296 个不在 sidecar 中，例子包括 A_log、conv1d.weight、dt_bias；反过来，sidecar 中 132 个 n-gram 张量全部不在主索引中。

主索引只列 7 个模型分片，`metadata.total_size` 为 52,358,593,436 bytes（48.76 GiB），不包含独立 `ngram_embedding.safetensors`。通过 HF 固定 revision 的 LFS 文件大小核查，仓库实际有 9 个 safetensors 文件：7 个模型分片、约 30.40 GiB 的 n-gram 文件和约 13 MB 的 `mtp_hyper_connection_mixer_patch.safetensors`，合计 **84,986,510,810 bytes（79.15 GiB）**。证据见 [主索引](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/69e33439ae950f17bcbe95c98f117d80f759ab6d/model.safetensors.index.json)及 [固定 revision 文件列表](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/69e33439ae950f17bcbe95c98f117d80f759ab6d)。MTP 补丁的适用范围、名字映射与覆盖顺序仍需按该包说明核实。

因此不能只按主索引宣称模型约 49 GiB，也不能把 sidecar 的 77.67 GiB 当作完整模型显存。约 79.15 GiB 的文件量提示单张本机 95.59 GiB Blackwell 有研究价值，但磁盘可能保留推理不使用的张量，加载后也可能出现 repack 副本、临时解码、CUDA graph 私有池及非 Torch 分配。必须测加载峰值、steady-state 常驻量和最大 prefill/并发时峰值，才可决定上下文、并发数及是否单卡可运行。

这也是 Qwen PLE 支持值得单独做的原因：压缩权重表有机会减少当前 CPU offload/UVA 数据流量，但其 row-wise 编码不是普通 16×16 linear trellis。需要对应的按行解码和原有 n-gram/hash/head-offset 语义，不能用普通 linear adapter 替代。上游另有 [ngram codec](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/modules/quant/exl3_lib/ngram_codec.py)。

**预填充和并发执行是最大的性能风险。** 上游 `LinearEXL3.forward` 在行数超过 144 时默认转到 reconstruction + HGEMM；输出特别宽时支持分片重建。单个 4096×14336 FP16 矩阵重建需要 112 MiB；一个 hidden=4096、intermediate=2048 的三投影专家为 48 MiB，还没有计入 activation 和路由缓冲。临时重建不等于全模型永久展开，但依然会吃掉低码率节省的部分容量和带宽。证据见 [上游 LinearEXL3](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/modules/quant/exl3.py)。

低 batch decode 可能受益于减少读取的权重字节；大 batch/prefill 中权重复用增强，Hadamard、码本生成、临时重建、scatter 和同步成本可能占主导。因此 SM80 上需要比较真实 GEMM/MoE 形状和请求组合，不能把 RTX 4090 或 DGX Spark 的某个内核加速倍数转写成这台机器的服务吞吐增幅。上游当前已存在 Ampere/Blackwell 分派与调优逻辑，早期 README 的性能描述也不足以代替当前版本测试。

还发现一个会影响精度基准的细节：**1.4.8 的 mul1 路径在适用条件下默认允许 plain INT8 activation GEMV。** `EXL3_INT8_GEMV` 未设置时默认值为 2，另有关闭和带残差的模式。这意味着“读取同一 EXL3 权重”与“严格保持 FP16 激活计算路径”并非同一个条件。首次数值基准建议明确关闭该快路径，随后对各加速模式分别测精度和速度，避免把新增激活误差归因于权重格式或 loader。证据见 [INT8 GEMV 分派](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/exllamav3_ext/quant/exl3_gemv_int8.cu)。

上游还通过每 GPU 单例持有 locks/workspace，有直接 `cudaMalloc`，并使用 cooperative kernel 与 autotuning。接入 vLLM 时需要在 profile/capture 之前初始化，显式管理 scratch 的生命周期和可重入性，核实当前 CUDA stream，并统计非 Torch 显存。某些 Python 分派含设备结果回传，会影响 full graph；把调用包装成 custom op 只解决一部分 compile 边界，不自动保证 CUDA graph capture/replay 正确。证据见 [设备上下文](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/exllamav3_ext/quant/exl3_devctx.cu)与 [autotuner](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/exllamav3/exllamav3_ext/quant/coop_autotune.cu)。

生产化适配应采用有 fake/meta 实现的 custom op，明确输出 shape/dtype、可变参数及工作区所有权。graph 重放必须更换输入、路由、top-k 权重和批次有效长度来检查；只重复同一输入不能发现捕获了旧路由或指针的错误。ExLlama 自有 Graph 对象的支持范围与 Torch CUDA graph 也不等价，需分别验证。

**本分支重点模型的工作量不同。**

| 模型/功能 | 额外适配与建议范围 |
| --- | --- |
| 常规 Llama/Qwen dense | 适合小模型数值原型；仍需 QKV/gate-up 融合、head、词表 padding 和 tied embeddings |
| GLM-5.3-Flash | 适合首个实际 MoE/PP4 目标；保留当前 router、KDA、sparse MLA、kpool 回滚，适配权重和专家计算 |
| Qwen3.8-Flash-Next | 原生包包含 dense 5-bit、expert 3-bit、mul1、PLE row-wise 表、MTP/vision，不能只做 routed experts 就声称全包支持 |
| DeepSeek V4 | 有 FP8 linear 与 MXFP4/FP8 experts 的特定分派及 scale dtype；EXL3 与剩余源格式应按实际模块组合 |
| MTP/投机解码 | 主模型、草稿层和共享 head 分别核对量化；重新测接受率、拒绝回滚和验证批次，不能沿用其他量化结果 |
| KV cache / prefix cache | 权重量化不自动改变 KV 表示；先复用现有 attention/KV 后端，再测试缓存与状态组合 |
| LoRA | 原理上可在原坐标中叠加低秩更新，但 vLLM 包装、融合和 TP 仍需适配；首期不承诺 |
| CPU/ROCm/其他设备 | 本次推荐路线限定 NVIDIA CUDA；其他后端应另行评估，不能由格式支持推定内核支持 |

具体而言，本仓库 Qwen 主模型的 [lm_head](/home/bul/dev/vllm-backport/vllm/models/qwen4_exp/nvidia/model.py:630)与 [MTP lm_head](/home/bul/dev/vllm-backport/vllm/models/qwen4_exp/nvidia/mtp.py:416)当前都没有传入 quant_config。PLE 的 [选择器](/home/bul/dev/vllm-backport/vllm/models/qwen4_exp/nvidia/ple_layer.py:252)主要处理 FP8；UVA embedding 还会删除 quant_method 并直接建立 dense weight 的 accelerator view。EXL3 n-gram 表接入必须重新适配这些位置，不能直接执行针对其他 fork 目录和类结构的补丁脚本。DeepSeek V4 的源格式选择则可参考现有 [quant_config](/home/bul/dev/vllm-backport/vllm/models/deepseek_v4/quant_config.py:28)。

**可选路线及取舍如下。**

| 路线 | 格式/质量与容量 | 评价 |
| --- | --- | --- |
| vLLM adapter + 上游 MIT EXL3 内核 | 保留 packed EXL3，量化误差由同一权重决定，运行模式另外验证 | 推荐；先固定上游 revision，建立清晰的可替换接口 |
| 安装已有第三方插件 | 可快速验证已有模型 recipe；受许可、接口和模型条件限制 | 可作为隔离的对照实验，不能当作本仓库已完成支持 |
| 加载后全量展开 FP16/BF16 | 可保留已解码 EXL3 的近似权重，但恢复不了原高精度权重 | 可作正确性参考；长期展开失去低显存优势 |
| EXL3 再转 AWQ/GPTQ | 需要重新量化，可能叠加误差，不能纯 repack | 可作为部署替代实验，不满足原生格式支持目标 |
| 自写全部 decoder/GEMM/MoE 内核 | 可控制 ABI、调度和工作区，维护成本最高 | 只在复用路径证实性能/并发瓶颈后考虑 |

权重转换与 serving 后端是两项工作。优先使用上游转换工具或已有原生 checkpoint；不把 Hessian/LDL/Viterbi 离线量化过程塞进 vLLM 启动流程。若要为“仅专家 EXL3、其他层保留原格式”制作检查点，需要固定源模型、校准数据与模块选择，并评测该混合产物；不能假定从不同量化版本拼出的包与原生全包质量等价。已有 AWQ/NVFP4 权重也不能通过无损换壳得到同等质量的 EXL3。

**建议按以下顺序实施和决定是否继续投入。**

| 阶段 | 交付范围 | 通过条件 |
| --- | --- | --- |
| 0：格式与运行契约 | 固定版本、原生 metadata/header 解析、支持矩阵、独立扩展构建；最小真实矩阵 | B/码本/padding/shape 完整校验；SM80 与 SM120 均能正确执行，严格模式没有 NaN/挂起 |
| 1：小模型原型 | 单 GPU dense、独立逻辑投影、有限 prefill fallback、eager | 同一 EXL3 权重对照重建参考及原生引擎；模型 logits/PPL 对齐，没有遗漏权重 |
| 2：实际部署子集 | GLM routed-expert EXL3 + 其余显式源格式、PP4/TP1；逐步打开 graph/MTP | 真实模型完整加载；与现有 AWQ/NVFP4 协议配对评测，显存和质量满足目标 |
| 3：容量目标 | Qwen 原生 3.05bpw 包，完善 PLE/head/MTP/vision；尝试单 SM120 | 全包语义正确，加载/预填充峰值可容纳，明确可承载上下文和并发 |
| 4：生产性能/通用性 | bounded scratch、按 B/码本/形状分派、prefill 优化；选择性 TP/EP | 压测、取消/槽复用、变路由 graph、长上下文和全部目标并行组合通过 |

首期第 2 阶段只支持明确的混合检查点，是 EXL3 支持的一个子集；第 3 阶段也只代表选定原生模型通过，不代表任意 EXL3 仓库已兼容。应在 loader 中对未支持组合报清晰错误，而不是悄悄把所有专家展开成高精度权重或忽略未知张量。

工作量只能粗估：假设一名熟悉 vLLM/CUDA 的工程师持续投入并有及时评审、直接复用上游内核，**单 GPU 原型约 1–2 周；限定 GLM + PP4/TP1 的可用版本累计约 4–8 周；含原生复杂模型、优化和较广 TP/EP 支持的生产版本通常需要数月。** 这些是范围估计，不是排期承诺。最大的工期变量是 cooperative kernel 与 runner 的并发兼容、SM80 prefill 表现，以及原生模型包需要的特殊模块数量。仅把某个现有插件 recipe 启动起来可能更快，但交付范围不同。

**验证必须同时分离格式正确性、模型质量和性能。**

| 层级 | 建议验证内容 |
| --- | --- |
| 格式单元检查 | 平均 bits 与逐张量 B、mcg/mul1 标记、legacy 拒绝、缺失/重复张量、HF revision、融合投影映射；复用现有量化注册/配置测试 |
| 内核数值 | 同一真实 EXL3 权重的 packed op 对照重建矩阵乘；记录 max error、NRMSE、有限值；测试两种 GPU、B=2/3/4/5 和对应码本，再按声明扩展 |
| 分派边界 | 覆盖 1/2/3/4/8/16/17/32/33/128/144/145/256/512/1024 等行数；不连续输入、非齐整输出、不同输入变换、bias |
| MoE / 并行 | 空专家、路由集中/均匀、不同 expert ids/weights、shared experts/clipping；PP 完整加载；启用的 TP/EP 组合单独对照 |
| 编译/graph | eager → compile → graph；重放时更换输入/路由/请求槽；检查指针生命周期、多 stream 冲突、取消与重启 |
| 模型质量 | 同一 EXL3 权重跨引擎的 logits/PPL；与源模型及既有 AWQ/NVFP4 的 GSM8K、代码执行、多语言、长上下文/检索、工具调用；多模态另测 |
| 投机/缓存 | MTP 开关与不同草稿数，接受率、每轮耗时、拒绝回滚、prefix hits、私有前缀恢复；保留首次失败和截断结果 |
| 服务性能 | 冷前缀 TTFT、prefill tokens/s、TPOT p50/p95、单请求速率、并发吞吐；短/长 prompt 与多并发；固定输出统计口径 |
| 资源 | 磁盘/主存、加载峰值、常驻量、临时重建、graph 池、非 Torch CUDA 分配、KV 容量；识别是否实际命中回退 |

不能只对比不同 bitrate 模型的 TPS，也不能以“输出看起来正常”替代数值和质量验证。精度误差阈值应先在 FP16 参考及严格 EXL3 模式中建立，再单独评估 INT8 activation 等加速模式。源码中有 kernel 或未执行的 GPU 测试，不等于该硬件/模型组合已通过。

后续实现沿用仓库 uv/.venv 开发规则；CUDA 修改参考 [增量构建流程](/home/bul/dev/vllm-backport/docs/contributing/incremental_build.md)。内核性能工作放到 benchmarks/kernels，正确性测试优先扩展 tests/quantization、tests/kernels/quantization 和现有 MoE 测试；模型评估复用 tests/evals 以及当前 [GLM](/home/bul/dev/vllm-backport/docs/validation/glm53-flash-optimization-20260909.md)和 [Qwen](/home/bul/dev/vllm-backport/docs/validation/qwen38next-optimization-20260909.md)协议。CMP 170HX 的既有记录显示 CUPTI 计时路径不可用，应沿用经过验证的 graph/event 测量方式并说明局限。

本项目的优先验收价值应是：以可接受质量在更少 GPU 上容纳目标模型，或在相同 GPU 上增加实际可用上下文/并发容量；然后再要求服务速度达到约定水平。如果严格数值路径不能稳定运行，或 prefill 工作区吞掉大部分容量收益，应在扩展模型支持前解决或暂停该方向。当前已有充足源码证据支持进入受控原型阶段，尚无本机 EXL3 实测证据支持直接替换已验证的 AWQ/NVFP4 部署。
