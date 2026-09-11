# EXL3 首轮集成与验证（2026-09-10）

本次已将 EXL3 权重格式接入当前 vLLM 的量化后端，直接执行原始 trellis 权重；模型结构、注意力、缓存及请求调度继续使用当前仓库的实现。测试覆盖指定的 Qwen3.8-27B dense 和 GLM-5.3-Flash MoE 检查点。当前定位是实验性实现：支持本地模型、TP1/EP1 和流水线并行，尚不具备任意分布式配置下的通用支持。

## 实现范围

- 通过 safetensors 头部读取每个矩阵的实际位宽、形状、码本及 Hadamard 向量，不以模型平均 bpw 分配张量。支持混合位宽及融合层内量化/未量化投影混合。
- 每个原始投影保持独立旋转边界；GLM 的整体 QKV 投影使用 tuple shard 加载，避免错误拆分 Hadamard。接入量化输出头，并忽略转换器写入的校准 `input_ids`。
- 直接导入 MIT 许可的 `exllamav3_ext`，不引入原版的模型加载器、生成器或分配器配置。未复制第三方 AGPL 适配层。
- 大批量线性 prefill 按列暂时重建权重；不会把整个模型长期扩展为 FP16。MoE prefill 按不超过 128 个 token 分块，避免原版融合内核跳过超过容量的热点专家。
- 小批量 MoE decode（1–8 token）使用原版 batched expert GEMM，保留动态 expert ID 与路由权重，减少排序和融合调度开销。
- GLM KDA、MLA 和视觉投影识别 EXL3 量化配置；已有 FP8 的 BF16 例外保持原行为。本轮整模型只验证文本推理，不能据此声称视觉输入已经验证。

共享专家必须与路由专家串行。首轮 GLM 测试在 32 题后卡住；源码检查发现原版 GEMM 使用设备级单例 `DevCtx` 的 lock/scratch，两个 CUDA stream 并发调用不安全。适配层据此声明 `supports_multi_stream=False`，让共享专家不创建辅助 stream，并明确拒绝 DBO。修复后完整跑过 12 次性能测量和 64 道评测题。该现象与工作区竞争一致；没有用 CUDA core dump 单独证明卡住位置。失败轮次保留为诊断材料，不进入有效性能比较。

## 比较协议与限制

- dense：单张 GPU 3，RTX PRO 6000 Blackwell（SM120，约 95 GiB）。MoE：GPU 0–2 CMP 170HX（SM80，各约 63 GiB）与 GPU 3，实际 decoder 分层两边都是 11/11/11/12。
- vLLM 用 PP4；原版用 `Model.load(use_per_device=[31,40,40,48])` 的按层顺序放置，未启用 tensor parallel。原版 embedding 在 CPU，vLLM embedding 在第一个 GPU。因此这是同设备、同分层的整引擎对照，不是相同调度实现的纯内核对照。
- vLLM：当前基线提交 `5fc5fa7c8cfd67640f83ff98215ae6e797b20426` 加本次未提交修改，Python 3.12.14、PyTorch 2.13.0+cu130、EXL3 扩展 1.4.8（本机编译，SM80/SM120）。
- 原版：`/home/bul/dev/exllamav3`，提交 `6ff3a17ea7f3d0026b273d43239398d57f71b788`，tag 1.4.8；conda `exllamav3`，Python 3.12.11、PyTorch 2.8.0+cu128。已更新匹配的 1.4.8 wheel、FLA 0.5.2 和 C++ 运行库，未修改原版源码。
- vLLM 模型激活使用 BF16，EXL3 乘积在边界转为 FP16；原版以 FP16 为主，并有个别 FP32 投影。因此逐 token 完全一致不是本轮对照前提，端到端差异也不能都归因于适配层。
- 两边固定 `EXL3_INT8_GEMV=0`、`OMP_NUM_THREADS=1`，无 MTP、无 prefix cache；原版每轮清空页表。记录并断言缓存命中为零。
- 输入长度 256 / 2048，batch 1 / 4，每请求强制生成 256 token，greedy，每项预热两次后测三次。主表为 `总输出 token / 完整 generate 耗时` 的三次中位数，包含 prefill 与调度；不是纯 decode TPS。
- 同一模型的两边使用字节完全相同的输入 JSON（含预先分词的 token ID）。最大上下文 4096、prefill chunk 512、最大请求数 4；vLLM 每 GPU 显式分配 2 GiB KV cache，原版缓存容量 16384 token。缓存策略与容量没有被假定完全等价。
- 开启 vLLM CUDA Graph；GLM 使用该架构现有的分段图执行。首次编译、磁盘缓存状态不同，加载时间只记录，不作启动性能结论。

## 数值与模型评测

算子测试使用独立 Sylvester Hadamard 参考，覆盖 2/3/4/5/6/8 bit、decode / prefill 切换、混合投影、缺失权重、热点专家超过 128 token、图重放后改变输入和路由，以及共享专家串行约束。真实 Qwen 投影的比较还覆盖 BF16 输入和原版同扩展调用。

GSM8K 使用 test 前 64 题、zero-shot chat、greedy，初始输出上限 768 token。Qwen 模板使用 `enable_thinking=False`；GLM 模板忽略该参数，所以在两边相同的输入 token 后显式补上 `</think>`，关闭开放的推理前缀。未运行完整 1319 题或模型 perplexity，因此本轮只能作为集成回归证据。

## 整模型性能

单位为生成 token/s，三次中位数；百分比为 vLLM 相对原版。完整逐次测量、首 token 时间和输出见[结果摘要](exl3-integration-20260910.json)及其指向的原始 JSON。

| 模型 | 输入 token | 并发 | vLLM | 原版 ExLlamaV3 | 差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 256 | 1 | 62.46 | 67.07 | -6.9% |
| Qwen3.8-27B | 256 | 4 | 214.11 | 232.61 | -8.0% |
| Qwen3.8-27B | 2048 | 1 | 56.18 | 58.29 | -3.6% |
| Qwen3.8-27B | 2048 | 4 | 148.73 | 152.94 | -2.8% |
| GLM-5.3-Flash | 256 | 1 | 36.97 | 33.79 | +9.4% |
| GLM-5.3-Flash | 256 | 4 | 45.78 | 61.43 | -25.5% |
| GLM-5.3-Flash | 2048 | 1 | 31.08 | 26.67 | +16.5% |
| GLM-5.3-Flash | 2048 | 4 | 37.28 | 38.79 | -3.9% |

当前 dense 吞吐约为原版的 92%–97%。MoE 单请求快于本机原版对照，但短输入四并发有明显差距，不能概括为“EXL3 集成已经比原版快”。后续应优先对该并发配置做整引擎 profiling，区分调度、跨卡传输、专家计算与 dtype 转换；本轮没有证据把差距归到单一因素。

## Prefill 与首 token 延迟

以下从同一批已保存的测量中提取，未重新运行 GPU 测试。单请求 prefill 阶段耗时：vLLM 为 `first_token_ts - scheduled_ts`，原版为 `time_first_token - time_first_prefill`。两者均计到首 token，包含引擎和首 token 相关开销；这里没有单独计量纯 GPU prefill kernel。吞吐为输入 token 数除以该阶段耗时，报告三次中位数。硬件、512-token prefill chunk、零缓存命中及 `EXL3_INT8_GEMV=0` 与主表一致。

| 模型 | 输入 token | vLLM 耗时 ms | 原版耗时 ms | vLLM 输入 token/s | 原版输入 token/s | 吞吐差异 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 256 | 151.0 | 98.3 | 1696 | 2605 | -34.9% |
| Qwen3.8-27B | 2048 | 583.5 | 655.7 | 3510 | 3123 | +12.4% |
| GLM-5.3-Flash | 256 | 518.9 | 498.5 | 493 | 514 | -3.9% |
| GLM-5.3-Flash | 2048 | 1697.6 | 2394.3 | 1206 | 855 | +41.0% |

4 并发不能把每个请求的 `prefill` 时间相加或平均后当成总吞吐：各请求计时区间重叠，且两个引擎交错执行 prefill/decode 的方式不同。下面改看每批四个请求中最大的 TTFT，再取三轮中位数，作为这一批最慢首 token 延迟；它包含排队及交错执行等待。

| 模型 | 每请求输入 token | vLLM 最慢 TTFT s | 原版最慢 TTFT s |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 256 | 0.346 | 0.388 |
| Qwen3.8-27B | 2048 | 2.375 | 2.649 |
| GLM-5.3-Flash | 256 | 1.230 | 1.737 |
| GLM-5.3-Flash | 2048 | 5.182 | 9.265 |

这些结果显示本轮 2048-token 输入下，vLLM 的单请求 prefill 阶段更快；256-token 的短输入则以原版为快。短输入受固定开销影响较大，具体差异归因仍需要 profiling。本节首轮数据仅覆盖 256/2048 token；后续的 8K、16K、32K、64K 实测见 [长上下文验证报告](exl3-long-context-20260910.md)。

## 模型评测结果与补测

初始评分使用输出中最后一个数值，与既有 GSM8K 评测方式一致；未要求生成完整的最终答案标记。因此表内“数值匹配”不能自动等同于正常完成。

| 模型/后端 | 数值匹配 / 64 | 输出截断数 | 正常结束且正确 | 正常结束但错误 |
| --- | ---: | ---: | ---: | ---: |
| Qwen / vLLM | 60 | 4 | 60 | 0 |
| Qwen / 原版 | 62 | 3 | 61 | 0 |
| GLM / vLLM | 64 | 0 | 64 | 0 |
| GLM / 原版 | 61 | 0 | 61 | 3 |

Qwen 原版第 12 题虽在 768 token 处被截断，但末尾数值恰好为标准答案，所以被原始评分器算作正确；这里仍将其列入截断。vLLM 截断题为 12/37/45/62，原版为 12/45/62。GLM 原版正常结束但错误的是 9/12/53：例如第 9 题写出 `400 + 60 = 500`，不是解析器漏掉正确答案。原始输出均保留，64 题样本不能支持普遍精度优劣结论。

对双方截断题的并集 12/37/45/62，使用相同输入将预算增为 3072 token，两边各复测一次。该补测独立保存，不覆盖初始分数，也不拼成一次“64 题新成绩”。

| 题号 | 标准答案 | vLLM 补测 | 原版补测 |
| --- | ---: | --- | --- |
| 12 | 13 | 13，正常结束，1416 token | 12，正常结束，1975 token |
| 37 | 2 | 2，正常结束，1025 token | 2，正常结束，655 token |
| 45 | 104 | 44，正常结束，2238 token | 44，正常结束，1426 token |
| 62 | 25000 | 再次截断，3072 token | 38125，正常结束，1664 token |

补测说明长度限制只解释了部分初始未完成情况，不能保证加长输出就恢复正确。两引擎在 dtype、注意力和算子归约上的差异，可能使 greedy 输出沿不同路径发展；没有在这里证明具体归因。

## 算子证据与检查

- CMP 170HX 上真实 Qwen 的 3-bit gate 与 5-bit QKV 矩阵，在 1/8/128/1024 行下与原版 `LinearEXL3` 使用同一扩展比较，输出相对 L2 为 0。计时使用 FlashInfer CUDA Graph + cold L2，包含 wrapper 内部转换与分配操作，排除输入构造和编译。
- 除首个 3-bit 单行样本外，大部分线性算子时间在约 1% 内。首个样本原版 134.86µs、适配层 105.88µs，存在首轮时钟/顺序影响，不能据此声称适配算法更快。
- Blackwell 默认 INT8 模式下，同样两块矩阵的 1/8/128/1024 行输出与原版同扩展也完全一致（相对 L2 为 0）；适配层计时相对原版在约 -0.3% 至 +3.0% 范围。它证明两侧调用一致，不证明 INT8 与 FP16 的整模型精度相同。
- GLM 第 3 层真实 288 专家在 CMP 上，decode 分支相对始终使用融合 MoE 的候选：1 token 为 334.23→227.02µs，4 token 为 1104.59→810.65µs，8 token 为 2133.20→1620.48µs；输出相对 L2 为 0.00110–0.00164。这是局部内核优化收益，不是相对原版整模型的吞吐提升。
- 最终 EXL3 回归：34 passed（CMP）。线性测试显式区分严格模式与默认 INT8 模式：前者相对 L2 <0.006，后者 <0.015；MoE 独立参考 <0.01。此前用严格阈值直接测默认 INT8 时，3-bit 单行出现约 0.006 的误差，已保留失败日志；这是不同精度模式的测试前提差异，没有放宽严格模式的阈值。
- 全部修改通过 pre-commit（含 Python 3.10 类型检查），手动 Python 3.12 mypy 也通过；`git diff --check` 通过。验证进程已退出，四张 GPU 已释放。
- 最终复核修正了旧格式 packed signs 与显式向量同时存在时的优先级，并增加两项加载用例。重新扫描确认本次两个模型的 409 / 36719 个量化矩阵均不含 packed signs，因此该末尾修正不改变已测量模型的执行路径。
- 现有模型注册测试有一项通过；需要下载 `meta-llama/Llama-3.2-1B-Instruct` 的另一项因配置的 Hugging Face mirror 返回 403 而未完成，不记为通过。

首轮不验证 MTP、LoRA、CPU offload、sleep、EPLB、视觉输入、长上下文容量或 TP/EP；长上下文后续已补充到 [64K 输入及四个请求](exl3-long-context-20260910.md)。MoE 仅支持 SiLU、每个投影在专家间形状/位宽一致、gate/up/down 码本相同的检查点。共享专家串行是当前上游工作区约束下的必要限制。默认 INT8 模式的整模型性能和完整评测未运行，主表不能代表默认 INT8 模式的速度。

## 重现与启动

依赖安装及通用用法见 [EXL3 后端文档](../features/quantization/exl3.md)。本机 vLLM 环境中的扩展使用 CUDA 13.0 和 SM80/SM120 编译；原版 conda 环境保持 PyTorch 2.8.0。原版运行库已更新为 conda-forge 的 libstdcxx/libgcc 16.2.0，不再需要 `LD_PRELOAD`；最初 dense 主测仍使用系统 libstdc++ 的 `LD_PRELOAD` 临时兼容方式，GLM 和后续补测使用更新后的 conda 库。

以下从当前仓库目录运行 dense 对照：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm \
  --inputs docs/validation/exl3-20260910/dense-inputs.json \
  --output /tmp/dense-vllm.json
```

MoE 使用同一脚本和 `moe-inputs.json`，额外设置：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=11,11,11,12 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm --pp 4 \
  --inputs docs/validation/exl3-20260910/moe-inputs.json \
  --output /tmp/moe-vllm.json
```

原版示例；dense 改为 GPU 3、`dense-inputs.json` 并去掉 `--pp 4`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 PYTHONPATH=/home/bul/dev/exllamav3 \
/home/bul/miniconda3/bin/conda run --no-capture-output -n exllamav3 \
/home/bul/.local/bin/uv run --no-project \
  --python /home/bul/miniconda3/envs/exllamav3/bin/python \
  python /home/bul/dev/vllm-backport/benchmarks/benchmark_exl3.py \
  --backend exllamav3 --pp 4 \
  --inputs /home/bul/dev/vllm-backport/docs/validation/exl3-20260910/moe-inputs.json \
  --output /tmp/moe-exllamav3.json
```

启动本机 GLM 文本服务，可沿用验证时的限制配置：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=11,11,11,12 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 \
.venv/bin/vllm serve /home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw \
  --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 2147483648 \
  --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'
```

测试命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest tests/quantization/test_exl3.py -q
```

## 证据文件

[JSON 摘要](exl3-integration-20260910.json)保存完整性检查、源码哈希、模型配置哈希和各次结果索引。[dense 输入](exl3-20260910/dense-inputs.json)与[MoE 输入](exl3-20260910/moe-inputs.json)保持测量时的原始字节，可检查两个后端的 `inputs_sha256`。[诊断日志](exl3-20260910/diagnostics.json)保留成功检查以及失败轮次。原始结果包含全部 token ID、文本、停止原因和各次计时；性能中每次生成长度及零缓存命中均已断言。

原版 dense 主测 JSON 的 `revision` 来自当时命令工作目录，记录的是 vLLM 提交；不将该字段误作 ExLlamaV3 版本。原版实际源码提交及 tag 已独立核实并记录在摘要中。生成的输出与诊断数据保留原文，不作拼写修正。
