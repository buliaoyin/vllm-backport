# DeepSeek V4 0731 性能优化与验证

日期：2026-09-08。本轮按用户要求将 `VLLM_DETERMINISTIC_MOE_ALIGN` 全局默认值改为 `0`，继续排查同一模型在当前仓库与 `vllm-dsv4-a100` 的性能差距。使用 `vllm-backport` conda 环境，工作由 AI 辅助完成。

## 本轮代码变更

- `vllm/envs.py`：类型提示与运行时默认值同步改为关闭确定性 MoE 分组；显式设置 `VLLM_DETERMINISTIC_MOE_ALIGN=1` 仍可启用稳定分组。对应的确定性测试显式开启该选项，不再依赖全局默认值。
- `vllm/v1/attention/ops/common.py`、`backend.py`：将 GPU 上的 query 长度计算、arange、repeat_interleave、拷贝及补零合并成一个 Triton 内核。按设备端 query 边界二分查找请求 ID，保留自适应验证对设备端边界的要求，支持零长度请求、补齐位置、缓冲区复用和非连续输出。CPU 路径保留原逻辑。
- `vllm/v1/worker/gpu/model_runner.py`：采样完成后立即发送已采样 token 与计数，随后生成草稿，再发送草稿快照。保持接收端的 collective 顺序：sampled → counts → draft。
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`：SM80 且 head tile 为 16 时，稀疏解码 partial 内核改用 8 warps，避免宽累加器的寄存器溢出；其他设备与 head tile 保留原配置。

本轮没有改模型权重、C4 默认 tile、DSpark 默认采样方法、草稿数或自适应验证设置。此前的视觉支持与 PP 状态修复仍保留。

## 对照配置

模型为 `/home/bul/dev/models0/DeepSeek/DeepSeek-V4-Flash-0731`。当前仓库基于 `9201d9936cf7c90be518d3a5fc880c32c32202c4`，参考仓库为 `/home/bul/dev/dsv4/vllm-dsv4-a100` 的 `a65c660dadca24a276986e6d2379f1ff2399c869`。参考仓库通过 `PYTHONPATH` 加载，源码与安装均未改动。

两边使用同一 conda 环境，PyTorch `2.13.0+cu130`，CUDA 13.0；GPU 0/1/2 均为 CMP 170HX，PP3/TP1、分层 `15,16,12`、`NCCL_P2P_DISABLE=1`、V2 runner、FP8 KV、显存比例 0.95、上下文 32768、`max-num-seqs=64`、`max-num-batched-tokens=2048`、`DSV4_LOGITS_ROW_CHUNK=64`，开启前缀缓存和分块 prefill。DSpark 配置为 `{"method":"dspark","num_speculative_tokens":7}`，默认 greedy，不开启自适应验证。

每题单请求输出 512 token，temperature=0、seed=42、thinking=false，三轮取中位数。速度口径为 `(completion_tokens-1)/(结束时间-首段内容时间)`。题目为 Python 异步缓存、TypeScript 调度器、SQL/Python 事务 outbox。这是固定工作负载对照，并非用户未提供的原始测速题目逐项复现；截取 512 token 不等于完整可执行代码质量评估。

## 代码生成速度

| 配置 | Python token/s | TypeScript token/s | SQL/Python token/s |
| --- | ---: | ---: | ---: |
| 修改前默认 1，前轮基线 | 126.78 | 154.14 | 123.76 |
| 仅关闭确定性分组，前轮对照 | 139.67 | 169.49 | 134.99 |
| 默认 0 + 融合映射 | 143.95 | 172.24 | 136.06 |
| 默认 0 + 融合映射 + 提前发送 | 149.89 | 172.18 | 139.28 |
| 最终：再调整 SM80 解码 warps | **152.57** | **178.35** | **149.26** |
| 参考仓库，前轮对照 | 151.42 | 174.75 | 148.20 |

关闭确定性分组是主要收益来源。融合映射与提前发送提供进一步的小幅收益，但上述 token/s 同时受文本分歧与草稿接受率影响，不能全部算作实现开销的下降。各组本轮测试均实际取消环境变量覆盖，使用新的默认值。最终组未加载事件追踪 hook，也未加载参考动态库。相对前轮默认 1 基线，最终三题分别提升约 20.35%、15.71%、20.61%；与参考组已处于同一水平，少量领先不构成稳定优势的统计结论。

| 配置 | Python 估算每轮 ms | TypeScript 估算每轮 ms | SQL/Python 估算每轮 ms |
| --- | ---: | ---: | ---: |
| 仅关闭确定性分组 | 32.85 | 32.91 | 33.04 |
| 再融合映射 | 32.59 | 32.59 | 32.86 |
| 再提前发送 | 32.60 | 32.39 | 32.59 |
| 最终：再调整 SM80 解码 warps | **31.25** | **31.07** | **31.17** |
| 参考仓库 | 31.44 | 31.19 | 31.23 |

每轮时间为平均接受长度除以 token/s 的近似值，含请求边界、CPU 与通信影响，并非直接测得的内核时间。提前发送在 Python 上没有证明每轮加速，Python token/s 的主要额外变化来自接受长度增加；不能只看吞吐提升来判断优化收益。

## 映射热点与微基准

首次 Torch profiler 记录到当前三个 rank 的 `aten::repeat_interleave` 事件分别为 512、512、716 次，参考为每个 rank 2 次。一次 API 调用可能产生多个嵌套事件，两边请求步数也不同，因此不能将事件数直接当成独立映射调用数或将 CPU 时间逐项相加。源码确认参考用 CPU 展开并上传，当前旧实现使用设备端展开；新内核保留设备端边界支持并去掉多次小操作。

CMP 170HX 的 CUPTI 返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)`，上述 trace 只有 CPU 活动，不能作为 GPU 内核时间线。微基准改用 CUDA Graph + CUDA Event，计时区间外清理 128 MiB L2，80 次交替测量；另测 eager 端到端调用开销，包括原路径的临时分配与 CPU 调度。先逐元素核对整数输出一致，再计时。

以下使用没有尾部补齐的输入；完整工件另外包含补齐 3 个 token 的测试。

| GPU / 形状 | 原映射 GPU μs | 融合映射 GPU μs | 原映射 eager μs | 融合映射 eager μs |
| --- | ---: | ---: | ---: | ---: |
| CMP 170HX，1 请求 / 8 token | 21.50 | 6.14 | 80.94 | 15.44 |
| CMP 170HX，64 请求 / 512 token | 19.46 | 7.17 | 86.87 | 15.84 |
| CMP 170HX，1 请求 / 2048 token | 20.48 | 6.14 | 88.36 | 15.44 |
| RTX PRO 6000 Blackwell，1 请求 / 8 token | 12.32 | 2.11 | 82.77 | 15.75 |

这是以启动延迟为主的微小整数操作，不是 GEMM 算力测试。其收益不能按层数直接相乘解释整模型加速；完整形状、精度、逻辑字节数和计时方法保存在结果 JSON。

## 剩余差距的阶段定位与修复

为绕开 CMP 的 CUPTI 限制，使用临时导入 hook 记录 CUDA Event。对同一 TypeScript 请求预热两次，跟踪随后 1024 token 的输出；目标计算图使用 8 query，DSpark 使用 7 query。目标前向统计剔除 PP 空轮次，图内计时按 query 形状区分目标与草稿。GPU 0/1/2 的主要运行频率均约 1470 MHz。

下面的当前组已经包含默认 0、融合映射和提前发送，尚未调整 warps；数值为 CUDA Event 中位数，单 rank 的目标图采样数为 96/96/48，草稿图为 48。

| 阶段 | 当前 ms | 参考 ms |
| --- | ---: | ---: |
| PP0 目标计算图 | 9.4367 | 9.0327 |
| PP1 目标计算图 | 9.3926 | 8.9400 |
| PP2 目标计算图 | 7.1030 | 6.8449 |
| 目标图分段中位数之和 | 25.9323 | 24.8176 |
| DSpark 草稿计算图 | 2.7213 | 2.6726 |
| 最后一级采样 | 0.7055 | 0.7035 |

分段中位数之和用于定位计算成本，不是请求端延迟或多 rank 同步后的全局单样本时间。元数据 CPU 耗时当前 PP0/PP1 约 1.22/2.65 ms，参考约 1.41/3.06 ms；草稿准备也接近。因此约 1.1 ms 的主要剩余差距在目标图内，不能再一概归因于元数据、MTP 接受率或 Python 调度。

两边 Marlin MoE CUDA 源码相同，但二进制校验值不同。临时加载参考 `_moe_C_stable_libtorch` 的对照因旧 `topk_softplus_sqrt` 接口只接收 10 个参数、当前视觉路由接口传入 12 个参数而未能启动；没有获得该组性能结果，也没有替换仓库中的动态库。

后续发现：两边 SM80 稀疏解码 partial 内核正文与 split 启发式相同，但当前启动使用 4 warps，参考使用 8。组合微基准同时比较 warps 与 K tile，并读取编译器资源统计，确认 4 warps 的寄存器溢出是实际成本来源。

以下包含 main SWA 128 行；C4 的索引缓冲区容量按每 query 512 条、C128 按 256 条保留，即使实际有效行较少，仍让 split 选择看到与 32768 上下文服务一致的容量。80 次交替 CUDA Graph 计时，计时外清理 128 MiB L2，FP32 参考校验在计时前执行。

| 压缩比 / query / 有效压缩行 | warps / K | 耗时 μs | 编译器 n_spills |
| --- | --- | ---: | ---: |
| C4 / 8 / 128 | 4 / 32 | 102.40 | 8 |
| C4 / 8 / 128 | 8 / 32 | 62.46 | 0 |
| C4 / 8 / 128 | 4 / 64 | 158.72 | 492 |
| C4 / 8 / 128 | 8 / 64 | 61.95 | 2 |
| C4 / 8 / 512 | 4 / 32 | 162.82 | 8 |
| C4 / 8 / 512 | 8 / 32 | 103.42 | 0 |
| C128 / 8 / 256 | 4 / 32 | 109.57 | 8 |
| C128 / 8 / 256 | 8 / 32 | 67.58 | 0 |

各组对 FP32 参考的最大绝对误差为 0.00195–0.00391。8 warps + K=32 消除了所测形状的寄存器溢出，C4 单 query 下也优于 K=64。因此保留 K=32 的默认值，只调整所验证的 SM80/head-tile=16 启动配置。此前“单改 K=64 变慢”发生在 4 warps 下，组合实验解释了该现象。

这项变更另外通过 SM80 attention 回归 64 passed、1 skipped，包含不同 split 数、压缩比、无 extra cache、attention sink、NaN 清理和输出布局。最终整模型估算每轮开销由约 32.4–32.6 ms 降至 31.1–31.3 ms，与阶段计时发现的约 1.1 ms 剩余差距相符。微基准百分比仍不能直接视作整模型加速率。

## 质量与状态回归

最终代码全量官方 GSM8K test 为 **1271/1319，96.36%**。配置为 5-shot、客户端并发 4、thinking=false、输出上限 4096，与调整 warps 前相同。全部 1319 个请求正常 stop，没有 length 截断，输出 138400 token，耗时 531.05 秒；草稿接受率 54.77%，平均接受长度 4.8338。最终组前 32 题抽测为 32/32。

与调整 warps 前的全量测试逐题核对，请求体 1319/1319 相同；1266 题两边都对、41 题都错，旧配置单独答对 7 题，新配置单独答对 5 题。最终少答对 2 题，即约 0.15 个百分点；616 题文本完全相同，其余文本存在差异。这里没有重复全量实验估计随机波动，不能宣称精度无损或将差异确定归因于某一个浮点算子。

最终版本再次通过混合结构化输出 48/48、64 并发短请求 64/64、21034 token 长输入检索。64 并发首轮用时 11.62 秒，日志同时记录了首次编译新的 sparse decode partial/reduce 形状；这是正确性与状态测试，不能将其用作已经预热的并发吞吐对照。首次遇到未覆盖形状仍会有 JIT 编译开销。

调整 warps 前，`DeepSeek-V4-Flash-0731` 在默认 0、融合映射与提前发送的组合下，全量官方 GSM8K test 为 **1273/1319，96.51%**。5-shot，客户端并发 4，thinking=false，输出上限 4096；所有 1319 个请求正常 stop，无截断，输出 138658 token，耗时 547.70 秒。平均草稿接受长度 4.8215，接受率 54.59%。测试期间未混入其他模型请求。

测试集 SHA256：`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。逐题请求、响应及错题索引均保留：调整 warps 前 46 题错，最终 48 题错。没有本配置修改前的全量配对基线，不能宣称全量准确率完全不变；此前 Vision 模型的 1271/1319 也不应作为这个不同 checkpoint 的配对比较。

- 融合映射组 GSM8K 前 32 题为 32/32；再提前发送组为 31/32，错题为 index 12。没有覆盖初次结果。
- 严格 Schema、JSON object、普通算术混跑：48/48。
- 同时提交 64 个短请求：64/64。
- 21034 token 长输入检索：返回正确口令。
- 注意力元数据测试：32 passed，含 CPU/GPU 边界不一致、零长度请求、补齐、非连续输出、缓存指针及 CUDA Graph 回放后动态边界更新。
- 环境变量、DSpark、V2 worker、异步输出测试：66 passed。
- MoE 分组及专家映射：25 passed；确定性测试改为显式开启后复测 1 passed。

最终代码还在前三张 CMP 170HX 上加载 `DeepSeek-V4-Flash-Vision-Exp`，PP3/TP1、DSpark 7、FP8 KV、`max-num-seqs=64`。该轮使用 `max-model-len=auto`，实际由 1048576 缩至 466432；这不表示已验证 466K 长度的实际请求。结果如下：

- 图像检查 20/20：19 个视觉检查加 1 个纯文本对照，覆盖颜色替换、OCR、空间关系与多图顺序。
- 混合结构化输出 48/48，GSM8K 前 32 题 32/32。
- 超过 2048 token prefill 分块边界的长图文请求 6/6。
- 取消 8 个流式请求后，多轮图像与严格 Schema 请求 16/16，服务没有运行错误。

以上是视觉功能与请求状态回归，不是公开 VQA 数据集的质量评测。所有本轮测试服务已关闭，显存已释放；参考仓库仍无改动。

本轮修改文件的 pre-commit 检查与手动 mypy 3.12 检查通过。

这些测试没有重跑 GLM/Qwen 的全量评测。全局默认值改变后，既有默认 1 下的逐字复现结论不能直接套用；显式设为 1 可恢复稳定分组选择，但仍不保证所有模型算子完全确定性。

## 工件

原始工件根目录：`/tmp/dsv4-perf-opt-20260908/`。包括逐题输出、启动日志、分阶段计时工具、CPU trace、各轮参数与运行代码补丁。前轮基线见 [此前速度分析](dsv4-code-speed-analysis-20260908.md)。汇总数据见 [结果 JSON](dsv4-perf-optimization-20260908-results.json)，包含各轮测速、完整形状、资源统计、错误索引及原始工件路径。`/tmp` 内的原始日志可能被系统清理，长期保留时需一并归档。

从仓库根目录复现微基准：

```bash
CUDA_VISIBLE_DEVICES=0 /home/bul/miniconda3/envs/vllm-backport/bin/python \
  -m benchmarks.kernels.benchmark_token_to_req_indices --output /tmp/token-map.json
CUDA_VISIBLE_DEVICES=0 /home/bul/miniconda3/envs/vllm-backport/bin/python \
  -m benchmarks.kernels.benchmark_dsv4_decode_launch --output /tmp/decode-launch.json
```

本轮最终文本模型启动命令：

```bash
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=15,16,12 \
CUDA_VISIBLE_DEVICES=0,1,2 VLLM_USE_V2_MODEL_RUNNER=1 \
DSV4_LOGITS_ROW_CHUNK=64 VLLM_DETERMINISTIC_MOE_ALIGN=0 \
/home/bul/miniconda3/envs/vllm-backport/bin/vllm serve \
  /home/bul/dev/models0/DeepSeek/DeepSeek-V4-Flash-0731 \
  --served-model-name local --host 127.0.0.1 --port 8001 \
  --pipeline-parallel-size 3 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
  --enable-prefix-caching --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

上面显式写出默认值 0 便于复现；实际测速启动脚本取消了该环境变量，验证的是代码默认行为。
