# DeepSeek V4 三卡优化与验证

日期：2026-09-08。本文由 AI 辅助实施、测试和整理，记录本地工作区结果，未发布 PR。

已完成适用的 PP/DSpark 缓冲区、草稿交接和 SM80 内核优化。三卡 PP3 全量 GSM8K 在 MTP 开启/关闭时均为 **1271/1319（96.36%）**；原先失败的 JSON 混跑从 **20/48 提高到 48/48**。视觉、长前缀多图及取消请求后的状态复用均通过。内核测量有收益，短文本整模型解码相对旧代码没有普遍提速。

## 范围与实现

基线为当前仓库 `codex/deepseek-v4-vision-tested` 分支的 `9201d9936cf7c90be518d3a5fc880c32c32202c4`。参考仓库为 `/home/bul/dev/dsv4/vllm-dsv4-a100`，分支 `170hx-dsv4f-pp3-1m`、提交 `a65c660da`。参考仓库未修改。

按照[前次差异分析](dsv4-a100-comparison-20260908.md)移植第 1–8 项，并按当前较新的视觉、调度和计算图实现调整：

| 改动 | 当前实现及限制 |
| --- | --- |
| PP 接收缓冲区复用 | V2、TP1 下直接接收至预分配输入区；代次与状态检查防止接收、主模型、DSpark 交错覆盖，另有可选 stream 检查 |
| DSpark 工作区复用 | 不再分配普通 MTP 的隐藏状态缓冲区；容量、dtype、设备及别名均允许时才借用已消费的 PP 输入区，其他情况保留分配回退 |
| C128 元数据合并 | decode/prefill 使用同一存储的不同区域，保留当前计算图要求的固定行步长 |
| 固页内存池 | 请求索引、query/logits 偏移和 grammar mask 分别复用暂存池；扩容时保留尚可能被 DMA 读取的旧存储 |
| 结构化输出正确性 | 草稿随产生它的输出返回，以请求 ID 和 producer step 匹配；缺失或过期草稿按无效后缀处理；V1 在生成 grammar mask 前等待在途状态推进 |
| 草稿广播快照 | 强制创建独立存储，避免 dtype 已符合时 `.to()` 返回原张量导致广播内容被后续步骤改写 |
| C128 prefill | 接通 SM80 query 分块；TP1 自动使用 block_m=4；存在图片双向可见区间时仍走原有视觉路径 |
| SM80 压缩内核 | C128 使用 16 warps，C4 保持 4；保留现有其他设备和 ROCm 处理 |
| Decode 候选参数 | 接通分层 tile 与固定 splits，实际测量后保留原 32 tile、原 splits 启发式作为默认 |

批次草稿匹配、V1 FSM 修复及 PP 接收复用分别参考 `a6d6ab46f`、`6e959b2ea`、`f73035cd5`、`94ad349dd`。最终实现没有采用临时禁用结构化输出推测解码的 `9dd006256` 方案，也没有覆盖当前视觉 token ID 路由、图片可见范围、GLM/Qwen 量化兼容和较新的计算图逻辑。

三卡 TP1 暂未移植按层 FP8→BF16 排除、INT8 all-reduce、局部 argmax all-reduce 和 Marlin 负收益实验。它们不能从本次结果推导出收益。

## 环境与复现

模型：`/home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp`。

模型服务仅使用 GPU 0、1、2，均为 CMP 170HX / SM80。部分独立回归测试使用 GPU 3；它不参与该模型的 PP 服务。环境为用户指定的 `vllm-backport` conda：Python 3.12，PyTorch `2.13.0+cu130`，CUDA runtime `13.0`，Triton `3.7.1`，Transformers `5.16.1`。为通信测试补装 `ray==2.56.1` 及 `msgpack==1.2.2`。

```bash
cd /home/bul/dev/vllm-backport
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_PP_REUSE_RECV_BUFFER_DEBUG=1 \
NCCL_P2P_DISABLE=1 \
VLLM_PP_LAYER_PARTITION=15,16,12 \
CUDA_VISIBLE_DEVICES=0,1,2 \
/home/bul/miniconda3/envs/vllm-backport/bin/vllm serve \
  /home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp \
  --served-model-name local \
  --pipeline-parallel-size 3 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --block-size 256 \
  --limit-mm-per-prompt '{"image":2}' \
  --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --host 127.0.0.1 --port 8001 \
  --speculative-config '{"method":"dspark","model":"/home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}'
```

关闭 MTP 时删去 `--speculative-config`；其余参数相同。同步调度验证另加 `--no-async-scheduling`。本报告中的 MTP 指 DSpark，3 个推测 token，不是另一种普通 MTP 实现。

实际验证边界仍为 8K、最多 4 个活跃序列、每请求最多 2 张图，不能据此宣称视觉模型三卡支持 1M 上下文。

## 可单独排查的开关

| 开关 | 默认与用途 |
| --- | --- |
| `VLLM_PP_REUSE_RECV_BUFFER` | 默认 1；设 0 可关闭 PP 直接接收复用 |
| `VLLM_PP_REUSE_RECV_BUFFER_DEBUG` | 默认 0；本次服务验证设 1，增加 stream 归属检查 |
| `VLLM_SPARSE_DENSE_QUERY_BLOCK` | 默认 auto（-1）；设 0 可让 C128 prefill 回到逐 query 路径 |
| `VLLM_DSV4_SM80_COMPRESSOR_TUNING` | 默认 1；设 0 使用原压缩器 warp 数 |
| `VLLM_DSV4_UNIFORM_DECODE_BLOCK_K` | 默认 1；设 0 试验 SM80 C4 split-k 路径的 64 tile，本机所测形状更慢 |
| `VLLM_DSV4_FIXED_DECODE_SPLITS` | 默认 0；正值覆盖 SM80 split-k 路径的启发式，最多 16 |

## 正确性结果

GSM8K 使用官方 test 的全部 1319 题，5-shot，temperature=0，seed=42，thinking=false，最大输出 4096 token，客户端并发 4。test 文件 SHA256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。答案采用仓库现有数字提取方式，不要求逐字一致。

| 配置 | GSM8K | 视觉及算术探针 | Schema / JSON / 普通请求混跑 |
| --- | --- | --- | --- |
| 优化前，MTP3 | 32/32（抽测） | 20/20 | 20/48 |
| 仅 PP/缓冲区/草稿交接优化，MTP3 | 32/32（抽测） | 20/20 | 48/48 |
| 全部采纳的优化，MTP3 | **1271/1319，96.36%** | **20/20** | **48/48** |
| 全部采纳的优化，MTP 关闭 | **1271/1319，96.36%** | **20/20** | **48/48** |

两次全量请求体逐题一致；1264 题两边都对、41 题两边都错，MTP 独自答对 7 题、关闭 MTP 独自答对 7 题。最终文本有 706 题不逐字相同，不能把相同准确率写成输出完全等价。关闭 MTP 同样没有截断，输出共 134315 token，耗时 1166.67 秒；本轮 MTP 总耗时缩短约 34.5%，速度比约 1.53×。

优化前的混跑中，严格 JSON Schema 仅 1/16、JSON object 3/16、普通算术 16/16 通过；日志出现 `Failed to advance FSM` 和 grammar 拒绝 token。修复后这三类均为 16/16，并未关闭受约束请求的推测解码。

MTP 全量 1319 个请求全部正常 stop，无 length 截断。草稿接受率 74.82%，平均接受长度（含目标模型 token）3.2446。全量耗时 764.26 秒、输出共 133288 token。该耗时包含请求调度与前缀缓存，也有独立代码测试在部分时间运行，不能作为严格隔离的内核 A/B。

额外视觉与状态复用测试：

- 原 20 个探针含 19 个视觉检查和 1 个纯算术对照，覆盖颜色替换、OCR、空间关系和多图顺序。
- 长前缀单图/多图 6/6；单图输入 5630 token，多图 5847 token，超过 2048 token 分块预算。图片完整区间仍由现有双向注意力路径处理。
- 主动断开 8 个流式请求后，多轮图片严格 Schema 输出 16/16。对话历史不预先提供目标颜色。
- MTP 关闭时也补跑同样的 6 个长图测试、8 次流断开和 16 个后续多轮 Schema，全部通过。
- 两种模式服务运行日志均未见 FSM 错误、PP 复用保护异常或 CUDA 非法访存。

优化前仅做 32 题质量基线，不能据此声称全量准确率相对旧代码完全不变。视觉探针也不等同于公开多模态基准。

## 性能与显存

### 内核：先数值检查，再冷 L2 测量

使用 `benchmarks/kernels/benchmark_dsv4_pp3_optimizations.py`，在一张 CMP 170HX 上模拟 TP1 的 64 heads、512 维。CUPTI 在该环境不可用，采用 CUDA graph + CUDA event，30 次中位数；每次显式清理 64 MiB L2，清理不计入计时间隔。早期未显式清 L2 的 `kernel-ab.json` 不作为本表依据，最终数据为 `kernel-ab-cold.json`。

| 内核及形状 | 原路径 | 采纳路径 | 加速比 |
| --- | ---: | ---: | ---: |
| C128 prefill，128 query，前缀 0 | 95.232 μs | 69.632 μs | 1.37× |
| C128 prefill，512 query，前缀 4096 | 500.736 μs | 338.944 μs | 1.48× |
| C128 prefill，2048 query，前缀 4096 | 1973.248 μs | 1247.232 μs | 1.58× |
| C128 压缩，1 个输出位置 | 267.264 μs | 40.960 μs | 6.53× |
| C128 压缩，16 个输出位置 | 286.720 μs | 44.544 μs | 6.44× |

Prefill 对 FP32 参考的抽样归一化最大误差为约 0.0030–0.0041，小于 0.01 门槛；补充单元测试覆盖多请求、压缩边界、非零 query 偏移。压缩内核四组 C4/C128 样例的最终量化字节与原配置完全一致。

**未采纳的默认设置：** C4 decode 单 query 从 BLOCK_K=32 的 55.296 μs 变为 BLOCK_K=64 的 118.784 μs，再固定 16 splits 为 125.952 μs；其他形状也没有证明普遍收益。因此 `VLLM_DSV4_UNIFORM_DECODE_BLOCK_K=1` 保持默认；`VLLM_DSV4_FIXED_DECODE_SPLITS=0` 表示沿用原启发式。以前默认注册的 16 没有被执行路径读取，接通后改成 0 是保留原来的实际行为。

以上只是所测形状的内核收益。基准按固定顺序测量配置，未做跨时段统计；不能换算成整模型加速率。

### 整模型短文本解码

单请求，中文文章、代码和英文设计题各连续输出 256 token，三轮中位数；速度为 `(输出 token 数 - 1) / 首次内容至结束时间`。

| 配置 | 中文 token/s | 代码 token/s | 英文 token/s |
| --- | ---: | ---: | ---: |
| 优化前 MTP3 | 71.64 | 106.74 | 84.10 |
| 仅 PP/缓冲区/交接优化 MTP3 | 71.57 | 107.09 | 84.20 |
| 全部采纳优化 MTP3 | 73.77 | 103.50 | 78.46 |
| 全部采纳优化，MTP 关闭 | 51.91 | 51.90 | 51.97 |

相对关闭 MTP，开启 MTP 的中文/代码/英文速度比分别为约 1.42× / 1.99× / 1.51×。

相对优化前代码，整模型短解码没有普遍加速：相对原始基线约为 +3.0%、-3.0%、-6.7%。各次输出及草稿接受率不同，不能将这些差异全部归因于内核执行时间。PP 内存优化的主要已证实收益是内存与正确性；C128 优化的加速结论目前限于上述内核测量。

### 显存

最后一级模型加载记录从 53.66 GiB 降到 53.60 GiB。0.5 秒间隔的 `nvidia-smi` 全卡占用采样中，最后一张卡从 61400 MiB 降到 61316 MiB，约减少 84 MiB。优化前、后 workload 长度不同，采样也可能错过瞬时峰值；这不是精确 CUDA allocator 峰值。

KV cache 仍为 20151 token，第一张卡的可用 KV cache 仍约 5.17 GiB。因此本次没有增加已经验证的上下文容量。

## 回归与检查

- PP/草稿/调度初轮：44 passed，4 skipped。
- GLM/Qwen 配置、量化后端选择、prefix cache、KV 管理、计算图、异步交接：252 passed。
- SM80 attention 数值回归：63 passed，覆盖分块 prefill、C4/C128 split-k 及输出布局约束。
- PP 接收张量复用及 TP all-gather 后处理：2 passed。首次收集缺少 Ray，补齐依赖后重新通过。
- 修改文件常规 pre-commit 与 Python 3.12 mypy 均通过。
- V1 多批次结构化输出集成：1 passed；新增/调整的 worker、DSpark、staging、metadata 及 grammar 测试复核：23 passed、4 skipped。
- 最终代码异步 MTP 重启复测：GSM8K 31/32，视觉 20/20，混跑 48/48。错题为 test index 12，将开始盈利的第 13 年答成回本的第 12 年；保留原结果，没有重跑覆盖。短解码三轮中位数为 73.76 / 103.25 / 78.98 token/s，与前次范围接近。
- 最终代码同步 MTP（`--no-async-scheduling`）：GSM8K 32/32，视觉 20/20，混跑 48/48；草稿接受率 73.40%，平均接受长度 3.2019。普通请求的草稿通过各批输出返回，未被同步路径丢弃。

首次较广的 attention 测试为 76 passed、16 skipped、3 failed。其中输出末维布局检查的缺失已修复，并在上述 63 项中通过；另两项是原有 `test_rocm_ragged_graph_buffer_view_tracks_source_width[False/True]` 仍调用已删除的 `_copy_ragged_to_graph_buffers`。基线 HEAD 的实现也没有该 helper，属于已存在的测试不匹配，未为通过测试恢复过期接口。不能将整套 attention suite 写成全部通过。

V1 集成测试首次失败于测试适配：直接构造请求没有调用当前版本的 `update_from_generation_config()` 初始化 EOS，语法终止后继续输出。补齐 EOS、保证至少 12 个输出 token，并在解码断言时跳过特殊 token 后通过；没有为这个测试额外修改运行代码。

末轮审查修正了移植过程中 offloader 初始化被误放到 DSpark 步骤的问题：恢复在模型加载末尾调用，并在现有 load_model 测试中检查该生命周期，5 项通过。本轮两次全量评测启动在这项修正之前；服务未启用权重 offload，使用 `NoopOffloader`，其 `post_init()` 为空操作，不改变数值和内存。最终代码另行重启进行了 MTP 与同步调度复测，结果已列在本节。

本轮没有重新跑 GLM/Qwen 大模型完整 GSM8K。252 项仅证明相关代码层回归通过，不替代原有模型级验证。

## 原始证据

机器可读的[结果与源码清单](dsv4-optimization-20260908-results.json)随报告保存。测试服务及采样进程已停止，四张卡显存回到空闲水平。

工作目录 `/tmp/dsv4-optimization-20260908/` 包含：

- `baseline/`、`phase1-retry/`、`optimized/`、`mtp-off/`、`final-async/`、`final-sync/`：每题请求、原始返回和汇总。
- `optimized-stress/`、`mtp-off-stress/`：长前缀多图、断开请求后的多轮视觉 Schema 记录。
- `*-serve.log`、`*-eval.log`、`*-memory.csv`：服务、测试与显存采样。
- `paired-gsm.json`：逐题配对统计；`source-manifest.json`：基线、环境与最终源码哈希。
- `kernel-ab-cold.json`：有效内核测量；`phase2-numerics.log`、`compatibility-unit.log`、`recv-unit-retry.log`、`pre-commit-final2.log`、`mypy312-final.log`：检查结果。
- `serve.sh`、`evaluate.py`、`stress.py`、`monitor.py`：本次运行驱动。评测器复用 `/tmp/deepseek-v4-vision-validation-20260908/` 的视觉资产和现有 GSM8K 入口。

复现内核测试：

```bash
PYTHONPATH=/home/bul/dev/vllm-backport CUDA_VISIBLE_DEVICES=0 \
/home/bul/miniconda3/envs/vllm-backport/bin/python \
  benchmarks/kernels/benchmark_dsv4_pp3_optimizations.py \
  --output /tmp/dsv4-kernel-ab.json
```

复现已启动服务的完整质量验证：

```bash
PYTHONPATH=/home/bul/dev/vllm-backport \
/home/bul/miniconda3/envs/vllm-backport/bin/python \
  /tmp/dsv4-optimization-20260908/evaluate.py rerun --count 1319
```

## 启动参数补充回归

用户随后提供的 DSpark 7 配置校验错误和 DSpark 6 在 `auto` 缩容后的 C128 捕获错误已修复，详见[两处启动错误修复报告](dsv4-startup-errors-20260908.md)。在前三张卡、`max-num-seqs=64` 下，两组均成功自动缩容至 466,432 token 并完成 CUDA Graph 捕获；32 题 GSM8K 抽测、视觉与混合结构化输出、64 个同时提交的短请求及 42,034 token 输入检索均通过。这是独立于上文固定 8K 全量测试的补充结果。
