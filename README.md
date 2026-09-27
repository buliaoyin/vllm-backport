# vLLM Backport

本分支整合 SM80 优化、EXL3 量化支持和 DeepSeek-V4.1 混合推理。
以下提供 DeepSeek-V4.1-Flash、EXL3、DeepSeek-V4-Flash（DSV4F）和
DeepSeek-V4-Flash-Vision-Exp（DSV4F-Vision）的启动示例。

## 安装与运行环境

三卡示例使用 **3 × NVIDIA CMP 170HX 64 GiB（SM80）**，采用流水线并行
`PP=3`、张量并行 `TP=1`。将模型路径替换为本地完整权重目录，按实际设备调整
`CUDA_VISIBLE_DEVICES`。各示例独立运行，默认监听 `0.0.0.0:8000`。

首次安装，在仓库根目录执行：

```bash
uv venv --python 3.12
source .venv/bin/activate
unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION
TORCH_CUDA_ARCH_LIST='8.0' MAX_JOBS=8 \
  uv pip install -e . --torch-backend=auto
```

已有可用环境时可跳过首次安装。运行前须编译本分支的原生算子；上游预编译 wheel
不能提供本分支新增的全部功能。复用现有 PyTorch、构建 wheel 和 CMake 增量编译见
[构建说明](docs/contributing/deepseek_v41_build.md)及
[增量编译指南](docs/contributing/incremental_build.md)。同时支持 SM120 时，
将 `TORCH_CUDA_ARCH_LIST` 改为 `8.0;12.0`，并使用支持这些架构的 CUDA toolkit。

下面使用 `.venv/bin/vllm`，避免调用其他环境的安装。三卡命令沿用实测配置中的
`NCCL_P2P_DISABLE=1`；启动前可用 `nvidia-smi` 核对设备编号和剩余显存。

## 三张 170HX：DeepSeek-V4.1-Flash 混合推理

DeepSeek-V4.1-Flash 使用 CPU/GPU 混合推理，分层由 `deepseek_v41_hybrid`
配置控制。它与 V4 Flash 使用不同的分层、解析器和 KV 预算参数。

当前支持：

- CPU/GPU 混合推理、自定义流水线分层，以及 DSpark、并发、视觉输入和工具调用。
- 按 KV token 预算预留显存，自动分配 GPU 专家缓存；支持 prefill/decode 热点更新，记录缓存覆盖率、命中率和专家替换信息。
- 优化 Engram／专家权重加载、CPU 专家算子和 prefill，并限制主机预打包缓存的内存占用。
- 可选 [Engram SSD 存储](docs/features/deepseek_v41_engram_ssd.md)，按需直接读取原权重文件，使用有界缓冲及热点缓存。
- CPU 专家按 NUMA 节点自动分片和绑核，适配单／双路 NPS1、NPS2、NPS4；共享 Engram 内存分散到可用节点，无需新增启动参数。
- 可通过 `--enable-prefix-caching` 复用相同前缀的 KV，兼容 CED 尾回放和 DSpark。

内存参考（此前三卡实测）：加载后 worker 主存占用约 **344 GiB（PSS）**，建议配置
**512 GiB 主存**。开启 DSpark 时三卡显存采样峰值合计约 **188–189 GiB**；
实际占用随分层、KV 预算和并发数变化。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 \
.venv/bin/vllm serve /model/path/DeepSeek/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port 8000 \
  --pipeline-parallel-size 3 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 8 \
  --max-model-len 524288 \
  --kv-cache-tokens 1048576 \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --default-chat-template-kwargs '{"reasoning_effort":50}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}' \
  --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[7,8,25],"cpu_threads":[24,24]}}'
```

`cpu_threads` 的两个值分别为 prefill 和 decode 的**总线程数**，并非每个 NUMA
节点的线程数。首次检查启动时，可先将 `--max-model-len` 降为 `32768`、
`--kv-cache-tokens` 降为 `131072`，再按负载增加预算。

重复长前缀、多轮对话可在上例增加 `--enable-prefix-caching`（混合模式默认关闭）。
命中后仍至少重算最后 **128 token**，按 KV 块边界对齐，以重建 CED 尾回放和
草稿上下文；日志中的 `Prefix cache hit rate` 反映前缀复用，与专家缓存命中率独立。
缓存使用已有 KV 预算，不额外扩大 `--kv-cache-tokens` 的内存预留。

内存不足时，可将上例的 `--additional-config` 替换为：

```bash
--additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[7,8,25],"cpu_threads":[24,24],"engram_storage":"ssd"}}'
```

此前三卡对照中，SSD 模式将 worker 主存由约 **342 GiB 降至 154 GiB**；默认行缓存
总计 **1 GiB**。冷请求的 prefill 会受 SSD 随机读取性能影响，详细配置和限制见
[Engram SSD 说明](docs/features/deepseek_v41_engram_ssd.md)。

## EXL3

### 依赖与单卡启动

EXL3 权重由 `config.json` 中的 `quantization_config.quant_method` 自动识别，
无需手动设置 `--quantization`。当前支持 SM80 及更新的 NVIDIA GPU、FP16/BF16
激活；`TP` 和专家并行度必须为 1，大模型使用流水线并行。

在 vLLM 的同一环境中安装 **ExLlamaV3 1.4.8 或更新版本**，其扩展必须匹配当前
PyTorch、CUDA 和 GPU 架构。以下以源码安装为例，替换源码路径和 `CUDA_HOME`：

```bash
uv pip install --python .venv/bin/python setuptools wheel ninja
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST='8.0' MAX_JOBS=8 \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
  /path/to/exllamav3
```

单卡示例，选择能装入显存的 EXL3 checkpoint，并按模型容量调整上下文长度：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve /model/path/exl3-model \
  --served-model-name exl3 \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-seqs 4
```

### 三张 170HX：GLM-5.3-Flash EXL3 与视觉输入

以下以 `GLM-5.3-Flash-exl3/4.05bpw` 为例，启用自适应 prefill：45 层分为
`16/15/14`，每卡预留 **2 GiB KV cache**，开启 1 个 MTP 草稿 token。
模型目录需包含视觉权重；启用 MTP 还需匹配的 `mtp.safetensors`，即使该文件
未列入主权重索引也可自动加载。MTP 位于最后一个流水线阶段，会额外占用显存。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 \
VLLM_PP_LAYER_PARTITION=16,15,14 VLLM_EXL3_MOE_PREFILL=auto \
.venv/bin/vllm serve /model/path/GLM-5.3-Flash-exl3/4.05bpw \
  --served-model-name glm-exl3 \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 3 \
  --max-model-len 66560 --max-num-seqs 4 \
  --kv-cache-memory-bytes 2147483648 --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --mm-processor-kwargs '{"max_image_tokens":1024}' \
  --chat-template-content-format string \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

原始 EXL3 聊天模板会丢弃 OpenAI 格式的图片占位符，因此上例使用
`--chat-template-content-format string`。若已替换为匹配的原版或 AWQ 模板，
可恢复默认 `auto` 格式。纯文本服务可将图片限制改为
`--limit-mm-per-prompt '{"image":0,"video":0}'`；关闭 MTP 则删除
`--speculative-config`。这些 MTP 和视觉选项针对 GLM5Next，不适用于所有 EXL3 模型。

上例使用 `VLLM_EXL3_MOE_PREFILL=auto`，不固定 `--max-num-batched-tokens`，
让调度器按输入长度使用自适应预算。`auto` 会在支持的 SM80 MoE 长 prefill 中
使用 INT8 计算，可能改变输出与精度；需要原生 prefill 时可改为 `native`。
更多支持范围、分层和调优说明见 [EXL3 文档](docs/features/quantization/exl3.md)。

## 三张 170HX：DSV4F 文本服务

以下配置使用 `DeepSeek-V4-Flash-0731`：43 层分为 `15/16/12`，最后一张卡
为草稿模型留出空间；使用 FP8 KV、32K 上下文和 DSpark 7。
直接加载检查点自带的量化权重，无需转换为 EXL3。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_PP_LAYER_PARTITION=15,16,12 \
DSV4_LOGITS_ROW_CHUNK=64 \
.venv/bin/vllm serve /model/path/DeepSeek/DeepSeek-V4-Flash-0731 \
  --served-model-name dsv4f \
  --host 0.0.0.0 --port 8000 \
  --pipeline-parallel-size 3 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --block-size 256 \
  --enable-prefix-caching --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

当前实现支持 DSpark 7，不要求草稿数是检查点草稿层数的整数倍。
关闭草稿推理可删除 `--speculative-config`。该配置的完整 GSM8K、并发和长输入结果见
DSV4F 三卡验证记录（本地归档：`docs/validation/dsv4-perf-optimization-20260908.md`）。

## 三张 170HX：DSV4F-Vision 图文服务

使用独立的 `DeepSeek-V4-Flash-Vision-Exp` 权重目录；给纯文本权重增加图片参数
不能获得视觉能力。下面沿用三卡图文验证配置：`15/16/12` 分层、8K 上下文、
最多 4 个并发序列、每个请求最多 2 张图，开启 DSpark 3。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_PP_LAYER_PARTITION=15,16,12 \
DSV4_LOGITS_ROW_CHUNK=64 \
.venv/bin/vllm serve /model/path/DeepSeek/DeepSeek-V4-Flash-Vision-Exp \
  --served-model-name dsv4f-vision \
  --host 0.0.0.0 --port 8000 \
  --pipeline-parallel-size 3 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --block-size 256 \
  --limit-mm-per-prompt '{"image":2}' \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}'
```

三卡配置通过了单并发和四并发图文检查，各 20/20；涵盖 OCR、多图和图文混排，
详见 Vision 三卡验证记录（本地归档：`docs/validation/deepseek-v4-vision-cmp170hx-20260908.md`）。
需要更长上下文或 DSpark 7 时，可参考
后续启动与长输入验证（本地归档：`docs/validation/dsv4-startup-errors-20260908.md`），
按启动时实际分配到的 KV 容量调整长度与并发。

## 容量与参数调整

- `--max-model-len` 包括提示词、图片 token 和输出；`--max-num-seqs` 是调度并发上限，
  不表示所有请求都能同时填满最大上下文。以启动日志中的 KV 容量为准。
- `--kv-cache-memory-bytes` 是每卡预算；`--kv-cache-tokens` 是 V4.1 混合推理的
  token 预算，不能按字节数照搬。减少 KV 预算或并发后，重新检查可用上下文长度。
- 流水线分层之和必须等于目标模型层数；DSV4F 的 `15,16,12` 不能直接用于
  GLM EXL3 或 V4.1。上述示例把分层变量限定在单条命令中，避免影响其他模型。
- 增加草稿 token 不一定提升吞吐；先使用对应示例，再按实际输入长度和并发测速。
  V4.1 使用 `deepseek_v41` 解析器，V4 Flash 与 Vision 使用 `deepseek_v4`。

通用用法与上游项目介绍见 [vLLM README](README.vllm.md)。
