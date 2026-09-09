# DeepSeek V4 两处启动错误修复

日期：2026-09-08。本次由 AI 辅助排查、修改和验证，改动保留在本地工作区。

用户提供的两份日志分别在配置校验和 CUDA Graph 捕获阶段退出。两处均已定位到代码：DSpark 误用了普通 MTP 的草稿长度限制；V2 执行器在 `auto` 缩短上下文后，没有同步所有持有长度副本的状态。

## 第一处：7 个草稿 token 被拒绝

日志错误为 `num_speculative_tokens:7 must be divisible by n_predict=3`，发生在权重加载之前。

该检查用于普通 MTP 的模块复用。当前模型的配置包含 `num_nextn_predict_layers=3`，转换草稿配置时得到 `n_predict=3`，但 DSpark 在一次并行前向中生成草稿块，草稿长度不受这 3 层的整除关系约束。模型还配置了 `dspark_block_size=5`；旧检查连 5 也会拒绝。

修复位于 `vllm/config/speculative.py`：仅让 `method="dspark"` 跳过普通 MTP 的整除检查。普通 MTP 和已有模型专属的草稿长度约束继续保留；不修改默认草稿长度。回归验证 DSpark 5、7 和普通 MTP 6 可建立配置，普通 MTP 7 仍被拒绝。

允许 6 或 7 不意味着它们比配置中的 5 更快或质量更高；本次测试只能证明所列输入上的运行结果。

## 第二处：auto 缩容后 C128 元数据断言失败

日志在权重加载成功后将 `max_model_len` 从 **1,048,576 自动缩短至 466,432**，随后在 CUDA Graph 捕获时报错：

```text
assert active_topk_width >= cm.max_seq_len // self.compress_ratio
```

元数据缓冲区使用缩容后的配置建立，但 `DefaultModelState.prepare_attn(for_capture=True)` 仍读取初始化时保存的 1,048,576。C128 捕获要求的压缩列数因此仍是 8192，而按新长度对齐分配的容量只有 3712，触发保护断言。

修复位于 `GPUModelRunner.update_max_model_len()`：更新执行器及请求状态时，同时更新模型状态与草稿执行器缓存的长度；草稿长度上限取原上限和新主模型上限的较小值。原有缓冲区容量断言保留，CUDA Graph 继续启用。

前次优化验证采用固定 8K、`max-num-seqs=4` 和 3 个草稿 token，没有覆盖这次 `auto + max-num-seqs=64 + 6/7 草稿` 的组合。两段有问题的逻辑在本次优化前的基线中也存在。

## 实机验证

使用用户指定的 `vllm-backport` conda 环境，模型为 `/home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp`。模型服务仅使用 GPU 0、1、2，PP3 分层 `15,16,12`，FP8 KV、显存利用率 0.95。保留用户的 `auto`、64 序列、前缀缓存、分块 prefill 和 DSpark 默认采样设置。

| 验证项 | DSpark 6 | DSpark 7 |
| --- | ---: | ---: |
| 自动上下文上限 | 466,432 | 466,432 |
| 三个 PP worker 的 CUDA Graph 捕获及 API 启动 | 通过 | 通过 |
| GSM8K 抽测 | 32/32 | 32/32 |
| 视觉与算术探针 | 20/20 | 20/20 |
| Schema / JSON / 普通请求混跑 | 48/48 | 48/48 |
| 同时提交 64 个短请求 | 64/64 | 64/64 |
| 42,034 token 输入检索 | 通过 | 通过 |

两组运行日志均未出现 C128 容量断言、FSM 错误、CUDA 非法访存或服务 ERROR。

短解码使用中文、代码、英文三道固定题，各输出 256 token，取三轮中位数：

| 配置 | 中文 token/s | 代码 token/s | 英文 token/s | GSM8K 草稿接受率 |
| --- | ---: | ---: | ---: | ---: |
| dspark6 | 58.70 | 103.93 | 71.02 | 49.44% |
| dspark7 | 63.02 | 107.16 | 72.82 | 43.19% |

这些是当前两个配置的测量，不是启动修复前后的性能 A/B；不能据此认定 7 个草稿 token 普遍更优。

GSM8K 本轮为官方测试集前 32 题的启动修复回归，5-shot、temperature=0、seed=42、thinking=false、输出上限 4096、客户端并发 4，**不是再次全量评测**。视觉套件包含 19 个视觉探针和 1 个算术对照，覆盖照片、颜色、OCR、空间位置和多图顺序。混跑包含 16 个严格 Schema、16 个 JSON object 和 16 个普通请求。

64 请求测试是客户端同时提交 64 个短请求，不代表持续 64 路吞吐基准。长输入测试在 42,034 token 中检索中间的指定字符串；自动容量与完成计算图捕获不等同于验证完整容量的长上下文质量，也不代表 64 路均可占满该上下文长度。

## 回归与检查

新增回归在修复前 **4 failed、1 passed**，修复后相关配置、V2 执行器、计算图和 DSpark/Qwen MTP 测试共 **49 passed**。格式与类型调整后配置文件再跑 **12 passed**。本次修改文件的常规 pre-commit 和 Python 3.12 mypy 均通过。

```bash
PYTHONPATH=/home/bul/dev/vllm-backport CUDA_VISIBLE_DEVICES=3 \
/home/bul/miniconda3/envs/vllm-backport/bin/python -m pytest \
  tests/config/test_speculative_draft_hf_overrides.py \
  tests/v1/worker/test_gpu_model_runner_v2.py \
  tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py \
  tests/transformers_utils/test_dspark_mla_config.py \
  tests/transformers_utils/test_speculators_dspark_config.py \
  tests/models/test_qwen3_5_mtp_config.py -q
```

验证启动命令如下，将 6 改成 7 即为另一组。测试使用本机监听地址；实际需要远程访问时可使用原命令的 `0.0.0.0`。

```bash
DSV4_LOGITS_ROW_CHUNK=64 \
VLLM_USE_V2_MODEL_RUNNER=1 \
NCCL_P2P_DISABLE=1 \
VLLM_PP_LAYER_PARTITION=15,16,12 \
CUDA_VISIBLE_DEVICES=0,1,2 \
/home/bul/miniconda3/envs/vllm-backport/bin/vllm serve \
  /home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp \
  --served-model-name local --host 127.0.0.1 --port 8001 \
  --pipeline-parallel-size 3 --max-model-len auto \
  --gpu-memory-utilization 0.95 --trust-remote-code \
  --enable-prefix-caching --enable-chunked-prefill --max-num-seqs 64 \
  --enable-auto-tool-choice --kv-cache-dtype fp8 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":6}'
```

日志中的可选 ROCm/AMD 导入提示以及 `--model` 写法弃用提示不是这两次退出的原因。

原始日志、请求与响应：`/tmp/dsv4-startup-errors-20260908/`。评测复用 `/tmp/dsv4-optimization-20260908/evaluate.py` 及前次文本/视觉探针；额外并发和长输入脚本为本目录的 `extra.py`。精简结果保存在 [dsv4-startup-errors-20260908-results.json](dsv4-startup-errors-20260908-results.json)。
