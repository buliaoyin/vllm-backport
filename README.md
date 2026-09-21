# vLLM Backport

## 当前分支改动

针对 DeepSeek-V4.1-Flash：

- 支持 CPU/GPU 混合推理、自定义流水线分层，以及 DSpark、并发、视觉输入和工具调用。
- 按 KV token 预算预留显存，自动分配 GPU 专家缓存；支持 prefill/decode 热点更新，记录缓存覆盖率、命中率和专家替换信息。
- 优化 Engram／专家权重加载、CPU 专家算子和 prefill，并限制主机预打包缓存的内存占用。
- 可选 [Engram SSD 存储](docs/features/deepseek_v41_engram_ssd.md)，按需直接读取原权重文件，使用有界缓冲及热点缓存。
- CPU 专家按 NUMA 节点自动分片和绑核，适配单／双路 NPS1、NPS2、NPS4；共享 Engram 内存分散到可用节点，无需新增启动参数。

内存参考（此前三卡实测）：加载后 worker 主存占用约 **344 GiB（PSS）**，建议配置 **512 GiB 主存**。开启 DSpark 时三卡显存采样峰值合计约 **188–189 GiB**；实际占用随分层、KV 预算和并发数变化。

内存不足时，可在下方 `deepseek_v41_hybrid` 中增加 `"engram_storage":"ssd"`。本轮三卡对照中，worker 主存由约 **342 GiB 降至 154 GiB**；默认行缓存总计 **1 GiB**。冷请求的 prefill 会受 SSD 随机读取性能影响，详细限制见上方说明。

编译安装见 [混合推理构建说明](docs/contributing/deepseek_v41_build.md)，支持官方源码安装、wheel 和 CMake 增量流程。

三张 CMP 170HX 推荐启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 vllm serve \
  /model/path/DeepSeek/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 \
  --port 8000 \
  --pipeline-parallel-size 3 \
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
