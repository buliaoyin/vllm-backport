# EXL3 MoE prefill 差异定位（2026-09-10）

主要瓶颈是当前 EXL3 专家路径：同一份 8K 输入、同机 PP4、外层 chunk=512 时，EXL3 首 token 为 6.22 秒，NVFP4 为 1.94 秒，AWQ 为 1.82 秒。EXL3 专家 wrapper 的 98.35% 时间在 fused kernel 内部；实际路由的 16 行 tile 有效行比例约 34%–35%。同权重消融证实，增大内部分组并配套扩大工作区能明显提高专家计算效率。

外层 chunk 与专家内部分组应分别处理：外层 chunk 从 512 提到 8192，使单请求从 16 个可流水执行的批次变成一个。专家 kernel 四卡总间隔仅从 13.91 变为 13.97 秒，重建/GEMM 分项还减少了，但请求墙钟从约 6.34 变为 16.30 秒。这是 PP 重叠减少的证据，不能把该退化归因于大批次线性重建。

## 协议与计时边界

- GPU 0–2：CMP 170HX / SM80；GPU 3：RTX PRO 6000 Blackwell / SM120。PP4 / TP1，层分配 11/11/11/12，无 NVLink，NCCL P2P 禁用。
- 同一架构：45 层，前三层 dense，42 层 MoE；288 routed experts、top-8、hidden 4096、MoE intermediate 2048、一个 shared expert。三个 tokenizer.json 的 SHA-256 一致。
- B1，8192 输入 token，强制生成 32 token；相同 token IDs、相同 BF16 dtype、max_model_len=16384、KV cache 4 GiB/GPU、max_num_seqs=4、无 MTP、无 prefix cache、相同 graph capture sizes 1/2/4/8。每组一次预热、三次计时，报告中位数。
- TTFT 是请求首 token 时间；prefill TPS 使用引擎 scheduled 到 first-token 的时间。加载、JIT 预热、profiler 均不计入结果。所有计时请求检查缓存命中为零、输出长度、停止符前的检索答案。
- Torch profiler 在本机返回 CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)，只有 CPU trace，不能据此计算 GPU 耗时。因此另发一个 1-token 输出请求，以 worker 内的 CUDA event 记录算子边界；原始吞吐计时在安装探针前完成。
- CUDA event 包含该 stream 上边界内的执行和发射间隙。嵌套的 MoE runner、专家 wrapper、内部 kernel 时间不可相加。每个 rank 单列；GPU 时间总和不直接等于 PP 请求墙钟延迟。

## 同配置整模型结果

| 格式 | 外层 chunk | TTFT 中位数（秒） | Prefill token/s | 检索检查 |
| --- | ---: | ---: | ---: | ---: |
| EXL3 | 512 | 6.2159 | 1318.5 | 3/3 |
| EXL3 | 8192 | 16.2489 | 504.3 | 3/3 |
| NVFP4 | 512 | 1.9360 | 4238.6 | 3/3 |
| NVFP4 | 8192 | 3.2318 | 2537.2 | 3/3 |
| AWQ | 512 | 1.8154 | 4520.2 | 3/3 |
| AWQ | 8192 | 3.2101 | 2554.4 | 3/3 |

默认 512 chunk 下，EXL3 的 TTFT 是 NVFP4 的 3.21 倍、AWQ 的 3.42 倍。外层 chunk 提升到 8192，在这组 B1 / PP4 输入上没有改善延迟。

## 专家路径的结构差异

当前主要瓶颈是 EXL3 fused 专家 kernel。512 chunk 的独立 CUDA-event 采样中，expert kernel / routed-wrapper 时间为 13.9071 / 14.1405 秒（四卡各自区间相加），占 98.35%。排序、类型转换和 wrapper 发射合计只占该边界剩余约 1.65%，不能把差距主要归为 Python 调用开销。

当前 `_exl3_moe_fused` 每 128 token 重新执行 dtype 转换、argsort、计数、索引构造、FP32 输出清零、专家 kernel、输出转换。外层调度 chunk=8192 也不会合并这些内部工作；每个 MoE 层仍要运行 64 次专家 kernel，42 层合计 2688 次。

EXL3 fused kernel 以 16 行为 GEMM tile，按专家组协同处理 gate、up、down；还包括 trellis 解码、输入/输出 Hadamard 和组间同步。128 token、top-8、288 专家在均匀路由下平均仅 3.56 行/专家。实际独立实验中 8192 token 的有效 assignment 为 65536，128 分组需 286320 个 tile 行，有效行占 22.89%；512 分组需 93040 行，占 70.44%。专家访问次数同时从 17895 降至 4608。

真实 8K 输入的 512-chunk 采样中，各 rank 的 16-row tile 有效行比例为 34.1%–35.1%；512 个输入 token 内单个专家最多收到 420 行。均匀路由消融的 22.89% 是更稀疏的专家分布，不能将它直接当成真实请求的利用率。

Marlin 对整个调度批次的路由结果分组，根据 M×topk/E 选择 8/16/32/48/64 的 M block；gate/up 合并为 w13，再计算 down。运行日志确认 NVFP4 和 AWQ 的量化专家都选择 Marlin。未设置 VLLM_MARLIN_INPUT_DTYPE；本轮为 W4A16，不是原生 W4A4 FP4 的比较。

PP4 的 engine 还有多个在途 batch 队列。8192 输入在 512 chunk 下形成 16 个调度批次，而 8192 chunk 下只有一个；前者可以把多个 chunk 的不同层阶段重叠。因而增大外层 chunk 并不必然提高 B1 prefill，尤其不能用四卡 GPU 时间之和替代 TTFT。

当前 EXL3 方法还关闭了 shared expert 的多流重叠，因为上游 cooperative GEMM 共用设备级 scratch/锁。该项未单独消融，不能给它分配一个耗时百分比。

128 来自当前工作区分配，并不是 EXL3 格式的硬限制。上游 CUDA 入口从 `temp_state_g.size(1)` 读取容量；超过容量的专家会被跳过。原版外层还有大专家 fallback，当前集成以 128-token 分块确保不超限。只提高 chunk 而不扩大工作区会破坏正确性。

同样使用外层 8192 chunk，四卡 routed-wrapper 区间之和为 EXL3 **14.192 秒**、NVFP4 **1.300 秒**、AWQ **1.257 秒**。这个边界包括各自的分组、专家计算和结果合并，说明差异主要落在专家路径。AWQ 包含的量化专家层为 41 层，其余两者为 42 层；跨 checkpoint 的实际路由也不保证完全相同。

## GPU 分项计时

| 格式 / chunk | rank | MoE runner 含路由/shared（ms） | routed wrapper（ms） | 专家 kernel/GEMM（ms） |
| --- | ---: | ---: | ---: | ---: |
| exl3-c512-events | 0 | 3327.35 | 3257.70 | 3207.71 |
| exl3-c512-events | 1 | 4570.91 | 4467.13 | 4395.44 |
| exl3-c512-events | 2 | 4759.33 | 4654.30 | 4582.11 |
| exl3-c512-events | 3 | 1821.41 | 1761.38 | 1721.88 |
| exl3-c8192-events | 0 | 3285.60 | 3252.84 | 3203.18 |
| exl3-c8192-events | 1 | 4545.97 | 4500.95 | 4432.68 |
| exl3-c8192-events | 2 | 4699.82 | 4655.09 | 4586.10 |
| exl3-c8192-events | 3 | 1802.32 | 1783.52 | 1747.19 |
| nvfp4-c8192 | 0 | 330.72 | 308.88 | 301.27 |
| nvfp4-c8192 | 1 | 457.11 | 427.04 | 416.45 |
| nvfp4-c8192 | 2 | 452.49 | 422.70 | 412.18 |
| nvfp4-c8192 | 3 | 155.84 | 141.27 | 130.95 |
| awq-c8192 | 0 | 325.53 | 303.32 | 295.60 |
| awq-c8192 | 1 | 445.31 | 415.02 | 404.48 |
| awq-c8192 | 2 | 439.73 | 409.69 | 399.18 |
| awq-c8192 | 3 | 158.91 | 129.41 | 119.98 |

以上三列是嵌套边界，不能相加。AWQ 的 routed / Marlin 列覆盖 41 个量化 MoE 层，runner 列还包含末层未量化专家；EXL3/NVFP4 routed 列覆盖 42 层。各组都是单独采样请求，最终首 token 与干净计时的首 token 一致。

`exl3-c512-events` 的 profile 请求墙钟为 6.3438 秒；四卡专家 wrapper 间隔之和 14.1405 秒，内部 expert kernel 间隔之和 13.9071 秒。其余 EXL3 扩展操作总间隔：

- `reconstruct_slice`：277.55 ms。
- `had_r_128`：189.70 ms。
- `hgemm`：698.54 ms。
- `exl3_gemm`：5.27 ms。

`exl3-c8192-events` 的 profile 请求墙钟为 16.3044 秒；四卡专家 wrapper 间隔之和 14.1924 秒，内部 expert kernel 间隔之和 13.9691 秒。其余 EXL3 扩展操作总间隔：

- `reconstruct_had_slice`：30.93 ms。
- `hgemm`：571.36 ms。
- `exl3_gemm`：0.33 ms。

## 消融的约束

固定模型第 3 层的 288 个专家权重，测试 512/8192 个 BF16 输入、top-8。随机种子为 20260910；均匀路由用独立均匀分数选 top-8；热门路由强制每个 token 都选择专家 0，另外七个 ID 仍互异。

分别将内部分组设为 128/512/2048/8192，并让工作区容量覆盖整组 token 数；数据与 workspace 构造在计时外。所有候选先与当前 128 分组实现比较，relative L2 必须小于 0.01，实际误差另外列出。使用 CUDA graph event 计时，15 个样本；CMP 两组每个 graph 包含 10 次调用，Blackwell 两组为 1 次长调用，每次结果均归一化到单次调用。请求 cold-L2；专家权重通过指针表访问，不能误称计时器克隆了底层权重，完整专家权重工作集本身远大于 L2。

这些结果衡量已量化专家算子的同权重分组效果，不包含注意力、shared expert、调度、PP 通信，也不代表整模型会按同样倍数加速。

## 8K 单层同权重消融

| GPU / 路由 | 分组 128（ms） | 分组 512（ms） | 分组 2048（ms） | 分组 8192（ms） | 512 相对 128 | 最大 relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| micro-gpu0-uniform | 604.00 | 202.21 | 158.12 | 142.73 | 2.99× | 4.79e-06 |
| micro-gpu0-hot | 609.58 | 198.12 | 158.25 | 143.42 | 3.08× | 1.50e-05 |
| micro-gpu3-uniform | 190.33 | 61.08 | 44.26 | 39.46 | 3.12× | 5.81e-06 |
| micro-gpu3-hot | 191.13 | 103.25 | 96.73 | 95.46 | 1.85× | 1.40e-05 |

两类 GPU、两种路由、两种输入长度、四种分组，共 32 个候选/基线输出比较均通过。更大工作区允许热门专家接收超过 128 行；极端用例覆盖了单专家 8192 行。

## 量化覆盖范围

通过 safetensors 文件头统计基础模型的 45 层，排除 MTP 和视觉层。EXL3 routed tensors 为 142.165 GiB，NVFP4 为 159.469 GiB，AWQ 为 173.497 GiB；包含对应 scales/signs/zero-points 等元数据。EXL3 的 shared tensors 约 0.740 GiB，而另两种约 1.969 GiB。NVFP4 保留更多 BF16 attention 投影，AWQ 同样保留 shared experts 和部分投影，最后一层 routed experts 也未量化。

运行时方法检查进一步确认：当前 GLM 实现把 AWQ 的压缩 attention 权重在加载时还原为 BF16，线性投影使用 `UnquantizedLinearMethod`；EXL3 则保留量化线性路径。文件头的存储字节数不能直接当作运行时显存或计算精度。

因此三个 checkpoint 只是架构和 tokenizer 一致，量化覆盖与重建后的权重并不相同。跨格式整模型结果不能直接作为纯 GEMM 性能或量化质量等价证明。

## 优化顺序

在本轮 B1/PP4 配置下，优先保留外层 512 chunk，同时将专家工作区容量与内部分组粒度一起调整，复用缓冲区，减少每 128 token 的重分组和小 M GEMM。512 分组对 CMP 只需将这组共享工作区从 24 MiB 增至 96 MiB；Blackwell 从 69 MiB 增至 276 MiB。更大的整组需要更多空间，且极端热门专家在 Blackwell 上仍表现出负载不均，不应只依据均匀路由选最大值。

后续再考虑按专家的 token 行分块、面向大 M 的 grouped GEMM、更好地复用 trellis 解码结果，以及融合转换/排序/回写。trellis 与 Hadamard 的独立比例尚未通过内核内部分项计时测得，不能把残差全部归于量化格式本身。

## 复现与原始数据

AWQ 8192-chunk 运行曾在权重加载阶段出现显存分配重试；最后一次在 18:39:00，加载随后完成，预热、三次测量和事件采样均完成，退出码为 0。计时阶段没有 OOM 重试。这些启动问题保留在日志中，不把加载耗时混入 prefill。

完整命令、环境、token IDs 哈希、源文件快照、结果和日志位于 [artifacts](exl3-prefill-20260910/sha256.json)，结构化汇总见 [summary.json](exl3-prefill-20260910/summary.json)。早期采样分别遇到禁用函数序列化、函数 pickle 失败和 helper 导入路径问题；已完成的干净计时被保留，错误发生在采样/启动阶段，不作为模型正确性失败。最终探针通过字符串 worker RPC 调用，不需要启用 pickle 序列化。

参考实现：本地 `vllm/model_executor/layers/quantization/exl3.py`、`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`，以及原版 `exl3_moe.cu` / `exl3_moe_kernel.cuh` / `block_sparse_mlp.py`。格式背景见 [EXL3 官方说明](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md)。

复现单组模型测量（替换输入文件及 chunk 即可运行另外两种格式）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=11,11,11,12 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 VLLM_DETERMINISTIC_MOE_ALIGN=0 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm --pp 4 \
  --inputs docs/validation/exl3-prefill-20260910/exl3-inputs.json.gz \
  --output /tmp/exl3-prefill-replay.json \
  --max-model-len 16384 --kv-cache-gib 4 --batch-size 4 \
  --chunk-size 512 --warmups 1 --repeats 3 --tokens 32 --skip-eval \
  --profile-dir /tmp/exl3-prefill-replay-profile --profile-tokens 1
```

复现单层消融：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 \
.venv/bin/python benchmarks/kernels/benchmark_exl3_moe.py \
  --checkpoint /home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw \
  --prefix model.language_model.layers.3.mlp.experts \
  --prefill-rows 512 8192 --prefill-chunks 128 512 2048 8192 \
  --expand-workspace --routing uniform --output /tmp/exl3-moe-replay.json
```

将设备改为 3 可验证 Blackwell；将 routing 改为 hot 可验证热门专家。当前脚本使用每个 CUDA graph 一次调用、15 个样本；原始 CMP 的每 graph 十次调用版本也已归档。

检查：相关文件 pre-commit 与 Python 3.12 mypy 通过；模型计时请求和独立算子正确性检查如上。此次诊断仅修改基准、探针与文档，较大的专家分组尚未应用于推理后端。
