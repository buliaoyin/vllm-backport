# EXL3 MoE prefill 优化实验（2026-09-10）

最终跨格式性能结论见[固定原始分层的复测](exl3-fixed-pp-comparison-20260910.md)：所有格式统一 PP=11/11/11/12、外层 chunk=512 时，8K B1 TTFT 为 EXL3 3.6831 秒、NVFP4 1.9374 秒、AWQ 1.8174 秒。固定分层和外层分块的 EXL3 代码收益约为 1.69 倍。下文 2.59 秒与约 2.4 倍提升是包含设备放置调整的历史硬件调优结果。

本轮保留两项推理代码改动：扩大并复用专家工作区，按实际 token 数优先运行大专家。配合当前三张 CMP 170HX 与一张 RTX PRO 6000 Blackwell 的层分配调整，GLM-5.3-Flash EXL3 的 8K 单请求首 token 从 6.22 秒降至 2.59 秒，64K 从 44.93 秒降至 18.71 秒。以上是整模型墙钟结果，包含分块和设备放置的共同收益。

量化权重、trellis/codebook、Hadamard scales/signs 和短 decode 计算路径保持原有表示。实验没有采用重新量化、永久展开整个模型或运行时 CPU 读取路由计数。基准中的候选实现与推理默认路径分开保留。

## 运行条件与计时

- 模型：`/home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw`，45 层，前三层 dense、42 层 MoE，288 专家、top-8、hidden=4096、expert intermediate=2048。
- GPU 0–2 为 CMP 170HX / SM80，CUDA 报告每卡 70 个 SM；原专家启动配置每卡使用 8×8=64 个 block。GPU 3 为 RTX PRO 6000 Blackwell / SM120，188 个 SM，原配置为 23×8=184 个 block。
- TP1、EP1、PP4；无 NVLink，`NCCL_P2P_DISABLE=1`，`CUDA_DEVICE_ORDER=PCI_BUS_ID`。BF16 输入，`EXL3_INT8_GEMV=0`，无 MTP、无 prefix cache，graph capture sizes 为 1/2/4/8。
- 软件：Python 3.12、PyTorch 2.13.0+cu130、ExLlamaV3 1.4.8。原版源代码 `/home/bul/dev/exllamav3` 的 revision 为 `6ff3a17`，该源目录与原版 conda 环境保持原状。
- 普通 8K 对照：max model len 16384，KV 4 GiB/GPU；长输入：max model len 66560，KV 8 GiB/GPU。max sequences=4。
- B1 为一个请求；B4 为四个同时提交的请求，报告四者中最大的 TTFT。每种输入/配置先预热一次，再测三次、取中位数；固定生成 32 token，计时请求不包含加载、评测、采样与编译。
- 所有计时请求检查实际输出长度、缓存命中为零，以及停止符前的检索答案。原始 token IDs、全部生成结果与逐次计时保存在 artifacts。
- CMP 不支持本次 CUPTI 采样（error 42），微基准使用 CUDA graph events，整模型分项使用 worker stream 上的 CUDA events。微基准包含专家 wrapper 的 GPU 操作，不包含 Python replay 调度；完整模型测量包含实际 CPU/调度/PP 开销。嵌套分项不可相加，各 rank 时间之和不是请求 TTFT。
- GPU 实验串行执行；独立 CUDA 变体的编译与精度控制实验可重叠，未与性能计时重叠。

## 保留的实现

`vllm/model_executor/layers/quantization/exl3.py` 根据工作区真实容量分块。`VLLM_EXL3_MOE_MAX_TOKENS` 默认为 1024，加载时受 scheduler `max_num_batched_tokens` 限制，必须为正。工作区缓存键包含设备、hidden/intermediate 维度和容量；匹配层共用四个 FP16 缓冲区。因为每个 token 的 top-k 专家 ID 互异，一个专家最多收到整块 token 数，不会超过容量后被上游 kernel 静默跳过。

`VLLM_EXL3_MOE_PRIORITY=1` 为默认值。当块长至少 256 且专家数大于并发组数时，按 count 降序稳定排序，同时重排 ID、count 和九张指针表。大专家先运行可以减轻尾部负载不均。小于等于八个 token 的 decode 路径沿用原实现。shared experts 继续串行，避免争用上游设备级锁与 scratch。

| 容量 | CMP 工作区 / 卡 | Blackwell 工作区 |
| ---: | ---: | ---: |
| 128 | 24 MiB | 69 MiB |
| 512 | 96 MiB | 276 MiB |
| 1024 | 192 MiB | 552 MiB |

这些是同形状层共享的工作区，不是每层都分配一份。外层 scheduler chunk 与专家容量独立配置；只调外层不会自动改善旧的 128-token 内部分组。

## 分块与专家优先级消融

固定原来的 PP=11/11/11/12、外层 chunk=512，只改变内部容量：

| 内部容量 | 8K B1 TTFT（秒） | 短 decode token/s |
| ---: | ---: | ---: |
| 128 | 6.2326 | 39.15 |
| 256 | 4.7960 | 39.04 |
| 512 | 3.9589 | 39.02 |

固定同样 PP、外层 chunk=1024 的消融：

| 候选 | 8K B1 TTFT（秒） |
| --- | ---: |
| 内部 128 | 6.7663 |
| 内部 512 | 4.2640 |
| 内部 1024 | 3.7665 |
| 1024 + 按专家负载排序 | 3.4709 |
| 1024 + Triton 路由与优先级 | 3.4863 |
| 强制融合线性重建 | 3.7833 |
| 线性权重缓存：旋转后的基底 | 3.7286 |
| 线性权重缓存：原始输入基底 | 3.6966 |
| 后者 + shared overlap | 3.6773 |
| Triton 优先级 + 原始基底线性缓存 | 3.4122 |

缓存分支覆盖非 routed-expert 线性层，排除本次 prefill 未使用的 LM head；每种表示约需额外 16 GiB 总显存（精确分项在 `workspace_sweep`）。额外复杂度和显存成本没有得到稳定且足够大的收益，未纳入推理默认路径。shared overlap 只在使用独立 workspace 的 `torch.mm` 缓存实验中启用。

## 当前硬件的层分配与外层 chunk

以下均使用保留的生产路径；PP 数字依次对应 GPU 0、1、2、3。将更多层放到 Blackwell 是当前异构硬件的运行参数，未硬编码到模型中。

| PP 分层 | 外层 / 内部容量 | 8K B1 TTFT（秒） | 8K B4 最大 TTFT（秒） |
| --- | ---: | ---: | ---: |
| 11/11/11/12 | 1024 | 3.4298 | 11.9579 |
| 10/8/8/19 | 1024 | 2.6840 | 9.0497 |
| 10/7/7/21 | 1024 | 2.5889 | 8.6881 |
| 10/7/7/21 | 512 | 2.6581 | 9.8415 |
| 10/7/7/21 | 2048 | 3.0012 | 8.7515 |

在这组测量中，短 decode 从均分的约 39.23 token/s 提高到 10/7/7/21 的约 44.04 token/s。这主要是设备放置收益；专家 prefill 优化没有替换 decode kernel。

继续把缓冲复用、Blackwell 热专家拆分、限制专家组数组合到优化后配置，B1 分别为 2.6329 / 2.5974 / 2.6139 秒，而前后两次基线为 2.5803 / 2.6983 秒。基线本身漂移约 4.6%，这些微小差异不足以支持额外默认分支。

## 长输入结果

“旧接入”为此前 PP=11/11/11/12、外层 512 / 内部 128；“本轮”为 PP=10/7/7/21、外层与内部 1024。旧结果来自同日已归档的长上下文实验，使用相同 token IDs、KV 预算及生成长度。

| 输入 | 旧接入 B1（秒） | 本轮 B1（秒） | 加速 | 旧接入 B4 最大（秒） | 本轮 B4 最大（秒） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 6.2179 | 2.5928 | 2.40× | 23.2671 | 8.6966 |
| 16384 | 11.8047 | 4.7970 | 2.46× | 45.8095 | 17.4134 |
| 32768 | 22.9422 | 9.2482 | 2.48× | 90.4863 | 35.4134 |
| 65536 | 44.9331 | 18.7067 | 2.40× | 178.7392 | 71.5881 |

对应的 B1 prefill 速度为：8K 3162.9、16K 3418.5、32K 3545.8、64K 3505.2 token/s，按 scheduled 到 first-token 的时间计算；TTFT 则还覆盖请求排队等首 token 前的时间。

64K 另测外层与内部 2048：B1 为 17.7036 秒，B4 最大 TTFT 为 68.8975 秒，均为三次中位数。更大 chunk 在长输入上略快、在 8K 上更慢，因此 1024 保留为当前默认容量。

原版 ExLlamaV3 1.4.8 的同日 64K 结果为 B1 78.5058 秒、B4 最大 315.2705 秒，见前一份[长上下文报告](exl3-long-context-20260910.md)。原版使用 11/11/11/12 放置和自己的多卡调度；这是整套运行配置的比较，不能视为相同调度下的纯 kernel 比较。

## 为什么这两项改动有效

8K、1024 chunk 的最终 CUDA-event 采样中，真实路由的 16-row tile 有效行比例为 78.1%–79.1%；将同一组路由按旧 128 切分后只有 34.0%–34.9%。分组减少了不满 16 行的计算和重复访问专家权重。42 层的专家启动次数从 42×64=2688 降为 42×8=336。

| rank | MoE runner 含路由/shared（ms） | routed wrapper（ms） | EXL3 专家 kernel（ms） |
| ---: | ---: | ---: | ---: |
| 0 | 1315.74 | 1271.43 | 1253.61 |
| 1 | 1311.64 | 1267.71 | 1248.93 |
| 2 | 1295.39 | 1251.91 | 1232.75 |
| 3 | 1511.41 | 1452.62 | 1422.00 |

三列为嵌套区间。分层后四卡的 MoE 工作时间接近，但剩余时间仍主要在专家 kernel。路由 Python 包装层无法解释剩余的大部分差距。

## 未采用的候选

- **专家拆分和并发组宽度**：测试全部专家拆分 2/4 份、仅拆分超过 64/128 行的热专家，以及不同 group hint。微基准中局部有收益，但整模型没有稳定超过已优化基线；加宽组还改变归约次序，relative L2 可达约 0.0016。
- **GPU 路由打包**：比较 Torch stable sort、Marlin align 和 Triton 计数/打包。整模型中 Triton 优先级版 3.4863 秒，Torch 版 3.4709 秒，保留较简单的路径。
- **临时缓冲复用**：有界复用 hidden、FP32 result、排序输出、计数和指针表，并保持输出独立所有权。1024 单层 CMP 43.700→43.682 ms、Blackwell 14.154→14.166 ms，整模型也没有稳定收益。
- **GPU 选择热专家后重建**：H=2/4/8，分别用 padded bmm 与 variable-M Triton grouped GEMM；无计时内 CPU count 读回。减少 GEMM padding 后仍更慢。例如 CMP 1024 分组基线 45.69 ms，对应 H2/H4/H8 为 47.62/49.06/52.42 ms；Blackwell 基线 15.46 ms，对应 16.05/16.86/18.00 ms。
- **热点权重跨 chunk 缓存**：缓存两个计时 chunk 中热点集合的并集，预先完整解码；这是缓存全部命中的乐观实验，不能代表未见输入的命中率。variable-M GEMM 直接索引缓存矩阵，避免重复解码和复制整个矩阵。CMP 基线 44.23 ms，H2/H4/H8 为 45.72/47.00/49.50 ms；Blackwell 基线 13.93 ms，对应 14.42/15.01/16.45 ms。缓存占每个所测层约 144–480 MiB，当前 gather/旋转/激活/回写仍有成本。这排除了当前缓存实现的收益，不证明所有缓存算法都无效。

2048 个 token、top-8 的三次专家矩阵乘法对应约 0.825 TFLOP 有效 GEMM 工作；43.78/14.02 ms 对应约 18.8/58.8 TFLOP/s。这里未把 padding、旋转、解码和同步计入 FLOP，不能把该比值当作 Tensor Core 利用率或实测显存带宽。

单层微基准在 GPU0 使用第 3 层真实权重/路由，在 GPU3 使用第 44 层。两者不是相同层，不能用它们的时间比值推导纯硬件性能比。原始路由来自实际完整模型 forward，非均匀随机合成。

## 共享内存、tile 和寄存器实验

进一步编译/测试 K tile=16/32、N tile=128/256、实际共享内存容量、group width=8/16，以及至少两个 block/SM 的 launch bounds，共 32 个微基准配置。

原版固定申请 90 KiB 动态共享内存。按 `exl3_gemm_inner.cuh` 的 A/B 三阶段缓冲与 C 布局，4-bit K32/N256 实际需要 31 KiB，K32/N128 为 17 KiB；K16/N256 和 K16/N128 分别为 23.5 和 12.5 KiB。但原 K32 kernel 为 512 线程、每线程 128 寄存器，CUDA occupancy API 仍报告只能驻留一个 block/SM，单独减共享内存不能解决寄存器限制。

K16 使用 256 线程；通过 `__launch_bounds__(256, 2)` 将寄存器限制在 128，实测可驻留两个 block/SM。候选启动的 grid 严格不超过 occupancy API 报告的容量，并遵守最多 64 个专家组，避免组间 barrier 死锁。实际 SM 数为 CMP 70、Blackwell 188，不能把原版的 64/184 个启动 block 误认为物理 SM 数。

- CMP：原 kernel 43.78 ms；K32/N128 约 64.00 ms；普通 K16/N256 约 57.61 ms；限制寄存器并使用 17 组的 K16/N256 为 42.01 ms，约快 4%。
- Blackwell：原 kernel 14.02 ms；只降到 31 KiB 为 13.61 ms；双驻留 K16/N256、47 组为 17.33 ms，加宽到 16 block/组为 14.56 ms。
- 新 tile 的最大 relative L2 约 0.00119，所有候选均通过阈值 0.01。完整寄存器、spill 与 occupancy 记录已保存。

整模型再次按“原版、31 KiB、CMP K16 + Blackwell 31 KiB、恢复原版”运行；所有配置使用保留的 1024 容量和优先级、PP=10/7/7/21：

| 专家 kernel | 8K B1 TTFT（秒） | 8K B4 最大 TTFT（秒） |
| --- | ---: | ---: |
| 原版，前测 | 2.5865 | 8.6725 |
| 31 KiB | 2.6025 | 8.7202 |
| CMP K16，Blackwell 31 KiB | 2.5919 | 8.6121 |
| 原版，后测 | 2.6974 | 8.9822 |

kernel 小收益没有形成稳定的整模型优势，基线前后有约 4% 波动。它们保留为实验，不进入生产路径。实验 launcher 依赖 ExLlamaV3 1.4.8 的私有 C++ 符号，显式检查版本；K16 模板在 `/tmp` 独立编译、使用独立符号名，未覆盖原版扩展。这也是不把实验 launcher 作为公共依赖的原因。

## 与 NVFP4 / AWQ 的硬件调优对照（最终差距见固定原分层复测）

均为外层 chunk=1024、max model len=16384、KV 4 GiB/GPU、max sequences=4、相同输入 token IDs；每行三次计时。按相同 PP 的两行比较。

| PP 分层 | 格式 | 8K B1 TTFT（秒） | B1 prefill token/s | 8K B4 最大 TTFT（秒） |
| --- | --- | ---: | ---: | ---: |
| 10/7/7/21 | EXL3 | 2.5889 | 3168.0 | 8.6881 |
| 10/7/7/21 | NVFP4 | 1.3669 | 6009.0 | 5.0071 |
| 11/8/8/18 | EXL3 | 2.8114 | 2916.9 | 9.5283 |
| 11/8/8/18 | AWQ | 1.3918 | 5897.2 | 5.1772 |

调优后的同 PP 比较中，EXL3 的 8K B1 TTFT 仍为 NVFP4 的 1.89 倍、AWQ 的 2.02 倍。此前 NVFP4/AWQ 的 1.94/1.82 秒使用不同的 PP/chunk 组合，不能直接与调优后的 EXL3 当作同配置比较。

在 10/7/7/21、1024 chunk 的独立事件采样中，EXL3 专家 kernel 的四卡区间和为 5.0082 GPU 秒，NVFP4 的 Marlin GEMM 为 1.8426 GPU 秒，约 2.72 倍。它们是各 rank 计算区间的和，不能当作请求墙钟。优化后的剩余瓶颈仍主要在专家计算。

AWQ 将 21 层放到 Blackwell 时，在模型构造阶段已用约 94.65 GiB，再申请 4.50 GiB 失败，未进入性能计时。采用 18 层后完成测试，EXL3 同配置补测。AWQ 的最后一层 routed experts 未量化，显存分配不能直接照搬 EXL3。该失败已归档，不将启动失败记录为低吞吐。

两种对照的量化专家都选择 Marlin W4A16；这不是原生 W4A4 FP4 性能对比。三个 checkpoint 的量化覆盖与重建后的权重不同，AWQ 还有一个未量化专家层，注意力/shared 的权重表示也不同。剩余差距包含 trellis 解码、Hadamard、M=16 GEMM 与组同步等实现成本；未做 kernel 内部拆时，不能给这些因素分别编造耗时百分比。量化覆盖详情见[此前定位报告](exl3-prefill-analysis-20260910.md)。

## 正确性与精度

本轮累计 394 个微基准配置通过数值比较；369/369 次计时请求的检索检查通过，输出长度与零缓存命中检查通过。计数包含不同配置及重复测量，不是同等数量的独立任务。相关文件 pre-commit 与 Python 3.12 mypy 通过。

`tests/quantization/test_exl3.py` 在 GPU0 和 GPU3 各通过 37 项。新增边界覆盖 129/512/513 token、工作区跨块、放在最后一个 ID 的热专家，以及改变输入和路由后的 CUDA graph replay；513-token 用例有 32 个专家，确保在 Blackwell 上也会触发优先级调度。

最终 1024 配置的 64 道 GSM8K 为 **63/64**，此前旧接入为 64/64；没有输出长度截断。差异为第 12 题“柠檬树何时开始盈利”，模型在回本的第 12 年与盈利的第 13 年之间给出了不同结论，正确答案为 13。

针对原始题目批次 12–15，分别在两种 PP 放置下测试容量 128/1024、优先级开/关，每组重复两次。**关闭优先级、恢复 128 容量，同一配置的两次结果也分别出现 13 和 12；默认 1024+优先级亦会变化。** 因此单次 63/64 不能归因为某一项优化，也不能据此宣称统计意义上的精度等价。所有控制实验的生成文本、token IDs、结束原因均保留。

## 后续需要更大 kernel 改造的方向

上游 M tile 固定为 16，`exl3_gemm_inner.cuh` 对 M=16 有静态约束，fragment 和 epilogue 布局也按其实现。M=32/64、真正面向大 M 的 grouped EXL3 GEMM、跨 M tile 复用 trellis 解码、进一步融合 Hadamard/gather/scatter，均需要新的 kernel 设计和正确性验证。本轮没有把这些当作只改常量即可获得的已验证收益。

gate/up 的独立量化旋转也使简单拼接权重不等同于直接合并两次乘法。若改为 AWQ/NVFP4，须另行评估重建权重、量化覆盖与精度，不能作为保持 EXL3 权重不变的优化来汇报。

## 复现

当前硬件推荐的 8K–64K 验证配置：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=10,7,7,21 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 VLLM_DETERMINISTIC_MOE_ALIGN=0 \
VLLM_EXL3_MOE_MAX_TOKENS=1024 VLLM_EXL3_MOE_PRIORITY=1 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm --pp 4 \
  --inputs docs/validation/exl3-long-20260910/moe-inputs.json.gz \
  --output /tmp/exl3-optimized-long.json \
  --max-model-len 66560 --kv-cache-gib 8 --batch-size 4 \
  --chunk-size 1024 --warmups 1 --repeats 3 --tokens 32 --skip-eval
```

恢复旧分块行为可设置 `VLLM_EXL3_MOE_MAX_TOKENS=128 VLLM_EXL3_MOE_PRIORITY=0`；恢复完整旧配置还需外层 chunk=512 和 PP=11/11/11/12。

GPU 正确性检查：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXL3_INT8_GEMV=0 \
.venv/bin/python -m pytest tests/quantization/test_exl3.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 EXL3_INT8_GEMV=0 \
.venv/bin/python -m pytest tests/quantization/test_exl3.py -q
```

结构化汇总、逐项结果、运行命令/环境、源文件快照与失败重试日志见 [artifacts 说明](exl3-opt-20260910/README.md) 和 [SHA-256 清单](exl3-opt-20260910/sha256.json)。本轮没有修改原版 ExLlamaV3 源目录，也没有提交或发布 PR。
