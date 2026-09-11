# EXL3 与 NVFP4/AWQ：固定原始 layer 分配的最终对照（2026-09-10）

模型为 GLM-5.3-Flash。全部格式重新使用最初的 **11/11/11/12** 分层。在原有 8K B1、外层 chunk=512 条件下，优化后 EXL3 的 TTFT 为 **3.6831 秒**，NVFP4 为 **1.9374 秒**，AWQ 为 **1.8174 秒**。EXL3 的延迟分别为后两者的 **1.90 倍 / 2.03 倍**。

此前 2.59 秒与约 2.4 倍提升包含 PP=10/7/7/21 的设备放置收益；该实验保留为硬件调优记录。与原 NVFP4/AWQ 的最终比较以本报告固定原始分层的结果为准。

## 分层与公共条件

基准脚本在计时前读取各 worker 的实际 decoder 模块，排除 `PPMissingLayer`，并对索引逐一断言。所有九次模型启动均通过：

| GPU / PP rank | 硬件 | 实际 decoder 索引（从零开始） | 层数 |
| ---: | --- | --- | ---: |
| 0 | CMP 170HX | 0–10 | 11 |
| 1 | CMP 170HX | 11–21 | 11 |
| 2 | CMP 170HX | 22–32 | 11 |
| 3 | RTX PRO 6000 Blackwell | 33–44 | 12 |

- TP1/EP1/PP4，BF16 输入，CUDA device order=PCI_BUS_ID；NCCL P2P 禁用，OMP threads=1，`EXL3_INT8_GEMV=0`，`VLLM_DETERMINISTIC_MOE_ALIGN=0`。GPU 实验串行执行。
- 模型、tokenizer 与 prompt token IDs 沿用原实验；三格式输入与 EOS 相同。8K 输入 token IDs 的原始 SHA-256 为 `0157c84c2a2d110089d9642fe2c2eedd42d1458a97b8ccdfdff1b0b8bf4762e0`。
- 实际并发为 **B1**；`max_num_seqs=4` 是容量上限。无 MTP、无 prefix cache，CUDA graphs 开启，capture sizes=1/2/4/8。
- 每种配置/长度预热一次，再测三次，固定输出 32 token，报告逐次指标中位数。加载、初始化、编译和独立事件采样均不进入性能请求。
- TTFT 为请求到首 token 的时间；prefill token/s 为输入 token 数除以 scheduled 到首 token 的时间。两者计时边界略有不同。
- 软件为 PyTorch 2.13.0+cu130、ExLlamaV3 1.4.8、NVIDIA 驱动 610.57.04；三张 CMP 的物理 SM 数为 70，Blackwell 为 188。未锁定 GPU 时钟。

## 8K：复用原协议，并统一比较外层 chunk

两列配置均为 max model len=16384、KV=4 GiB/GPU。512 是原 NVFP4/AWQ 对比采用的外层分块；1024 是三格式共同改变外层分块的独立对照。两组分层始终不变。

| 格式 | chunk 512 TTFT（秒） | chunk 512 prefill token/s | chunk 1024 TTFT（秒） | chunk 1024 prefill token/s |
| --- | ---: | ---: | ---: | ---: |
| EXL3 | 3.6831 | 2225.6 | 3.4221 | 2395.8 |
| NVFP4 | 1.9374 | 4235.4 | 1.7408 | 4715.1 |
| AWQ | 1.8174 | 4514.4 | 1.6804 | 4883.5 |

外层 1024 时，EXL3 的 TTFT 为 NVFP4 的 1.97 倍、AWQ 的 2.04 倍。不能把 EXL3 的 1024 分块行与另两者的 512 分块行混算为同配置差距。

此前原协议 NVFP4 / AWQ 的 TTFT 为 1.9360 / 1.8154 秒；本次重新测量为 1.9374 / 1.8174 秒。

## EXL3 代码收益：固定分层和 512 外层分块

同一已加载模型依次运行生产配置、恢复旧参数、恢复生产配置；每组均预热一次、测三次。三组使用同一个原生 fused 函数，没有启用微基准候选 kernel。

| 配置 | 内部容量 / 专家优先级 | 8K TTFT（秒） |
| --- | --- | ---: |
| 生产配置，前测 | 512 / 开 | 3.6831 |
| 旧参数复现 | 128 / 关 | 6.2365 |
| 生产配置，后测 | 512 / 开 | 3.7132 |

扩大工作区和按负载安排专家带来 **1.68–1.69 倍**加速，前测 TTFT 降低 **40.9%**。生产配置前后漂移为 0.82%。旧参数复现与此前旧接入的 6.2159 秒相近。该控制测量恢复旧分块/优先级设置，并非切换整个仓库到旧 revision。

启动环境的容量上限为 1024，实际容量取 scheduler 上限，因此 512 外层对应内部容量 512；每张 CMP 工作区 96 MiB，Blackwell 276 MiB。外层 1024 对应内部容量 1024，分别为 192 / 552 MiB；同形状层共用工作区。

## 长输入：8K 到 64K，三格式固定同一分层

全部使用外层/EXL3 内部容量=1024、max model len=66560、KV=8 GiB/GPU、B1，其余条件相同。长输入表的 8K 行也在此 KV 预算下重新测量，不与上表的 4 GiB 预算混用。

| 输入 token | EXL3：TTFT 秒 / prefill token/s | NVFP4：TTFT 秒 / prefill token/s | AWQ：TTFT 秒 / prefill token/s |
| ---: | ---: | ---: | ---: |
| 8192 | 3.4277 / 2391.9 | 1.7541 / 4676.9 | 1.6830 / 4877.2 |
| 16384 | 6.2188 / 2636.4 | 3.1921 / 5139.4 | 3.0599 / 5362.2 |
| 32768 | 11.8868 / 2758.3 | 6.1370 / 5345.3 | 5.8938 / 5567.4 |
| 65536 | 23.4243 / 2799.2 | 12.1481 / 5400.3 | 11.7093 / 5602.6 |

64K 时，EXL3 的 TTFT 为 NVFP4 的 **1.93 倍**、AWQ 的 **2.00 倍**。本轮长输入只比较 B1；此前 B4 的硬件调优结果不属于这个固定分层对照。

此前同样 PP=11/11/11/12 的旧接入 64K 为 44.9331 秒，本次为 23.4243 秒，约 1.92 倍加速。该数值包含外层 chunk 从 512 改到 1024 的收益；固定 512 外层的代码收益仍以上面的 1.68–1.69 倍控制实验为准。此前 PP=10/7/7/21 的 18.7067 秒属于另一个设备放置配置。

## 剩余耗时位于哪里

8K、512 外层的独立 CUDA-event 请求，以下为每个 rank 的专家算子区间总和（ms）。EXL3 采样在恢复生产配置后进行；所有事件 instrumentation 都在干净性能测量结束后安装。

| PP rank | EXL3 专家 kernel | NVFP4 Marlin GEMM | AWQ Marlin GEMM |
| ---: | ---: | ---: | ---: |
| 0 | 1635.0 | 672.2 | 589.6 |
| 1 | 2329.6 | 963.3 | 851.0 |
| 2 | 2351.0 | 978.3 | 836.2 |
| 3 | 827.4 | 411.1 | 358.3 |

固定原分层后，CMP 上的 EXL3 专家计算仍明显更耗时。NVFP4/AWQ 的量化专家采用 Marlin W4A16，表格不代表原生 W4A4 FP4 性能。EXL3 含 42 个量化 MoE 层，AWQ 的 Marlin 列只覆盖 41 个量化层，末层未量化专家不在该列中。跨 checkpoint 的路由、量化覆盖与重建权重有差异。

这些 GPU 区间不能跨 rank 相加后当作墙钟 TTFT，嵌套 wrapper/kernel 区间也不能相加。剩余成本包括 trellis 解码、Hadamard、固定 M tile、专家调度与同步；本轮没有 kernel 内部拆时，不为这些因素分配未经测量的百分比。

## 验证与复现

60/60 次计时请求均通过：输入长度正确、输出恰好 32 token、缓存命中为零、停止符之前的检索答案正确。这个计数包含重复测量，不代表 60 道独立评测题。三种格式每次启动均验证实际 layer 索引，九次运行退出码均为零。加载阶段的显存分配重试保留在日志中；校验确认计时阶段没有此类重试。

本轮只为 benchmark 增加了实际分层校验，没有修改推理实现；`exl3.py` 与 `envs.py` 的内容与前一轮最终快照一致。基准文件 pre-commit 和 Python 3.12 mypy 通过。推理数值/模型评测沿用[优化报告](exl3-optimization-20260910.md)中的记录及其 GSM8K 波动限制，本轮没有新增精度等价结论。

复现原协议的单个格式，以 EXL3 为例：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=11,11,11,12 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 VLLM_DETERMINISTIC_MOE_ALIGN=0 \
VLLM_EXL3_MOE_MAX_TOKENS=1024 VLLM_EXL3_MOE_PRIORITY=1 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm --pp 4 --expected-layer-counts 11 11 11 12 \
  --inputs docs/validation/exl3-fixed-pp-20260910/exl3-inputs.json.gz \
  --output /tmp/exl3-fixed-pp.json \
  --max-model-len 16384 --kv-cache-gib 4 --batch-size 4 \
  --chunk-size 512 --warmups 1 --repeats 3 --tokens 32 --skip-eval
```

替换 inputs 为 `nvfp4-inputs.json.gz` 或 `awq-inputs.json.gz` 即可运行另两种格式。长输入使用相应 `*-long-b1-inputs.json.gz`，并改为 max model len=66560、KV=8、chunk=1024。全部命令、逐次结果、源码快照、运行日志与 SHA-256 清单见 [artifacts](exl3-fixed-pp-20260910/README.md)。
