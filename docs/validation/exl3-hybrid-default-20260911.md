# EXL3 混合专家 decode 默认路径验证（2026-09-11）

混合专家 decode 已纳入正式 `_exl3_C` 组件，`VLLM_EXL3_MOE_DECODE` 默认值为 `hybrid`。SM80 使用普通 INT8 DP4A，SM120 使用带激活残差补偿的 INT8 DP4A。该策略适用于 BF16、1–8 行、每行选择 1–8 个专家、uniform 4-bit mul1、输入/中间维度为 256 的倍数且在 [256, 8192]；其他配置回退上游专家路径。可在启动前设 `VLLM_EXL3_MOE_DECODE=native` 关闭专家 INT8。

本轮只改变专家 decode。两组普通线性层均使用上游默认 `EXL3_INT8_GEMV=2`，并保持安全 M32、2048 chunk/workspace、优先调度一致。因此本报告中的收益与[普通 GEMV 0/2 对照](exl3-int8-combination-20260911.md)是不同实验。

## 实现与回退

- 生产路径直接调用 PyTorch stable API 注册的算子，不需要实验 worker 或临时 `decode.so`。M32 与专家 decode 共用可单独构建的 `_exl3_C` 组件。
- 保留已测原型的计算指令、投影精度与逐架构 grid。gate/up 为 FP16，down 中间结果为 FP32，路由加权后输出 BF16。
- 固定 scratch 的 assignment stride，并取两个投影方向和各 batch grid 的最大需求，防止 CUDA Graph 重放或切换 batch 时旧 partial sums 覆盖同步计数器。不同层共享工作区，沿用单流执行限制。
- 对本次 GLM，专家 decode scratch 为 SM80 每卡 5.25 MiB、SM120 9.25 MiB。主 prefill workspace 仍为 384/384/384/1104 MiB，SM80 M32 锁缓冲另约 4 MiB/卡。
- `native`、`plain`、`residual` 均可覆盖默认专家策略；普通 `EXL3_INT8_GEMV` 与专家策略独立。要验证不含激活 INT8 的路径，需同时设置 `EXL3_INT8_GEMV=0 VLLM_EXL3_MOE_DECODE=native`。

## 实验设置

GLM-5.3-Flash checkpoint 为 `/home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw`。GPU0–2 为 CMP 170HX（SM80），GPU3 为 RTX PRO 6000 Blackwell（SM120）；Torch 2.13.0+cu130、CUDA 13.0、ExLlamaV3 1.4.8。

固定 PP `11/11/11/12`，逐进程检查实际层索引为 0–10、11–21、22–32、33–44。KV 每 rank 8 GiB，max_model_len=66560，max_num_seqs=4，prefix caching 关闭，greedy。所有请求的缓存命中为 0。

按 `native 前测 → 默认 hybrid → native 后测` 顺序串行启动三个独立进程。前两组每项 1 次预热、3 次测量，后测每项 1 次预热、2 次测量。控制值为前后共 5 个样本中位数，候选为 3 个样本中位数；各进程和原始样本均保留。性能计时没有并行 GPU 任务或 CUDA 编译。

模型运行使用普通 profile worker，仅采集已有状态，不安装实验 kernel。`--chunk-size` 未传值，实际解析为 2048；M tile、工作区上限、优先调度和普通 GEMV 环境变量均未设置。hybrid 组也不设置专家策略变量，直接验证默认值。加载后及全部请求结束后都检查了实际分层、M tile=32/32/32/16、capacity=2048、priority=true、专家模式 plain/plain/plain/residual；控制组模式均为 native。

## 吞吐

B1 输出 TPS = `(输出 token 数−1)/(最后 token 时间−首 token 时间)`。输入 TPS = `输入 token 数/TTFT`，包含调度和首 token 开销。2048-token B1 生成 256 token，8K/64K 生成 32 token。

| 输入 token | 输入 native | 输入 hybrid | 输出 native | 输出 hybrid | 输出增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1415.86 | 1410.23 | 39.96 | 45.03 | +12.69% |
| 8192 | 2891.47 | 2883.99 | 40.18 | 45.26 | +12.65% |
| 65536 | 4086.38 | 4063.81 | 39.42 | 44.35 | +12.52% |

B4 每请求输入 2048 token、输出 128 token，下表是完整 generate 的输出吞吐，包含 prefill。PP 调度及请求重叠影响该值，不能称为固定 M=4 的纯 decode。

| 设置 | 中位数 [最小, 最大]，token/s |
| --- | ---: |
| native | 32.98 [32.96, 32.99] |
| hybrid | 36.30 [36.30, 36.32] |

中位数变化 +10.09%。

前测→后测的 native B1 decode 中位数为 2048 token：39.96→39.95；8192 token：40.19→40.16；65536 token：39.43→39.41 token/s。hybrid 相对前、后两个独立控制进程均提升，范围为 12.48%–12.72%。这仍是本机少量重复测量，不代表其他硬件或并发的保证值。

## 输出质量

同一组 64 道 GSM8K，按 B1 逐题生成，每题最多 768 token。保留完整回答、token IDs、抽取答案、停止原因和错题。

| 专家策略 | 正确数 | 错题 ID | 输出截断 ID |
| --- | ---: | --- | --- |
| native | 63/64 | [9] | [] |
| hybrid | 64/64 | [] | [] |

hybrid 新增错题 []，修正题 [9]。最终抽取答案相同 63/64，完整 token 序列完全相同 21/64。

8K/64K 固定访问码检索合计 **16/16**，其中默认 hybrid 为 **6/6**。

64 题和指定访问码只能排查本次样本中的明显回归，不能证明精度无损、总体准确率提升或完整长上下文能力。INT8 会改变输出路径，生成过程也存在运行非确定性。本轮没有运行完整 GSM8K、困惑度或其他学科评测。

## 内核与工程验证

- SM80 `tests/quantization/test_exl3.py`：125 passed。
- SM120 同一套测试：85 passed、40 skipped（不适用于该架构的实验）。
- 默认专家路径覆盖不同 batch、top-k=2/8、非对称投影 256→512→256、零输入、零路由权重、修改输入/路由后的 CUDA Graph replay、切换 batch 共用 scratch、显式覆盖和不支持输入回退。相对独立 FP32 旋转权重参考的误差阈值为 plain 2%、residual 1%。
- SM120 Compute Sanitizer：memcheck 28 passed、0 errors；racecheck 28 passed、0 errors / 0 warnings。CMP 170HX 的 sanitizer 支持限制见前次报告，未把 SM120 检查当作 SM80 的直接内存检查。
- 两个架构共 10 个专家 GEMV/reduction 内核的反汇编计算指令与已验证原型逐条相同。源文件、构建命令、二进制哈希和 SASS 对照均保存。
- 完整相关改动通过仓库 pre-commit；Python 3.12 mypy 通过。规范检查修正了一个局部缓存键变量的类型推断，以及 vendored 头文件格式。格式化后重新构建，全部 CUDA 内核计算指令与计时二进制一致；重建组件在 SM80/SM120 上各通过 28 项默认 decode 测试。

## 最终默认参数

| 项目 | 默认值 | 范围 |
| --- | --- | --- |
| 普通线性层 GEMV | `EXL3_INT8_GEMV=2` | 上游默认，独立于专家策略 |
| 专家 decode | `VLLM_EXL3_MOE_DECODE=hybrid` | SM80 plain / SM120 residual；支持条件外回退 |
| 专家 prefill | `VLLM_EXL3_MOE_M_TILE=32` | 已支持的 SM80 使用安全 M32、FP32 累加，其他配置走上游 |
| 调度 chunk | 未显式指定时 `2048` | EXL3 默认；保留显式用户设置及既有调度模式调整 |
| 专家 workspace 上限 | `VLLM_EXL3_MOE_MAX_TOKENS=2048` | 同时受调度 token budget 限制 |
| 专家优先调度 | `VLLM_EXL3_MOE_PRIORITY=1` | 大 prefill 批次优先计算热点专家 |
| INT8 Tensor Core prefill | 不启用 | 仍是独立实验，本轮专家 decode 用 DP4A |

本机固定分层下，默认开启混合专家 decode 带来约 12.5%–12.7% 的 B1 输出吞吐提升，输入吞吐基本不变。结合上述工程检查与有限模型评测，采用 hybrid 作为本次已支持配置的默认策略，并保留 native 回退。Dense 不使用路由专家，本轮变更不影响其路径；本轮未重复测试 AWQ/NVFP4 或原版 ExLlamaV3。

## 原始证据

[机器可读汇总](exl3-hybrid-default-20260911/summary.json.gz)、[执行配置与时间](exl3-hybrid-default-20260911/model-status.json.gz)、[输入数据](exl3-hybrid-default-20260911/inputs.json.gz)、[基线前测](exl3-hybrid-default-20260911/native-before.json.gz)、[默认 hybrid](exl3-hybrid-default-20260911/hybrid-default.json.gz)、[基线后测](exl3-hybrid-default-20260911/native-after.json.gz)。

同目录保存测试/构建日志、SASS 对照、源码指纹及统计脚本。文件 SHA256 清单见 [manifest](exl3-hybrid-default-20260911/manifest.json)。
