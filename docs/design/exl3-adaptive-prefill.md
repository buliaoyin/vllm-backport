# EXL3 自适应 MoE prefill

2026-09-12：INT8 专家 prefill 已接入正常组件构建及量化后端，可显式启用。
三卡并发 GSM8K 对照出现 64/64 → 61/64 的退化，因此最终默认保留原生 prefill，
不因吞吐改善而自动启用新增量化。最终测量见
[验证记录](../validation/exl3-adaptive-prefill-20260912.md)。

## 为什么不直接用 16K 或 batch 总长度

同一 PP `16/15/14`、三张 CMP 170HX 的固定策略对照表明：B1 的 8K / 16K
不适合固定 INT8，20K 的收益约 2%，24K / 32K / 64K 则有明确收益。但四卡固定分层下 24K 出现约 2% 回退，
因此最终统一取 32K 为自动起点，接受三卡在 24K 放弃部分收益。
4 个 8K 请求合计 32K，固定 INT8 吞吐只改善约 1%，第一个请求 TTFT 却增加约 23%。
8K + 32K 混跑也会回退。批次总长度不能替代对单个请求和已有 decode 的考虑。

按剩余总工作量每步切换的原型同样被否决：20K TTFT 从原生 7.00 秒退化到
7.72 秒，24K 从 8.26 秒退化到 8.52 秒。大 chunk 计划需要保持到尾部。
这些测量只用于选择候选策略，不能当作最终 auto 的性能。

## 自动策略

`VLLM_EXL3_MOE_PREFILL=native` 是默认值。显式设置 `auto` 后，策略在 scheduler 中运行：

1. 在等待队列的缓存查询之后，使用 `num_prompt_tokens - num_computed_tokens`
   得到尚需计算的输入量；运行中的请求使用其已推进的计算位置。
2. 第一个可调度 prefill 尚有至少 32768 token、且本步预算允许至少 4096 行时，
   采用最多 6144 token 的 chunk，并记住该请求的计划。其他 prefill 使用最多 2048。
3. 已选择大 chunk 的请求继续该计划，直到剩余量小于 4096。不会在剩余量经过
   32K、24K 或 16K 时反复切换。
4. 存在运行中的 decode 时，本步使用最多 2048 token，限制长 prefill 对输出间隔的影响。
5. 用户的 token budget、`long_prefill_token_threshold`、Mamba 对齐及其他调度限制
   仍是上限。请求结束、抢占及 streaming 输入更新会清除旧计划。

因此多个短请求不会因合计超过阈值而启用大 chunk，短请求在前也不会因后面的长请求
扩大本步预算。该策略优先保留短请求及 decode 的延迟；大规模批处理可显式选择 `int8`
模式，使用固定调度预算，接受其延迟取舍。

32768 是兼顾本次两种拓扑的保守起点，不是所有模型、PP 拓扑上的理论交点。
配置为较短 `max_model_len` 的 auto 引擎不预留不可达的 INT8 路径。

## 执行与图

MoE 算子仍按本次实际 GEMM 行数选择实现：达到 `VLLM_EXL3_MOE_INT8_MIN_TOKENS`
（默认 4096）且有可用 workspace 时使用 INT8，否则使用原生。INT8 最大处理 6144 行，
更大的显式批次分块，小尾块回退。原生 SM80 prefill 保持 M32，INT8 grouped GEMM
使用 M64 / N128 / K64。普通 Linear 与专家 decode 的策略独立。

整个 batch 共享一次专家权重重建；无需按请求切开、复制和重新合并激活，也无需把
请求分类逐层传入 model runner。调度结果在 PP 各 rank 上保持一致，MRV1 / MRV2
复用现有输入路径，本轮整模型实测使用 MRV2。FULL CUDA Graph 使用原生算术，避免大 padding 使小 decode
误用 INT8。eager / piecewise 的算子选择只依赖形状和启动时确定的参数。
不在请求执行时修改环境变量，也不为选择后端执行 GPU 到 CPU 同步。

## 算术与能力范围

重建、Hadamard、激活与 scatter 固化在 `_exl3_C`；接口使用 PyTorch stable API，
检查 device、dtype、layout、尺寸和容量，并使用调用方的 CUDA stream。
Triton row quantization 与 grouped INT8 Tensor Core GEMM 位于生产模块；调用时
显式选择输入设备并恢复调用方设备。正常运行不导入 `benchmarks`，不依赖 `ctypes`、
外部 ExLlamaV3 checkout 或 `/tmp/helpers.so`。

首期加速范围为 SM80、uniform 4-bit mul1、SiLU、top-k ≤8、总专家数 ≤65535，
维度为 256 的倍数且位于
[256, 8192] 的 MoE。其他设备、格式使用原生专家实现，SM120 保留已有混合 decode。
既有 TP=EP=1、共享单 stream、禁止 DBO 等约束继续适用。

INT8 会量化激活及重建后的码本值，并非无损优化。数值参考、模型能力及真实长输入
验证必须分别记录，不能用 auto 下走原生的短 GSM8K 题代替 INT8 精度验证。

## 显存与配置

临时专家权重按 device / expert count / dimensions / capacity / top-k 共享，
在加载结束、KV cache 预算确定前预分配。容量固定，不为 4096、6144 等实际长度
分配多份池。GLM 的 6144 容量每张 SM80 约 3.188 GiB，另有原生 workspace；
没有普通投影 FP16 缓存或全模型永久展开。

Auto 遇到 workspace 分配 OOM 时记录原生回退；显式 `int8` 模式会报告该 OOM。
这不替代 vLLM 对整个模型、KV cache 和用户显存预算的检查。三卡实测使用每 rank
2 GiB KV cache，并记录实际峰值。Sparse MQA-only MLA 不再为不可执行的 dense
prefill 投影预留临时空间；有 dense prefill backend 的层仍保留该 profiling 分配。

可用控制项：

| 设置 | 含义 |
| --- | --- |
| `VLLM_EXL3_MOE_PREFILL=auto` | 按未缓存请求工作量及 decode 状态选择 chunk |
| `VLLM_EXL3_MOE_PREFILL=native` | 默认；禁用新 INT8 prefill 及其 workspace |
| `VLLM_EXL3_MOE_PREFILL=int8` | 固定预算下按行数使用 INT8，用于吞吐场景及对照 |
| `VLLM_EXL3_MOE_INT8_MIN_TOKENS=4096` | 内核行数门槛；精度评估可显式降至 9 |
| `VLLM_EXL3_MOE_MAX_TOKENS=2048` | 原生 workspace / 内部分块容量 |
| `VLLM_EXL3_MOE_M_TILE=32` | 原生 SM80 内核 tile |
| `VLLM_EXL3_MOE_DECODE=hybrid` | SM80 plain / SM120 residual 专家 decode |
| `EXL3_INT8_GEMV=2` | 上游普通投影 GEMV 设置 |

设置在进程启动时确定。`int8` 不强制不支持的 rank 使用 INT8，也不会将小尾块强行补齐
到 4096。PP 分层是部署配置，三卡对照为 `16/15/14`，四卡为 `11/11/11/12`。

## 验证

扩展现有量化、scheduler、arg-utils、MLA 测试，覆盖阈值、缓存命中、并发短输入、
持续 chunk 计划、decode、请求 ID 复用、数值参考、容量边界、图重放、非默认 stream
和分配失败。完整模型验证保存输入 IDs、输出、每请求 TTFT / decode、逐次输出到达时间、
分层、workspace 与峰值。测试、性能、质量结果及限制统一记录在验证报告中。
