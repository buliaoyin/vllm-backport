# EXL3 自适应 prefill 正式接入与并发验证（2026-09-12）

## 最终决定

生产后端和自动调度已接入，但 **默认仍为 `VLLM_EXL3_MOE_PREFILL=native`，chunk 2048**。
显式 `auto` 可以按未缓存的请求长度及 decode 状态选择 INT8；无需实验 worker 或 `/tmp` 动态库。

原因分为两部分：吞吐及并发策略通过验证；新增 INT8 算术尚未通过无退化的精度门槛。
同轮三卡、B4、64 道 GSM8K：原生 64/64，强制新 INT8 61/64，新增错题为 9、12、21。
没有截断或解析失败。不能拿旧四卡 B1 的 64/64 覆盖这次结果。

此精度对照把 INT8 行数阈值临时降到 9，以使短题真正经过新算术；它不是默认
4096 门槛的 auto 运行结果，也不能外推为所有长输入必然下降相同比例。
但它足以否决“只要长输入更快，就可以默认切换且宣称精度不变”的结论。
当前提供可显式启用的实现，完整长上下文能力评测及误差补偿仍是提升为默认前的工作。

## 代码与策略

- `_exl3_C` 的正常 CMake target 编入 trellis→INT8 重建、Hadamard gather/activation/scatter。
  Triton grouped IMMA 位于生产模块；不依赖实验 worker、`ctypes` 或外部 checkout。
- 自动选择在 prefix cache 查询后进行。首个可调度请求尚有至少 **32768** 输入 token，
  且当前预算至少 4096 时，使用至多 **6144** 的 chunk；记住选择直到不足 4096 的尾部。
  多个短请求的长度不相加，不因为后面的长请求扩大前面短请求的预算。
- 有运行中 decode 时，每步预算收回至 **2048**。显式用户预算、Mamba 对齐等仍是上限。
  请求结束、抢占、streaming 输入更新清除计划。实际 MoE 行数不足 4096 走原生。
- FULL CUDA Graph 保留原生算术，避免 padding 把小 decode 误判为大 prefill。
  支持当前 stream 和非当前设备输入；共享 workspace 在加载后、KV profiling 前预分配。
- 初期加速范围：SM80、uniform 4-bit mul1、SiLU、top-k ≤8、总专家数 ≤65535，
  维度是 256 的倍数且位于 [256,8192]。
  SM120 和其他不支持的格式走原生。TP=EP=1、禁止 DBO 等原有限制继续生效。

完整策略见 [设计](../design/exl3-adaptive-prefill.md)，用法见
[EXL3 文档](../features/quantization/exl3.md)。

## 测量条件与口径

GLM 路径：`/home/bul/dev/models1/zai/turboderp/GLM-5.3-Flash-exl3/4.05bpw`。
三卡是 GPU0–2 的 CMP 170HX / SM80，固定 PP **16/15/14**；四卡增加 GPU3 的
RTX PRO 6000 Blackwell / SM120，固定 PP **11/11/11/12**。
每 rank KV 2 GiB，max model len 66560，max seqs 4，BF16，prefix cache 关闭，
NCCL P2P 关闭；普通 `EXL3_INT8_GEMV=2`，专家 decode hybrid，没有普通投影 FP16 缓存。
Torch 2.13.0+cu130、CUDA 13.0、ExLlamaV3 1.4.8；日志确认使用 V2 Model Runner。
实际层号及每 rank 状态保存在原始结果。

每个性能请求固定输出 32 token、temperature 0。同组使用相同输入 IDs 和访问码：
B1 输入吞吐 = 输入 token / TTFT；同时提交的 B2/B4 = 总输入 token / 最后一个首 token 延迟。
Decode 吞吐 = (输出 token−1) / (最后 token 时间−首 token 时间)，不含 prefill。
异步错峰测试单列每请求 TTFT 和输出间隔，不误用同时到达批次的分母。

原生主扫描及固定 INT8 各 2 次测量；auto 各 3 次。原生主扫描与固定 INT8 每形状
预热 1 次，auto 预热 2 次，早期预热只输出 16 token；检查 timed MEASURE 段没有
JIT 编译告警。后续原生确认、四卡、dense 改为同样输出 32 token 的预热，各测 3 次。
均报告中位数，不是最优单次。事件 profile 在 clean 性能测量后单独运行。

## 三卡单请求

| 输入 | 原生输入 tok/s | 固定 INT8 输入 tok/s | auto 输入 tok/s | auto / 原生 | auto decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 8K | 2416.5 | 1964.3 | 2374.1 | -1.8% | 38.34 |
| 16K | 2841.0 | 2775.7 | 2798.4 | -1.5% | 38.12 |
| 20K | 2926.0 | 2975.2 | 2889.5 | -1.2% | 38.11 |
| 24K | 2974.8 | 3188.0 | 2956.4 | -0.6% | 38.10 |
| 32K | 3053.7 | 3398.6 | 3366.0 | +10.2% | 38.01 |
| 64K | 3184.3 | 3860.1 | 3883.4 | +22.0% | 37.76 |

8K/16K 固定 INT8 分别回退约 19%/2%；20K 只有约 2% 收益，24K 起收益更明确。
24K 候选在三卡原本达到 3156 tok/s，但四卡约回退 2%，因此最终 auto 统一取 32K 起点。
上表 auto 的 24K/32K/64K 来自最终三次复测，其余行来自策略行为相同的初轮扫描。
以下独立原生确认用于检查热态和顺序漂移：

| 输入 | 原生独立确认 tok/s | auto tok/s | 差异 |
| --- | --- | --- | --- |
| 8K | 2363.2 | 2374.1 | +0.5% |
| 64K | 3213.4 | 3883.4 | +20.8% |

短输入在独立确认中基本持平；64K 的收益仍约 21%。Decode 保持约 38 tok/s。
原生重复测量的差异意味着不应把主扫描中短输入约 1%–2% 的差距直接解释为新策略开销。

## 并发吞吐和首 token

| 同时提交 | 原生输入 tok/s | 固定 INT8 输入 tok/s | auto 输入 tok/s |
| --- | --- | --- | --- |
| 8K + 8K | 2778.3 | 2372.5 | 2785.7 |
| 8K + 8K + 8K + 8K | 3035.4 | 3064.1 | 3032.2 |
| 16K + 16K | 3045.0 | 3058.9 | 3040.4 |
| 16K + 16K + 16K + 16K | 3175.3 | 3652.4 | 3168.9 |
| 32K + 32K | 3182.6 | 3870.6 | 3350.7 |
| 2K + 32K | 2772.4 | 3034.5 | 2773.9 |
| 8K + 32K | 3084.2 | 2907.3 | 3081.8 |
| 8K + 8K + 8K + 32K | 3126.2 | 3526.9 | 3120.0 |

只看总吞吐会遗漏前面请求的延迟。例如 4×8K 固定 INT8 吞吐只改善约 1%，
第一个请求 TTFT 却从约 3.47 秒增至 4.26 秒。auto 各请求 TTFT 保持原生水平。
固定 INT8 在 4×16K、2×32K 的批处理吞吐更高，auto 刻意保留已有 decode 的调度预算。
这是吞吐与延迟之间的取舍，auto 不保证每个并发批次的总吞吐最优。

| 4×8K 模式 | 各请求 TTFT 秒 |
| --- | --- |
| 原生 | 3.474 / 6.459 / 8.883 / 10.795 |
| 固定 INT8 | 4.261 / 7.974 / 9.312 / 10.694 |
| auto | 3.477 / 6.468 / 8.894 / 10.807 |

## 已有 decode 时到达长请求

先提交 2K 请求，观察其输出 4 token 后再提交 32K 请求，两个请求均输出 32 token。

| 模式 | 首请求 TTFT 秒 | 后到长请求 TTFT 秒 | 首请求最大输出间隔秒 |
| --- | --- | --- | --- |
| 原生确认 | 1.652 | 10.854 | 2.842 |
| 固定 INT8 | 1.653 | 9.838 | 6.415 |
| auto | 1.652 | 10.885 | 2.847 |

固定 INT8 最大停顿约 6.4 秒；auto 恢复到原生约 2.84 秒的水平。
这仍不是低延迟交互服务的理想水平，不能说已消除 PP 的 prefill 干扰。
仅看 p95 也会漏掉单次长停顿，因此保存了全部输出到达时间并单列最大间隔。

## 四卡固定分层回归

| 输入 | 原生输入 tok/s | auto 输入 tok/s | 差异 | auto decode tok/s |
| --- | --- | --- | --- | --- |
| 8K | 2842.3 | 2851.1 | +0.3% | 44.98 |
| 16K | 3451.8 | 3461.4 | +0.3% | 44.89 |
| 20K | 3602.6 | 3614.1 | +0.3% | 44.80 |
| 24K | 3696.4 | 3757.6 | +1.7% | 44.90 |
| 32K | 3833.8 | 4009.2 | +4.6% | 44.74 |
| 64K | 4088.6 | 4715.3 | +15.3% | 44.32 |

四卡仍固定 11/11/11/12，没有靠改变层分配放大收益。SM120 不分配新的 INT8 prefill 池。
其原生 workspace 也保留 2048，64K 性能接近此前约 4705 tok/s 的无普通投影缓存实验。
初轮 auto24 在四卡 24K 仅为 3608.6 tok/s，比原生低约 2%，而三卡有约 6% 收益。
最终统一改为 32K 起点，接受放弃三卡 24K 的收益；表中 24K 使用最终边界复测，
其余行的策略没有变化，保留初轮三次样本。

## Dense 回归

Qwen 路径：`/home/bul/dev/models1/Qwen/turboderp/Qwen3.8-27B-exl3/SC_4.00bpw_H5`。
单张 GPU3，max model len 33792，KV 4 GiB；其余普通投影配置相同。

| 输入 token | native 输入 / decode tok/s | auto 输入 / decode tok/s |
| --- | --- | --- |
| 8192 | 4406.3 / 69.10 | 4411.9 / 69.14 |
| 32768 | 3950.6 / 64.46 | 3951.3 / 64.46 |

Dense 两种模式的实际 scheduler capacity 均为 2048，不分配专家 INT8 workspace。
保存并比较 token IDs；具体输出一致性见下表。

| Dense case | 跨配置所有重复 token IDs 一致 |
| --- | --- |
| p8192_b1 | True |
| p32768_b1 | True |

## 精度与内核路径

| 配置 | GSM8K 正确 | 截断 | 新增错题 |
| --- | --- | --- | --- |
| 原生，B4 | 64/64 | 0 | — |
| 强制 INT8，B4，行数门槛 9 | 61/64 | 0 | 9、12、21 |

存档运行共 287 个访问码检查均通过（含重复和被否决的候选），
这只验证指定检索答案，不能替代完整长上下文能力评测。

原生和 INT8 使用同一组 64 题、B4、最多 768 输出 token。错误包括把 400+60 写成 500，
以及把 37−23 写成 16；原始文本保留，不能归因于评测解析。
独立事件诊断确认三卡各 13/15/14 个 MoE 层实际调用了新 INT8 算子。
64K auto 诊断中分别为 143/165/154 次 INT8 MoE 调用；算子区间有嵌套，不把它们相加
当作整请求延迟。

量化测试使用独立旋转权重参考、边界行数 4095/4096/4097/6145、不同维度及真实路由形式，
通过相对 L2 <2% 的误差上限检查。这个数值门槛不等于模型答案质量无损。

为区分移植错误与近似算术影响，另用同一输入比较正式后端与冻结的旧原型：

| 行数 | hidden | 逐元素一致比例 | 相对 L2 | 最大绝对差 |
| --- | --- | --- | --- | --- |
| 97 | 768 | 1.000000 | 0 | 0 |
| 400 | 768 | 1.000000 | 0 | 0 |
| 4096 | 768 | 1.000000 | 0 | 0 |
| 6144 | 768 | 1.000000 | 0 | 0 |

该比较验证列出的内核形状，不足以把三卡 B4 与历史四卡 B1 的所有模型结果差异归结为
单一因素。三卡多了 12 层经过 SM80 INT8 的 MoE（42 对 30），批次和普通投影缓存也不同。
旧结果只作历史记录，本轮精度判断使用本轮的配对对照。

## 显存

6144 容量的 GLM INT8 临时池每 SM80 **3.188 GiB**，原生池另外 0.375 GiB。
临时展开一个专家投影并跨层复用，不永久展开整个模型；容量不随实际 chunk 长度追加多份。

| 三卡 auto rank | 原生池 GiB | INT8 池 GiB | 测量后 allocated GiB | 峰值 allocated GiB |
| --- | --- | --- | --- | --- |
| 0 | 0.375 | 3.188 | 53.164 | 55.804 |
| 1 | 0.375 | 3.188 | 58.106 | 60.745 |
| 2 | 0.375 | 3.188 | 55.547 | 57.999 |

Auto 的 workspace 分配若 OOM 则记录原生回退；显式 `int8` 报错。整个模型和 KV 预算
仍须满足 vLLM 的显存约束，本轮验证的是显式每 rank KV 2 GiB，而非所有自动 KV 配置。
修正 sparse MQA-only MLA 的 profiling，使它不再预留不会执行的 dense context 投影；
有 dense prefill backend 的层仍保留该临时分配。

## 被否决的候选与最终参数

按“所有等待请求剩余工作量合计超过 16K”每步切换的初版被否决。
20K TTFT 为 7.72 秒（原生 7.00），24K 为 8.52 秒（原生 8.26），
64K 为 17.99 秒（固定 INT8 16.98）。中途预算缩小使吞吐变差，且排队短请求会改变
前面请求的策略。原始 `auto-remaining` 结果保留供复核，不能混作最终 auto 成绩。

| 参数 | 最终默认 / 可选行为 |
| --- | --- |
| 普通 `EXL3_INT8_GEMV` | 2 |
| 专家 decode | hybrid：SM80 plain / SM120 residual |
| SM80 原生 M tile | 32 |
| 原生 workspace / 默认 chunk | 2048 |
| 专家 prefill | native；auto、int8 均为显式选项 |
| auto 大 chunk / 起点 | 6144 / 32768 未缓存输入 token |
| INT8 实际行数门槛 | 4096，FULL graph 回退原生 |
| 普通投影 FP16 缓存 | 不启用 |

## 测试与复现

```bash
.venv/bin/python -m pytest tests/quantization/test_exl3.py -q
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  tests/v1/core/test_async_scheduler.py tests/engine/test_arg_utils.py \
  tests/v1/attention/test_mla_prefill_selector.py -q
.venv/bin/python -m pytest tests/engine/test_arg_utils.py -k exl3 -q
```

核心四个测试文件 369 passed；随后默认保留 native 的参数选择 60 passed，
最终 32K 策略及默认参数合计 115 passed。
最终量化全套 106 passed、174 skipped；跳过项为历史 opt-in 实验，不能算作已验证。
pre-commit 对本轮代码及文档通过。分配失败、缓存命中、短请求并发、decode、计划保持、
请求 ID 复用、图重放、非默认 stream、非当前 GPU 均有对应检查。

[tests-exl3-final-default.log.gz](exl3-adaptive-prefill-20260912/tests-exl3-final-default.log.gz)

[tests-final-core.log.gz](exl3-adaptive-prefill-20260912/tests-final-core.log.gz)

[tests-default-native.log.gz](exl3-adaptive-prefill-20260912/tests-default-native.log.gz)

[tests-policy32.log.gz](exl3-adaptive-prefill-20260912/tests-policy32.log.gz)

[port-equivalence.log.gz](exl3-adaptive-prefill-20260912/port-equivalence.log.gz)

组件使用独立增量构建：

```bash
TORCH_CUDA_ARCH_LIST='8.0;12.0' cmake -S csrc/libtorch_stable/quantization/exl3 \
  -B /tmp/exl3-adaptive-20260912/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DVLLM_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build /tmp/exl3-adaptive-20260912/build --target _exl3_C -j3
cmake --install /tmp/exl3-adaptive-20260912/build --component _exl3_C
```

完整命令、环境、输入 hash、每次输入输出、GPU 分层、内存和 profile 在以下原始记录。
执行脚本按原样 gzip 归档，复现时先解压；公用 benchmark 入口仍在 `benchmarks/`。
同轮扫描期间 benchmark 增加了输出时间戳、同长度 warmup，正式模块最后增加了设备 guard；
这些变化不改变 INT8 公式。四卡、dense 和最终内核检查使用最终代码及重编译后的组件。
auto-final/mixed-auto 是 24K 门槛候选，auto32-confirm/mixed-auto32-confirm 是最终门槛复测。
源码与组件最终 hash 见 manifest；基础 revision 不是未提交改动的完整源码版本。

[来源与文件哈希](exl3-adaptive-prefill-20260912/manifest.json)

[汇总数据](exl3-adaptive-prefill-20260912/summary.json)

[三卡原始执行脚本](exl3-adaptive-prefill-20260912/run.py.gz)

[四卡/dense 原始执行脚本](exl3-adaptive-prefill-20260912/regression.py.gz)

- native2048: [结果](exl3-adaptive-prefill-20260912/native2048.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/native2048-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/native2048.log.gz)

- int8-6144: [结果](exl3-adaptive-prefill-20260912/int8-6144.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/int8-6144-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/int8-6144.log.gz)

- auto-remaining: [结果](exl3-adaptive-prefill-20260912/auto-remaining.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/auto-remaining-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/auto-remaining.log.gz)

- auto-final: [结果](exl3-adaptive-prefill-20260912/auto-final.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/auto-final-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/auto-final.log.gz)

- native-confirm: [结果](exl3-adaptive-prefill-20260912/native-confirm.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/native-confirm-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/native-confirm.log.gz)

- int8-accuracy: [结果](exl3-adaptive-prefill-20260912/int8-accuracy.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/int8-accuracy-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/int8-accuracy.log.gz)

- mixed-auto: [结果](exl3-adaptive-prefill-20260912/mixed-auto.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/mixed-auto-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/mixed-auto.log.gz)

- mixed-native: [结果](exl3-adaptive-prefill-20260912/mixed-native.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/mixed-native-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/mixed-native.log.gz)

- dense-native: [结果](exl3-adaptive-prefill-20260912/dense-native.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/dense-native-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/dense-native.log.gz)

- dense-auto: [结果](exl3-adaptive-prefill-20260912/dense-auto.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/dense-auto-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/dense-auto.log.gz)

- auto32-confirm: [结果](exl3-adaptive-prefill-20260912/auto32-confirm.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/auto32-confirm-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/auto32-confirm.log.gz)

- mixed-auto32-confirm: [结果](exl3-adaptive-prefill-20260912/mixed-auto32-confirm.json.gz) · [命令/环境](exl3-adaptive-prefill-20260912/mixed-auto32-confirm-status.json.gz) · [日志](exl3-adaptive-prefill-20260912/mixed-auto32-confirm.log.gz)

性能输入、GSM8K 输入和错峰输入均存为 gzip JSON。未新增 AWQ/NVFP4 测量，
也没有把旧不同 chunk 或分层的结果拼作本轮对照。完整长上下文推理质量、多机、
TP/EP、speculative decode、多模态及 MRV1 的整模型回归不在本次已验证范围。
