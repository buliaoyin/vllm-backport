# 固定原始 PP 层分配的验证数据

对应[报告](../exl3-fixed-pp-comparison-20260910.md)。全部运行使用 GPU0/1/2/3 的 11/11/11/12 decoder 层，worker 实际索引在每份结果的 `runtime_before[].decoder_layers` 中，开始计时前已断言。

- `protocol.json`：8K 原协议的 512 分块主对照及统一 1024 分块对照，含输入/tokenizer 哈希和源码快照哈希。
- `long-protocol.json`：8K、16K、32K、64K 的 B1 长输入协议，三格式相同输入、分层、KV 和分块。
- `summary.json`：逐组中位数和每次 TTFT。`prefill_tokens_per_s` 使用 scheduled 到首 token 的时间，`ttft_s` 使用请求到首 token 的时间。
- `validation.json`：实际层索引、源文件不变、公共参数和计时阶段无显存分配重试的交叉校验。
- `benchmark-placement-check.patch`：本轮相对于上一轮基准文件的独立补丁；`placement-argument-checks.json` 为参数拒绝检查。
- `status.json` / `long-status.json`：完整运行命令、环境变量、起止时间、退出码和后置检查结果。加载不进入请求计时。
- `*-pp11-11-11-12-c*.json`：每次生成的 token IDs、文本、答案检查与指标；8K 普通配置还有计时完成后的独立 CUDA-event 采样。
- `exl3-c512-variants.json`：原生生产配置、旧参数对照、恢复生产配置。主表使用 `production-before`，采样使用 `production-after`；配置切换没有替换原生 fused 函数。
- `*-inputs.json.gz`：实际使用的输入。长输入仅保留 B1，用例来自上一轮长上下文验证；全部格式 token IDs 与 EOS 一致。
- `sources/`：运行时源码快照。`tracked-diff.patch.gz` 是运行结束时整个工作区已有的 tracked diff，包含此前工作，不能当作本轮专属补丁。
- `run_fixed_pp.py.gz` / `run_fixed_pp_long.py.gz`：本机驱动脚本；先运行普通配置，再运行长输入。脚本中的工作区和 `/tmp` 路径可按需调整。
- `summarize.py.gz` / `archive.py.gz` / `write_report.py.gz`：校验、汇总与归档脚本。`pre-commit.log.gz`、`mypy.log.gz` 保存本轮基准改动的检查结果。
- `gpu-hardware.csv` / `gpu-topology.txt`：GPU 型号、驱动、PCIe 拓扑与设备上限；未锁定时钟。EXL3 前后控制测量用于检查漂移。
- `sha256.json`：除该清单自身外的全部归档文件哈希。

60/60 次计时请求通过长度、零缓存命中和停止符前检索检查。包含重复测量，不能把这个计数当作 60 道独立评测题。此次为性能复测，精度结论沿用前一轮报告中的限制。
