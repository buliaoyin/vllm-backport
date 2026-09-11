# EXL3 长上下文验证至 64K（2026-09-10）

本轮将上一阶段的输入长度扩展到 8192、16384、32768、65536 token，分别提交 1 / 4 个请求，对照 vLLM 与原版 ExLlamaV3。模型仍为用户指定的 Qwen3.8-27B EXL3 dense 与 GLM-5.3-Flash EXL3 MoE。推理实现未为本轮长度测试修改；只扩展了测试脚本的上下文、缓存、进度和输入校验能力。

## 协议

- dense 使用单张 RTX PRO 6000 Blackwell（GPU 3）；MoE 使用 3×CMP 170HX + 1×RTX PRO 6000，vLLM PP4/TP1/EP1，decoder 分层 11/11/11/12。原版为单进程按层放置，启动时显式核对分层。
- 两边均使用同一份预先分词的输入，保存为 gzip JSON，按文件 SHA-256 核对。模型路径可能显示为解析符号链接后的 `/media/bul/ext1/dev/models/...`，仍是同一检查点。
- 最大上下文设为 66560，为完整 65536-token 输入及输出留出空间。每个请求固定生成 32 token，无 MTP；prefill chunk 保持 512，`EXL3_INT8_GEMV=0`，`OMP_NUM_THREADS=1`。vLLM 保持 CUDA Graph。
- 每个长度/并发组合预热一次（生成 16 token），再测三次（每次生成 32 token）。每次都验证实际输出 32 token、缓存命中为零。原版每次重置页表，默认状态检查点机制仍存在；不假定两边的内部缓存管理完全一致。
- vLLM dense 分配 32 GiB KV cache，启动日志报告约 505250 token 容量；MoE 每 GPU 分配 8 GiB，启动日志报告约 1881832 token 容量。原版缓存总容量为 266240 token，足以容纳四个完整 64K 请求及输出。缓存字节数不直接对等，因为布局不同；本轮关注在容量充足时的速度和正确性。
- vLLM 环境为 PyTorch 2.13.0+cu130，模型激活 BF16，EXL3 乘法使用 FP16 操作数；原版 conda `exllamav3` 为 PyTorch 2.8.0+cu128，以 FP16 为主。两边均使用 ExLlamaV3 1.4.8 扩展，对照不隔离 PyTorch、dtype、调度和注意力实现差异。
- 单请求吞吐为输入 token 数除以“开始执行到首 token”的阶段墙钟时间，包含首 token 与引擎开销。4 并发报告每批最大的请求 TTFT，不把各请求重叠的 prefill 时间相加。结果均为三次中位数。
- 原版 wrapper 每 30 秒打印一次已处理输入 token 数，用于区分长时间 prefill 与停滞；不修改原版源码。dense vLLM 在加入这项日志前已启动，两份实际脚本均保存快照，计算逻辑相同。

输入为确定性生成的归档文本。每个长度准备四条独立请求，将不同的六位随机码分别放在输入的约 10% / 35% / 65% / 90% 处；尾部只问对应记录的码，不包含答案。生成器断言答案在整段输入中只出现一次，且每条输入长度精确匹配目标 token 数。B1 使用第一条请求，B4 使用全部四条。

这是一项基本长距离检索及跨请求随机码混淆检查：每个模型/后端有 16 条不同的长提示词，共 60 次计时输出检查，包含重复测量及 B1/B4 对第一条输入的重复使用。最终匹配规则是在首个配置的停止 token 之前的回答中出现完整六位码，不等同于完整长上下文任务准确率。

原始 wrapper 对完整显示文本使用单词边界匹配。vLLM MoE 先返回正确码及 `<|user|>` 停止标记，但强制生成 32 token 会继续生成；显示文本隐藏停止标记后形成类似 `752228We…` 的拼接，导致该检查误报。汇总阶段统一对所有四组运行按 token ID 截到首个停止标记后重新解码和判分，排除强制续写的内容。原始文本、原始判分和计时完整保留，并额外保存重新解码的回答及停止 token 位置；没有为修正判分重测或修改速度数据。最终共纠正 vLLM MoE 的 59 条完整文本匹配误报；首个停止标记前的 60 条回答全部通过，其他三组的判分不变。

本轮输出预算从之前的 256 改为 32，以聚焦 prefill，同时能检查返回的码。特别是 4 并发的 prefill/decode 交错行为会受输出预算影响，因此不要把跨轮差异都归因于输入长度；本轮两个引擎的参数相同。

## 单请求 prefill

单位为输入 token/s；越高越好。耗时计入首 token 与引擎开销，非独立 GPU kernel 计时。

| 模型 | 输入 token | vLLM token/s | 原版 token/s | vLLM / 原版 | vLLM 耗时（秒） | 原版耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen dense | 8192 | 3625.7 | 3247.1 | 1.117× | 2.259 | 2.523 |
| Qwen dense | 16384 | 3412.0 | 3035.7 | 1.124× | 4.802 | 5.397 |
| Qwen dense | 32768 | 3175.0 | 2824.9 | 1.124× | 10.321 | 11.600 |
| Qwen dense | 65536 | 2873.1 | 2489.7 | 1.154× | 22.810 | 26.323 |
| GLM MoE | 8192 | 1318.1 | 809.5 | 1.628× | 6.215 | 10.119 |
| GLM MoE | 16384 | 1388.4 | 815.8 | 1.702× | 11.801 | 20.083 |
| GLM MoE | 32768 | 1428.7 | 822.9 | 1.736× | 22.936 | 39.818 |
| GLM MoE | 65536 | 1458.9 | 834.8 | 1.748× | 44.923 | 78.503 |

## 四并发首 token

每行提交四个同长度、不同内容的请求。报告一批中最大的请求 TTFT，越低越好，包含排队及交错生成的等待。

| 模型 | 每请求输入 token | vLLM 最慢 TTFT（秒） | 原版最慢 TTFT（秒） | vLLM 延迟变化 |
| --- | ---: | ---: | ---: | ---: |
| Qwen dense | 8192 | 9.382 | 10.335 | -9.2% |
| Qwen dense | 16384 | 19.963 | 21.734 | -8.2% |
| Qwen dense | 32768 | 42.227 | 46.305 | -8.8% |
| Qwen dense | 65536 | 93.367 | 104.454 | -10.6% |
| GLM MoE | 8192 | 23.267 | 40.818 | -43.0% |
| GLM MoE | 16384 | 45.809 | 80.765 | -43.3% |
| GLM MoE | 32768 | 90.486 | 160.094 | -43.5% |
| GLM MoE | 65536 | 178.739 | 315.270 | -43.3% |

## 功能检查与数据完整性

| 模型/后端 | 随机码匹配 | 输出中出现同批其他请求的码 |
| --- | ---: | ---: |
| dense-vllm | 60/60 | 0 |
| dense-exllamav3 | 60/60 | 0 |
| moe-vllm | 60/60 | 0 |
| moe-exllamav3 | 60/60 | 0 |

每个后端/模型完成 24 次批次测量，每项三轮；每轮均检查输入长度、32 个实际输出 token、保存的 token ID 数、缓存命中为零，以及计时有效。答案匹配由汇总脚本再次从原始输出重新计算，未只信任运行时布尔结果。不同后端的输入文件哈希一致，实际 benchmark 源码哈希与对应保存快照一致。

完整三次样本、请求级延迟、生成文本、四次运行的完整压缩日志及源码快照见 [汇总 JSON](exl3-long-context-20260910.json) 和 [原始数据目录](exl3-long-20260910/)。目录还保存压缩输入、生成器、两版实际运行脚本及环境协议。

首个停止标记之前没有检出同批其他请求的随机码。固定 32-token 生成会继续越过停止标记，因此后续强制生成的文本不用于回答正确性评分。

本轮未修改模型推理代码，因此沿用 [首轮集成报告](exl3-integration-20260910.md) 中的算子正确性与模型回归结果；新增验证针对更长输入的执行、速度与基本检索。

## 复现

在仓库根目录执行。输入文件已包含预先分词的 token ID 和本机模型路径。若更换路径，先生成输入，并确保两个后端读取同一个文件。硬件和环境参数见协议。

```bash
# vLLM dense: GPU 3, PP1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm --pp 1 \
  --inputs docs/validation/exl3-long-20260910/dense-inputs.json.gz \
  --output /tmp/exl3-long-dense-vllm.json \
  --max-model-len 66560 --kv-cache-gib 32 \
  --warmups 1 --tokens 32 --skip-eval

# vLLM MoE: GPUs 0..3, PP4
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 VLLM_PP_LAYER_PARTITION=11,11,11,12 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 \
.venv/bin/python benchmarks/benchmark_exl3.py \
  --backend vllm --pp 4 \
  --inputs docs/validation/exl3-long-20260910/moe-inputs.json.gz \
  --output /tmp/exl3-long-moe-vllm.json \
  --max-model-len 66560 --kv-cache-gib 8 \
  --warmups 1 --tokens 32 --skip-eval
```

原版在自己的源码目录及 conda 环境执行，设置 `PYTHONPATH` 指向该目录。MoE 命令如下；dense 改为 `CUDA_VISIBLE_DEVICES=3`、`--pp 1`，替换输入/输出文件，去掉 `--exl-memory` 和 `--expected-layer-counts`。

```bash
cd /home/bul/dev/exllamav3
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 EXL3_INT8_GEMV=0 PYTHONPATH=/home/bul/dev/exllamav3 \
/home/bul/miniconda3/bin/conda run --no-capture-output -n exllamav3 \
/home/bul/.local/bin/uv run --no-project \
  --python /home/bul/miniconda3/envs/exllamav3/bin/python \
  python /home/bul/dev/vllm-backport/benchmarks/benchmark_exl3.py \
  --backend exllamav3 --pp 4 \
  --inputs /home/bul/dev/vllm-backport/docs/validation/exl3-long-20260910/moe-inputs.json.gz \
  --output /tmp/exl3-long-moe-original.json \
  --max-model-len 66560 --exl-cache-tokens 266240 \
  --exl-memory 32 41 41 49 --expected-layer-counts 11 11 11 12 \
  --warmups 1 --tokens 32 --skip-eval
```

## 检查范围

两个模型、两个后端均完成全部长度和请求数组合，没有 OOM 或请求异常。
单请求的 64K prefill：vLLM dense 约快 15.4%，MoE 约快 74.8%；
四请求的最慢首 token 延迟分别降低约 10.6% 和 43.3%。
这些数字适用于本报告的硬件、512-token chunk、32-token 固定输出和严格 FP16 EXL3 运算设置。
两边 PyTorch、激活 dtype、调度及注意力实现不同，不能据此归因为某一个量化 kernel 的优势。

输入、脚本和原始结果的 SHA-256 保存在汇总 JSON，原始判分误报也完整保留。
所有测试进程均已退出；原版 ExLlamaV3 源码保持干净。

本轮脚本、文档和数据快照通过全部 pre-commit 检查；Python 3.12 类型检查及 `git diff --check` 均通过。具体文件列表、命令和输出见 [检查记录](exl3-long-20260910/validation_checks.json)。

判分修复使用最终脚本回放全部 240 条已保存输出，核对停止位置之前的解码文本与汇总结果一致，且未检出同批其他请求的码。测量脚本快照、最终脚本和全部保存文件的哈希均已核对。
