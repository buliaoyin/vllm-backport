# DeepSeek V4 Flash Vision Exp：混合 GPU 验证报告

日期：2026-09-08。测试使用 conda 环境 `vllm-backport`。

## 结论

本机已能通过 PP4 运行模型的真实视觉推理。关闭 MTP 和启用 DSpark 3 时，中英文 OCR、自然图片、表格、计数、位置关系、多图顺序和图文混排均通过功能检查。推荐使用 V2 执行器和默认计算图；eager 模式速度明显较低。

本次 GSM8K 是测试集前 32 题抽查，**不是 1,319 题全量成绩**。视觉检查是明确答案的功能探针，**不是 OCRBench/MMMU 等标准视觉基准成绩**。

## 机器和模型

| 项目 | 配置 |
| --- | --- |
| GPU 0–2 | NVIDIA CMP 170HX，约 64 GiB/卡，SM80 |
| GPU 3 | NVIDIA RTX PRO 6000 Blackwell，约 96 GiB，SM120 |
| Python / PyTorch | Python 3.12 / PyTorch 2.13.0+cu130 |
| 环境 | `/home/bul/miniconda3/envs/vllm-backport` |
| 模型 | `/home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp` |
| 权重 | 48 个 safetensors 分片，约 156.29 GiB |
| 量化 | MoE 专家 FP4，其余相应权重 FP8；使用检查点原始权重 |
| 并行 | PP4 / TP1；层分配 `10,11,11,11` |
| 推理范围 | 最大上下文 8,192；最多 4 条序列；每次最多 2 张图片 |
| KV cache | `fp8`，block size 256 |
| 内存配置 | `gpu-memory-utilization=0.90`；`max-num-batched-tokens=2048` |

## 为什么需要修改代码

原分支不能直接运行该检查点：初次启动要求显式设置 FP8 KV；补齐 KV 配置后，在加载 `aligner` 权重时失败。仅跳过视觉权重只能验证文本，不能证明模型能看图。

完整支持移植自 [vLLM PR #54566](https://github.com/vllm-project/vllm/pull/54566)，合并提交为 `1356635d837c4ef002ec98c1a0296e7ff60be3c1`。远端 `origin/dsv4-vision-exp` 的 `f8e055927` 仅包含文本加载和独立视觉编码器，尚未接通完整视觉请求链路。

本地额外适配包括：

- 将图像预处理、ViT、投影层、图片标记和视觉路由偏置接入完整模型；保留视觉 RMSNorm 的 FP32 参数精度。
- 为 Ampere 使用的 Triton 注意力路径增加图片区域双向可见范围，并保留既有 DSpark 非因果草稿逻辑。
- 为 V2 执行器补齐完整图片标记区间和前导对齐 padding 的处理；图片区间超过普通滑动窗口时交由注意力内核处理。
- 在 PP 的每个阶段保留视觉路由所需的原始 token ID，实际执行和 CUDA graph 捕获使用相同规则。未修正时，V2 在预热中报 `vision MoE routing requires input_ids`。
- 适配本地权重加载后处理，确保 MHC 广播参数和注意力输入矩阵融合执行；避免重复映射权重名称。
- 为 SM80 与 SM120 编译并安装新的 `_moe_C_stable_libtorch` 扩展。此改动包含 CUDA 算子接口，不能只更新 Python 文件。

## 输出验证结果

每轮 20 项由 **19 项视觉检查和 1 项纯文本算术检查**组成。四并发轮次包含图片缓存与前缀缓存复用；保存的请求只向模型提供图片像素和问题，没有把预期答案作为提示发送。

| 配置 | 单并发功能检查 | 四并发功能检查 | GSM8K 前 32 题 | GSM8K 耗时 / s | GSM8K 草稿接受率 |
| --- | --- | --- | --- | --- | --- |
| V1 / eager / 关闭 MTP | 20/20 | 20/20 | 31/32 | 260.10 | — |
| V2 / 计算图 / 关闭 MTP | 20/20 | 20/20 | 31/32 | 29.42 | — |
| V2 / 计算图 / DSpark 3 | 20/20 | 20/20 | 32/32 | 19.40 | 73.2% |

GSM8K 使用官方测试集、5-shot、并发 4、temperature 0、seed 42、`thinking=false`、最多生成 4,096 token。标签文件 SHA256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。沿用仓库的题目构造函数，并保存每题请求与原始响应。

关闭 MTP 时唯一错误为从零开始的第 12 题：模型答 12，官方标签为 13；DSpark 本轮答对该题。样本很小，且启用草稿验证会改变目标模型的计算批次形状，**不能将 31/32 与 32/32 的差异解读为 MTP 提高了模型能力**。

| 视觉项目 | 检查内容 | 结果 |
| --- | --- | --- |
| 自然图片 | 模型自带胡萝卜、玉米图片 | 正确识别 |
| 换图对照 | 相同问题分别输入红色、蓝色矩形图片 | 回答随像素变化 |
| 英文 OCR | 图片中的 `ZX-4827` | 正确 |
| 中文 OCR | `订单编号：视觉7284`、`下午三点` | 正确 |
| 表格 | 梨的价格 19；总价 167 | 正确 |
| 图表 | 最高柱为 B | 正确 |
| 计数与空间关系 | 3 个红圆；蓝方块在下方 | 正确 |
| 多图 | 胡萝卜/玉米交换输入顺序 | 按各自顺序回答 |
| 图文混排 | 文字、图片、文字、图片 | 正确区分两图 |
| 图像尺寸 | 宽图、竖图、31×19 小图 | 正确识别左右颜色 |
| 缓存复用 | 重复图片和问题、并发不同图片 | 未出现串图 |

DSpark 另完成 4 个并发长前缀图片请求，正确 4/4，每条实际输入 5,635 token。前缀后交替放入红色和蓝色图片，均识别了对应图片颜色。

## 速度与 MTP

固定中文、代码、英文三个提示；每条强制生成 256 token，单并发，`thinking=false`。解码速度使用 `(输出 token 数−1)/(结束时间−首个输出时间)`，排除首 token 等待时间。这是预热后的单轮测量，不代表所有输入长度或并发下的吞吐量。

| 配置 | 中文 token/s | 代码 token/s | 英文 token/s |
| --- | --- | --- | --- |
| V1 / eager / 关闭 MTP | 3.72 | 3.65 | 3.65 |
| V2 / 计算图 / 关闭 MTP | 46.76 | 46.78 | 46.74 |
| V2 / 计算图 / DSpark 3 | 74.70 | 107.80 | 78.97 |

在相同 V2/计算图配置下，DSpark 的速度分别为关闭 MTP 时的 **1.60×、2.30×、1.69×**。这三个提示的草稿 token 接受率分别为 **38.1%、71.6%、43.2%**，说明收益与生成内容有关。

DSpark 使用检查点自带的 3 层草稿权重，设置 `num_speculative_tokens=3`、`draft_sample_method=probabilistic`。**PP4 暂不支持自适应验证，必须设置 `enable_adaptive_verification=false`**；启用该功能的尝试被现有配置校验拒绝，未绕过限制。

## 与参考实现的视觉数值对照

使用模型目录中自带的 `inference/image_processor.py` 和 `inference/vision.py`，直接读取真实 ViT 与 aligner 权重：

- 8 张图片的像素张量、ViT/语言网格完全一致；每张图检查 8 个不同起始位置，图片 token 类型和排列均一致。
- 5 张真实输入在采用相同的 PyTorch 注意力、激活运算后，完整视觉编码和投影结果**逐元素完全一致**。这是独立的算子对照，不是服务运行时替换。
- 服务使用的加速注意力和融合激活存在浮点舍入差异：SM80 的展平特征余弦相似度约 0.9956–0.9990，SM120 约 0.9969–0.9994。因此不宣称加速实现的视觉特征逐位一致；最终功能结论还依赖上面的真实图片请求。

## 代码检查和回归

| 检查 | 结果 |
| --- | --- |
| SM80 路由、初版视觉索引、分词和配置组合 | 1,732 passed |
| SM120 路由矩阵 | 1,669 passed |
| 扩宽图片窗口后的最终视觉索引测试 | SM80 21 passed；SM120 21 passed |
| V2 执行器、缓存及图捕获回归，含图片边界测试 | 36 passed |
| pre-commit | 通过，含 Ruff、Clang format、mypy 和文档检查 |
| 旧 DeepSeek 文本测试对照 | 原分支与移植分支均为 31 passed / 11 failed / 29 skipped；失败项完全相同 |

旧测试失败包括调用已改名的注意力方法、测试桩缺少 `modules()`、缺少 `vllm.third_party.deep_gemm.utils`。这些失败已在未修改的 `bbf878805` 上复现。既有 DSpark 稀疏注意力测试还受硬件条件限制，在本机跳过；不将跳过计为通过。GLM/Qwen 之前的全量模型评测未在本次重复执行，原中文报告继续保留。

## 当前工作区复核

已将实现同步到 `/home/bul/dev/vllm-backport`，当前分支为 `codex/deepseek-v4-vision-tested`，代码提交为 `7337f1434`。原 `codex/glm5next-serving-fixes` 分支仍保留在 `bbf878805`。新版 MoE 扩展已安装，原二进制备份在原始记录目录的 `binary-backup/` 中。

从当前工作区直接调用 `vllm-backport/bin/vllm`，启用 DSpark 3 后重新启动成功；四并发复核的 8 项图片、OCR、表格和图文混排请求全部通过。记录为 `serve-installed-dspark3.log` 和 `probes-installed-dspark3/summary.json`。复核后已停止测试服务并释放 GPU。

## 可复现启动命令

先确认当前源码包含本次移植，并已同步新版 MoE 扩展。以下为实际验证的配置范围：

```bash
NCCL_P2P_DISABLE=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_PP_LAYER_PARTITION=10,11,11,11 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/bul/miniconda3/envs/vllm-backport/bin/vllm serve \
  /home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp \
  --served-model-name local \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --limit-mm-per-prompt '{"image":2}' \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","model":"/home/bul/dev/models1/DeepSeek/DeepSeek-V4-Flash-Vision-Exp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}' \
  --host 127.0.0.1 \
  --port 8001
```

关闭 DSpark 时删除 `--speculative-config` 参数。用于本报告探针的请求还设置 `chat_template_kwargs={"thinking": false}`。

## 原始记录和范围

全部日志、图片、请求、响应、构建记录和测量脚本位于：

```text
/tmp/deepseek-v4-vision-validation-20260908/
```

关键文件包括 `provenance.json`、`upstream-pr-54566.patch`、`moe-build.log`、`regression-comparison.json`；功能结果在 `probes-*/summary.json`，文本结果在 `text-*/gsm-summary.json` 和 `text-*/bench-summary.json`。失败启动和服务尚未就绪时的请求记录单独保留，不混入通过率。测试工具为 `compare_vision.py`、`vision_probes.py`、`long_vision.py`、`text_eval.py`。

本报告覆盖 8K 配置下的文本与静态图片、最多两图、并发 1/4，以及固定 3-token DSpark。尚未验证更长上下文、大并发、视频、多机、其他 TP 组合或标准视觉评测全量结果。代码由 AI 辅助移植和适配，未向远端提交 PR。
