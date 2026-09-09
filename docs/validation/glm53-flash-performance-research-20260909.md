# GLM-5.3-Flash 性能优化调研（2026-09-09）

后续实现与本机配对测量见[优化验证报告](glm53-flash-optimization-20260909.md)。本篇保留调研时的源码与远端状态快照。

结论：还有优化空间。对当前 **3 × CMP 170HX（SM80）+ 1 × RTX PRO 6000（SM120）、PP4/TP1**，应先处理 PP 草稿回传的隐式同步、GLM 重复 router 计算和 KDA/NoPE 的不必要拷贝，再评估更换推测解码方案。已有可参考的代码；尚未在本机测出这些候选的增益，不能把其他机器的提升比例当成本机预测。

本轮完成公开资料检索、补丁阅读和本地源码核对，没有启动新的模型评测，也没有修改推理实现。下列 PR 状态和分支 SHA 均为本轮查询快照。

## 对比范围与本地基线

- 本地分支：`codex/deepseek-v4-vision-tested`，HEAD `9d12faf89bfbcd7280e966b4f1479b001bc85864`。
- 远端 `wtdcode/vllm-backport/master` 查询时为 `85f227da62bbcc851cdf32a841b4b92017a0f02f`。
- 枚举了 backport 的 **27 个公开 fork**，逐一读取分支列表；完整读取上游“2026-08-25 起 GLM 相关 PR”搜索的两页，共 158 条结果，再筛选性能、MTP、缓存和硬件相关条目。另检索了 kpool、HiSparse、DFlash2、SM80/SM120 及独立部署仓库。158 条是搜索命中数，不代表 158 个可用优化或逐个完成代码审查。
- 对重点 PR 下载了正文和 diff，对部分 PR 阅读了讨论；对 `artlair` 的五个 GLM 分支在临时仓库中取得源码，对 `promisezackr` 的 24 个补丁取得固定快照。没有合并外部分支。
- 模型仍是用户指定的 cyankiwi AWQ-INT4 和 LibertAIDAI NVFP4。GLM 实际类为 `Glm5NextForConditionalGeneration`：45 层中 34 层 KDA、11 层稀疏 MLA，`index_kpool=4`，主 MLA 的 RoPE 维度为 0。

最近一次完整 GLM 配对测速来自 9 月 7 日；后续共享代码已有改动，以下仅作历史参照，**不是当前 HEAD 重新跑出的基线**。完整协议见[既有验证报告](glm-qwen-cmp170hx-20260907.md)。

| 模型 | MTP | 中文解码 token/s | 代码解码 token/s | 英文解码 token/s | GSM8K 测试集 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 0 | 45.36 | 45.29 | 45.29 | 1282/1319 |
| NVFP4 | 3 | 53.08 | 74.74 | 65.08 | 1280/1319 |
| AWQ | 0 | 45.83 | 45.75 | 45.75 | 1282/1319 |
| AWQ | 3 | 51.09 | 68.49 | 56.79 | 1283/1319 |

当时 PP 分层 `11,11,11,12`，32K 上下文，最多 16 条序列；测速为单请求固定 512 个实际输出 token，GSM8K 并发 16。服务日志确认使用 V2 Model Runner；SM80 使用 `TRITON_MLA_SPARSE` 和 Marlin。GSM8K 的草稿接受率约 82%，已经不是此前 embedding 漏加载导致的低接受率状态。

## 优先候选：当前源码中确实存在对应开销

| 顺序 | 候选及来源 | 本地差异 | 适用范围与判断 |
| --- | --- | --- | --- |
| 1 | PP 草稿 token 回传改为 Triton scatter。[promisezackr 补丁 0007](https://github.com/promisezackr/glm53-flash-170hx-pp8/tree/90ec72e9525e90be701e742c70a20c4154418307/patches)、[artlair 440f6f84c](https://github.com/artlair/vllm-backport/commit/440f6f84c) | `gpu/model_runner.py:1165` 仍按 GPU 布尔 mask 索引 `idx_mapping[valid]` 和 `draft_tokens[valid]`。 | 动态 mask 索引引入取非零位置及主机同步，使非末级 PP 的 CPU/GPU 重叠受损。SM80/SM120 均可采用固定网格 scatter，是最值得先做的一项。别人的“2×”来自较慢 CPU 和 PP8 的旧基线，不是本机增益。 |
| 2 | GLM 解码优化 [vLLM PR #55736](https://github.com/vllm-project/vllm/pull/55736)，**Open** | `Glm5NextMoE.forward` 先执行 gate，同时将同一 gate 传给 `FusedMoEFactory`；`MoERunner` 又执行一次。KDA wrapper 仍对 q/k/v/beta 调用 `.contiguous()`。 | 可去掉重复 router GEMM，并让 recurrent kernel 直接读取 token stride；需要保留本地状态槽位及 int64 边界修复。覆盖主模型；router 也覆盖 MTP 层。 |
| 3 | 同一 PR 的 token-major MLA query，以及 [#55738](https://github.com/vllm-project/vllm/pull/55738) 的 NoPE 拷贝处理，均 **Open** | 本地 W_UK bmm 仍写入 head-major 缓冲后转置；Triton 后端继承的 `XPUMLASparseImpl.forward_mqa` 仍对零宽 RoPE 执行 `torch.cat`。 | 需要同时改 query 布局和当前实际使用的后端。#55736 的后端 hunk 在 FlashInfer 文件内，只套该 hunk 不会优化本机 Triton 路径。应保留有 RoPE、padding head、其他 dtype 的回退。 |
| 4 | indexer 工作区按 kpool 压缩率计量，[#55222](https://github.com/vllm-project/vllm/pull/55222)，**Open** | `glm5next/nvidia/attention.py:303` 仍直接使用 `get_max_prefill_buffer_size()`；没有除以 `index_kpool`。 | 通用内存优化，独立于该 PR 的 SM90 dtype 修复。可释放 KV 预算，主要帮助长上下文和并发，不应称为短文本解码提速。 |

PR #55736 作者在 **4 × GB300、TP4、FP8、无 prefix cache** 的配对测试中，报告单请求 decode 约 +1.8%，并发 64/256 约 +4.3%/+3.1%。这是三项改动一起的结果，不能拆成每项收益，也不能外推到本机 PP4。其每个版本都做了全量 GSM8K，但协议与本地 chat/reasoning 评测不同，绝对分数不可横比。[PR 测试记录](https://github.com/vllm-project/vllm/pull/55736)

工作区的源码名义容量为 `40 × max_model_len × 132 bytes`。kpool=4 时，32K 上下文对应 165 MiB → 41.25 MiB；1M 对应约 5.16 GiB → 1.29 GiB。**这些是工作区尺寸计算，不是实测可回收显存**，不能按 11 个 MLA 层相乘；共享 workspace、其他峰值以及额外 logits 预留会影响实际收益。本地还有单独的 profiler logits 预留，应一起核对其计量单位，但不能直接删除预留来掩盖真实峰值。

## PP/MTP 调度：有参考实现，需要单独验证

`artlair/vllm-backport` 的 `glm53-launch-cost` 固定在 `60170fe84eace9a5b529186c953a6ab5d301646e`。其改动中以下三项本地没有对应机制；这个分支含大量调试和早期移植内容，应挑选实现思路，不能整体覆盖当前仓库。

| 改动 | 来源 | 当前情况与验证重点 |
| --- | --- | --- |
| 用 CUDA event 建立草稿主流与辅助流的依赖，替代阻塞主机的同步 | [ac1bba532](https://github.com/artlair/vllm-backport/commit/ac1bba532e4d604abdeda079b9b0c51eb92a5a26) | 本地 `AutoRegressiveSpeculator` 仍有为修复 IMA 保留的 `current_stream().synchronize()`。必须证明 event 覆盖所有读写依赖及缓冲复用；不能直接删 fence。来源默认关闭，仅属候选。 |
| 对形状稳定的 decode，缓存 PP 中间张量元数据，减少 Gloo 对象往返 | [7a1693188](https://github.com/artlair/vllm-backport/commit/7a16931887f932521bbd626fb918e7ae45d9db04) | 本地仍通过 `send_object`/`recv_object` 交换。需要验证签名覆盖 tensor keys、形状、dtype、图 padding；自适应验证、prefill 混批时回退。 |
| KDA/GDN 纯 uniform speculative batch 的元数据快路 | [f9c3ba19c](https://github.com/artlair/vllm-backport/commit/f9c3ba19cb0be1c026a1d6f214b1205fe9443b6a) | 本地仍构造通用 mask、索引和状态映射。可针对固定形状复用缓冲；必须测试不同 batch 交替、混批及持久请求槽位，不能照搬旧的行索引假设。 |

另一项是限制每个 decode 微批的请求数，让并发请求分散到 PP 在途批次。`promisezackr` 的补丁 0020 提供 `VLLM_PP_MAX_DECODE_REQS_PER_BATCH`；`artlair` 的 [73d354ea9](https://github.com/artlair/vllm-backport/commit/73d354ea9) 提供另一版 PP decode microbatch 实现。本地未提供这两种开关。它们针对**并发吞吐和调度公平性**，不是提升单请求上限的开关；应在 C1/4/8/16 配对验证，包含长 prefill 与短 decode 混合场景。

不要优先移植所谓“整段 MTP 循环一个 CUDA graph”。上游 [#54790](https://github.com/vllm-project/vllm/pull/54790) 主要覆盖 FlashMLA/DSV3.2 indexer；本机还有 Triton sparse 和 kpool tail 元数据。`promisezackr` 的补丁 0006 尝试补齐这些 hook，随后 **0008 默认禁用**：记录了并发 IMA，而且修好 scatter 后没有测到额外收益。这个负向结果比单看补丁标题更有参考价值。

## 更换 drafter：NVFP4 值得实验，AWQ 保留原生 MTP 对照

| 实现或 PR | 查到的证据 | 对本机的意义 |
| --- | --- | --- |
| [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) | 8 张 SM80、同名 LibertAIDAI NVFP4 权重、PP8。作者报告代码场景 MTP3 约 71 → DFlash2 k7 约 123 token/s；prose 则约 43 → 37。提供 aux 跨 PP 传递、图输出 staging、草稿 embedding 和独立 KV 分组补丁。 | 最贴近本机硬件和权重的 DFlash2 参考。提升高度依赖题材；其质量检查只有小样本，不能替代本地全量 GSM8K 和代码正确率。 |
| [PixelML 四张 170HX 实测](https://github.com/PixelML/club-170hx/blob/main/docs/models/glm-5.3-flash.md) | AWQ W4A16、PP4：代码 MTP3 58.37 → DFlash2 k7 36.21 token/s，prose 60.54 → 32.45；DFlash 代码还出现重复。它使用 wtdcode AWQ，**不是本地 cyankiwi AWQ**。 | 支持“AWQ 需要独立验证”，不支持“DFlash 必然提升”或“AWQ 本身是根因”。其 NVFP4 同机配对未完成，因此量化导致的解释仍是假设。 |
| [vLLM #55423](https://github.com/vllm-project/vllm/pull/55423) | **Open / Draft**；当前正文包含 mHC 完成态捕获和草稿 SWA KV 布局。4 × GB300 测试报告约 +13.77% 输出吞吐；保留未解释的输出差异，未验证 PP。 | 不足以直接作为 PP4 上可投入使用的实现。须按当前 patch 评审，不能沿用旧评论对早期 `hidden + residual` 的批评来描述新 head。 |
| [vLLM #55682](https://github.com/vllm-project/vllm/pull/55682) | **Open**；侧重通过 `hc_post` 后 `hc_contract` 捕获完成态，H200 TP4 验证。显式声明 `supports_aux_hidden_states_over_pp=False`，草稿 KV 仍有配套工作。 | 可借鉴捕获语义和测试，单独合入不能解决本机 PP。 |
| [SGLang #36708](https://github.com/sgl-project/sglang/pull/36708) | **Merged，2026-08-27**；GLM DFlash aux capture 适配；基础模型 PR [#36507](https://github.com/sgl-project/sglang/pull/36507) 于 2026-09-06 合并。 | 可交叉核对 mHC 收缩和层编号语义，不是可直接 cherry-pick 的 vLLM 补丁。 |

本地 GLM 类目前没有对应的 EAGLE3 aux capture 协议，也没有 GLM 的跨 PP aux relay 和外部 drafter KV 分组支持。要做 DFlash2，必须一起解决这三项，以及 graph 输出缓冲生命周期、PP embedding 加载。首轮建议用 NVFP4、独立开关、固定 k，再讨论按接受率自适应；AWQ 另跑，保留失败和重复输出。

`tonyd2wild` 的 [DFlash2 说明](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/main/docs/DFLASH2-SPECULATIVE-DECODING.md) 和 `alexellis` 的 [双 Spark 配方](https://github.com/alexellis/glm-5.3-flash-2x-dgx-spark-switchless) 提供了相同功能在 SM121/TP2 上的参考，后者还保留按题材的多次测量。它们同源性较强，不能当成多份独立证据来叠加收益。

## 末张 SM120 可单独尝试的优化

[local-inference-lab 的 GLM 配方](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md) 对应大量 B12X/Blackwell 优化。取得的 `voipmonitor/vllm` 集成分支 SHA 为 `9ff42d83938e74018f9c255e8cfa7ca6df6921b0`。

其中较适合本机挑出来的是[独立量化 MTP vocabulary head](https://github.com/voipmonitor/vllm/blob/9ff42d83938e74018f9c255e8cfa7ca6df6921b0/vllm/models/glm5next/nvidia/mtp_draft_head.py)：从 BF16 目标 head 创建仅供草稿使用的 NVFP4 副本，通过 FlashInfer W4A16 执行，代码显式要求 SM120。目标模型仍保留 BF16 head；草稿分布和接受率可能改变。

本地目前共享目标 head，没有这个独立副本。MTP 恰好位于 GPU3，因此不必让三张 SM80 都支持该内核。应先量出 head 在草稿耗时中的占比，再比较每轮耗时、每个位置的接受率、最终 token/s 和显存；新增副本会消耗显存。该路径本机尚未验证，没有可承诺的提速比例。

另外两组上游补丁的适用面有限：

- [#55737 FlashKDA prefill](https://github.com/vllm-project/vllm/pull/55737)，**Open**：源码只选择 SM90/SM10x/SM12x。作者的 1.7–3.8× 是 KDA 内核比较；4 × GB300 整机 prefill TTFT 改善约 7.9%–13.2%。本地虽然有 FlashKDA 构建配置，GLM KDA 还没有接入；前三张 SM80 仍须用 Triton，所以只能评估 GPU3 局部收益及混合后端数值。
- [#55738 NoPE dense/masked-MHA prefill](https://github.com/vllm-project/vllm/pull/55738)，**Open**：GB300 上的独立增量 TTFT 改善约 3.5%–8.5%，依赖对应 FA/FlashInfer 路径。本机 Triton sparse 后端并不自动获得这套路径，需要重新接入并选阈值；NoPE 无效拷贝部分可以先单独处理。

## 增加 MTP 深度前需补的正确性检查

[vLLM #55219](https://github.com/vllm-project/vllm/pull/55219)，**Open / Draft**，同时重构 GLM KV 布局和修复 tail ring。值得单独提取的是：tail 不能只保存一个 kpool；speculative token 在拒绝后可能覆盖下一次重放需要的已提交 key。该 PR 按 `kpool × ceil((kpool + num_spec) / kpool)` 分配环，配套更新寻址与测试。

当前本地 `Glm5NextTailCache` 仍使用 `block_size=index_kpool`，未包含这个 speculative ring 扩容。对 kpool=4，MTP3 对应候选容量 8，MTP7 对应 12。**这是源码层面的风险命中，不是已经证明本地曾经的错题由此导致**；先用拒绝跨池边界的定向测试复现，再决定修复。不建议为了这一点整体移植 34 个文件的 KV 布局重构。

已有的 prefix-cache、请求槽位和 PP embedding 修复不能代替这项测试。最低覆盖：不同起始位置 mod 4、接受长度从 0 到 k、拒绝后重放、多个请求交错、取消后槽位复用，以及 eager/CUDA graph 两种模式。

## 已具备或暂不优先的方向

| 来源 | 判断 |
| --- | --- |
| backport [#51 PP mHC 交接](https://github.com/wtdcode/vllm-backport/pull/51)，已合并；[#50 草稿 embedding](https://github.com/wtdcode/vllm-backport/pull/50)、[#63 Mamba 状态](https://github.com/wtdcode/vllm-backport/pull/63)，仍 Open | 本地已有 PP 交接、GLM 专用 embedding 加载完整性检查和请求槽位/缓存修复。不能把这些再次列为新增性能收益；跨仓库 commit 不同不代表行为缺失。 |
| SM80 Triton sparse MLA、软件 FP8 编码、Marlin NVFP4、mHC post/pre 融合、KDA 融合投影/卷积/门控、MTP index sharing | 本地已有。`VLLM_DETERMINISTIC_MOE_ALIGN` 默认也已是 0；不再重复推荐。 |
| [#55442 延迟分配临时 MTP head](https://github.com/vllm-project/vllm/pull/55442)，Open | 本地未延迟构造。可减少加载峰值；后续 head 本来就共享，不能算稳定运行时 token/s 优化，也与“独立量化草稿 head”不同。 |
| [#54951 TP 分片 indexer prefill](https://github.com/vllm-project/vllm/pull/54951)，Open | 长上下文 TP 下有实测，但当前 TP1 没有多个 TP rank 可分工。保持 PP4/TP1 时不优先。 |
| [#54524 CuTeDSL BF16 默认 GEMM](https://github.com/vllm-project/vllm/pull/54524)，Open | 针对 SM100a；不是 SM80/SM120 的通用替代。当前 `glm52_low_latency_gemm.py` 也专门限制 SM103 和 GLM5.2 形状，不能因为文件名含 GLM 就直接启用。 |
| [#54929 SM12x Triton sparse fallback](https://github.com/vllm-project/vllm/pull/54929)，Open | 可用于将来内核对比，本地已经有另一套 Triton split-KV fallback。对方主要验证其他 DSA 模型，没有证据证明替换后本机 GLM 更快。 |
| [#53969 NoPE SM120](https://github.com/vllm-project/vllm/pull/53969)、[#55277](https://github.com/vllm-project/vllm/pull/55277)，Open | 有助 SM120 FlashInfer/FP8 KV 支持；不能给 SM80 增加原生 FP4/FP8 算力。本地 Triton sparse 基类还拒绝 FP8 KV，整机切换需另外实现。不要通过削减 `index_topk` 来绕过宽度错误。 |
| [#53781 HiSparse](https://github.com/vllm-project/vllm/pull/53781)，Open | 重点是 host-resident KV 与长上下文容量。9 月 7 日官方博客讨论的是 8 × H200 上完整 GLM-5.3，不能直接作为 Flash/本机结果。短上下文单请求不是优先场景。 |
| PixelML 对 NVFP4 的失败记录 | 只说明其检查点/加载路径在四张 SM80 上没有成功，不能推导“SM80 不能运行 NVFP4 权重”。本地两种量化已通过 Marlin 完成全量评测。 |

## 建议实施和验证顺序

1. **重建当前 HEAD 的 GLM 基线**。复用原 conda `vllm-backport`，AWQ/NVFP4 都跑 MTP0/3；固定模型文件、模板、采样参数、分层和并发。已有报告是 9 月 7 日数据，不能直接作本轮代码 A/B 的 A。
2. **先回归 kpool 拒绝边界，再做 scatter、重复 router、KDA stride、NoPE query**。每项独立开关/提交，先跑定向正确性和阶段计时，最后测合并后的效果。共享 kernel 要回归 Qwen，PP 通路要回归 DeepSeek。
3. **测量 PP 主机等待和末级草稿成本**，再选择元数据缓存、event fence 或独立量化草稿 head。PP 层分配按逐级计时和 KV/权重显存决定；不能照抄 PixelML 的 `14,12,12,7`，因为我们末张是更快且显存更大的 SM120。
4. **另做并发与长上下文优化**：微批 C1/4/8/16、workspace 缩容、GPU3 FlashKDA。分别报告吞吐、TTFT、显存和抢占；不能以其中一项改善代替其余项。
5. **最后试 NVFP4 DFlash2**，再独立试 AWQ。完成 aux 语义、跨 PP relay 和 KV 分组后，以原生 MTP3 为对照；不只报告最佳代码题。

测速至少包含中文解释、多个代码题、英文设计和低熵结构化对照，固定 512/1024 个实际输出 token，预热后每类至少 5 次，交替 A/B 顺序。记录 TTFT、解码中位数和波动、MTP 各位置接受率、每轮目标验证/草稿时间，防止把更容易接受的输出当成内核提速。

验证分短上下文和长上下文：2K、16K、32K 固定 decode；64K/128K 做未命中 prefix cache 的 prefill/TTFT；按显存再扩展。功能覆盖工具调用、长前缀恢复、请求取消/复用、混合 prefill/decode；多模态路径或缓存公共代码发生变化时补真实图像检查。

最终对影响模型执行的方案，AWQ/NVFP4 分别跑完整 **GSM8K 测试集 1319 题**；代码另用有可执行断言的题集检查。保留新增错误、截断和重复输出，不能只看净分。CMP 上此前 CUPTI 不可用，可沿用 CUDA event 阶段计时；它包含流等待，不能称为独立内核耗时。

## 调研资料

- [来源状态和分支快照](glm53-flash-performance-research-20260909.json)。
- PR 正文、diff、讨论、fork refs、外部补丁与临时比较仓库保存在 `/tmp/glm53-research-20260909/`；临时目录不保证长期保留，正文的远端链接和 SHA 可用于重新取得来源。
- 本轮只新增这份中文报告及来源 JSON，未合入优化、未修改模型文件、未宣称本机取得新增性能收益。
