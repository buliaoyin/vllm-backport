# DeepSeek V4.1 混合推理专家缓存

GPU 常驻热点专家，其余路由由 CPU 处理。缓存根据 verification-group
路由反馈和跨请求 EWMA 更新专家选择，已有 host packed/LRU 缓存、收益阈值和
替换预算。热点持久化默认保存驻留名单和学习状态，供同一模型重启时恢复，也可关闭。

## 配置

以下字段位于 `--additional-config` 的 `deepseek_v41_hybrid` 对象中：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `expert_profile` | 按模型指纹生成路径 | 读取、保存学习状态的 JSON 路径，支持 `~`；`false` 关闭持久化。 |
| `expert_profile_interval` | `16` | 两次保存尝试之间的请求组结束次数，正整数；默认路径也可设置，关闭持久化时不可设置。 |
| `overlap_decode` | `true` | 方案 A：先提交 draft，再完成 CPU 专家错误检查、采样计数及缓存维护。只接受布尔值。 |

未指定 `expert_profile` 时，自动使用
`$VLLM_CACHE_ROOT/expert_profiles/deepseek_v41/<模型指纹>.json`。
沿用 vLLM 缓存目录约定：`VLLM_CACHE_ROOT` 优先，其次为 `$XDG_CACHE_HOME/vllm`，
均未设置时使用 `~/.cache/vllm`。同一模型重启复用文件，不同指纹使用不同文件。
启动日志中的 `Expert cache profile` 显示实际路径；显式配置的路径优先于默认路径。

例如三张 CMP 170HX：

```json
{
  "deepseek_v41_hybrid": {
    "pipeline_layers": [7, 8, 25],
    "cpu_threads": [24, 24],
    "engram_storage": "ram",
    "expert_profile": "/absolute/path/dsv41f-expert-profile.json"
  }
}
```

四卡的分层、CPU 线程和存储应使用对应硬件实测配置。RAM Engram 比 SSD 多占约
189 GiB 主存；内存不足时参见 [Engram SSD 存储](deepseek_v41_engram_ssd.md)。
新增缓存选项不要求重建 native 库；已有混合推理的构建要求见
[构建说明](../contributing/deepseek_v41_build.md)。

## 学习和恢复

1. 文件不存在时按默认选择启动。发送实际业务的代表性请求，让原有在线策略学习。
2. worker 的混合请求组全部结束后保存；第一份有效学习结果立即保存，之后按间隔保存。
   持续存在在途请求会推迟保存。停止进程不会额外强制保存尚未达到间隔的状态。
3. 同模型、同路径重启，出现 `Restored expert cache profile` 日志表示恢复成功。
   仍需重新加载、打包和上传专家，profile 不会省略模型加载或恢复 KV cache。
4. 恢复之前的驻留名单；容量缩小时优先保留历史更热的已选专家，增大时按历史补齐。
   各层容量和 GPU 放置沿用原有公平分配。

## 持久化边界

文件版本为 `1`，大小最多 1 MiB。仅保存各层 `selected` 专家 ID、`history` 调用频率、
`requests` 和 EWMA `mass`，不保存输入、输出文本、权重、GPU tensors 或 pending counters。
没有活跃 GPU cache 的层保留之前的历史记录；这些记录不表示当前驻留，也不会继续学习。

checkpoint fingerprint 包含模型路径/ID、revision、HF commit、配置，以及本地 safetensors
文件名、大小、mtime 和 index 内容摘要。它没有完整哈希数百 GB 权重；保留所有元数据却
手工替换权重内容不在这个快速检查的保证范围内。

文件缺失时重新学习；身份、版本、层号、专家 ID、频率或大小校验失败时记录 warning 并
忽略整份文件。保存使用同目录临时文件和 `os.replace`，失败不打断生成。各独立部署使用
独立路径；多个写入者共享路径时最后一次原子写入生效，不合并学习状态。

移除 `expert_profile` 即可使用默认路径；设为 `false` 并移除 `expert_profile_interval`
即可关闭持久化，恢复每次启动重新学习的行为。
GPU/CPU 既有数值舍入可能因缓存放置变化而改变生成内容和 DSpark 接受率，需要做模型评测。

## Decode 后处理重叠（方案 A）

`deepseek_v41_hybrid` 默认启用方案 A，不需要 benchmark worker、额外调度器或
重建 native 库。在此对象中设置 `"overlap_decode": false` 可恢复原同步路径；
删除字段会恢复默认开启。预填充和仅编码阶段沿用原处理方式。

方案 A 保留 GPU 状态更新的提交顺序，将 decode 的主机检查推迟到 draft 提交后。
主机等待对应输出的完成事件，复用已回传的采样计数；CPU 回调错误在输出交付前抛出。
每步完成函数保存请求和步数，缓存替换及线程切换仍保留原来必要的等待。

3×170HX、PP `[7,8,25]`、24 CPU 线程、原自适应 DSpark 的实验版本结果：

- 冻结缓存、32K 输入/1024 输出、三类单请求各五对：综合 +6.40%，95% CI `[+0.88%, +12.22%]`。
- 在线学习、统一 GPU/学习历史/空 host LRU 初态、各十对：综合 +4.86%，95% CI `[+1.66%, +8.16%]`。
  在线收益未达到预定 5% 默认采用门槛；并发吞吐也没有通过该门槛。
- 自然 GSM8K 256题：基线227题正确，A为231题；准确率差值区间 `[-1.17, +4.30]`
  个百分点，不能认证质量等价。调度时序影响验证形状，既有浮点归约差异可能改变输出。

上述结果来自迁移前的实验实现，不等于任意设备、上下文或流量的性能承诺。
本分支默认开启；更换设备或负载时应重新检查性能和模型质量。

正式实现与实验 A 的迁移回归使用固定 K=3、稳定 MoE 排序及在线学习：
256/256 行 logits、批次形状和输入完全一致，20 层缓存名单及学习状态一致。
16 道自然生成双方均为12/16正确，16/16条完整 token 序列一致；此小样本只用于
迁移检查。并发8请求取消后继续生成384个 token、请求排空及在线缓存更新检查通过。

## Strata 来源

参考 [Niko1221/Strata](https://github.com/Niko1221/Strata)，固定 commit
`99f3dbd0b21d1401b3769e0c0d963913607f380b`：

- [热点排序、profile 保存与原子提交](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/src/core/expert_cache.cpp#L72-L140)
  和[容量变化后的恢复](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/include/strata/core/expert_cache.hpp#L58-L72)。
本分支结合 DeepSeek 的现有学习状态实现热点保存和恢复，未直接移植 Strata 内核。
CPU miss fallback、CPU/GPU 并行、路由反馈、LRU 和 cost-aware admission 在修改前已经存在。
