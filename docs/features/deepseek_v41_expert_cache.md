# DeepSeek V4.1 混合推理专家缓存

GPU 常驻热点专家，其余路由由 CPU 处理。缓存根据 verification-group
路由反馈和跨请求 EWMA 更新专家选择，已有 host packed/LRU 缓存、收益阈值和
替换预算。本分支增加可选的热点持久化、按历史需求分配容量，并减少替换时的复制开销。

## 配置

以下字段位于 `--additional-config` 的 `deepseek_v41_hybrid` 对象中：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `expert_profile` | 不设置 | 读取、保存学习状态的 JSON 路径，支持 `~`。 |
| `expert_profile_interval` | `16` | 两次保存尝试之间的请求组结束次数，正整数，要求设置 profile。 |
| `expert_allocation` | `"fair"` | `"fair"` 公平分配；`"profile"` 按历史需求分配，要求设置 profile。 |

例如三张 CMP 170HX：

```json
{
  "deepseek_v41_hybrid": {
    "pipeline_layers": [7, 8, 25],
    "cpu_threads": [24, 24],
    "engram_storage": "ram",
    "expert_profile": "/absolute/path/dsv41f-expert-profile.json",
    "expert_allocation": "profile"
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
4. `fair` 保留之前的驻留名单；容量缩小时优先保留历史更热的已选专家，增大时按历史补齐。
   `profile` 的初始专家名单直接取完整历史排名前 N 个。

`profile` 分配在启动时执行：保留公平方案的 GPU 放置，以八个 slots 为一组，优先增加
预计能省下更多专家调用的层。历史不完整、全零或预测收益未超过公平方案时回退。
它不按 GPU kernel、远端通信或 CPU 实测时间重新建模，也不在运行中跨层迁移容量。
业务分布变化后，应重新比较两种模式的端到端效果。

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

移除 `expert_profile`、`expert_profile_interval` 并省略 `expert_allocation` 即可回到默认行为。
GPU/CPU 既有数值舍入可能因缓存放置变化而改变生成内容和 DSpark 接受率，需要做模型评测。

## 替换工作区

同一 GPU 的各层共享一个 pinned host buffer 和一个 device buffer，每个大小为一个原始专家。
此模型每份为 18,800,640 字节。四个权重/scale views 通过一次连续 H2D 复制传输，再进行
MXFP4/Marlin 打包。缓存容量预算为每个使用的 GPU 预留这一份 device scratch。

共享锁保护完整更新，重用 host buffer 前等待 DMA 完成；权重复制、expert map 和 origin
membership 的发布通过 stream 依赖排序，更新结束后同步。异常路径也在释放锁前同步。
最终 packed weights、映射与 IO 缓冲区地址保持稳定，可继续被 CUDA Graph 使用。
策略的 NumPy 稳定排序保留原有准入公式、预算和 tie-breaking。

更新仍是同步操作；这里没有实现跨多个 decode rounds 的后台 pending admission。
长期复用工作区也不消除 PCIe 权重传输。CMP 的 PCIe Gen2 上，小 verification-group 的冷
专家直接搬上 GPU 未必比 CPU 快，不能只按 GPU 算力决定 miss 路径。

## Strata 来源

参考 [Niko1221/Strata](https://github.com/Niko1221/Strata)，固定 commit
`99f3dbd0b21d1401b3769e0c0d963913607f380b`：

- [热点排序、profile 保存与原子提交](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/src/core/expert_cache.cpp#L72-L140)
  和[容量变化后的恢复](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/include/strata/core/expert_cache.hpp#L58-L72)。
- [长期复用 staging/scratch](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/include/strata/core/verify.hpp#L254-L266)
  和 [HIP pinned staging](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/src/core/expert_cache.cpp#L145-L158)。

本分支为 DeepSeek 的 MXFP4/Marlin 与现有学习状态重新实现这些思路，未直接移植 Strata 内核。
CPU miss fallback、CPU/GPU 并行、路由反馈、LRU 和 cost-aware admission 在修改前已经存在。
