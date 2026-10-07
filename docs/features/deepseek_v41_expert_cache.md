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

## Strata 来源

参考 [Niko1221/Strata](https://github.com/Niko1221/Strata)，固定 commit
`99f3dbd0b21d1401b3769e0c0d963913607f380b`：

- [热点排序、profile 保存与原子提交](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/src/core/expert_cache.cpp#L72-L140)
  和[容量变化后的恢复](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/include/strata/core/expert_cache.hpp#L58-L72)。
本分支结合 DeepSeek 的现有学习状态实现热点保存和恢复，未直接移植 Strata 内核。
CPU miss fallback、CPU/GPU 并行、路由反馈、LRU 和 cost-aware admission 在修改前已经存在。
