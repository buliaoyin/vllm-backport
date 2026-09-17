# DeepSeek-V4.1 混合推理构建

Linux x86-64 的 CUDA 源码构建会自动编译并安装 `libdsv41_ik.so` 和
`libdsv41_cuda.so`，无需单独运行专家库脚本或复制二进制文件。
IK 源码固定版本并校验下载摘要，自动应用 SiLU 截断补丁；CPU 路径使用
AVX2、FMA、F16C、OpenMP，在 Release / RelWithDebInfo 下使用 `-O3`。
CPU 必须支持这些指令。GPU 架构仍由 vLLM 的 `TORCH_CUDA_ARCH_LIST` 控制。

## 源码安装

按 [vLLM 官方 CUDA 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
使用 `uv`。以下命令在仓库根目录、已激活的 `.venv` 中运行：

```bash
unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION
MAX_JOBS=8 uv pip install -e . --torch-backend=auto
```

需要复用已安装的 PyTorch 时，采用官方的无构建隔离流程：

```bash
.venv/bin/python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
MAX_JOBS=8 uv pip install --no-build-isolation -e .
```

`MAX_JOBS` 只限制编译并行度，不改变推理线程数。保留现有的 `CC`、`CXX`、
`CUDA_HOME`、`CMAKE_BUILD_TYPE`、`CMAKE_ARGS` 和编译缓存配置方式。
仅部署三张 CMP 170HX 可设置 `TORCH_CUDA_ARCH_LIST='8.0'`；兼顾本机
Blackwell 时设置 `TORCH_CUDA_ARCH_LIST='8.0;12.0'`，并使用支持该架构的 CUDA toolkit。
不设置时沿用 vLLM 的架构检测逻辑。

构建可分发 wheel：

```bash
MAX_JOBS=8 uv build --wheel
```

生成的 wheel 包含两个混合推理库及 IK 的许可证。默认不使用 `-march=native`，
避免在 AVX-512 主机上构建后，拿到 AVX2 主机运行时出现指令集不兼容。

## 增量编译

使用 [官方 CMake 增量流程](https://docs.vllm.ai/en/latest/contributing/incremental_build/)
生成并配置 `release` preset 后，常规 `install` 目标包含这两个库：

```bash
cmake --preset release
cmake --build --preset release --target install
```

只修改 CPU 专家实现或 CUDA 回调时，可以单独构建、安装对应目标。
以下假设 preset 的构建目录为 `cmake-build-release`：

```bash
cmake --build --preset release --target libdsv41_ik libdsv41_cuda --parallel 8
cmake --install cmake-build-release --component libdsv41_ik
cmake --install cmake-build-release --component libdsv41_cuda
```

离线构建可通过 CMake 的 `FETCHCONTENT_SOURCE_DIR_DSV41_IK` 指定预先准备的
IK 源码目录，版本为 `3bb386eb68ffee0a5dc7db21da0735d594929eeb`。

## 预编译安装的范围

`VLLM_USE_PRECOMPILED=1` 按官方含义跳过原生编译。使用本分支构建的匹配 wheel
时，会一并提取混合推理库；上游 wheel 不含本分支新增的原生代码，安装时会提示。
这种情况下需改用源码构建，或完成上述增量构建；只改 Python 不能补齐原生算子。
修改 C++ 后，应重新编译并重启服务。

## CPU NUMA / NPS

IK 专家后端读取 Linux 暴露的节点、插槽和物理核拓扑，自动适配单／双路的
NPS1、NPS2、NPS4。权重按行分配到节点，加载与计算采用相同的分工；改变
`cpu_threads` 的 prefill/decode 线程数时保持权重位置稳定，不复制整份专家。
默认内存策略下，Engram 和主机预打包 LRU 在可用节点间交错分配，
原有容量与主存余量限制仍生效。

无需新增 NPS 参数。`cpu_threads` 的两个值分别是 prefill 和 decode 的**总线程数**，
不是每个节点或插槽的线程数。默认利用 worker 当前可用的物理核和节点；
`taskset`、容器 cpuset 或 `--numa-bind` 会缩小其可用范围，线程数不得超过该范围。
显式 `numactl --membind`、`--interleave` 等策略及节点范围保持生效。
默认策略下专家权重首选本地分配，节点内存不足时
允许回退到其他可用节点；显式绑定的内存范围仍由用户的设置决定。

启动日志包含 `CPU expert NUMA` 的节点/CPU 列表、分片状态，以及加载后权重页
的本地/远端采样比例。共享内存交错分配使用系统 `libnuma`；缺少该库、
容器禁止内存策略或页归属查询时会说明降级原因。
多节点下禁用专家缓冲区的透明大页，避免大页跨越行分片；并发验证与尾回放
按最多 128 token 分块计算，限制中间工作区。
初始 GPU 专家缓存打包临时用单线程执行 Torch CPU 拼接，结束后恢复原设置，
避免与原生导出反复切换 OpenMP 团队；`cpu_threads` 配置不受影响。

本次已验证拓扑模拟、原生专家数值、线程切换、权重导出和 CUDA 图回放。
混合 worker 会在通信初始化后恢复原有 CPU 范围，避免 GPU 邻近节点绑定使
专家初始化误判可用核数。原生专家入口还会在创建 OpenMP 团队前恢复已配置的
CPU 范围，并在输入量化前完成节点绑核；新建和复用的工作线程在最终同步前
均释放到已配置的 CPU 范围，避免挤在单节点等待。调用后恢复调用线程原有范围。
单路 NPS4 已通过完整启动和 32K 请求性能验证；双路
配置仍需对应硬件实测。
修改 C++ 后须按上文重建 `libdsv41_ik` 并重启服务，旧库不能提供 NUMA 分片。
