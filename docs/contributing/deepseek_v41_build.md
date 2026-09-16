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
