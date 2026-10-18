# CAT-Surface GPU v0.3.0

[English](RELEASE_v0.3.0.md) · [简体中文](RELEASE_v0.3.0.zh-CN.md)

v0.3.0 新增 CPU 和 Metal 后端，并加速 CUDA 球面生成与配准。支持 Linux、Windows 和 Apple Silicon macOS。

## 主要变化

- CUDA 面积平滑使用共享内存分块和保序更新，默认仍执行 4,999 次平滑。
- 几何装配保留 FP64，特征求解和 GPU 常驻 DARTEL 使用 FP32；PCG 仍为 40 次，最终指数映射保留 FP64。
- 固定拓扑和目标特征按内容及实现版本缓存。两个 CLI 复用已验证的拓扑，避免每次启动重新建表。
- CPU、Metal 共用配准编排。Metal 通过 MLX 执行平滑、特征计算、球面查询、旋转代价及 DARTEL。
- 保留现有命令和 CUDA 数组 API。软件包独立运行，无需 CAT 外部程序。

## 平台与安装

| 平台 | 后端 | 安装 |
| --- | --- | --- |
| 配备 NVIDIA GPU 的 Linux / Windows | CUDA / CuPy | `python -m pip install ".[cuda]"` |
| Apple Silicon macOS | Metal / MLX | `python -m pip install ".[metal]"` |
| Linux / Windows / macOS | CPU / NumPy / Numba | `python -m pip install .` |

使用 Python 3.11 或更新版本。CUDA 可选依赖对应 CUDA 12；其他运行时版本请单独安装匹配的 CuPy。Metal 需要原生 Apple Silicon Python，以及 [MLX 支持的 macOS 版本](https://ml-explore.github.io/mlx/build/html/install.html)。

安装和运行前，将临时文件及缓存放在项目内。Linux/macOS：

```bash
mkdir -p .work/v030/{tmp,cache}
export TMPDIR="$PWD/.work/v030/tmp" TMP="$PWD/.work/v030/tmp" TEMP="$PWD/.work/v030/tmp"
export XDG_CACHE_HOME="$PWD/.work/v030/cache"
export PIP_CACHE_DIR="$PWD/.work/v030/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/v030/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/v030/cache/cupy"
export CUDA_CACHE_PATH="$PWD/.work/v030/cache/cuda"
export CAT_SURFACE_CACHE_ROOT="$PWD/.work/v030/cache/targets"
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force .work/v030/tmp, .work/v030/cache | Out-Null
$env:TMPDIR = "$PWD/.work/v030/tmp"
$env:TMP = $env:TMPDIR
$env:TEMP = $env:TMPDIR
$env:XDG_CACHE_HOME = "$PWD/.work/v030/cache"
$env:PIP_CACHE_DIR = "$env:XDG_CACHE_HOME/pip"
$env:NUMBA_CACHE_DIR = "$env:XDG_CACHE_HOME/numba"
$env:CUPY_CACHE_DIR = "$env:XDG_CACHE_HOME/cupy"
$env:CUDA_CACHE_PATH = "$env:XDG_CACHE_HOME/cuda"
$env:CAT_SURFACE_CACHE_ROOT = "$env:XDG_CACHE_HOME/targets"
```

两个命令都支持 `--device cuda`、`--device cuda:1`、`--device metal` 和 `--device cpu`。`--device auto` 自动选择可用后端；为兼容旧调用，默认仍为 `cuda`。显式指定的设备不可用时直接报错。

```bash
cat-surf2sphere-gpu --input-surface white.gii --output-surface sphere.gii --device metal
cat-surfwarp-gpu --source-surface white.gii --source-sphere sphere.gii --target-surface template.white.gii --target-sphere template.sphere.gii --output sphere.reg.gii --device metal
```

数组接口使用 `surf_to_sphere(..., device="cpu" | "cuda" | "metal")`。既有 `surfwarp_cupy(...)` 新增 `backend="cpu" | "cupy" | "metal"`，保留函数名以兼容旧调用。Metal 当前使用固定迭代 PCG（`pcg_check_interval=0`）；CUDA/CPU 保留显式 PCG/FMG 选择。

## 性能

在 RTX 2080 Ti、Xeon Gold 6230、Python 3.11、8 线程上，使用同一组 163,842 顶点 fsaverage 输入，PCG 固定 40 次。以下为三次交错运行的中位数，计时从新进程启动到退出，包含输入读取、验证、已有缓存加载、计算和压缩写出。每个数值对应单份表面。

| 完整命令 | v0.2.0 | v0.3.0 | 提速 |
| --- | ---: | ---: | ---: |
| Surf2Sphere | 7.968 s | 2.612 s | 3.05× |
| SurfWarp | 20.479 s | 4.983 s | 4.11× |

双半球测试使用重采样后的公共 fsaverage 表面：61,442 顶点生成球面，升采样到 245,762 顶点后配准。输入和固定拓扑预先加载，左右半球串行，四个压缩 GIFTI 全部写完为 **9.731 秒**；其中内部生成与配准计时相加为 **5.771 秒**。

输入摘要、参数、首次调用、重复计时和数值对比见 [v0.3.0 基准记录](benchmark-v0.3.0.json)。单份表面的 CLI 与双半球 CAT 流程计时范围不同，分别报告。

## 数值检查

Linux 回归测试结果为 **60 项通过、1 项跳过**，覆盖 CPU 和 CUDA 计算。跳过项需要 macOS 和 Metal 环境。CPU CI 矩阵配置覆盖 Linux、Windows 和 macOS。

新版 FP32 路径保留参数和拓扑，但浮点结果有所变化。在公共 fsaverage 测试中，相对 v0.2.0，生成球面最大角差约 0.00023°，配准球面约 0.094°。这些是该输入的输出差异，不作为解剖精度指标。拓扑缓存改动单独检查坐标与三角形数组是否逐元素相同。

`CAT_SURFACE_CUDA_OPTIMIZED=0` 可显式选择原 CUDA 数值路径进行诊断，运行失败不会自动切换。独立包保留 v0.2.0 的参数校验、GIFTI 检查以及固定迭代 PCG 的零残差处理。

源码版本与文件摘要见[源码清单](../upstream/SOURCE_MANIFEST.json)。
