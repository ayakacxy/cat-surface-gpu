# CAT-Surface GPU

[English](README.md) · [简体中文](README.zh-CN.md)

[![Release](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![CI](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

使用 Python 进行皮层曲面球面化与配准，支持 CPU、CUDA 和 Apple Metal。

CAT-Surface GPU 读取 GIFTI 格式的皮层曲面，生成用于表面分析的球面。软件包提供两条命令：

- **`cat-surf2sphere-gpu`**：膨胀皮层曲面并将其投影到球面。
- **`cat-surfwarp-gpu`**：通过深度特征、初始旋转和 DARTEL，将球面配准到模板。

软件包独立运行，皮层曲面和模板由用户提供。MRI 分割和皮层重建不在本项目范围内。

<a id="install"></a>

## 安装

使用 Python 3.11 或更新版本，建议创建独立环境。克隆仓库后，按所需后端安装：

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
```

| 平台 | 后端 | 在仓库目录中安装 |
| --- | --- | --- |
| 配备 NVIDIA GPU 的 Linux 或 Windows | CUDA / CuPy | `python -m pip install ".[cuda]"` |
| Apple Silicon macOS | Metal / MLX | `python -m pip install ".[metal]"` |
| Linux、Windows 或 macOS | CPU / NumPy / Numba | `python -m pip install .` |

CUDA 可选依赖安装适用于 CUDA 12.x 的 CuPy，需要兼容的 NVIDIA 驱动。使用其他 CUDA 运行时版本时，请安装基础包及对应的 CuPy。Metal 需要原生 Apple Silicon Python 环境。

预构建 wheel 和源码包可从[发布页面](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)下载。wheel 安装、开发环境配置和源码构建方法见[构建与安装说明](docs/BUILDING.zh-CN.md)。

<a id="run"></a>

## 快速开始

以下示例使用 CUDA。使用其他后端时，将 `--device cuda` 替换为 `--device metal` 或 `--device cpu`。

### 生成球面

```bash
cat-surf2sphere-gpu --input-surface white.gii --output-surface sphere.gii --device cuda
```

`white.gii` 应为适合球面映射的闭合三角网格。输出保留输入的三角形索引。

### 配准到模板

```bash
cat-surfwarp-gpu --source-surface white.gii --source-sphere sphere.gii --target-surface template.white.gii --target-sphere template.sphere.gii --output sphere.reg.gii --device cuda
```

每份曲面及其对应球面必须具有相同的顶点顺序和三角形。被试与模板可以使用不同大小的网格。`sphere.reg.gii` 保存配准后的球面，拓扑与被试的输入曲面相同。

### 选择设备

| 选项 | 行为 |
| --- | --- |
| `--device cuda` | 使用默认 NVIDIA GPU，也是命令的默认设置。 |
| `--device cuda:1` | 使用编号为 1 的 NVIDIA GPU。 |
| `--device metal` | 使用 Apple Metal。 |
| `--device cpu` | 使用 CPU。 |
| `--device auto` | 优先使用可用的 GPU 后端，否则使用 CPU。 |

两条命令均支持 `--threads`（1–8，默认 8）和 `--quiet`。运行进度写入标准错误，最终 JSON 摘要写入标准输出。完整选项可通过 `--help` 查看。

首次运行包含内核编译。拓扑和目标缓存默认保存在工作目录下的 `.work/cat-surface/cache`，可通过 `CAT_SURFACE_CACHE_ROOT` 指定其他位置。

## Python 接口

```python
from cat_surface_gpu import GiftiSurface, load_gifti_surface, save_gifti_surface, surf_to_sphere

surface = load_gifti_surface("white.gii")
vertices, timings = surf_to_sphere(surface.vertices, surface.faces, device="cuda")
save_gifti_surface(
    GiftiSurface(vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

配准调用、参数、返回值和数值精度见[接口指南](docs/IMPLEMENTATION.zh-CN.md)。

## 性能

在 RTX 2080 Ti 和 Xeon Gold 6230 上，使用 8 个 CPU 线程及 163,842 顶点的 fsaverage 表面，完整命令耗时如下：

| 操作 | 耗时中位数 |
| --- | ---: |
| 球面生成 | 2.612 秒 |
| 球面配准 | 4.983 秒 |

每个数值对应单份表面，是已有缓存条件下三次新进程运行的中位数，包含启动、输入读取、计算和压缩写出。配准使用 40 次 PCG 迭代。输入详情、首次调用耗时和数值对比见[性能报告](docs/BENCHMARKS_v0.3.0.zh-CN.md)。

## 文档与支持

- [文档目录](docs/README.zh-CN.md)
- [接口与实现](docs/IMPLEMENTATION.zh-CN.md)
- [性能报告](docs/BENCHMARKS_v0.3.0.zh-CN.md)与[复现方法](docs/REPRODUCIBILITY.zh-CN.md)
- [迁移指南](docs/MIGRATION.zh-CN.md)与[更新记录](docs/CHANGELOG.zh-CN.md)
- [参与开发](docs/CONTRIBUTING.zh-CN.md)、[使用支持](docs/SUPPORT.zh-CN.md)与[安全报告](docs/SECURITY.zh-CN.md)

## 引用与许可

引用本软件时请使用 [CITATION.cff](CITATION.cff)。算法基于 Christian Gaser 开发的 [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface)，其中 DARTEL 源于 John Ashburner 的工作。致谢和源码参考见[第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)。

CAT-Surface GPU 采用 [GPL-3.0-or-later](LICENSE) 许可。
