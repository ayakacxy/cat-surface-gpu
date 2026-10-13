<div align="center">

# CAT-Surface GPU

面向 `CAT_Surf2Sphere` 与 `CAT_SurfWarp` 曲面配准的 GPU 加速实现，使用 Python 和 CuPy。

[![许可证：GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![正式版本](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![CI](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![已测试 CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](docs/RELEASE_v0.2.0.zh-CN.md)
[![Linux x86-64](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](docs/IMPLEMENTATION.zh-CN.md)

[English](README.md) · [文档目录](docs/README.zh-CN.md) · [详细性能报告](docs/BENCHMARKS_v0.2.0.zh-CN.md) · [源码构建](docs/BUILDING.zh-CN.md) · [上游致谢](THIRD_PARTY_NOTICES.zh-CN.md)

⚡ **GPU 加速** · 🧠 **曲面配准** · 🧪 **CUDA 实测** · 📦 **Python 软件包**

</div>

本仓库提供 CAT-Surface/SimNIBS 皮层曲面配准中的两个组件：

- `CAT_Surf2Sphere`：皮层曲面膨胀与球面投影。
- `CAT_SurfWarp`：深度特征、初始旋转、球面重采样、DARTEL 配准、平均以及配准球面输出。

**0.2.0** 采用原生 Python 实现，使用 NumPy/Numba 完成 CPU 计算，使用 CuPy 执行 CUDA kernel。这里的“原生 Python”是指不再调用 CAT 外部程序，也不加载本项目专用的 C/Cython 扩展；Numba 和 CuPy 仍会在运行时编译计算内核，完整流程需要 NVIDIA GPU。PyTorch 和 Triton 已不再是依赖。

## ⚡ 验证与性能概览

在 NVIDIA GeForce RTX 2080 Ti、Python 3.11 和 CUDA 12.6 环境测量，每份 v0.2.0 输入包含 163,842 个顶点、327,680 个三角形：

| 组件 | CPU 中位数 | GPU 中位数 | CPU/GPU 耗时比 | 最大逐坐标差 |
| --- | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | 216.722 s | 10.340 s | 20.96x | 0.057757 |
| `CAT_SurfWarp`，PCG（默认） | 182.182 s | 32.550 s | 5.60x | 0.166157 |
| `CAT_SurfWarp`，FMG | 182.182 s | 164.697 s | 1.11x | 0.165802 |

中位数来自三次完整命令运行，每次使用新进程，磁盘缓存已建立；计时包括启动、输入输出和计算。CPU 参考是使用 FMG 的 CAT-Surface `b0c2aa23`。默认 GPU PCG 的比较包含求解器差异，输出并非逐元素相同；误差沿用 GIFTI 坐标单位。耗时范围、首次运行成本、全部数值差异及复现命令见 [v0.2.0 基准](docs/BENCHMARKS_v0.2.0.zh-CN.md)。

本版还通过了以下检查：

| 检查项 | 结果 |
| --- | --- |
| 测试套件 | **57 项通过**，包含 8 项 CUDA 测试 |
| 安装后的命令 | 完成完整 fsaverage 模板的球面生成，并将生成球面传入后续配准 |
| 输入网格 | 每个输入 163,842 个顶点、327,680 个三角面 |
| 输出检查 | 坐标均为有限值，三角面数组保持不变 |

这些检查确认了发布包可以完成相应流程。详情见 [v0.2.0 版本说明](docs/RELEASE_v0.2.0.zh-CN.md)和[验证记录](docs/validation-v0.2.0.json)。

以下是 **v0.1.0** 的历史数据，该版使用 PyTorch/Triton 和 CPU 辅助程序：

| 组件 | v0.1.0 输入与参数 | CPU 参考程序 | GPU | 加速比 |
| --- | --- | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | LH，61,442 点，`stop_at=10` | 54.84 s | 10.8119 s | 5.07x |
| `CAT_SurfWarp` | LH，`steps=2`、`runs=2`、`avg` | 119.9142 s | 45.0848 s | 2.66x |
| `CAT_SurfWarp` | RH，参数同上 | 119.4056 s | 46.9862 s | 2.54x |

同一环境下，v0.1.0 双半球 CAT 配准流程实测为 **55.2573 秒**，不包含完整 CHARM 流程。硬件、参数、计时范围和误差数据保留在 [历史性能记录](docs/BENCHMARKS.zh-CN.md) 中；这些数字不能用来代表 v0.2.0。

## 🧩 环境要求

- Linux x86-64，为当前测试平台
- NVIDIA GPU，以及与所安装 CuPy 包兼容的驱动
- Python 3.11 或更新版本
- NumPy、SciPy、Numba 和 nibabel
- 完整 CUDA 流程所需的 CuPy

`cuda` 可选依赖安装面向 CUDA 12.x 的 `cupy-cuda12x`；其他 CUDA 环境需安装对应的 CuPy 包。wheel 使用 `py3-none-any` 标记，表示本项目没有随包编译扩展，GPU 计算仍需满足以上运行条件。

<a id="install"></a>

## 🐍 Conda 环境安装

克隆仓库，在项目内设置缓存目录，再创建环境：

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
git checkout v0.2.0

mkdir -p .work/setup/{tmp,cache/pip,cache/conda,cache/numba,cache/cupy}
export TMPDIR="$PWD/.work/setup/tmp" TMP="$PWD/.work/setup/tmp" TEMP="$PWD/.work/setup/tmp"
export XDG_CACHE_HOME="$PWD/.work/setup/cache"
export PIP_CACHE_DIR="$PWD/.work/setup/cache/pip"
export CONDA_PKGS_DIRS="$PWD/.work/setup/cache/conda"
export NUMBA_CACHE_DIR="$PWD/.work/setup/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/setup/cache/cupy"

conda env create -f environment.yml --prefix "$PWD/.work/setup/env"
conda activate "$PWD/.work/setup/env"
python -m pip install '.[cuda,test]'
```

已有 Python 环境时可以直接使用相同的 `pip` 命令安装。wheel 和源码包的构建方法见 [构建说明](docs/BUILDING.zh-CN.md)。

检查环境并运行测试：

```bash
python - <<'PY'
import cupy

print("CuPy:", cupy.__version__)
print("CUDA runtime:", cupy.cuda.runtime.runtimeGetVersion())
print("CUDA devices:", cupy.cuda.runtime.getDeviceCount())
PY

python -m pytest -q --basetemp .work/setup/tmp/pytest
```

没有 GPU 时可以运行 CPU 测试，CUDA 测试会跳过；跳过不表示 GPU 结果已经验证。

<a id="run"></a>

## 🌐 运行 `CAT_Surf2Sphere`

输入和输出均为 GIFTI 曲面。输入应为适合球面映射的闭合三角网格，个体与模板文件需自行提供。

先在项目内设置工作目录和计算内核缓存目录：

```bash
mkdir -p .work/example/{tmp,cache/numba,cache/cupy,output}
export TMPDIR="$PWD/.work/example/tmp" TMP="$PWD/.work/example/tmp" TEMP="$PWD/.work/example/tmp"
export XDG_CACHE_HOME="$PWD/.work/example/cache"
export NUMBA_CACHE_DIR="$PWD/.work/example/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/example/cache/cupy"
```

生成球面：

```bash
cat-surf2sphere-gpu \
  --input-surface /path/to/white.gii \
  --output-surface .work/example/output/sphere.gii \
  --stop-at 10 \
  --device cuda \
  --threads 8
```

默认 `stop_at=10` 保留 4,999 次面积平滑。程序会显示阶段进度，首次运行还包括内核编译时间。

## 🧭 运行 `CAT_SurfWarp`

将球面配准到模板：

```bash
cat-surfwarp-gpu \
  --source-surface /path/to/white.gii \
  --source-sphere .work/example/output/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --output .work/example/output/sphere.reg.gii \
  --steps 2 \
  --runs 2 \
  --average \
  --solver-backend pcg \
  --solver-iterations 40 \
  --device cuda \
  --threads 8
```

每个解剖曲面及其对应球面必须具有相同的顶点顺序和三角面；个体与模板之间可以采用不同网格。两个命令在 CUDA 不可用时均直接报错，完整参数见 `--help`。

## 📦 Python 软件包

| 版本 | 实现方式 | CAT-Surface 参考版本 |
| --- | --- | --- |
| [v0.2.0](docs/RELEASE_v0.2.0.zh-CN.md) | Python、NumPy/Numba 和 CuPy，不附带 CAT 可执行文件 | [`b0c2aa23`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d) |
| [v0.1.0](docs/RELEASE_v0.1.0.zh-CN.md) | PyTorch/Triton，配合随包发布的 CPU 参考程序和辅助程序 | [`628b6851`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) |

v0.2.0 wheel 包含 Python 实现和内嵌 CUDA 内核源码，不包含解剖数据或模板。发行文件另提供项目源码包、固定版本的 CAT-Surface 源码包、依赖清单和校验文件。

两个主要命令名称保留，后端选项和 Python 接口已有调整。从 v0.1.0 升级时请参考[迁移说明](docs/MIGRATION.zh-CN.md)。

## 🎯 数值检查与 Python 接口

- 三角面数组保留源网格的索引。
- 默认配准使用双精度 DARTEL，每次线性求解执行 40 次 PCG 迭代。
- PCG 与上游 FMG 的计算过程不同，比较结果时需记录求解器和参考版本。
- 当前配准支持使用 type-5 特征的一步或两步流程，具体设置见[实现说明](docs/IMPLEMENTATION.zh-CN.md)。
- v0.1.0 使用的误差阈值不能直接套用到 v0.2.0。

安装后的公共 Python 命名空间仍为 `cat_surface_gpu`：

```python
from cat_surface_gpu import (
    GiftiSurface,
    Surf2SphereParameters,
    load_gifti_surface,
    save_gifti_surface,
    surf_to_sphere_cupy,
)

surface = load_gifti_surface("white.gii")
sphere_vertices, timings = surf_to_sphere_cupy(
    surface.vertices, surface.faces, Surf2SphereParameters(stop_at=10)
)
save_gifti_surface(
    GiftiSurface(sphere_vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

配准接口为 `SurfWarpParameters` 和 `surfwarp_cupy`，完整调用示例及输入要求见[实现说明](docs/IMPLEMENTATION.zh-CN.md)。

## 🗂️ 仓库结构

```text
src/cat_surface_gpu/          Python 接口、命令、几何计算和 CUDA 内核
tools/                        合成 GIFTI 测试数据生成器
scripts/                      构建与发行检查工具
tests/                        CPU 和 CUDA 测试
upstream/                     源码记录与上游许可
docs/                         使用说明、验证、性能记录与版本说明
```

## 🧪 合成测试数据

仓库不发布个体解剖数据。可以生成可复现、不含解剖信息的 GIFTI 文件，用于文件读写和安装检查：

```bash
python tools/generate_synthetic_fixture.py --output .work/example/smoke
python scripts/verify_release.py
```

该小型数据用于结构检查，不能证明解剖配准精度，也不能替代 v0.2.0 已记录的完整 fsaverage 流程检查。

## 🛠️ 开发与发行检查

- [构建说明](docs/BUILDING.zh-CN.md)：wheel、源码包和发行文件的构建方法。
- [复现说明](docs/REPRODUCIBILITY.zh-CN.md)：数值比较与计时方法。
- [贡献指南](docs/CONTRIBUTING.zh-CN.md)：开发要求与验证记录。
- [安全说明](docs/SECURITY.zh-CN.md)：漏洞报告方式。
- [更新日志](docs/CHANGELOG.zh-CN.md)：各版本的主要变化。

## 🙏 上游致谢和许可证

本项目基于 Christian Gaser 开发的 [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface) 及 [SimNIBS](https://github.com/simnibs/simnibs) 使用的 CAT 曲面配准流程。DARTEL 源于 John Ashburner 的工作，具体来源、固定版本和致谢见[第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)。

本项目代码采用 [GPL-3.0-or-later](LICENSE)，上游组件保留各自许可和版权说明。
