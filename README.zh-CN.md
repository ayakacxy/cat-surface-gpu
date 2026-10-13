# CAT-Surface GPU

CAT-Surface 球面膨胀与皮层曲面配准的 Python 实现，使用 CuPy CUDA kernel 加速。

[![许可证：GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![正式版本](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)

[English](README.md) · [文档目录](docs/README.md) · [版本说明](docs/RELEASE_v0.2.0.md) · [上游致谢](THIRD_PARTY_NOTICES.md)

**0.2.0** 将 FlashCHARM 中的 CAT 实现移植为独立软件包，包含：

- `CAT_Surf2Sphere`：皮层曲面膨胀与球面投影。
- `CAT_SurfWarp`：深度特征、初始旋转、球面重采样、DARTEL 配准、平均以及配准球面输出。

本版使用 NumPy/Numba 完成 CPU 计算，使用 CuPy 执行 CUDA kernel。这里的“原生 Python 实现”是指不再调用 CAT 外部程序，也不加载本项目专用的 C/Cython 扩展；wheel 中发布的是 Python 源码。Numba 和 CuPy 仍会在运行时编译计算内核，完整流程需要 NVIDIA GPU。PyTorch 和 Triton 已不再是依赖。

## 版本区别

| 版本 | 实现方式 | CAT-Surface 参考版本 |
| --- | --- | --- |
| [0.2.0](docs/RELEASE_v0.2.0.md) | Python、NumPy/Numba 和 CuPy，不附带 CAT 可执行文件 | [`b0c2aa23`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d) |
| [0.1.0](docs/RELEASE_v0.1.0.md) | PyTorch/Triton 加速，配合随包发布的 CPU 参考程序和辅助程序 | [`628b6851`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) |

v0.1.0 的性能与误差数据保留在[历史测试记录](docs/BENCHMARKS.md)中，不能用来代表 v0.2.0。新版同时更换了上游参考版本和默认线性求解器，比较结果时应使用对应版本的参考实现。升级方法见[迁移说明](docs/MIGRATION.md)。

## 安装

需要 Python 3.11 或更新版本，以及与 CuPy 匹配的 NVIDIA 驱动。当前测试平台为 Linux。`cuda` 可选依赖安装面向 CUDA 12.x 的 `cupy-cuda12x`；使用其他 CUDA 环境时，请安装对应的 CuPy 包。

从源码安装：

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
git checkout v0.2.0

mkdir -p .work/setup/{tmp,cache/pip,cache/numba,cache/cupy}
export TMPDIR="$PWD/.work/setup/tmp" TMP="$PWD/.work/setup/tmp" TEMP="$PWD/.work/setup/tmp"
export XDG_CACHE_HOME="$PWD/.work/setup/cache"
export PIP_CACHE_DIR="$PWD/.work/setup/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/setup/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/setup/cache/cupy"

python -m pip install '.[cuda]'
python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())'
```

仓库也提供 Conda 用户可用的[环境文件](environment.yml)。wheel 和源码包的构建方法见[构建说明](docs/BUILDING.md)。

## 运行

输入和输出均为 GIFTI 曲面。每个解剖曲面及其对应球面必须具有相同的顶点顺序和三角面；个体与模板之间可以采用不同网格。个体与模板文件需自行提供。

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
  --stop-at 10 --device cuda --threads 8
```

配准到模板：

```bash
cat-surfwarp-gpu \
  --source-surface /path/to/white.gii \
  --source-sphere .work/example/output/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --output .work/example/output/sphere.reg.gii \
  --steps 2 --runs 2 --average \
  --solver-backend pcg --solver-iterations 40 \
  --device cuda --threads 8
```

命令会显示阶段进度。CUDA 不可用时直接报错，首次运行还包括内核编译时间。完整参数见 `--help`；支持范围和与 CAT-Surface 的数值差异见[实现说明](docs/IMPLEMENTATION.md)。

## Python 接口

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

配准接口为 `SurfWarpParameters` 和 `surfwarp_cupy`，完整示例见[实现说明](docs/IMPLEMENTATION.md)。

## 测试与开发

仓库提供合成曲面和 CPU/CUDA 测试。跳过 CUDA 测试不能证明 GPU 结果正确。比较数值时需使用相同输入、参数和上游版本，逐项检查三角面，并报告顶点坐标差异。具体方法见[复现说明](docs/REPRODUCIBILITY.md)和[贡献指南](docs/CONTRIBUTING.md)。

## 致谢和许可证

本版移植自 FlashCHARM 的 CAT 模块，基于 Christian Gaser 开发的 [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface) 及 [SimNIBS](https://github.com/simnibs/simnibs) 使用的曲面配准流程。DARTEL 源于 John Ashburner 的工作。

本项目代码采用 [GPL-3.0-or-later](LICENSE)，上游组件保留各自许可和版权说明。来源、固定版本及致谢见[第三方说明](THIRD_PARTY_NOTICES.md)。
