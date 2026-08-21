<div align="center">

# CAT-Surface GPU

面向 `CAT_Surf2Sphere` 与 `CAT_SurfWarp` 兼容曲面配准链路的 GPU 加速实现。

[![许可证：GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![正式版本](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![已测试 CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](BENCHMARKS.md)
[![Linux x86-64](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](bin/linux-x86_64)

[English](README.md) · [详细性能报告](BENCHMARKS.md) · [上游致谢](THIRD_PARTY_NOTICES.md)

⚡ **GPU 加速** · 🧠 **曲面配准** · 🧪 **Reference 验证** · 📦 **开箱即用二进制**

</div>

本仓库开源 CAT-Surface/SimNIBS 曲面配准中的两个 GPU 加速组件：

- `CAT_Surf2Sphere`：使用保持官方顶点依赖顺序的 Triton kernel，加速面积保持球面膨胀循环；迭代次数不变。
- `CAT_SurfWarp`：使用 PyTorch/Triton 加速初始旋转、曲面重采样、DARTEL、`-avg` 和最终 warp；对应 SimNIBS 曲面配准流程中 `CAT_WarpSurf` 承担的功能。

CPU reference 始终显式保留。请求 CUDA 但 GPU 不可用，或请求 Triton 但 kernel 失败时，程序会直接报错，不会静默回退到 CPU。

## ⚡ 性能概览

测试平台为 NVIDIA GeForce RTX 2080 Ti、Python 3.11、PyTorch `2.6.0+cu126` 和 Triton `3.2.0`。所有 A/B 均使用同一输入、参数、输出拓扑和显式顶点误差合同。

| 组件 | 输入与参数 | CPU reference | GPU | 加速比 | 最大顶点误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | LH，61,442 点，`stop_at=10` | 54.84 s | 10.8119 s | 5.07x | 1.14e-5 |
| `CAT_SurfWarp` | LH，`steps=2`、`runs=2`、`avg` | 119.9142 s | 45.0848 s | 2.66x | 2.85e-5 |
| `CAT_SurfWarp` | RH，同上 | 119.4056 s | 46.9862 s | 2.54x | 3.41e-4 |

按 SimNIBS 默认方式同时运行左右半球时，完整 CAT 配准链路实测为：

| 阶段 | 双侧墙钟时间 |
| --- | ---: |
| `CAT_Surf2Sphere` | 15.7609 s |
| 官方拓扑上采样 | 4.5208 s |
| `CAT_SurfWarp` | 34.9756 s |
| **CAT 配准链路总计** | **55.2573 s** |

最终 faces 逐元素一致；LH/RH `sphere.reg` 最大顶点误差分别为 `1.38e-6` 和 `1.81e-4`，均低于 `1e-3`。完整硬件、形状、参数、计时边界和阶段热点见 [BENCHMARKS.md](BENCHMARKS.md)。

## 🐍 Conda 环境安装

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu

conda env create -f environment.yml
conda activate cat-surface-gpu
```

安装与本仓库实测一致的 CUDA 12.6 PyTorch，并安装项目：

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[cuda,test]'
```

检查环境和测试：

```bash
python - <<'PY'
import torch
import triton

print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pytest -q
```

没有 GPU 时可以运行 CPU 单元测试，CUDA 测试会跳过；不能据此声称 GPU 性能或正确性已经验证。

## 🌐 运行 `CAT_Surf2Sphere`

```bash
python tools/run_cat_surf2sphere_gpu.py \
  --reference-cli bin/linux-x86_64/CAT_Surf2Sphere \
  --input-surface /path/to/coarse.white.gii \
  --output-surface /path/to/sphere.gii \
  --stop-at 10 \
  --device cuda \
  --dtype float32 \
  --kernel triton \
  --preprocess-kernel cpu \
  --areal-schedule ordered \
  --areal-arithmetic cat
```

严格 CPU/GPU A/B：

```bash
python tools/benchmark_cat_surf2sphere_gpu.py \
  --reference-cli bin/linux-x86_64/CAT_Surf2Sphere \
  --input-surface /path/to/coarse.white.gii \
  --reference-output /tmp/sphere.cpu.gii \
  --gpu-output /tmp/sphere.gpu.gii \
  --device cuda \
  --kernel triton \
  --areal-schedule ordered \
  --areal-arithmetic cat \
  --max-vertex-error 1e-3 \
  --max-mean-error 1e-3 \
  --max-p99-error 1e-3
```

## 🧭 运行 `CAT_SurfWarp`

```bash
python tools/run_cat_surface_gpu_pipeline.py \
  --source-surface /path/to/white.gii \
  --source-sphere /path/to/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --output /path/to/sphere.reg.gii \
  --rotation-depth-probe bin/linux-x86_64/cat_surface_rotation_depth \
  --rotation-feature-backend cuda-official-depth \
  --stencil-builder bin/linux-x86_64/cat_surface_stencil_builder \
  --rotated-stencil-builder bin/linux-x86_64/cat_surface_stencil_builder \
  --device cuda \
  --kernel triton \
  --dartel-dtype float64 \
  --squaring-kernel triton \
  --steps 2 \
  --runs 2 \
  --avg \
  --cuda-graph
```

使用 `tools/benchmark_cat_surface_end_to_end.py` 可以在同一输入上，将 GPU pipeline 与仓库内 `bin/linux-x86_64/CAT_SurfWarp` CPU reference 做完整 A/B。

## 📦 仓库内二进制

`bin/linux-x86_64/` 包含四个已经剥离调试信息的 x86-64 ELF：

| 文件 | 用途 |
| --- | --- |
| `CAT_Surf2Sphere` | 球面膨胀混合后端使用的上游 CPU reference |
| `CAT_SurfWarp` | 完整 warp A/B 使用的上游 CPU reference |
| `cat_surface_rotation_depth` | 导出与上游一致的初始旋转 raw depth 特征 |
| `cat_surface_stencil_builder` | 构建确定性重采样 stencil，使用 32 个 CPU 线程编译 |

四个程序均基于 [CAT-Surface commit `628b6851`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede)，使用 `-O2 -fPIC` 构建；运行时只链接 glibc、libm 和 pthread，要求 glibc 2.29 或更新版本。SHA-256 见 [`bin/linux-x86_64/SHA256SUMS`](bin/linux-x86_64/SHA256SUMS)，两个 helper 的源码位于 [`tools/cat_surface_c/`](tools/cat_surface_c/)。

## 🎯 数值合同

- faces/拓扑必须与 CPU reference 完全一致；
- `CAT_Surf2Sphere stop_at=10` 保持官方 4,999 次 sweep；
- 推荐 warp 路径使用 FP64 DARTEL，保留 `steps/runs/loop/cycles/nit/its/code/avg` 语义；
- 最终顶点默认验收为 `max/mean/p99 <= 1e-3`；
- all-GPU feature 等实验路径继续保持显式 opt-in，不进入上述推荐命令。

安装后的公共 Python 命名空间为 `cat_surface_gpu`：

```python
from cat_surface_gpu import run_cat_surf2sphere_gpu, run_cat_surface_gpu_pipeline
```

## 🙏 上游致谢和许可证

本项目建立在 Christian Gaser 开发的 [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface) 以及 [SimNIBS](https://github.com/simnibs/simnibs) 使用的 CAT 曲面配准流程之上。CAT-Surface 的 DARTEL、曲面 I/O 和网格处理实现是本项目能够完成等价 GPU 加速的基础。完整来源和许可证说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本仓库新增代码采用 [GPL-3.0-or-later](LICENSE)。上游派生二进制和外部组件继续遵守各自许可证。
