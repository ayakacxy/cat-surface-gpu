# 第三方说明

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

## CAT-Surface

本项目算法源于耶拿大学医院 Christian Gaser 开发的 [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface)。

| 本项目版本 | 固定上游提交 |
| --- | --- |
| v0.3.0 / v0.2.0 | [`b0c2aa2395ea4957a331b9444fb54c7716b6572d`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d) |
| v0.1.0 | [`628b6851d8638f3ab773cd25c0ec406d0ec61ede`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) |

相关参考源码包括 `Progs/CAT_Surf2Sphere.c`、`Progs/CAT_SurfWarp.c` 和 `Lib/CAT_SurfWarpDartel.c`。DARTEL 实现源于 John Ashburner 在 SPM 中的工作，并按适用的 GPL 条款随 CAT-Surface 分发。

CAT-Surface 原始代码采用 GPL-2.0-or-later 许可，也可由版权持有人另行提供商业许可。保留的原始许可说明位于 [upstream/CAT-Surface-LICENSE.txt](upstream/CAT-Surface-LICENSE.txt)。上游源码中的第三方组件保留各自许可。

v0.2.0 和 v0.3.0 包含 Python 和内嵌 GPU 内核源码，不包含 CAT-Surface 可执行文件、原生辅助程序或完整上游源码树。发行文件单独提供固定版本的上游源码包，用于参考和来源说明。源码修订版本与文件摘要见 [UPSTREAM.lock](UPSTREAM.lock) 和[源码清单](upstream/SOURCE_MANIFEST.json)。

v0.1.0 分发了四个 Linux 可执行文件，使用 `-O2 -fPIC` 从提交 `628b6851` 构建。其辅助程序源码和对应上游源码仍可从 [v0.1.0 发行页面](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0)获取。

v0.3.0 包含 CPU、CUDA 和 Metal 实现。源码来源及改编记录保存在 [upstream/](upstream/)。Metal 使用外部依赖 [MLX](https://github.com/ml-explore/mlx)。

## SimNIBS

[SimNIBS](https://github.com/simnibs/simnibs) 提供头模型构建流程，其中使用 `CAT_Surf2Sphere` 和 `CAT_WarpSurf` 进行皮层曲面配准。本软件包不重新分发 SimNIBS，也不修改 SimNIBS 安装。

## Python 依赖

v0.2.0 和 v0.3.0 使用 [NumPy](https://github.com/numpy/numpy)、[SciPy](https://github.com/scipy/scipy)、[Numba](https://github.com/numba/numba)、[nibabel](https://github.com/nipy/nibabel) 和 [CuPy](https://github.com/cupy/cupy) 作为外部依赖。这些项目保留各自许可。

v0.1.0 使用 [PyTorch](https://github.com/pytorch/pytorch) 和 [Triton](https://github.com/triton-lang/triton)，v0.2.0 和 v0.3.0 不再需要这两项依赖。

本项目代码采用 [GPL-3.0-or-later](LICENSE) 许可，并保留上游版权与许可说明。
