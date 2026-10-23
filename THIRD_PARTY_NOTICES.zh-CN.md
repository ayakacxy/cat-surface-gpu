# 第三方声明

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

CAT-Surface GPU 采用 [GPL-3.0-or-later](LICENSE) 许可。上游组件保留各自的版权和许可声明。

## CAT-Surface 与 DARTEL

球面生成和配准算法基于耶拿大学医院 Christian Gaser 开发的 [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface)。参考版本为 [`b0c2aa2395ea4957a331b9444fb54c7716b6572d`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d)，记录在 [UPSTREAM.lock](UPSTREAM.lock) 中。

参考源码包括 `Progs/CAT_Surf2Sphere.c`、`Progs/CAT_SurfWarp.c` 和 `Lib/CAT_SurfWarpDartel.c`。DARTEL 源于 John Ashburner 在 SPM 中的工作，按适用的 GPL 条款随 CAT-Surface 分发。

CAT-Surface 原始代码采用 GPL-2.0-or-later 许可，也可向版权持有人申请单独的商业许可。商业许可不涵盖第三方组件。原始声明及各组件的许可条款保留在 [upstream/CAT-Surface-LICENSE.txt](upstream/CAT-Surface-LICENSE.txt) 中。

软件包包含 Python 代码和内嵌 GPU 内核，不附带 CAT-Surface 可执行程序或完整上游源码树。固定版本的上游源码包作为独立发行附件提供。源码摘要和改编记录保存在 [upstream/](upstream/) 中，其中[源码清单](upstream/SOURCE_MANIFEST.json)记录了文件摘要。

## 外部依赖

以下库单独安装，并保留各自的许可：

| 库 | 用途 |
| --- | --- |
| [NumPy](https://github.com/numpy/numpy) | 数组运算 |
| [SciPy](https://github.com/scipy/scipy) | 科学计算与空间查询 |
| [Numba](https://github.com/numba/numba) | CPU 内核编译 |
| [nibabel](https://github.com/nipy/nibabel) | GIFTI 文件读写 |
| [CuPy](https://github.com/cupy/cupy) | NVIDIA CUDA 后端 |
| [MLX](https://github.com/ml-explore/mlx) | Apple Silicon Metal 后端 |

[SimNIBS](https://github.com/simnibs/simnibs) 在皮层表面配准流程中使用 CAT-Surface 工具。本软件包不重新分发 SimNIBS。

## 早期发行包

[v0.1.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0) 使用 PyTorch 和 Triton，附带四个从 CAT-Surface 提交 [`628b6851d8638f3ab773cd25c0ec406d0ec61ede`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) 以 `-O2 -fPIC` 构建的 Linux 可执行文件。对应的辅助程序和上游源码仍可在该发行页面下载。这些依赖和可执行文件不属于当前软件包。
