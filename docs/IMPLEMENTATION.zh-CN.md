# 实现说明与 Python 接口

[English](IMPLEMENTATION.md) · [简体中文](IMPLEMENTATION.zh-CN.md)

v0.3.0 在 `cat_surface_gpu` 中实现球面膨胀和皮层曲面配准，算法参考 CAT-Surface 固定提交 [`b0c2aa2395ea4957a331b9444fb54c7716b6572d`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d)。

## 运行方式

NumPy 和 Numba 负责 CPU 上的几何、拓扑及辅助计算。CuPy RawKernel/RawModule 内核负责球面膨胀和配准等 CUDA 计算。软件包独立运行，不调用 CAT 命令行程序，也不加载本项目专用的原生扩展。CUDA 内核源码嵌在 Python 模块中。

完整流程可显式使用 CPU、CUDA 或 Metal。CPU 使用 NumPy/Numba，CUDA 使用 CuPy，Apple Silicon 上的 Metal 使用 MLX。首次调用包含内核编译。

## 曲面和坐标

- 每个输入 GIFTI 文件必须包含一个顶点坐标数组和一个三角面数组。
- 顶点数组形状为 `(N, 3)`；三角面数组形状为 `(M, 3)`，索引从零开始。
- 源解剖曲面与对应球面必须具有相同的三角面和顶点顺序，模板曲面与对应球面也一样。
- 源曲面与模板可以具有不同的顶点数和三角面数。
- 曲面必须是适合球面映射的闭合三角网格，每个三角面的面积都必须非零。本软件包不负责从 MRI 重建皮层，也不修复任意输入的拓扑。
- 输出顶点采用 float32，三角面采用 int32；配准球面保留源曲面的三角面数组。
- 写出 GIFTI 时，如提供参考图像，会保留图像元数据、标签表，以及顶点坐标和三角面数组的元数据、坐标系变换。额外的数据数组，例如逐顶点测量值，不会复制。

程序会检查输入和输出坐标是否为有限值，以及三角面是否存在零面积。非常粗糙的合成网格可能在膨胀过程中退化，程序会拒绝这类结果。

计算使用输入中提供的坐标，不会推断 MRI 仿射变换或转换解剖坐标系。

## 球面膨胀

```python
from cat_surface_gpu import (
    GiftiSurface,
    Surf2SphereParameters,
    load_gifti_surface,
    save_gifti_surface,
    surf_to_sphere,
)

surface = load_gifti_surface("white.gii")
vertices, timings = surf_to_sphere(
    surface.vertices,
    surface.faces,
    parameters=Surf2SphereParameters(stop_at=10),
    device="cuda",
)
save_gifti_surface(
    GiftiSurface(vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

`surf_to_sphere(vertices, faces, parameters=None, topology=None, *, device="cuda", progress=None)` 返回新顶点和计时字典。`Surf2SphereParameters` 默认设置为 `stop_at=10`，该参数控制上游膨胀阶段和面积平滑的迭代安排。

`device` 支持 `cpu`、`cuda`、`metal` 和 `auto`。旧接口 `surf_to_sphere_cupy(...)` 保留为 CUDA 入口。

## 配准

```python
from cat_surface_gpu import (
    GiftiSurface,
    SurfWarpParameters,
    load_gifti_surface,
    save_gifti_surface,
    surfwarp_cupy,
)

source = load_gifti_surface("white.gii")
source_sphere = load_gifti_surface("sphere.gii")
target = load_gifti_surface("template.white.gii")
target_sphere = load_gifti_surface("template.sphere.gii")

result = surfwarp_cupy(
    source.vertices,
    source_sphere.vertices,
    source.faces,
    target.vertices,
    target_sphere.vertices,
    target.faces,
    parameters=SurfWarpParameters(steps=2, runs=2, average=True),
    solver_iterations=40,
    solver_backend="pcg",
    backend="cupy",
)
save_gifti_surface(
    GiftiSurface(result.vertices, source.faces, source_sphere.reference),
    "sphere.reg.gii",
)
```

返回结果包含 `vertices`、`velocity`、`deformation`、`timings` 和 `objectives`。两个主要 Python 函数均接受可选的 `progress` 回调，该回调接收阶段名称字符串。Python 调用要求传入已经配对的曲面和球面数组；命令行工具还会检查每一对的三角面是否一致。

`backend` 支持 `cpu`、`cupy` 和 `metal`。Metal 使用固定迭代 PCG，要求 `pcg_check_interval=0`；CPU 和 CUDA 支持 PCG 与 FMG。

| 设置 | 默认值 |
| --- | --- |
| 步数 / 运行次数 | `2 / 2` |
| 初始旋转 / 平均 | 开启 / 开启 |
| DARTEL 循环数 | `6` |
| 球面映射尺寸 | `512 x 256` |
| 每个 DARTEL 层级的迭代次数 | `3` |
| 线性求解器 | `pcg`，`40` 次迭代 |
| PCG 收敛检查间隔 | `0` |
| 旋转分块点数 | `16384` |
| 旋转细化 | 开启，最多 `500` 次迭代 |
| 命令行 CPU 线程数 | `8`，支持范围为 `1–8` |

当前配准支持使用 type-5 深度特征的一步或两步流程，采用膜正则化（`regularization_type=1`），并使用 `code=1`、`multires_levels=0`、`lambda_value=0`、`surface_fwhm=0`、`resampled_triangles=81920` 和 `cycles=3`。超出已实现范围的设置会被拒绝，本版未实现上游 CAT 的全部命令行模式。

默认 PCG 求解器的计算过程与上游全多重网格（FMG）不同。`pcg_check_interval=0` 只关闭中途的 CPU 收敛检查，不会减少所请求的 40 次迭代。`solver_iterations` 仅用于 PCG。`solver_backend="fmg"` 可选择已有的 FMG 实现，但不能仅根据求解器名称就认定其与 CAT-Surface 逐位一致。每次比较都应记录使用的求解器。

将 `rotation_point_chunk` 从默认的 `16384` 减小到例如 `12288`，可以缩小旋转计算的临时批次。显存占用和耗时取决于输入及 GPU。

## 命令行输出

两个命令均支持 `--device cpu`、`--device cuda`、`--device cuda:N`、`--device metal` 或 `--device auto`、`--threads` 和 `--quiet`。阶段进度写入标准错误，最终 JSON 摘要写入标准输出。完整选项列表见 `--help`。

[返回文档目录](README.zh-CN.md) · [返回项目首页](../README.zh-CN.md)
