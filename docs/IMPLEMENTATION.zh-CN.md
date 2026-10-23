# Python API 与实现

[English](IMPLEMENTATION.md) · [简体中文](IMPLEMENTATION.zh-CN.md)

`cat_surface_gpu` 提供基于数组的球面生成和曲面配准函数，使用 GIFTI 读写曲面，支持 CPU、NVIDIA CUDA 和 Apple Metal。安装方法与命令行示例见[项目首页](../README.zh-CN.md)。

## 曲面数据

曲面由形状为 `(N, 3)` 的顶点数组和形状为 `(M, 3)` 的整数三角面数组组成，顶点索引从 0 开始。输入应是适合球面映射的闭合三角网格，坐标必须为有限值，且不能有零面积三角形。

配准需要源和模板各自的解剖表面及对应球面。每对表面与球面的顶点顺序和三角面必须一致；源与模板之间可以使用不同的网格分辨率。输出球面保留源表面的顶点顺序和三角面。

`load_gifti_surface(path)` 读取一个点集数组和一个三角面数组，返回 `GiftiSurface`。`save_gifti_surface(surface, path)` 将顶点写为 float32、三角面写为 int32。提供参考 GIFTI 时，保存图像元数据、标签、数组元数据和坐标系变换，但不复制其他数据数组。计算直接使用输入坐标，不应用 MRI 仿射变换，也不转换坐标系。

## 球面生成

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

函数签名：

```text
surf_to_sphere(
    vertices, faces, parameters=None, topology=None,
    *, device="cuda", progress=None,
)
```

返回值为 `(vertices, timings)`。`stop_at=10` 执行完整膨胀流程；较小的值会提前结束，其中小于 5 的值生成中间表面。对同一网格连接关系反复计算时，可以通过 `topology` 传入预先构建的 `SurfaceTopology`。

`device` 接受 `cpu`、`cuda`、`metal` 或 `auto`。自动选择模式在支持的 macOS 系统上使用 Metal，在其他有可用 NVIDIA GPU 的系统上使用 CUDA，否则使用 CPU。显式选择的后端不可用时会报错。

`surf_to_sphere_cupy(...)` 是专用于 CUDA 的入口。在 Python 中选择某张 NVIDIA GPU 时，将 CUDA 函数调用放在 `cupy.cuda.Device(index)` 上下文中。

## 曲面配准

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
    parameters=SurfWarpParameters(),
    backend="cupy",
)
save_gifti_surface(
    GiftiSurface(result.vertices, source.faces, source_sphere.reference),
    "sphere.reg.gii",
)
```

`surfwarp_cupy` 虽然名称带有 `cupy`，但支持三个后端：CUDA 使用 `"cupy"`，另外两个是 `"cpu"` 和 `"metal"`。该函数不接受 `"auto"`；命令行支持 `--device auto`，会先选择设备，再调用对应后端。

`SurfWarpResult` 包含以下字段：

| 字段 | 内容 |
| --- | --- |
| `vertices` | 配准后的源球面；启用极点平均时包含平均结果 |
| `velocity` | 最后一次主配准的速度场 |
| `deformation` | 该次配准的形变场 |
| `timings` | 各阶段耗时和旋转设置 |
| `objectives` | 配准过程中记录的目标函数值 |

最终配准球面应使用 `vertices`。多次运行或启用极点平均时，`velocity` 和 `deformation` 不代表从原始输入球面到最终结果的完整变换。

球面生成和配准均接受 `progress` 回调，传入的参数是描述当前阶段的字符串。文件读写独立于数组 API，其耗时不计入函数返回的计时字典。

### 配准参数

主要参数由 [`SurfWarpParameters`](../src/cat_surface_gpu/parameters.py) 定义：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `steps` | `2` | type-5 深度特征的尺度数，可选 1 或 2 |
| `runs` | `2` | 连续配准次数 |
| `rotate` | `True` | 执行初始刚性旋转搜索 |
| `average` | `True` | 旋转极点后再配准，并对结果取平均 |
| `map_size` | `(512, 256)` | 球面映射图的宽和高 |
| `loop` | `6` | DARTEL 调度层数 |
| `iterations` | `3` | 每层 DARTEL 更新次数 |
| `curvature_fwhm` | `5.0` | 初始特征平滑尺度 |
| `mu` / `mu_rate` / `mu_change` | `0.125` / `1.25` / `4` | 膜正则化调度参数 |
| `lm_regularization` | `0.001` | Levenberg–Marquardt 正则化参数 |

`surfwarp_cupy` 的其他关键字参数控制求解器和旋转：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `solver_backend` | `"pcg"` | PCG 或 FMG；Metal 支持 PCG |
| `solver_iterations` | `40` | PCG 迭代上限；FMG 不使用此项 |
| `pcg_check_interval` | `0` | 收敛检查间隔；0 表示不进行中间检查 |
| `rotation_point_chunk` | `16384` | 每批旋转处理的点数 |
| `rotation_refine` | `True` | 细化初始旋转估计 |
| `rotation_max_iter` | `500` | 旋转细化的最大迭代次数 |

Metal 要求 `solver_backend="pcg"` 且 `pcg_check_interval=0`。减小 `rotation_point_chunk` 可以降低旋转阶段的临时内存用量，不会改变输入顶点数。

当前支持的模型采用 type-5 深度特征和膜正则化。以下 `SurfWarpParameters` 参数为固定值：`code=1`、`regularization_type=1`、`multires_levels=0`、`resampled_triangles=81920`、`lambda_value=0`、`surface_fwhm=0`、`cycles=3`。不支持的设置会抛出 `ValueError`。

## 实现概览

球面生成依次执行解剖表面平滑、球面投影和面积平滑。配准先估计初始旋转，再将深度特征重采样到球面映射图上，优化 DARTEL 速度场，最后将形变映射回源顶点。默认启用多次配准和极点平均。

NumPy 和 Numba 负责 CPU 几何与数值计算。CUDA 内核内嵌于 Python 模块，由 CuPy 编译；Metal 后端使用 MLX。默认 CUDA 和 Metal 配准路径中的特征求解与 DARTEL 更新使用 FP32，最终速度场指数映射使用 FP64。内核编译以及拓扑、模板缓存会影响首次运行耗时。

算法参考版本为 [CAT-Surface `b0c2aa23`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d)。参考实现使用 FMG，本项目默认使用 PCG。比较结果时，应同时记录求解器设置、后端和精度。比较方法见[复现指南](REPRODUCIBILITY.zh-CN.md)，实测结果见[性能测试](BENCHMARKS_v0.3.0.zh-CN.md)。

[文档目录](README.zh-CN.md) · [项目首页](../README.zh-CN.md)
