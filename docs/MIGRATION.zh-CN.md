# 从 v0.1.0 升级到 v0.2.0

[English](MIGRATION.md) · [简体中文](MIGRATION.zh-CN.md)

0.2.0 使用 NumPy、Numba 和 CuPy 的原生 Python 实现，替换了旧版 PyTorch/Triton 实现及 CAT 辅助程序。如果需要保留 v0.1.0 进行比较，请将新版安装到独立环境。

## 依赖和文件

| v0.1.0 | v0.2.0 |
| --- | --- |
| PyTorch 和 Triton | NumPy、SciPy、Numba、nibabel 和 CuPy |
| 随包发布 CAT 参考程序和辅助程序 | 不附带 CAT 可执行文件或本项目专用的原生扩展 |
| Linux 平台 wheel | 仅含 Python 源码的 `py3-none-any` wheel，运行时仍需 CUDA |
| CAT-Surface `628b6851` | CAT-Surface `b0c2aa23` |

软件包独立运行，GIFTI 曲面和模板需自行提供。

## 命令

两个主要命令名称及其输入、输出文件参数保持不变：

- `cat-surf2sphere-gpu --input-surface ... --output-surface ...`
- `cat-surfwarp-gpu --source-surface ... --source-sphere ... --target-surface ... --target-sphere ... --output ...`

请移除 v0.1.0 的后端选项，包括 `--kernel triton`、`--dtype`、`--preprocess-kernel`、`--areal-schedule`、`--areal-arithmetic`、`--rotation-feature-backend`、`--stencil-threads`、`--dartel-dtype`、`--squaring-kernel` 和 `--cuda-graph`。v0.2.0 不再通过这些选项选择后端。CPU 计算使用 `--threads` 设置线程数，配准求解器使用 `--solver-backend` 和 `--solver-iterations` 设置。

`--average` 及其别名 `--avg` 用于开启平均计算，该功能默认已经开启；使用 `--no-average` 可关闭。初始旋转也默认开启，可使用 `--no-rotate` 关闭。

旧版独立旋转命令不再提供，旋转由 `cat-surfwarp-gpu` 完成。当前用法见 [README 示例](../README.zh-CN.md#run)。

## Python 接口

使用 `surf_to_sphere_cupy` 替代 `run_cat_surf2sphere_gpu`，使用 `surfwarp_cupy` 替代 `run_cat_surface_gpu_pipeline`。新函数接收顶点和三角面数组，通过 `Surf2SphereParameters` 或 `SurfWarpParameters` 设置参数，与旧版接收文件路径的函数不能直接互换。具体用法见[接口示例](IMPLEMENTATION.zh-CN.md)。

## 结果与性能

本版同时更换了上游源码版本和具体实现。v0.2.0 的 DARTEL 默认执行 40 次 PCG 迭代，官方参考实现使用 FMG。v0.1.0 的误差阈值和计时只适用于该版本；新版应与对应上游版本使用相同输入进行比较，再确定误差容限。

历史记录中的双半球 55.2573 秒属于 v0.1.0 的 CAT 配准流程。v0.2.0 的完整命令耗时和输出差异见[当前基准](BENCHMARKS_v0.2.0.zh-CN.md)。

[返回文档目录](README.zh-CN.md) · [返回项目首页](../README.zh-CN.md)
