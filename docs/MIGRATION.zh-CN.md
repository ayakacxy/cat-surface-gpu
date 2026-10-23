# 版本迁移指南

[English](MIGRATION.md) · [简体中文](MIGRATION.zh-CN.md)

首次安装请从[项目首页](../README.zh-CN.md#install)开始。本指南说明升级时需要调整的脚本和环境。

## 从 v0.2 升级到 v0.3

命令名称、文件参数和 CUDA 数组 API 保持兼容。默认设备仍为 `cuda`，已有 CUDA 命令不需要增加设备参数。

CPU 和 Apple Metal 现已支持完整流程：

| 后端 | 从源码目录安装 | CLI 选项 |
| --- | --- | --- |
| CPU | `python -m pip install .` | `--device cpu` |
| NVIDIA CUDA | `python -m pip install ".[cuda]"` | `--device cuda` 或 `--device cuda:N` |
| Apple Metal | `python -m pip install ".[metal]"` | `--device metal` |

`--device auto` 选择可用的 GPU，没有受支持的 GPU 时使用 CPU。显式选择的后端不可用时会报错。

### Python 调用

球面生成调用示例为 `surf_to_sphere(..., device="cuda")`，设备可选 `cpu`、`cuda`、`metal` 和 `auto`。已有 CUDA 代码可以继续调用 `surf_to_sphere_cupy(...)`。

配准调用示例为 `surfwarp_cupy(..., backend="cupy")`，后端可选 `cpu`、`cupy` 和 `metal`。CUDA 后端名称为 `cupy`，配准函数不接受 `auto`。Metal 要求 PCG 且 `pcg_check_interval=0`，默认设置已满足。完整示例和返回字段见 [Python API](IMPLEMENTATION.zh-CN.md)。

### 数值结果

默认 CUDA 路径的特征求解和 DARTEL 更新改用 FP32，最终速度场指数映射仍使用 FP64。内核优化和缓存复用也改变了运行耗时。数值差异及同输入计时比较见 [v0.3 性能测试](BENCHMARKS_v0.3.0.zh-CN.md)。

## 从 v0.1 升级

Python 软件包不再需要 PyTorch、Triton 或 CAT 辅助程序。基础依赖为 NumPy、SciPy、Numba 和 nibabel，GPU 支持通过 CUDA 或 Metal 可选依赖安装。如需保留最初的实现用于比较，请使用独立环境。

### 命令行变化

两条主要命令及其文件参数保持不变：

```text
cat-surf2sphere-gpu --input-surface ... --output-surface ...
cat-surfwarp-gpu --source-surface ... --source-sphere ... --target-surface ... --target-sphere ... --output ...
```

删除旧选项 `--kernel`、`--dtype`、`--preprocess-kernel`、`--areal-schedule`、`--areal-arithmetic`、`--rotation-feature-backend`、`--stencil-threads`、`--dartel-dtype`、`--squaring-kernel` 和 `--cuda-graph`。使用 `--device` 选择后端、`--threads` 设置 CPU 线程数，使用 `--solver-backend` 和 `--solver-iterations` 设置配准求解器。

初始旋转和平均默认开启，可通过 `--no-rotate` 和 `--no-average` 关闭；`--avg` 是 `--average` 的别名。旋转已包含在 `cat-surfwarp-gpu` 中，不再提供独立的旋转命令。

### Python 与参考实现变化

以文件为输入的 `run_cat_surf2sphere_gpu` 和 `run_cat_surface_gpu_pipeline` 已替换为数组 API。使用 `load_gifti_surface` 读取 GIFTI，调用 `surf_to_sphere` 或 `surfwarp_cupy`，然后通过 `save_gifti_surface` 保存结果。示例见 [API 指南](IMPLEMENTATION.zh-CN.md)。

算法参考版本从 CAT-Surface `628b6851` 更新为 `b0c2aa23`，默认配准求解器采用 PCG，参考实现采用 FMG。更新数值基线时，应使用相同输入比较输出。历次发布记录可从[更新日志](CHANGELOG.zh-CN.md)查阅。

[文档目录](README.zh-CN.md) · [项目首页](../README.zh-CN.md)
