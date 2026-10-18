## CAT-Surface GPU v0.2.0

[English](RELEASE_v0.2.0.md) · [简体中文](RELEASE_v0.2.0.zh-CN.md)

本文记录 v0.2.0。v0.3.0 的平台支持、安装与性能见[当前版本说明](RELEASE_v0.3.0.zh-CN.md)。

v0.2.0 使用 NumPy、Numba 和 CuPy 原生 Python 实现 `CAT_Surf2Sphere` 和 `CAT_SurfWarp`。球面膨胀与曲面配准可独立运行，无需 v0.1.0 使用的 CAT 辅助程序。

### 主要变化

- 使用 Python 实现球面膨胀、深度特征、初始旋转、球面重采样、DARTEL 配准、平均和 GIFTI 输出
- 使用 CuPy CUDA 内核和 NumPy/Numba 几何计算，不再依赖 PyTorch 或 Triton
- 保留 `cat-surf2sphere-gpu` 和 `cat-surfwarp-gpu` 命令，并显示各阶段进度
- 提供基于数组的 Python API：`Surf2SphereParameters`、`SurfWarpParameters`、`surf_to_sphere_cupy` 和 `surfwarp_cupy`
- 提供仅含 Python 源码的 wheel、源码归档、固定版本的 CAT-Surface 源码、CycloneDX 依赖清单和 SHA-256 校验文件
- 保留 GIFTI 元数据，检查退化三角形，并修复 PCG 残差精确为零时的除零问题

### 兼容性

CAT-Surface 参考版本固定为 `b0c2aa2395ea4957a331b9444fb54c7716b6572d`。配准默认每次线性求解执行 40 次双精度 PCG 迭代，其计算过程与上游 FMG 不同。两个主要命令名称保留，Python API 和后端选项有所变化，详见[迁移指南](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/MIGRATION.zh-CN.md)和[实现说明](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/IMPLEMENTATION.zh-CN.md)。

Numba 和 CuPy 会在运行时编译内核。`py3-none-any` wheel 包含 Python 源码，完整流程仍需要兼容的 NVIDIA GPU 和 CuPy/CUDA 环境。v0.1.0 的历史基准数据仅适用于旧版。

### 性能

使用完整 fsaverage 模板，将发布 wheel 与固定版本 CPU 参考程序进行比较。三次运行均使用新进程和已建立的磁盘缓存，计时包括启动、读取、计算和写入：

| 组件 | CPU 中位数 | GPU 中位数 | CPU/GPU 耗时比 | 最大逐坐标差 |
| --- | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | 216.722 s | 10.340 s | 20.96x | 0.057757 |
| `CAT_SurfWarp`，PCG（默认） | 182.182 s | 32.550 s | 5.60x | 0.166157 |
| `CAT_SurfWarp`，FMG | 182.182 s | 164.697 s | 1.11x | 0.165802 |

CPU 使用 FMG，默认 GPU PCG 的耗时比包含求解器差异。误差是 GIFTI 坐标单位中的逐坐标绝对差，表中取三次比较的最大值。[完整结果与复现说明](BENCHMARKS_v0.2.0.zh-CN.md)还列出耗时范围、首次运行成本和顶点距离。这些测量在发布后补充，不据此判定解剖配准精度相同。

### 验证结果

- RTX 2080 Ti 上 **57 项测试通过**，包含 8 项 CUDA 测试，无跳过项
- 安装后的 wheel：两个命令完成了完整 fsaverage 模板测试，生成的球面直接用于后续配准
- 每份输出保留 **163,842 个顶点和 327,680 个三角形**，坐标有限，三角形数组不变，无零面积三角形
- 已检查 wheel 和源码归档内容、Python 源码摘要、命令入口及发布文件校验值

[验证记录](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/validation-v0.2.0.json)包含环境、输入摘要和参数。安装检查覆盖所列模板；上方单独的性能比较提供了完整命令耗时和数值差异。

### 安装

需要 Python 3.11 或更新版本，以及与 CuPy 匹配的 NVIDIA 驱动。文档中的 CUDA 12.x 环境使用 `cupy-cuda12x`。Conda 和 wheel 安装命令见[中文 README](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.zh-CN.md)，也可查阅[英文版本](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.md)。

项目采用 GPL-3.0-or-later 许可。CAT-Surface 的署名和许可说明见 `THIRD_PARTY_NOTICES.zh-CN.md` 及对应源码归档。
