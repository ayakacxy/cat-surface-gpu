# CAT-Surface GPU v0.3.0

[English](RELEASE_v0.3.0.md) · [简体中文](RELEASE_v0.3.0.zh-CN.md)

发布日期：2026-10-18。

本版本新增 CPU 和 Apple Metal 支持，并提升 CUDA 球面生成与配准的性能。

## 新增功能

- 基于 NumPy 和 Numba 的 CPU 后端，支持 Linux、Windows 和 macOS。
- 基于 MLX 的 Metal 后端，支持 Apple Silicon。
- 两个命令均支持设备选择：`cuda`、`cuda:N`、`metal`、`cpu` 和 `auto`。
- 新增 `surf_to_sphere()` 数组接口，通过 `device` 参数选择设备。
- 新增 Linux、Windows 和 macOS 的 CPU CI。

## 性能改进

- CUDA 面积平滑使用共享内存分块，保留有序更新和默认的 4,999 次平滑。
- 优化后的 GPU 实现在特征求解和 DARTEL 中使用 FP32；几何装配和最终指数映射使用 FP64，默认 PCG 迭代次数保持为 40。
- 缓存固定拓扑和目标特征，在多轮配准中复用输入检查与拓扑数据。

在 RTX 2080 Ti 上，单份表面的球面生成和配准命令分别耗时 **2.612 秒**和 **4.983 秒**。测试使用 163,842 顶点的 fsaverage 表面，缓存已建立，结果为三个新进程的耗时中位数。参数、首次调用耗时和输出对比见[性能报告](BENCHMARKS_v0.3.0.zh-CN.md)。

## 兼容性

需要 Python 3.11 或更新版本。两个命令名和已有 CUDA 数组接口继续可用。命令行默认设备为 `cuda`；使用 `--device auto` 可自动选择可用后端。显式指定的设备必须可用。

`surfwarp_cupy()` 现在接受 `backend="cpu"`、`"cupy"` 或 `"metal"`。Metal 使用 PCG，要求 `pcg_check_interval=0`；CPU 和 CUDA 也支持 FMG。接口变化见[迁移指南](MIGRATION.zh-CN.md)，安装方法见 [README](../README.zh-CN.md)。

优化后的 GPU 实现与此前实现存在小幅浮点差异。参数校验、GIFTI 检查和固定迭代 PCG 的零残差处理均保留。Linux CPU/CUDA 回归测试通过 60 项；Metal 测试在 Linux 上跳过。

[下载 v0.3.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.3.0)
