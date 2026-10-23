# 更新记录

[English](CHANGELOG.md) · [简体中文](CHANGELOG.zh-CN.md)

## [0.3.1] - 2026-10-23

### 文档

- 重写中英文 README、用户指南、开发指南和版本说明。
- 为已有的 v0.3.0 性能测量增加独立报告。
- 集中整理版本历史，更新文档链接。
- 重新构建包含新版文档的发行包，算法和数值行为保持不变。

[版本说明](RELEASE_v0.3.1.zh-CN.md)

## [0.3.0] - 2026-10-18

### 新增

- Linux、Windows 和 macOS 上的 CPU 后端。
- 面向 Apple Silicon 的 MLX Metal 后端。
- 两个命令和 `surf_to_sphere()` 接口中的设备选择功能。
- Linux、Windows 和 macOS 的 CPU CI。

### 改进

- 使用共享内存分块加速 CUDA 平滑。
- 特征求解和 DARTEL 使用 FP32，几何装配和最终指数映射保留 FP64。
- 缓存固定拓扑和目标特征，减少配准内部的重复输入检查。

[版本说明](RELEASE_v0.3.0.zh-CN.md) · [性能报告](BENCHMARKS_v0.3.0.zh-CN.md)

## [0.2.0] - 2026-10-13

### 新增

- 几何计算、深度特征、旋转、重采样和 DARTEL 配准的 Python 实现。
- 球面生成与配准的参数类和数组接口。
- 退化三角形检查和 GIFTI 元数据保留。

### 调整

- 用 NumPy、Numba 和 CuPy 替代 PyTorch、Triton 及随包可执行程序。
- 将 CAT-Surface 参考提交更新为 `b0c2aa23`。
- 默认配准求解器改为 40 次 FP64 PCG 迭代。
- 改用纯 Python wheel，并更新命令选项。

### 修复

- 固定迭代 PCG 的残差精确为零时发生的除零错误。

[版本说明](RELEASE_v0.2.0.zh-CN.md) · [性能报告](BENCHMARKS_v0.2.0.zh-CN.md)

## [0.1.0] - 2026-08-21

### 新增

- 基于 PyTorch/Triton 的球面生成和 FP64 DARTEL 配准。
- 初始旋转、球面重采样、平均和 GIFTI 输出。
- 基于 CAT-Surface `628b6851` 的 Linux x86-64 CPU 参考程序与辅助程序，以及对应源码和构建脚本。
- 命令行工具、Conda 环境、CPU/CUDA 测试和性能对比。
- 平台 wheel、Linux 分发包、源码包、依赖清单和 SHA-256 校验和。

[版本说明](RELEASE_v0.1.0.zh-CN.md) · [性能报告](BENCHMARKS.zh-CN.md)

[0.3.1]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.3.1
[0.3.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.3.0
[0.2.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0
