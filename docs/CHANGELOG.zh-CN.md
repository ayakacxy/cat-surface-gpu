# 项目更新记录

[English](CHANGELOG.md) · [简体中文](CHANGELOG.zh-CN.md)

本文件记录主要变化。

## [0.2.0] - 2026-10-13

### 新增与调整

- 将 PyTorch/Triton 实现及随包 CAT 可执行程序替换为使用 NumPy、Numba 和 CuPy 的原生 Python 实现。
- 新增 NumPy/Numba 几何与拓扑计算、CuPy CUDA 内核、type-5 深度特征、初始旋转、球面重采样及 DARTEL 配准。
- 固定 CAT-Surface 参考版本从 `628b6851` 更新为 `b0c2aa23`。
- 新增 `Surf2SphereParameters`、`SurfWarpParameters`、`surf_to_sphere_cupy` 和 `surfwarp_cupy` 接口。
- 保留 `cat-surf2sphere-gpu` 和 `cat-surfwarp-gpu` 命令，选项调整为适配新实现。
- 修复 PCG 残差精确为零时的除零问题，包括相同曲面配准的情况。迭代上限与非零情况下的计算保持不变。
- 新增输入和输出的零面积三角形检查，对退化网格给出明确错误。
- 改为仅包含 Python 源码的 wheel。CUDA 计算仍需要 CuPy 和 NVIDIA GPU。
- 在保留原有文档风格的基础上更新安装与使用说明，并单独标明 v0.1.0 的历史性能数据。

### 兼容性

公共接口与后端选项有所变化。默认 DARTEL 求解执行 40 次 PCG 迭代，与上游 FMG 的数值计算过程不同。详见[迁移说明](MIGRATION.zh-CN.md)和 [v0.2.0 版本说明](RELEASE_v0.2.0.zh-CN.md)。

## [0.1.0] - 2026-08-21

首个独立发布版本使用 PyTorch 和 Triton 加速 CAT-Surface，同时保留上游 CPU 程序和两个辅助可执行程序。

### 新增

- `CAT_Surf2Sphere` 的顺序 GPU 面积平滑。
- GPU `CAT_SurfWarp` 配准，包含 FP64 DARTEL、CUDA Graph 支持、初始旋转、平均和输出写入。
- Linux x86-64 参考程序和辅助程序、相应源码，以及可复现的构建脚本。
- 命令行工具、Conda 环境、CPU/CUDA 测试，以及数值和性能对照。
- 平台 wheel、项目源码包、Linux 分发包、上游源码包、SBOM 和 SHA-256 清单。

### 兼容性

参考提交为 `628b6851d8638f3ab773cd25c0ec406d0ec61ede`。[v0.1.0 版本说明](RELEASE_v0.1.0.zh-CN.md)和[历史性能记录](BENCHMARKS.zh-CN.md)描述该实现。这些说明在 v0.2.0 文档中调整了表述，v0.1.0 标签和发行文件保持不变。

[0.2.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0
