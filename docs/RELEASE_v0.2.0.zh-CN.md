# CAT-Surface GPU v0.2.0

[English](RELEASE_v0.2.0.md) · [简体中文](RELEASE_v0.2.0.zh-CN.md)

发布日期：2026-10-13。

本版本改用 Python、NumPy、Numba 和 CuPy 实现，替代 PyTorch/Triton 实现及随包提供的 CAT 可执行程序。

## 主要变化

- 在 Python 中实现球面生成、深度特征、初始旋转、球面重采样、DARTEL 配准、平均和 GIFTI 输出。
- 新增 `Surf2SphereParameters`、`SurfWarpParameters`、`surf_to_sphere_cupy()` 和 `surfwarp_cupy()` 接口。
- 保留 `cat-surf2sphere-gpu` 和 `cat-surfwarp-gpu` 命令，增加进度显示并调整后端选项。
- 保留 GIFTI 元数据，检查输入和输出网格中的零面积三角形。
- 修复固定迭代 PCG 的残差精确为零时发生的除零错误。
- 提供纯 Python wheel、源码包、固定版本的 CAT-Surface 源码、依赖清单及 SHA-256 校验和。

## 兼容性

完整流程需要 Python 3.11 或更新版本、NVIDIA GPU 和匹配的 CuPy 安装，不再依赖 PyTorch、Triton 或随包 CAT 辅助程序。Python 接口及后端选项有所变化，详见[此版本的迁移指南](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/MIGRATION.zh-CN.md)。

CAT-Surface 参考提交为 `b0c2aa2395ea4957a331b9444fb54c7716b6572d`。配准中的每次线性求解默认执行 40 次 FP64 PCG 迭代，与参考程序采用的 FMG 求解器不同。

## 验证与性能

RTX 2080 Ti 上的 57 项测试全部通过，其中包含八项 CUDA 测试。安装后的两个命令完成了 fsaverage 模板流程，并将生成的球面直接用于配准。输出保留 163,842 个顶点和 327,680 个三角形，坐标有限、三角形数组不变、无零面积三角形。

发布后补充的性能测试使用三个新进程和已建立的缓存测量完整命令。GPU 耗时中位数分别为：球面生成 10.340 秒、PCG 配准 32.550 秒、FMG 配准 164.697 秒。[性能报告](BENCHMARKS_v0.2.0.zh-CN.md)包含 CPU 对比、求解器差异、首次调用耗时和复现命令；输入摘要与参数见[验证记录](validation-v0.2.0.json)。

[下载 v0.2.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0) · [此版本的安装说明](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.zh-CN.md)
