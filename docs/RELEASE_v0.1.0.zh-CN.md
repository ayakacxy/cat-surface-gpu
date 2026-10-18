## CAT-Surface GPU v0.1.0

[English](RELEASE_v0.1.0.md) · [简体中文](RELEASE_v0.1.0.zh-CN.md)

本文记录 v0.1.0。v0.3.0 的平台支持、安装与性能见[当前版本说明](RELEASE_v0.3.0.zh-CN.md)。

首个版本使用 PyTorch 和 Triton 加速 `CAT_Surf2Sphere` 与 `CAT_SurfWarp`，同时保留 CAT-Surface CPU 参考程序和辅助程序。本项目是独立的社区项目，参考 CAT-Surface 固定提交 `628b6851d8638f3ab773cd25c0ec406d0ec61ede`。

### 主要功能

- 提供 `cat-surf2sphere-gpu`、`cat-surfwarp-gpu` 和 `cat-surface-rotation-gpu` 三个命令
- 球面膨胀的初始阶段由随包提供的 CPU 参考程序完成，面积平滑由 Triton 按顺序执行
- 使用 PyTorch/Triton 完成曲面配准，包括双精度 DARTEL、初始旋转、重采样、平均和结果输出
- 提供可按文档重建的 Linux x86-64 参考程序、辅助程序及相应源码
- 提供合成测试数据生成工具、CPU CI、CodeQL 和手动触发的 CUDA 工作流
- 提供平台 wheel、Linux 文件包、源码归档、CycloneDX 依赖清单和 SHA-256 校验文件

### 发布时的性能检查

发布检查使用真实左半球输入，硬件为 NVIDIA GeForce RTX 2080 Ti，PyTorch 版本为 2.6.0+cu126：

| 组件 | CPU 参考实现 | GPU | 加速比 | 最大顶点误差 |
| --- | ---: | ---: | ---: | ---: |
| `CAT_SurfWarp` | 126.658 s | 35.061 s | 3.613x | 5.481e-5 |
| `CAT_Surf2Sphere` | 54.985 s | 10.689 s | 5.144x | 1.144e-5 |

三角形数组完全一致，球面膨胀执行了全部 4,999 次扫描。上表是单次发布检查结果。另一项双半球 CAT 配准测量耗时 **55.2573 秒**，覆盖球面膨胀、拓扑上采样和配准。输入、参数、计时范围和成对比较见 [v0.1.0 基准记录（历史英文原文）](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BENCHMARKS.md)；当前文档也提供了[中文基准说明](BENCHMARKS.zh-CN.md)。

以上测量仅适用于 v0.1.0 及所记录的环境，不代表 v0.2.0 性能或完整 CHARM 流程耗时。

### 验证结果

- CPU 测试：33 项通过，2 项跳过
- CUDA 主机测试：34 项通过，1 项预期跳过项用于检查无 GPU 时的行为
- v0.1.0 发布提交通过 CI 和 CodeQL 检查
- 已验证 wheel 安装、三个命令入口、随包程序摘要和发布文件校验值

### 安装

按 [v0.1.0 中文 README](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/README.zh-CN.md)创建 Conda 环境，安装匹配的 CUDA 版 PyTorch 和平台 wheel。如需 Python/Numba/CuPy 实现，请使用 [v0.2.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0)。

项目采用 GPL-3.0-or-later 许可。上游 CAT-Surface 的署名和许可信息保留在 `THIRD_PARTY_NOTICES.md` 和对应源码归档中。

本说明于 2026-10-13 修订，v0.1.0 标签和发行文件保持不变。
