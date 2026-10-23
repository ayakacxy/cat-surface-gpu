# CAT-Surface GPU v0.1.0

[English](RELEASE_v0.1.0.md) · [简体中文](RELEASE_v0.1.0.zh-CN.md)

发布日期：2026-08-21。

首个版本使用 PyTorch 和 Triton 加速 CAT-Surface 球面生成与配准，同时提供 CPU 参考程序和辅助程序。

## 功能

- 提供 `cat-surf2sphere-gpu`、`cat-surfwarp-gpu` 和 `cat-surface-rotation-gpu` 命令。
- 使用 Triton 实现有序面积平滑，初始膨胀阶段由 CPU 参考程序完成。
- 配准包含 FP64 DARTEL、初始旋转、重采样、平均和输出写入。
- 提供 Linux x86-64 参考程序及辅助程序、对应源码和可复现的构建脚本。
- 提供平台 wheel、Linux 分发包、源码包、CycloneDX 依赖清单和 SHA-256 校验和。

CAT-Surface 参考提交为 `628b6851d8638f3ab773cd25c0ec406d0ec61ede`，软件采用 GPL-3.0-or-later 许可证。

## 发布检查

在 RTX 2080 Ti 和 PyTorch 2.6.0+cu126 环境中，同一左半球输入的对照结果如下：

| 组件 | CPU 参考程序 | GPU | CPU/GPU 耗时比 | 记录的最大误差 |
| --- | ---: | ---: | ---: | ---: |
| Surf2Sphere | 54.985 s | 10.689 s | 5.144× | 1.144e-5 |
| SurfWarp | 126.658 s | 35.061 s | 3.613× | 5.481e-5 |

这些数据来自随包程序重新构建后的单次检查。三角形数组完全一致，球面生成执行了全部 4,999 次平滑。另一次双半球流程测试耗时 55.2573 秒，包含球面生成、拓扑升采样和配准。输入规模、计时范围及辅助程序的重复测量见[性能报告](BENCHMARKS.zh-CN.md)。

CPU 测试通过 33 项、跳过两项；CUDA 主机测试通过 34 项，跳过一项仅适用于无 GPU 环境的测试。发布时检查了 CI、CodeQL、wheel 安装、命令入口和发行文件校验和。

[下载 v0.1.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0) · [此版本的安装说明](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/README.zh-CN.md)
