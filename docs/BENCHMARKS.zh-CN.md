# 性能报告：v0.1.0

[English](BENCHMARKS.md) · [简体中文](BENCHMARKS.zh-CN.md)

本报告记录 PyTorch/Triton 实现及随包 CAT CPU 程序的测量结果。输入为一组匿名皮层表面和 SimNIBS fsaverage 模板，项目不分发被试数据。

## 环境与计时

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti，11 GiB |
| Python / PyTorch | 3.11 / 2.6.0+cu126 |
| CUDA 运行时 / Triton | 12.6 / 3.2.0 |
| CAT-Surface 参考提交 | `628b6851d8638f3ab773cd25c0ec406d0ec61ede` |
| 参考程序构建 | GCC，`-O2 -fPIC`，不使用 `-ffast-math` |
| 配准精度 | FP64 DARTEL |
| 球面生成精度 | FP32 存储，双精度中间计算 |

CPU 和 GPU 使用相同的 GIFTI 输入及参数。实际耗时包含解析、CPU 辅助计算、设备准备、同步后的计算、输出传输和 GIFTI 写出。双半球测试由两个进程共用一块 GPU，流程止于球面配准，不包含分割和皮层表面重建。

## 球面生成

### 粗网格

每个半球包含 61,442 个顶点和 122,880 个三角形。前五个阶段由随包 CPU 参考程序完成，随后使用 Triton 进行有序面积平滑。`stop_at=10` 时执行全部 4,999 次平滑。

| 半球 | CPU 参考程序 | GPU 总耗时 | 面积平滑 | 绝对误差：最大 / 均值 / p99 |
| --- | ---: | ---: | ---: | ---: |
| LH | 54.84 s | 10.8119 s | 3.7084 s | 1.1444e-5 / 1.7699e-6 / 7.6294e-6 |
| RH | 本次未测量 | 10.6667 s | 3.5142 s | 1.5259e-5 / 1.6752e-6 / 7.6294e-6 |

三角形数组完全一致，LH 的 CPU/GPU 耗时比为 5.07×。两个半球并发时，每个进程约耗时 13.81 秒，双半球阶段实际耗时 15.7609 秒。

### 全分辨率网格

较早的一组测量使用每半球 245,762 个顶点、491,520 个三角形，并显式选择 FP32/Triton 实现。中间球面的最大、均值和 p99 误差阈值分别为 `5e-3 / 1e-3 / 3e-3`。

| 半球 | CPU 参考程序 | GPU | CPU/GPU 耗时比 | 绝对误差：最大 / 均值 / p99 |
| --- | ---: | ---: | ---: | ---: |
| LH | 234.00 s | 37.8887 s | 6.18× | 3.073e-3 / 6.20e-4 / 2.407e-3 |
| RH | 236.35 s | 37.4264 s | 6.32× | 3.170e-3 / 6.12e-4 / 2.398e-3 |

三角形数组完全一致。双半球流程采用粗网格有序实现，满足更严格的 `1e-3` 最大误差阈值。

## 球面配准

源表面每份包含 245,762 个顶点和 491,520 个三角形，目标表面每份包含 163,842 个顶点和 327,680 个三角形。参数为 `steps=2`、`runs=2`、`avg=true`、`loop=6`、`cycles=3`、`nit=3`、`its=3`、`code=1`，DARTEL 映射网格为 `512 × 256`。

深度特征由上游辅助程序计算，热扩散滤波和归一化使用 CUDA。DARTEL 和平方运算使用 FP64 Triton，并开启 CUDA Graph。下列成对测量使用 32 个模板构建线程。

| 半球 | CPU 参考程序 | GPU 总耗时 | CPU/GPU 耗时比 | 绝对误差：最大 / 均值 / p99 |
| --- | ---: | ---: | ---: | ---: |
| LH | 119.914245 s | 45.084841 s | 2.659746× | 2.8491e-5 / 7.3001e-8 / 9.4995e-7 |
| RH | 119.405634 s | 46.986172 s | 2.541293× | 3.4094e-4 / 5.8125e-6 / 4.9412e-5 |

两个输出均完整保留三角形数组。这是 CPU/GPU 成对单次测量，不是重复运行的中位数。

一份 RH 分阶段记录显示：源表面各轮计算与平均耗时 19.380 秒，初始模板构建和特征计算 9.155 秒，旋转后的模板构建 4.442 秒，输出写入 0.823 秒。主要耗时来自重复求解和模板构建。

### 模板构建线程数

辅助程序通过 `--threads` 指定线程数，Python 命令对应选项为 `--stencil-threads`，默认为 8。在同一全分辨率 LH 输入上，每种配置启动三个新进程，结果如下：

| 线程数 | 实际耗时中位数 | CPU 时间中位数 |
| ---: | ---: | ---: |
| 8 | 4.28 s | 7.59 s |
| 16 | 4.13 s | 8.39 s |
| 32 | 4.03 s | 10.21 s |

九个输出文件逐字节相同，SHA-256 均为 `bffbb212d752d4c12f4eb77d03944edc4797ca449d8e36e6e3a35db925b9dd56`。

采用默认 8 线程的完整 LH 成对测量，CPU 耗时 110.8468 秒、GPU 耗时 37.0151 秒，耗时比为 2.9946×。三角形数组一致，最大、均值和 p99 误差为 `7.06e-6 / 4.75e-8 / 4.02e-7`。另两次仅 GPU 测量在 32 线程和 8 线程下分别耗时 31.5073 秒和 31.6381 秒，差异不足以支持提高默认线程数。

## 发行程序重建后的检查

按[此版本的构建说明（英文）](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BUILDING.md)重新构建四个随包 ELF 程序后，单次 LH 成对检查结果如下：

| 组件 | CPU | GPU | CPU/GPU 耗时比 | 绝对误差：最大 / 均值 / p99 |
| --- | ---: | ---: | ---: | ---: |
| 球面生成，61,442 顶点 | 54.9849 s | 10.6890 s | 5.1440× | 1.1444e-5 / 1.6734e-6 / 7.6294e-6 |
| 球面配准 | 126.6581 s | 35.0607 s | 3.6125× | 5.4806e-5 / 2.1548e-7 / 3.4571e-6 |

三角形数组完全一致，执行了全部 4,999 次球面平滑。CUDA 主机测试通过 34 项、跳过一项无 GPU 测试；无 GPU 环境通过 33 项、跳过两项。

## 双半球流程

两个半球先并发生成球面，再进行拓扑升采样，最后并发完成配准。

| 阶段 | 实际耗时 |
| --- | ---: |
| 双半球球面生成 | 15.7609 s |
| 拓扑升采样 | 4.5208 s |
| 双半球配准 | 34.9756 s |
| **合计** | **55.2573 s** |

与 CPU 参考程序的最终输出比较：

| 半球 | 绝对误差：最大 / 均值 / p99 | 三角形索引 |
| --- | ---: | --- |
| LH | 1.3821e-6 / 7.0854e-8 / 5.5507e-7 | 完全一致 |
| RH | 1.8090e-4 / 6.7359e-6 / 5.7042e-5 | 完全一致 |

## 复现方法

单独检出 [v0.1.0 标签](https://github.com/ayakacxy/cat-surface-gpu/tree/v0.1.0)，按该版本说明安装。以下示例使用该标签中的脚本和程序，并将工作文件保存在项目目录内。

```bash
mkdir -p .work/v0.1-comparison/{tmp,cache/triton,cache/torch,output}
export TMPDIR="$PWD/.work/v0.1-comparison/tmp" TMP="$PWD/.work/v0.1-comparison/tmp" TEMP="$PWD/.work/v0.1-comparison/tmp"
export XDG_CACHE_HOME="$PWD/.work/v0.1-comparison/cache"
export TRITON_CACHE_DIR="$PWD/.work/v0.1-comparison/cache/triton"
export TORCH_HOME="$PWD/.work/v0.1-comparison/cache/torch"

python tools/benchmark_cat_surface_end_to_end.py \
  --reference-cli bin/linux-x86_64/CAT_SurfWarp \
  --source-surface /path/to/white.gii \
  --source-sphere /path/to/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --cpu-output .work/v0.1-comparison/output/sphere.cpu.gii \
  --gpu-output .work/v0.1-comparison/output/sphere.gpu.gii \
  --rotation-depth-probe bin/linux-x86_64/cat_surface_rotation_depth \
  --rotation-feature-backend cuda-official-depth \
  --stencil-builder bin/linux-x86_64/cat_surface_stencil_builder \
  --rotated-stencil-builder bin/linux-x86_64/cat_surface_stencil_builder \
  --stencil-threads 8 \
  --device cuda \
  --kernel triton \
  --dartel-dtype float64 \
  --squaring-kernel triton \
  --steps 2 \
  --runs 2 \
  --avg \
  --max-vertex-error 1e-3
```

如果三角形索引不同或最大误差超过指定阈值，命令将以错误状态退出。
