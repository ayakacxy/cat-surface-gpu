# v0.1.0 历史性能与精度记录

[English](BENCHMARKS.md) · [简体中文](BENCHMARKS.zh-CN.md)

以下测量针对 **v0.1.0**，即使用 PyTorch/Triton 并附带 CAT CPU 程序的实现。这些记录用于复现旧版结果，不是 v0.2.0 Python/Numba/CuPy 实现的测量。结果仅适用于下列硬件、软件、输入和参数。

## 已验证的环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti，11 GiB |
| Python | 3.11 |
| PyTorch | 2.6.0+cu126 |
| CUDA 运行时 | 12.6 |
| Triton | 3.2.0 |
| CAT-Surface 参考版本 | `628b6851d8638f3ab773cd25c0ec406d0ec61ede` |
| 参考程序构建 | GCC，`-O2 -fPIC`，不使用 `-ffast-math` |
| GPU DARTEL 精度 | FP64 |
| `CAT_Surf2Sphere` 精度 | FP32 存储，中间计算使用与 CAT 兼容的双精度 |

测试输入为一对匿名化皮层曲面和 SimNIBS fsaverage 模板。本仓库不分发被试数据。

## 测量规则

- CPU 和 GPU 使用相同的 GIFTI 输入、网格拓扑、CAT-Surface 参数和输出检查。
- 完整流程的墙钟时间使用 `time.perf_counter()` 测量，包含输入解析、CPU 辅助计算、主机到设备的数据准备、同步后的 CUDA 计算、输出传输和 GIFTI 写入。
- 读取计时器之前，先同步 CUDA 阶段。
- 双半球墙钟时间对应两个独立半球进程共享一张 RTX 2080 Ti，与 SimNIBS 的进程池调度方式一致。
- `CAT_Surf2Sphere stop_at=10` 仍请求 5,000 次平滑迭代，实际执行上游循环中的 4,999 次遍历（`k=1..4999`）。
- 以下结果均不外推到完整 SimNIBS CHARM 流程。

## `CAT_Surf2Sphere`

### SimNIBS 默认粗球面

每个半球的输入为 61,442 个顶点、122,880 个三角形。v0.1.0 后端保留随包 CPU 参考程序中的前五个上游阶段，并通过保持顺序的 Triton 调度执行面积平滑。

| 半球 | 固定版本 CPU 参考程序 | GPU 总耗时 | 仅面积平滑 | 最大 / 平均 / p99 绝对误差 | 三角面 |
| --- | ---: | ---: | ---: | ---: | --- |
| LH | 54.84 s | 10.8119 s | 3.7084 s | 1.1444e-5 / 1.7699e-6 / 7.6294e-6 | 精确一致 |
| RH | 未在同一次独立运行中记录 | 10.6667 s | 3.5142 s | 1.5259e-5 / 1.6752e-6 / 7.6294e-6 | 精确一致 |

LH 相对固定版本 CPU 参考程序的加速比为 5.07x。RH 表中没有借用另一次 CPU 运行的时间来推算加速比。

LH 和 RH 在同一张 GPU 上并发运行时，计算内核共享设备，每个进程报告约 13.81 秒；实测双半球阶段墙钟时间为 15.7609 秒。

### 全分辨率分析

每个半球的输入为 245,762 个顶点、491,520 个三角形。这次较早的分析显式使用 FP32/Triton 路径，并采用较宽的中间球面误差限制 `5e-3 / 1e-3 / 3e-3`。

| 半球 | 固定版本 CPU 参考程序 | GPU | 加速比 | 最大 / 平均 / p99 绝对误差 | 三角面 |
| --- | ---: | ---: | ---: | ---: | --- |
| LH | 234.00 s | 37.8887 s | 6.18x | 3.073e-3 / 6.20e-4 / 2.407e-3 | 精确一致 |
| RH | 236.35 s | 37.4264 s | 6.32x | 3.170e-3 / 6.12e-4 / 2.398e-3 | 精确一致 |

上面的粗球面顺序调度采用更严格的 `1e-3` 误差限制，因此它是 v0.1.0 中对应 SimNIBS 默认设置的方案。

## `CAT_SurfWarp`

每个半球的输入与设置：

- 源曲面的顶点/三角面：`(245762, 3)` / `(491520, 3)`；
- 目标曲面的顶点/三角面：`(163842, 3)` / `(327680, 3)`；
- DARTEL 映射尺寸：`512 x 256`；
- 参数：`steps=2`、`runs=2`、`avg=true`、`loop=6`、`cycles=3`、`nit=3`、`its=3`、`code=1`；
- 初始特征后端：上游原始深度特征，加上 CUDA 热扩散与归一化；
- 插值模板辅助程序：历史主要结果使用 32 个 CPU 工作线程，最终 v0.1.0 支持运行时配置，默认使用 8 个；
- DARTEL 和平方运算：FP64 Triton，启用 CUDA Graph。

### 成对端到端 A/B 对照

| 半球 | CPU 参考程序 | GPU 总耗时 | 加速比 | 最大 / 平均 / p99 绝对误差 | 三角面 |
| --- | ---: | ---: | ---: | ---: | --- |
| LH | 119.914245 s | 45.084841 s | 2.659746x | 2.8491e-5 / 7.3001e-8 / 9.4995e-7 | 精确一致 |
| RH | 119.405634 s | 46.986172 s | 2.541293x | 3.4094e-4 / 5.8125e-6 / 4.9412e-5 | 精确一致 |

v0.1.0 README 使用的是这组配对结果。后续单次探索性分析中，LH 达到 38.15 秒，RH 达到 37.26 秒；但这些运行没有通过交错顺序的重复基准测试验证，因此不作为主要性能结论。

### 代表性 RH GPU 时间分布

| 阶段 | 秒 | 占比 |
| --- | ---: | ---: |
| 输入 | 0.337 | 0.72% |
| 初始插值模板与上游特征 | 9.155 | 19.48% |
| 旋转索引上传 | 2.535 | 5.39% |
| 旋转搜索 | 3.475 | 7.40% |
| 旋转后的插值模板 | 4.442 | 9.45% |
| 预热 | 1.472 | 3.13% |
| 设备准备与其余上传 | 5.368 | 11.42% |
| 源曲面多轮计算与平均所需的 DARTEL / 形变 | 19.380 | 41.25% |
| 输出 | 0.823 | 1.75% |

在这次分析中，主要剩余开销来自反复求解和构建插值模板，而非 GIFTI 写入或平均合并。

### 运行时配置插值模板工作线程

v0.1.0 随包辅助程序接受 `--threads N`，Python 运行入口通过 `--stencil-threads` 提供相同设置。默认值为 8，线程数不再在编译二进制文件时固定。

在同一份全分辨率 LH 球面上，每种设置分别启动三次新的辅助程序，结果如下：

| CPU 工作线程 | 墙钟时间中位数 | CPU 时间中位数 | 输出 |
| ---: | ---: | ---: | --- |
| 8 | 4.28 s | 7.59 s | 逐字节一致 |
| 16 | 4.13 s | 8.39 s | 逐字节一致 |
| 32 | 4.03 s | 10.21 s | 逐字节一致 |

九个插值模板文件的 SHA-256 均为 `bffbb212d752d4c12f4eb77d03944edc4797ca449d8e36e6e3a35db925b9dd56`。将线程数从 8 增加到 32，这个独立辅助程序仅节省约 0.25 秒，但累计 CPU 时间明显增加。

使用可在运行时配置线程数的辅助程序，并保持默认 8 个工作线程，重新完成一次完整 LH CPU/GPU A/B 对照：CPU 参考程序为 110.8468 秒，GPU 路径为 37.0151 秒，加速比为 2.9946x。三角面精确一致，最终最大/平均/p99 顶点误差为 `7.06e-6 / 4.75e-8 / 4.02e-7`。

后续仅运行 GPU 的测量中，32 个工作线程为 31.5073 秒，8 个工作线程为 31.6381 秒。0.13 秒的差异处于完整流程的运行波动范围内，不能视为稳定的端到端加速。默认 8 个工作线程因此保留了基本相同的实测端到端性能，并减少辅助程序与双半球同时运行时的线程争用。

将软件包统一到 `cat_surface_gpu` 命名空间后，最终候选发布版使用随包辅助程序和默认 8 个工作线程完成了一次冒烟测试，耗时 30.6743 秒。与同输入 CPU 输出相比，三角面精确一致，最大/平均/p99 顶点误差为 `7.06e-6 / 4.74e-8 / 4.02e-7`。这次仅运行 GPU 的测试用于验证打包后的运行路径，不作为主要加速比结论。

### 最终 v0.1.0 可复现构建审计

按照 [v0.1.0 构建步骤（历史英文原文）](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BUILDING.md)重新构建全部四个随包 ELF 文件后，在同一张 RTX 2080 Ti 上重新完成成对 LH 测试：CPU 参考程序为 `126.6581 s`，GPU 路径为 `35.0607 s`，加速比为 `3.6125x`。三角面精确一致，最大/平均/p99 顶点误差为 `5.4806e-5 / 2.1548e-7 / 3.4571e-6`。这是单次发布审计，不替代前面较保守的主要配对结果。

重新构建的 `CAT_Surf2Sphere` 也在 61,442 个顶点的 LH 输入上重跑：CPU 为 `54.9849 s`，GPU 为 `10.6890 s`，加速比为 `5.1440x`。完整执行了上游的 4,999 次遍历，三角面精确一致，最大/平均/p99 顶点误差为 `1.1444e-5 / 1.6734e-6 / 7.6294e-6`。

可访问 GPU 的主机上，测试结果为 `34 passed, 1 skipped`；唯一跳过的是检查 CUDA 不可用时行为的测试。无 GPU 环境中的测试结果为 `33 passed, 2 skipped`。

## SimNIBS 默认双半球 CAT 流程

这项基准测试让两个半球进程并发执行 `CAT_Surf2Sphere`，随后执行与 SimNIBS 相同的拓扑上采样，再并发执行两个半球的 `CAT_SurfWarp`。

| 阶段 | 墙钟时间 |
| --- | ---: |
| 双半球 `CAT_Surf2Sphere` | 15.7609 s |
| 拓扑上采样 | 4.5208 s |
| 双半球 `CAT_SurfWarp` | 34.9756 s |
| **合计** | **55.2573 s** |

最终输出与固定版本 CPU 参考程序的对照：

| 半球 | 最大 / 平均 / p99 绝对误差 | 三角面 |
| --- | ---: | --- |
| LH | 1.3821e-6 / 7.0854e-8 / 5.5507e-7 | 精确一致 |
| RH | 1.8090e-4 / 6.7359e-6 / 5.7042e-5 | 精确一致 |

55.26 秒仅覆盖 CAT 球面配准流程，不包含皮层曲面估计、分割更新，也不包含完整 SimNIBS `Surface creation` 阶段中的其他工作。

## 复现测量

请单独检出 [v0.1.0 标签](https://github.com/ayakacxy/cat-surface-gpu/tree/v0.1.0)，并使用该版本的安装说明。以下命令使用该版的工具和二进制程序，它们不属于 v0.2.0。运行单半球对照前，先设置项目内的工作目录：

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

如果拓扑不一致，或最大顶点误差超过指定限制，基准测试会以非零退出码结束。
