# 性能报告：v0.3.0

[English](BENCHMARKS_v0.3.0.md) · [简体中文](BENCHMARKS_v0.3.0.zh-CN.md)

测量日期：2026-10-18。在 RTX 2080 Ti 上，单份表面的球面生成和配准完整命令耗时中位数分别为 **2.612 秒**和 **4.983 秒**。各次测量、命令、输入摘要及输出对比见 [JSON 记录](benchmark-v0.3.0.json)。

## 环境与输入

| 项目 | 配置 |
| --- | --- |
| 平台 | Linux x86-64 |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| CPU | Intel Xeon Gold 6230 @ 2.10 GHz |
| Python / CuPy | 3.11.15 / 14.2.0 |
| NumPy / SciPy / Numba / nibabel | 2.3.0 / 1.17.1 / 0.64.0 / 5.3.3 |
| CPU 线程数 | 8 |
| 输入 | 公共 fsaverage 表面，每份含 163,842 个顶点、327,680 个三角形 |
| 球面生成参数 | `stop_at=10`，包含全部 4,999 次面积平滑 |
| 配准参数 | 两轮配准，开启平均和初始旋转，映射网格为 `512 × 256` |
| 求解器 | PCG，40 次迭代，`pcg_check_interval=0` |
| 旋转批次大小 | `rotation_point_chunk=12288` |

球面生成使用 `lh.white.gii`。配准以已有的 LH white/sphere 文件为源、RH white/sphere 文件为目标。两个命令分别测量；配准测试不使用前一个命令生成的球面。

## 完整命令耗时

每次测量均启动新进程，从启动计时至退出，包含导入、输入读取与检查、缓存加载、计算和压缩 GIFTI 写出。输出对比在计时结束后进行。下表汇总缓存建立后的三次测量；每轮交替两个版本的执行顺序。

| 命令 | 中位数 | 范围 |
| --- | ---: | ---: |
| 球面生成 | 2.612 s | 2.601–2.648 s |
| 球面配准 | 4.983 s | 4.967–5.116 s |

首次球面生成和配准分别耗时 **14.050 秒**和 **11.700 秒**。球面生成前应用缓存为空，随后配准可复用已编译的内核；操作系统文件缓存和驱动状态没有重置。首次调用不计入上表中位数。

### 版本对比

同一次测试还使用相同文件和 PCG 参数测量了 v0.2.0：

| 命令 | v0.2.0 中位数 | v0.3.0 中位数 | 耗时比 |
| --- | ---: | ---: | ---: |
| 球面生成 | 7.968 s | 2.612 s | 3.05× |
| 球面配准 | 20.479 s | 4.983 s | 4.11× |

该对比反映完整实现的性能变化，包含数值计算优化和准备工作减少带来的收益。

## 输出对比

两个版本均保留三角形索引，输出坐标有限，无零面积三角形。每个版本的三次重复运行均得到逐元素相同的坐标。

| 版本间差异 | 球面生成 | 球面配准 |
| --- | ---: | ---: |
| 最大逐坐标绝对差 | 0.000244141 | 0.001388580 |
| 最大顶点距离 | 0.000286091 | 0.001632973 |
| 最大角差 | 0.000227° | 0.093562° |
| 角差第 99 百分位 | 0.000195° | 0.048106° |
| 对应三角形法向量点积为负的数量 | 0 | 34 |

坐标和距离采用各 GIFTI 文件中存储的单位，角差按归一化后的顶点方向计算。最后一行比较两个输出的对应法向量，不等同于独立判定出的翻转三角形数量。这些几何比较不衡量解剖配准精度。

## 双半球流程

另一组测试将公共 fsaverage 白质表面重采样到 61,442 顶点网格。每个半球依次完成球面生成、升采样至 245,762 顶点及配准。左右半球在同一进程内串行运行，输入、模板和固定拓扑预先加载。

| 计时范围 | 中位数 |
| --- | ---: |
| 双半球完整处理，包含四个压缩 GIFTI 文件写完 | 9.731 s |
| 内部球面生成与配准计时之和 | 5.771 s |

表中数值为三次已有缓存运行的中位数，不计首次调用。第一行包含内部计时之外的准备与检查，但不包含进程启动、导入和重采样测试输入的准备。此处的网格规模及计时范围与单份表面命令不同，不能将前面的命令耗时相加或乘二作为双半球结果。

## 复现命令对比

先按 [README](../README.zh-CN.md) 安装 CUDA 依赖，再从仓库根目录运行。输入可使用公共 fsaverage 文件或其他适合的源/目标表面。基线目录需要直接包含 `cat_surface_gpu` 包，以下示例从 v0.2.0 标签提取源码。

```bash
mkdir -p .work/version-benchmark/{tmp,cache,baseline}
export TMPDIR="$PWD/.work/version-benchmark/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PWD/.work/version-benchmark/cache"
export PYTHONDONTWRITEBYTECODE=1

git archive v0.2.0 src | tar -x -C .work/version-benchmark/baseline

python tools/benchmark_versions.py \
  --baseline-source .work/version-benchmark/baseline/src \
  --candidate-source src \
  --source-surface /path/to/fsaverage/lh.white.gii \
  --source-sphere /path/to/fsaverage/lh.sphere.gii \
  --target-surface /path/to/fsaverage/rh.white.gii \
  --target-sphere /path/to/fsaverage/rh.sphere.gii \
  --work-dir .work/version-benchmark/results \
  --device cuda --repeats 3 --threads 8
```

结果目录必须尚不存在。脚本为两个版本分别设置项目内缓存，记录一次首次调用和三次重复调用，生成 `report.json`、命令日志及表面文件，同时检查重复结果并比较输出几何。输入文件不会被修改。
