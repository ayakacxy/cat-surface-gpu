# v0.2.0 性能与输出比较

[English](BENCHMARKS_v0.2.0.md) · [简体中文](BENCHMARKS_v0.2.0.zh-CN.md)

本文记录 v0.2.0。v0.3.0 的平台支持、安装与性能见[当前版本说明](RELEASE_v0.3.0.zh-CN.md)。

本次使用已发布的 **v0.2.0** wheel，以及未经修改的 CAT-Surface CPU 参考程序，固定提交为 `b0c2aa2395ea4957a331b9444fb54c7716b6572d`。以下结果记录于 **2026-10-13**。PyTorch/Triton 旧版数据保留在 [v0.1.0 历史基准](BENCHMARKS.zh-CN.md)中。

## 环境与输入

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti，11 GiB |
| CPU | Intel Xeon Gold 6230 @ 2.10 GHz，双路 |
| Python / CuPy | 3.11.15 / 14.2.0 |
| NumPy / SciPy / Numba / nibabel | 2.3.0 / 1.17.1 / 0.64.0 / 5.3.3 |
| CUDA 运行时 | 12.6 |
| Numba CPU 线程数 | 8 |
| CPU 参考构建 | GCC 9.4.0，`-O2 -fPIC`，静态库 |
| 输入 | 公开 fsaverage 模板；每份 163,842 个顶点、327,680 个三角形 |
| 球面生成参数 | `stop_at=10`，完整执行 4,999 次面积平滑扫描 |
| 配准参数 | `steps=2`、`runs=2`，启用平均和旋转，`512 x 256` 映射，type-5 特征 |
| GPU 配准求解器 | 默认 PCG，40 次迭代；另行测量 FMG |

球面生成使用 `lh.white.gii`。配准将固定的左半球 white/sphere 文件对映射到固定的右半球 white/sphere 文件对，CPU 和 GPU 使用相同文件。两个组件分别计时，不串联新生成的球面，也不并发运行双半球。

C 参考程序沿用其内部线程设置。本次未固定 CPU 或 GPU 时钟，估计实际耗时时应一并考虑所记录的耗时范围。

## 完整命令耗时

每行报告首次 GPU 调用之后的 **三次独立进程运行**，此时磁盘内核缓存和操作系统输入文件缓存已建立。计时包括进程启动、导入、输入读取、CPU 前处理、CUDA 初始化、编译或缓存加载、计算、数据传输和 GIFTI 写入；不计参考程序编译、软件安装及输出检查。各轮交替调整 CPU/GPU 执行顺序。

| 组件 | CPU 中位数 | GPU 中位数 | CPU/GPU 耗时比 | CPU 范围 | GPU 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | 216.722 s | 10.340 s | 20.96x | 212.922–263.969 s | 9.907–13.642 s |
| `CAT_SurfWarp`，默认 PCG | 182.182 s | 32.550 s | 5.60x | 180.606–186.225 s | 28.075–35.514 s |
| `CAT_SurfWarp`，FMG | 182.182 s | 164.697 s | 1.11x | 180.606–186.225 s | 129.926–179.939 s |

CPU 参考使用 FMG。默认 GPU PCG 这一行比较的是两套完整默认流程，耗时比包含求解器差异的影响，不能单独归因于 GPU 硬件。即使选择 GPU FMG，Python 实现的数值结果也不一定与 C 参考程序完全相同。

GPU 首次调用耗时分别为：球面生成 **17.766 秒**、PCG 配准 **48.985 秒**、FMG 配准 **130.208 秒**。本次使用新建的任务缓存，首次球面调用前只执行了环境导入和整数向量 CUDA 检查，未预热曲面内核；之后 PCG 和 FMG 顺序使用同一缓存。这些首次调用未计入中位数，也不是各求解器相互独立的冷缓存测试。重复运行仍计入导入及新进程需要执行的编译。

## 输出差异

CPU/GPU 输出均保留输入三角形数组，坐标有限，且无零面积三角形。下表各项取三次比较中出现的最大统计值，单位沿用 GIFTI 文件中的坐标单位。

| 组件 | 逐坐标绝对差：最大 / 平均 / p99 | 顶点距离：最大 / 平均 / p99 |
| --- | ---: | ---: |
| `CAT_Surf2Sphere` | 0.057757 / 0.018752 / 0.050117 | 0.062416 / 0.036491 / 0.061337 |
| `CAT_SurfWarp`，PCG | 0.166157 / 0.012807 / 0.096529 | 0.166649 / 0.027569 / 0.122179 |
| `CAT_SurfWarp`，FMG | 0.165802 / 0.012483 / 0.095415 | 0.166345 / 0.026878 / 0.122087 |

逐坐标绝对差按数组元素计算；顶点距离是对应三维顶点的欧氏距离。GPU 平滑按着色后的独立顶点组更新，参考程序按其串行顺序更新，因此实现与求解器差异可能产生不同坐标。本次如实报告观察到的差异，不据此判定解剖配准精度相同，也不沿用 v0.1.0 的误差阈值。

[JSON 记录](benchmark-v0.2.0.json)包含每次耗时、参数、输入和实现摘要、参考程序摘要，以及全部输出比较。测试使用的 GPU 源码与发布 wheel 一致。计时覆盖所列 CAT 命令，不包含完整 CHARM 流程。

## 复现比较

使用当前仓库中的基准脚本和已安装的 v0.2.0 软件包。该脚本在发布后补充，不在最初的 v0.2.0 源码归档中。输入可使用公开模板或其他适合球面映射的 GIFTI 曲面。

先设置项目内临时目录和缓存，再构建固定版本的 CPU 参考程序。以下示例需要 GNU Autotools、C 编译器、Make 和 GitHub CLI：

```bash
mkdir -p .work/performance/{tmp,cache/pip,cache/numba,cache/cupy,cache/cuda,source}
export TMPDIR="$PWD/.work/performance/tmp" TMP="$PWD/.work/performance/tmp" TEMP="$PWD/.work/performance/tmp"
export XDG_CACHE_HOME="$PWD/.work/performance/cache"
export PIP_CACHE_DIR="$PWD/.work/performance/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/performance/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/performance/cache/cupy"
export CUDA_CACHE_PATH="$PWD/.work/performance/cache/cuda"
export PYTHONPYCACHEPREFIX="$PWD/.work/performance/cache/pycache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8

gh release download v0.2.0 --repo ayakacxy/cat-surface-gpu \
  --pattern 'CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz' \
  --dir .work/performance/source

tar -xf .work/performance/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz \
  -C .work/performance/source
CAT_REFERENCE_ROOT="$PWD/.work/performance/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d"
(
  cd "$CAT_REFERENCE_ROOT"
  NOCONFIGURE=1 ./autogen.sh
  CFLAGS='-O2 -fPIC' ./configure --disable-shared --enable-static
  make -j8 libCAT.la
  make -C Progs -j8 CAT_Surf2Sphere CAT_SurfWarp
)
```

执行完整命令比较：

```bash
python tools/benchmark_cat_surface.py \
  --reference-sphere "$CAT_REFERENCE_ROOT/Progs/CAT_Surf2Sphere" \
  --reference-warp "$CAT_REFERENCE_ROOT/Progs/CAT_SurfWarp" \
  --source-surface /path/to/fsaverage/lh.white.gii \
  --source-sphere /path/to/fsaverage/lh.sphere.gii \
  --target-surface /path/to/fsaverage/rh.white.gii \
  --target-sphere /path/to/fsaverage/rh.sphere.gii \
  --work-dir .work/performance/results \
  --repeats 3 --threads 8 --include-fmg
```

输出目录应尚未存在。脚本在每条命令完成后记录进度，并报告三角形是否一致、坐标差异及各求解器耗时。`--include-fmg` 会增加单独的 FMG 对照。输入文件保持只读。
