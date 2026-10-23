# v0.2.0 性能与输出比较

[English](BENCHMARKS_v0.2.0.md) · [简体中文](BENCHMARKS_v0.2.0.zh-CN.md)

测量日期：2026-10-13。测试使用已发布的 wheel，以及未经修改的 CAT-Surface CPU 参考程序 `b0c2aa2395ea4957a331b9444fb54c7716b6572d`。各次测量、参数、输入摘要和输出对比见 [JSON 记录](benchmark-v0.2.0.json)。

## 环境与输入

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti，11 GiB |
| CPU | Intel Xeon Gold 6230 @ 2.10 GHz，双路 |
| Python / CuPy | 3.11.15 / 14.2.0 |
| NumPy / SciPy / Numba / nibabel | 2.3.0 / 1.17.1 / 0.64.0 / 5.3.3 |
| CUDA 运行时 | 12.6 |
| Numba CPU 线程数 | 8 |
| CPU 参考程序构建 | GCC 9.4.0，`-O2 -fPIC`，静态库 |
| 输入表面 | 公共 fsaverage 模板，每份含 163,842 个顶点、327,680 个三角形 |
| 球面生成参数 | `stop_at=10`，执行全部 4,999 次面积平滑 |
| 配准参数 | `steps=2`、`runs=2`，开启平均和旋转，`512 × 256` 映射网格，type-5 特征 |
| GPU 求解器 | 默认 PCG，40 次迭代；另行测量 FMG |

球面生成使用 `lh.white.gii`。配准将已有的 LH white/sphere 文件映射到已有的 RH white/sphere 文件，两种实现使用完全相同的输入。此处分别测量两个单表面命令，没有将生成的球面传入后续配准。

C 参考程序保留其原有线程设置，测量时未锁定 CPU 和 GPU 时钟。

## 完整命令耗时

下表为首次 GPU 调用之后三个新进程的测量结果，内核缓存和输入文件缓存已经建立，每轮交替执行顺序。计时包含进程启动、导入、读取、准备、CUDA 初始化、缓存加载或编译、计算、传输和 GIFTI 写出，不包含参考程序编译、软件安装和输出对比。

| 组件 | CPU 中位数 | GPU 中位数 | CPU/GPU 耗时比 | CPU 范围 | GPU 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 球面生成 | 216.722 s | 10.340 s | 20.96× | 212.922–263.969 s | 9.907–13.642 s |
| 配准，默认 PCG | 182.182 s | 32.550 s | 5.60× | 180.606–186.225 s | 28.075–35.514 s |
| 配准，FMG | 182.182 s | 164.697 s | 1.11× | 180.606–186.225 s | 129.926–179.939 s |

CPU 参考程序采用 FMG，因此与默认 GPU PCG 的对比同时包含求解器和实现方式的变化。GPU FMG 属于同类求解器，但结果仍不与 C 参考程序逐元素相同。

首次 GPU 调用分别耗时 **17.766 秒**（球面生成）、**48.985 秒**（PCG 配准）和 **130.208 秒**（FMG 配准）。这三项按上述顺序共用一个新建应用缓存，之前仅进行了环境导入和整数向量 CUDA 检查。后续调用可以复用此前的内核编译结果，因此它们不是三次独立的冷缓存测量，也不计入上表中位数。

## 输出对比

三角形数组完全一致，坐标均为有限值，没有零面积三角形。下表给出三次比较中各统计量的最大值。坐标差按数组元素计算，顶点距离为对应点之间的欧氏距离，单位均采用 GIFTI 坐标单位。

| 组件 | 绝对坐标差：最大 / 均值 / p99 | 顶点距离：最大 / 均值 / p99 |
| --- | ---: | ---: |
| 球面生成 | 0.057757 / 0.018752 / 0.050117 | 0.062416 / 0.036491 / 0.061337 |
| 配准，PCG | 0.166157 / 0.012807 / 0.096529 | 0.166649 / 0.027569 / 0.122179 |
| 配准，FMG | 0.165802 / 0.012483 / 0.095415 | 0.166345 / 0.026878 / 0.122087 |

GPU 平滑按相互独立的顶点分组更新，C 参考程序则按串行顺序更新。更新顺序和求解器的差异都会影响坐标。这些结果是几何比较，不能据此判断解剖配准精度等价。

## 复现方法

安装 v0.2.0 软件包，并使用当前仓库中的 `tools/benchmark_cat_surface.py`。此脚本在发布后加入，不包含在原始源码包中。

以下 Linux 示例需要 GNU Autotools、C 编译器、Make 和 GitHub CLI。参考源码、构建文件和缓存均存放于 `.work/performance`。

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

使用公共 fsaverage 模板或其他适合的 GIFTI 文件进行对比：

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

结果目录必须尚不存在。脚本记录每个完成的命令，并比较三角形索引和坐标。`--include-fmg` 增加单独的 FMG 测量，输入文件不会被修改。
