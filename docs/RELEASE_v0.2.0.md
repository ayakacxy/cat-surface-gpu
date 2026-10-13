# v0.2.0 — Python implementation from FlashCHARM

Version 0.2.0 replaces the earlier PyTorch/Triton workflow and bundled CAT programs with the Python CAT implementation from FlashCHARM.

- Spherical inflation, type-5 depth features, initial rotation, resampling, DARTEL registration, averaging, and GIFTI output are implemented in Python using NumPy/Numba and CuPy CUDA kernels.
- The package neither runs CAT executables nor loads project-specific C/Cython extensions. Numba and CuPy still compile kernels at runtime; a compatible NVIDIA GPU is required for the full workflow.
- The wheel contains Python source and uses the `py3-none-any` tag. PyTorch, Triton, FlashCHARM, and SimNIBS are not runtime dependencies.
- The two main command names are retained. The new Python API exposes `Surf2SphereParameters`, `SurfWarpParameters`, `surf_to_sphere_cupy`, and `surfwarp_cupy`.

The fixed CAT-Surface reference is `b0c2aa2395ea4957a331b9444fb54c7716b6572d`. Registration defaults to 40 double-precision PCG iterations per linear solve; this differs from upstream FMG. Supported settings are described in the implementation guide and cover the type-5 registration path.

Install with Python 3.11 or newer and a matching CuPy/CUDA environment. See the [README](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.md), [implementation details](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/IMPLEMENTATION.md), and [migration guide](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/MIGRATION.md).

Historical v0.1.0 benchmarks and FlashCHARM integrated-workflow timings are not measurements of this standalone release.

## Validation

On an RTX 2080 Ti, all **57 tests passed**, including eight CUDA tests. Both commands also completed from an installed wheel on full fsaverage templates, with the sphere output passed directly to registration.

A complete comparison against the source FlashCHARM Python modules used fsaverage surfaces with 163,842 vertices and 327,680 triangles. With the default scientific settings and rotation batches of 12,288 points, sphere coordinates, registration coordinates, velocity, deformation, and objective values were element-for-element identical in this comparison; triangle arrays were unchanged. This verifies the package migration on these templates, not equivalence to every upstream CAT mode or a speedup claim. [Parameters, input hashes, and results](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/validation-v0.2.0.json) are included.

This release also fixes division by zero when PCG reaches an exactly zero residual, including registration of identical surfaces. The regression tests cover CPU and CUDA calculations without reducing the iteration limit.

## 中文说明

v0.2.0 将 FlashCHARM 的 CAT 模块移植为独立 Python 包，覆盖球面膨胀、深度特征、初始旋转、球面重采样、DARTEL 配准、平均和 GIFTI 输出。

本版使用 NumPy/Numba 和 CuPy CUDA kernel，不再调用 CAT 外部程序或加载本项目专用的 C/Cython 扩展。Numba、CuPy 仍需在运行时编译内核，完整流程需要 NVIDIA GPU；无需安装 PyTorch、Triton、FlashCHARM 或 SimNIBS。

上游参考版本更新为固定提交 `b0c2aa23`，配准默认每次线性求解执行 40 次双精度 PCG 迭代，与上游 FMG 的计算过程存在差异。两个主要命令名称保留，旧版后端选项与 Python 接口需按迁移说明更新。v0.1.0 的历史性能数据及 FlashCHARM 集成流程的计时均不作为本版独立软件包的性能结果。

本轮验证：RTX 2080 Ti 上 57 项测试全部通过，包含 8 项 CUDA 测试；安装后的两个命令也完成了完整 fsaverage 模板的球面生成与后续配准。完整 fsaverage 模板上本次对照的球面、配准坐标、速度场、形变场及目标函数值均与 FlashCHARM 来源实现逐元素一致。另修复了 PCG 精确零残差时的除零问题，未减少迭代次数。详细参数和输入摘要见上方验证记录。
