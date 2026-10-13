# v0.1.0 — PyTorch/Triton acceleration with CAT helper programs

The first release provides GPU acceleration for CAT-Surface spherical inflation and surface registration, using CAT-Surface revision `628b6851d8638f3ab773cd25c0ec406d0ec61ede`.

- `CAT_Surf2Sphere` runs the initial stages with the bundled CPU reference and accelerates ordered area smoothing with Triton.
- `CAT_SurfWarp` uses PyTorch/Triton for rotation, resampling, FP64 DARTEL, averaging, and output generation, with CPU helper programs for supporting work.
- The release includes Linux x86-64 reference/helper executables, Python command-line tools, corresponding source, build instructions, and checksum manifests.

On the recorded RTX 2080 Ti setup, the bilateral CAT registration chain took **55.2573 s**, including spherical inflation, topology upsampling, and registration. This is a v0.1.0 measurement of the CAT stages, not a complete CHARM runtime. Input sizes, settings, and numerical results are documented in the [v0.1.0 benchmarks](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BENCHMARKS.md).

For the Python/Numba/CuPy implementation without CAT helper executables, see [v0.2.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0). Its algorithm reference and solver differ, so v0.1.0 performance and error thresholds should not be applied to it.

This description was clarified on 2026-10-13. The v0.1.0 tag and release files are unchanged.

## 中文说明

v0.1.0 是基于 PyTorch/Triton 的首个 GPU 加速版本，参考 CAT-Surface 固定提交 `628b6851`，运行时仍使用随包提供的 CPU 参考程序和辅助程序。

球面膨胀的前处理由 CPU 参考程序完成，面积平滑由 Triton 加速；曲面配准使用 PyTorch/Triton 完成旋转、重采样、双精度 DARTEL 和结果输出。发行文件包括 Linux x86-64 可执行文件、Python 命令、相应源码及校验记录。

RTX 2080 Ti 上记录的双半球 CAT 配准时间为 **55.2573 秒**，覆盖球面膨胀、拓扑上采样和配准，不包含完整 CHARM 流程。该数字只属于 v0.1.0，不能用来代表 v0.2.0。本次仅修订版本说明，保留原有标签和发行文件。
