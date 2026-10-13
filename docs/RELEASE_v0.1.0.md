## CAT-Surface GPU v0.1.0

The first release accelerates `CAT_Surf2Sphere` and `CAT_SurfWarp` with PyTorch and Triton while retaining CAT-Surface CPU reference and helper programs. This is an independent community project, with CAT-Surface revision `628b6851d8638f3ab773cd25c0ec406d0ec61ede` as its reference.

### Highlights

- Installed CLIs: `cat-surf2sphere-gpu`, `cat-surfwarp-gpu`, and `cat-surface-rotation-gpu`
- Ordered Triton area smoothing for spherical inflation, with the initial stages run by the bundled CPU reference
- PyTorch/Triton surface registration with FP64 DARTEL, initial rotation, resampling, averaging, and output generation
- Reproducibly built Linux x86-64 reference and helper programs, with corresponding source
- Synthetic fixtures, CPU CI, CodeQL, and a manual CUDA workflow
- Platform wheel, Linux bundle, source archives, CycloneDX dependency inventory, and SHA-256 manifest

### Audited performance

The release audit used real left-hemisphere inputs on an NVIDIA GeForce RTX 2080 Ti with PyTorch 2.6.0+cu126:

| Component | CPU reference | GPU | Speedup | Max vertex error |
| --- | ---: | ---: | ---: | ---: |
| `CAT_SurfWarp` | 126.658 s | 35.061 s | 3.613x | 5.481e-5 |
| `CAT_Surf2Sphere` | 54.985 s | 10.689 s | 5.144x | 1.144e-5 |

Triangle arrays matched exactly, and spherical inflation ran all 4,999 sweeps. These are single-run release checks. The separate bilateral CAT registration measurement was **55.2573 s**, covering spherical inflation, topology upsampling, and registration. See [v0.1.0 benchmarks](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BENCHMARKS.md) for inputs, settings, timing scope, and paired comparisons.

All measurements above describe v0.1.0 and the recorded environment. They are not v0.2.0 performance results or complete CHARM runtimes.

### Verification

- CPU test suite: 33 passed, 2 skipped
- CUDA host suite: 34 passed, 1 expected skip for the no-GPU behavior test
- CI and CodeQL passed on the v0.1.0 release commit
- Wheel installation, three CLI entry points, bundled helper hashes, and release checksums verified

### Installation

Follow the [v0.1.0 README](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/README.md) to create the Conda environment and install the matching CUDA-enabled PyTorch stack and platform wheel. For the Python/Numba/CuPy implementation, see [v0.2.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0).

Licensed under GPL-3.0-or-later. Upstream CAT-Surface attribution and license information are included in `THIRD_PARTY_NOTICES.md` and the corresponding source archive.

### 中文说明

v0.1.0 是基于 PyTorch/Triton 的首个 GPU 加速版本。球面膨胀的前处理由随包提供的 CPU 参考程序完成，面积平滑由 Triton 加速；曲面配准使用 PyTorch/Triton 完成旋转、重采样、双精度 DARTEL 和结果输出，并使用 CPU 辅助程序。

上表保留了当时 RTX 2080 Ti 上的单次发布检查结果。另一次双半球 CAT 配准测量为 55.2573 秒，覆盖球面膨胀、拓扑上采样和配准，不包含完整 CHARM 流程。这些数字仅适用于 v0.1.0。

本说明于 2026-10-13 修订，原有 v0.1.0 标签和发行文件保持不变。
