# CAT-Surface GPU v0.1.0

[English](RELEASE_v0.1.0.md) · [简体中文](RELEASE_v0.1.0.zh-CN.md)

Released on 2026-08-21.

The first release provides GPU acceleration for CAT-Surface spherical inflation and registration using PyTorch and Triton, with bundled CPU reference and helper programs.

## Features

- `cat-surf2sphere-gpu`, `cat-surfwarp-gpu` and `cat-surface-rotation-gpu` commands.
- Ordered Triton area smoothing, with the initial inflation stages handled by the CPU reference.
- Surface registration with FP64 DARTEL, initial rotation, resampling, averaging and output writing.
- Linux x86-64 reference and helper executables, corresponding source and reproducible build scripts.
- Platform wheel, Linux bundle, source archives, CycloneDX dependency inventory and SHA-256 checksums.

The CAT-Surface reference is `628b6851d8638f3ab773cd25c0ec406d0ec61ede`. The software is licensed under GPL-3.0-or-later.

## Release checks

A paired left-hemisphere check on an RTX 2080 Ti with PyTorch 2.6.0+cu126 produced the following results:

| Component | CPU reference | GPU | CPU/GPU ratio | Maximum reported error |
| --- | ---: | ---: | ---: | ---: |
| Surf2Sphere | 54.985 s | 10.689 s | 5.144× | 1.144e-5 |
| SurfWarp | 126.658 s | 35.061 s | 3.613× | 5.481e-5 |

These are single-run checks after rebuilding the bundled programs. Triangle arrays matched exactly, and sphere generation executed all 4,999 sweeps. The separate bilateral workflow took 55.2573 seconds for sphere generation, topology upsampling and registration. See the [benchmark report](BENCHMARKS.md) for input sizes, timing definitions and repeated helper measurements.

The CPU test suite completed with 33 passes and two skips; the CUDA host suite completed with 34 passes and one expected no-GPU-test skip. CI, CodeQL, wheel installation, CLI entry points and release checksums were checked for the release.

[Download v0.1.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0) · [Installation for this release](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/README.md)
