# CAT-Surface GPU v0.3.0

[English](RELEASE_v0.3.0.md) · [简体中文](RELEASE_v0.3.0.zh-CN.md)

Released on 2026-10-18.

This release adds CPU and Apple Metal support and improves CUDA performance for spherical inflation and registration.

## Added

- CPU execution on Linux, Windows and macOS using NumPy and Numba.
- Metal execution on Apple Silicon using MLX.
- Device selection in both commands: `cuda`, `cuda:N`, `metal`, `cpu` and `auto`.
- The `surf_to_sphere()` array API with an explicit `device` argument.
- CPU CI on Linux, Windows and macOS.

## Performance improvements

- CUDA area smoothing uses shared-memory tiles while preserving the ordered updates and 4,999 default sweeps.
- Feature solves and DARTEL use FP32 on the optimized GPU path. Geometry assembly and final exponentiation retain FP64; the default PCG budget remains 40 iterations.
- Fixed topology and target features are cached. Repeated registration runs reuse input validation and topology setup.

On an RTX 2080 Ti, complete single-surface commands take a median **2.612 seconds** for sphere generation and **4.983 seconds** for registration. These measurements use 163,842-vertex fsaverage surfaces, populated caches and three fresh processes. See the [benchmark report](BENCHMARKS_v0.3.0.md) for settings, first-call times and output comparisons.

## Compatibility

Python 3.11 or later is required. Both command names and the existing CUDA array APIs remain available. The default command-line device is `cuda`; choose `--device auto` to select an available backend automatically. An explicitly requested device must be available.

`surfwarp_cupy()` now accepts `backend="cpu"`, `"cupy"` or `"metal"`. Metal uses PCG with `pcg_check_interval=0`; CPU and CUDA also support FMG. See the [migration guide](MIGRATION.md) for API changes and the [README](../README.md) for installation.

The optimized GPU path produces small floating-point differences from the previous implementation. Parameter validation, GIFTI checks and zero-residual handling in fixed-iteration PCG are retained. Linux CPU/CUDA regression tests completed with 60 passes; the Metal test was skipped on Linux.

[Download v0.3.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.3.0)
