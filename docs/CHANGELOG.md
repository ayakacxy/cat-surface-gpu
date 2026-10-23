# Project Changelog

[English](CHANGELOG.md) · [简体中文](CHANGELOG.zh-CN.md)

## [0.3.0] - 2026-10-18

### Added

- CPU execution on Linux, Windows and macOS.
- MLX Metal backend for Apple Silicon.
- Device selection in both commands and the `surf_to_sphere()` API.
- CPU CI on Linux, Windows and macOS.

### Changed

- Accelerated CUDA smoothing with shared-memory tiles.
- Added FP32 feature solves and DARTEL while retaining FP64 geometry assembly and final exponentiation.
- Cached fixed topology and target features; removed repeated input checks within registration.
- Revised the English and Chinese guides and synchronized the packaged documentation.

[Release notes](RELEASE_v0.3.0.md) · [Benchmarks](BENCHMARKS_v0.3.0.md)

## [0.2.0] - 2026-10-13

### Added

- Python implementations of geometry, depth features, rotation, resampling and DARTEL registration.
- Parameter classes and array APIs for spherical inflation and registration.
- Degenerate-triangle checks and GIFTI metadata preservation.

### Changed

- Replaced PyTorch, Triton and bundled executables with NumPy, Numba and CuPy.
- Updated the CAT-Surface reference to `b0c2aa23`.
- Set the default registration solver to 40 FP64 PCG iterations.
- Switched to a Python-only wheel and updated command options.

### Fixed

- Division by zero when a fixed-iteration PCG solve reaches exactly zero residual.

[Release notes](RELEASE_v0.2.0.md) · [Benchmarks](BENCHMARKS_v0.2.0.md)

## [0.1.0] - 2026-08-21

### Added

- PyTorch/Triton spherical inflation and FP64 DARTEL registration.
- Initial rotation, spherical resampling, averaging and GIFTI output.
- Linux x86-64 CPU reference and helper programs based on CAT-Surface `628b6851`, with source and build scripts.
- Command-line tools, Conda environment, CPU/CUDA tests and benchmarks.
- Platform wheel, Linux bundle, source archives, dependency inventory and SHA-256 checksums.

[Release notes](RELEASE_v0.1.0.md) · [Benchmarks](BENCHMARKS.md)

[0.3.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.3.0
[0.2.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0
