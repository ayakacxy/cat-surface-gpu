# Version Migration Guide

[English](MIGRATION.md) · [简体中文](MIGRATION.zh-CN.md)

For a new installation, start with the [README](../README.md#install). This guide covers changes that affect existing scripts and environments.

## From v0.2 to v0.3

Command names, file arguments, and CUDA array APIs remain compatible. The default device is still `cuda`; existing CUDA commands do not need a new device argument.

CPU and Apple Metal are now available for the complete workflow:

| Backend | Installation from a checkout | CLI option |
| --- | --- | --- |
| CPU | `python -m pip install .` | `--device cpu` |
| NVIDIA CUDA | `python -m pip install ".[cuda]"` | `--device cuda` or `--device cuda:N` |
| Apple Metal | `python -m pip install ".[metal]"` | `--device metal` |

`--device auto` selects an available GPU, or CPU if no supported GPU is available. Explicit device selection reports an error when that backend is unavailable.

### Python calls

Use `surf_to_sphere(..., device="cuda")` for inflation. Accepted device values are `cpu`, `cuda`, `metal` and `auto`. `surf_to_sphere_cupy(...)` remains available for existing CUDA code.

Registration uses `surfwarp_cupy(..., backend="cupy")`, with `cpu`, `cupy` and `metal` as the accepted backend values. The CUDA backend name is `cupy`, and this function does not accept `auto`. Metal requires PCG with `pcg_check_interval=0`, which is the default. See the [Python API](IMPLEMENTATION.md) for complete examples and result fields.

### Numerical results

The default CUDA path now uses FP32 feature solves and DARTEL updates, while final velocity exponentiation remains FP64. Optimized kernels and cache reuse also change runtime. Numerical differences and same-input timing comparisons are reported in the [v0.3 benchmarks](BENCHMARKS_v0.3.0.md).

## From v0.1

The Python package no longer needs PyTorch, Triton, or CAT helper executables. Its base dependencies are NumPy, SciPy, Numba, and nibabel; GPU support is installed through the CUDA or Metal extra. Use a separate environment if you need to keep the original implementation available for comparison.

### Command-line changes

The two main command names and their file arguments are unchanged:

```text
cat-surf2sphere-gpu --input-surface ... --output-surface ...
cat-surfwarp-gpu --source-surface ... --source-sphere ... --target-surface ... --target-sphere ... --output ...
```

Remove the old options `--kernel`, `--dtype`, `--preprocess-kernel`, `--areal-schedule`, `--areal-arithmetic`, `--rotation-feature-backend`, `--stencil-threads`, `--dartel-dtype`, `--squaring-kernel`, and `--cuda-graph`. Select a backend with `--device`, CPU threads with `--threads`, and the registration solver with `--solver-backend` and `--solver-iterations`.

Initial rotation and averaging are enabled by default. `--no-rotate` and `--no-average` disable them; `--avg` is an alias for `--average`. Rotation is part of `cat-surfwarp-gpu`; there is no separate rotation command.

### Python and reference changes

The file-oriented functions `run_cat_surf2sphere_gpu` and `run_cat_surface_gpu_pipeline` were replaced by array APIs. Load GIFTI files with `load_gifti_surface`, call `surf_to_sphere` or `surfwarp_cupy`, and write results with `save_gifti_surface`. See the [API guide](IMPLEMENTATION.md).

The algorithm reference changed from CAT-Surface `628b6851` to `b0c2aa23`, and the default registration solver is PCG rather than the reference FMG. Compare outputs on the same inputs when updating a numerical baseline. Original release records remain available in the [changelog](CHANGELOG.md).

[Documentation](README.md) · [Project README](../README.md)
