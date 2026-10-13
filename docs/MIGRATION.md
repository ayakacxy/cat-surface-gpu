# Migrating from v0.1.0 to v0.2.0

[English](MIGRATION.md) · [简体中文](MIGRATION.zh-CN.md)

Version 0.2.0 replaces the earlier PyTorch/Triton implementation and CAT helper programs with a native Python implementation using NumPy, Numba, and CuPy. Install it in a separate environment when keeping v0.1.0 available for comparisons.

## Dependencies and files

| v0.1.0 | v0.2.0 |
| --- | --- |
| PyTorch and Triton | NumPy, SciPy, Numba, nibabel, and CuPy |
| Bundled CAT reference/helper executables | No bundled CAT executables or project-specific native extensions |
| Linux platform wheel | Python-only `py3-none-any` wheel; CUDA runtime still required |
| CAT-Surface `628b6851` | CAT-Surface `b0c2aa23` |

The package runs independently. Supply your own GIFTI surfaces and template.

## Commands

The main command names and input/output file arguments are retained:

- `cat-surf2sphere-gpu --input-surface ... --output-surface ...`
- `cat-surfwarp-gpu --source-surface ... --source-sphere ... --target-surface ... --target-sphere ... --output ...`

Remove v0.1.0 backend options such as `--kernel triton`, `--dtype`, `--preprocess-kernel`, `--areal-schedule`, `--areal-arithmetic`, `--rotation-feature-backend`, `--stencil-threads`, `--dartel-dtype`, `--squaring-kernel`, and `--cuda-graph`. They do not select backends in v0.2.0. Use `--threads` for CPU work and `--solver-backend` / `--solver-iterations` for the registration solver.

`--average` and its `--avg` alias enable averaging, which is already enabled by default. `--no-average` disables it. Initial rotation is enabled by default and can be disabled with `--no-rotate`.

The old standalone rotation command is not provided. Rotation is part of `cat-surfwarp-gpu`. See the current [README examples](../README.md#run).

## Python API

Replace `run_cat_surf2sphere_gpu` with `surf_to_sphere_cupy` and `run_cat_surface_gpu_pipeline` with `surfwarp_cupy`. The new functions operate on vertex and triangle arrays and use `Surf2SphereParameters` or `SurfWarpParameters`. They are not drop-in wrappers for the earlier file-oriented functions. See [API examples](IMPLEMENTATION.md).

## Results and performance

Both the upstream source revision and the implementation changed. The default v0.2.0 DARTEL solver uses 40 PCG iterations, whereas the official reference uses FMG. Keep v0.1.0 thresholds and timings attached to v0.1.0; establish new tolerances from same-input comparisons against the v0.2.0 reference revision.

The historical 55.2573-second bilateral measurement describes the v0.1.0 CAT registration chain. Measured v0.2.0 command timings and output differences are recorded in the [current benchmarks](BENCHMARKS_v0.2.0.md).

[Documentation index](README.md) · [Project README](../README.md)
