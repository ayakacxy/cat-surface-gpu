# CAT-Surface GPU v0.2.0

[English](RELEASE_v0.2.0.md) · [简体中文](RELEASE_v0.2.0.zh-CN.md)

Released on 2026-10-13.

This release replaces the PyTorch/Triton implementation and bundled CAT executables with Python, NumPy, Numba and CuPy.

## Changes

- Implements spherical inflation, depth features, initial rotation, spherical resampling, DARTEL registration, averaging and GIFTI output in Python.
- Adds the `Surf2SphereParameters`, `SurfWarpParameters`, `surf_to_sphere_cupy()` and `surfwarp_cupy()` APIs.
- Retains the `cat-surf2sphere-gpu` and `cat-surfwarp-gpu` commands, with progress reporting and updated backend options.
- Preserves GIFTI metadata and rejects zero-area triangles in input and output meshes.
- Fixes division by zero when a fixed-iteration PCG solve reaches exactly zero residual.
- Ships a Python-only wheel, source archive, pinned CAT-Surface source, dependency inventory and SHA-256 checksums.

## Compatibility

The full workflow requires Python 3.11 or later, an NVIDIA GPU and a matching CuPy installation. PyTorch, Triton and the bundled CAT helper programs are no longer required. The Python API and backend options changed; see the [version-specific migration guide](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/MIGRATION.md).

The CAT-Surface reference is `b0c2aa2395ea4957a331b9444fb54c7716b6572d`. Registration defaults to 40 FP64 PCG iterations per linear solve. This differs from the reference's FMG solver.

## Validation and benchmarks

The RTX 2080 Ti test suite passed all 57 tests, including eight CUDA tests. Both installed commands completed a workflow on fsaverage templates, passing the generated sphere to registration. Outputs retained 163,842 vertices and 327,680 triangles, with finite coordinates, unchanged triangle arrays and no zero-area triangles.

A subsequent benchmark measured complete commands in three fresh processes with populated caches. GPU medians were 10.340 seconds for sphere generation, 32.550 seconds for PCG registration and 164.697 seconds for FMG registration. The [benchmark report](BENCHMARKS_v0.2.0.md) includes CPU comparisons, solver differences, first-call costs and reproduction commands. The [validation record](validation-v0.2.0.json) contains input hashes and settings.

[Download v0.2.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0) · [Installation for this release](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.md)
