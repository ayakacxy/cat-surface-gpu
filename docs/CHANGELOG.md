# Changelog

## [0.2.0] - 2026-10-13

- Replaced the PyTorch/Triton implementation and bundled CAT executables with the Python CAT implementation from FlashCHARM.
- Added NumPy/Numba geometry and topology routines, CuPy CUDA kernels, type-5 depth features, initial rotation, spherical resampling, and DARTEL registration.
- Updated the fixed CAT-Surface reference from `628b6851` to `b0c2aa23`.
- Added the `Surf2SphereParameters`, `SurfWarpParameters`, `surf_to_sphere_cupy`, and `surfwarp_cupy` API.
- Retained the `cat-surf2sphere-gpu` and `cat-surfwarp-gpu` commands with options for the new implementation.
- Fixed division by zero in PCG when the residual is exactly zero, including registration of identical surfaces. Iteration limits and nonzero calculations are unchanged.
- Added checks for zero-area triangles in inputs and outputs, with clear errors for degenerate meshes.
- Changed packaging to a Python-only wheel. CUDA execution still requires CuPy and an NVIDIA GPU.
- Rewrote installation and usage documentation and separated historical v0.1.0 benchmarks from current release information.

The public API and backend options changed. Default DARTEL solves use 40 PCG iterations; they do not follow the same numerical path as upstream FMG. See [migration notes](MIGRATION.md) and [v0.2.0 release notes](RELEASE_v0.2.0.md).

## [0.1.0] - 2026-08-21

The first standalone release accelerated CAT-Surface with PyTorch and Triton while retaining upstream CPU programs and two helper executables.

- Ordered GPU area smoothing for `CAT_Surf2Sphere`.
- GPU `CAT_SurfWarp` registration with FP64 DARTEL, CUDA Graph support, initial rotation, averaging, and output writing.
- Linux x86-64 reference/helper executables, corresponding source, and reproducible build scripts.
- Command-line tools, Conda environment, CPU/CUDA tests, and numerical/performance comparisons.
- Platform wheel, source archive, Linux bundle, upstream source archive, SBOM, and SHA-256 manifests.

The reference revision was `628b6851d8638f3ab773cd25c0ec406d0ec61ede`. The [v0.1.0 release notes](RELEASE_v0.1.0.md) and [benchmarks](BENCHMARKS.md) describe that implementation. Their wording has been clarified for the v0.2.0 documentation; the v0.1.0 tag and release files remain unchanged.

[0.2.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0
