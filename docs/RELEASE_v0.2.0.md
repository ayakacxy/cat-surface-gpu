## CAT-Surface GPU v0.2.0

[English](RELEASE_v0.2.0.md) · [简体中文](RELEASE_v0.2.0.zh-CN.md)

Version 0.2.0 provides a native Python implementation of `CAT_Surf2Sphere` and `CAT_SurfWarp` using NumPy, Numba, and CuPy. Spherical inflation and surface registration run independently, without the CAT executable helpers used by v0.1.0.

### Highlights

- Python implementations of spherical inflation, depth features, initial rotation, spherical resampling, DARTEL registration, averaging, and GIFTI output
- CuPy CUDA kernels and NumPy/Numba geometry routines; no PyTorch or Triton dependency
- Installed CLIs: `cat-surf2sphere-gpu` and `cat-surfwarp-gpu`, with stage progress reporting
- Array-based Python API: `Surf2SphereParameters`, `SurfWarpParameters`, `surf_to_sphere_cupy`, and `surfwarp_cupy`
- Python-only wheel, source archive, pinned CAT-Surface source, CycloneDX dependency inventory, and SHA-256 manifest
- Preserved GIFTI metadata, checks for degenerate triangles, and a fix for division by zero at an exactly zero PCG residual

### Compatibility

The fixed CAT-Surface reference is `b0c2aa2395ea4957a331b9444fb54c7716b6572d`. Registration defaults to 40 double-precision PCG iterations per linear solve, which follows a different numerical path from upstream FMG. The two main command names are retained, while Python APIs and backend options have changed. See the [migration guide](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/MIGRATION.md) and [implementation details](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/IMPLEMENTATION.md).

Numba and CuPy compile kernels at runtime. The `py3-none-any` wheel contains Python source; the full workflow still requires a compatible NVIDIA GPU and CuPy/CUDA installation. Historical v0.1.0 benchmarks describe that version only.

### Verification

- RTX 2080 Ti test suite: **57 passed**, including eight CUDA tests, with no skips
- Installed wheel: both CLIs completed on full fsaverage templates, passing the generated sphere directly to registration
- Each output retained **163,842 vertices and 327,680 triangles**, with finite coordinates, unchanged triangle arrays, and no zero-area triangles
- Wheel and source contents, Python source hashes, CLI entry points, and release checksums verified

The [validation record](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/docs/validation-v0.2.0.json) includes the environment, input hashes, and settings. These checks cover the stated templates; they do not establish accuracy across subjects or a speedup over the official C implementation.

### Installation

Use Python 3.11 or newer and an NVIDIA driver compatible with your CuPy build. The documented CUDA 12.x setup installs `cupy-cuda12x`. See the [English README](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.md) or [简体中文说明](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.2.0/README.zh-CN.md) for Conda and wheel installation commands.

Licensed under GPL-3.0-or-later. CAT-Surface attribution and license information are included in `THIRD_PARTY_NOTICES.md` and the corresponding source archive.
