# CAT-Surface GPU

Python implementation of CAT-Surface spherical inflation and cortical surface registration, with CuPy CUDA kernels.

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![CI](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Release notes](docs/RELEASE_v0.2.0.md) · [Upstream credits](THIRD_PARTY_NOTICES.md)

Version **0.2.0** brings the CAT implementation from FlashCHARM into this standalone package:

- `CAT_Surf2Sphere`: cortical surface inflation and projection onto a sphere.
- `CAT_SurfWarp`: depth features, initial rotation, spherical resampling, DARTEL registration, averaging, and registered-sphere output.

The implementation uses NumPy and Numba for CPU work and CuPy for CUDA kernels. It does not call CAT executables or load project-specific C/Cython extensions. The wheel contains Python source; Numba and CuPy still compile kernels at runtime, and the full workflow requires an NVIDIA GPU. PyTorch and Triton are no longer dependencies.

## Versions

| Version | Implementation | CAT-Surface source |
| --- | --- | --- |
| [0.2.0](docs/RELEASE_v0.2.0.md) | Python, NumPy/Numba, and CuPy; no bundled CAT programs | [`b0c2aa23`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d) |
| [0.1.0](docs/RELEASE_v0.1.0.md) | PyTorch/Triton acceleration with bundled CPU reference and helper programs | [`628b6851`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) |

The v0.1.0 measurements remain available in [historical benchmarks](docs/BENCHMARKS.md). They describe that release and do not measure v0.2.0. The upstream revision and default linear solver changed in v0.2.0, so outputs should be compared with the corresponding reference. See [migration notes](docs/MIGRATION.md).

## Install

Use Python 3.11 or newer and an NVIDIA driver compatible with your CuPy installation. Linux is the tested platform. The `cuda` extra installs `cupy-cuda12x` for CUDA 12.x; other CUDA configurations require a matching CuPy package.

From a checkout:

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
git checkout v0.2.0

mkdir -p .work/setup/{tmp,cache/pip,cache/numba,cache/cupy}
export TMPDIR="$PWD/.work/setup/tmp" TMP="$PWD/.work/setup/tmp" TEMP="$PWD/.work/setup/tmp"
export XDG_CACHE_HOME="$PWD/.work/setup/cache"
export PIP_CACHE_DIR="$PWD/.work/setup/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/setup/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/setup/cache/cupy"

python -m pip install '.[cuda]'
python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())'
```

The repository also provides an [environment file](environment.yml) for Conda users. For a wheel or source archive, see [building and packaging](docs/BUILDING.md).

## Run

Inputs and outputs are GIFTI surfaces. Each anatomical surface must share vertex ordering and triangles with its corresponding sphere. The source and template may have different meshes. Template and subject files are supplied by the user.

Before running, set writable work and kernel-cache directories inside the project:

```bash
mkdir -p .work/example/{tmp,cache/numba,cache/cupy,output}
export TMPDIR="$PWD/.work/example/tmp" TMP="$PWD/.work/example/tmp" TEMP="$PWD/.work/example/tmp"
export XDG_CACHE_HOME="$PWD/.work/example/cache"
export NUMBA_CACHE_DIR="$PWD/.work/example/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/example/cache/cupy"
```

Create a sphere:

```bash
cat-surf2sphere-gpu \
  --input-surface /path/to/white.gii \
  --output-surface .work/example/output/sphere.gii \
  --stop-at 10 --device cuda --threads 8
```

Register it to a template:

```bash
cat-surfwarp-gpu \
  --source-surface /path/to/white.gii \
  --source-sphere .work/example/output/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --output .work/example/output/sphere.reg.gii \
  --steps 2 --runs 2 --average \
  --solver-backend pcg --solver-iterations 40 \
  --device cuda --threads 8
```

The commands report stage progress and return an error if CUDA is unavailable. First runs include kernel compilation. Use `--help` for options and [implementation details](docs/IMPLEMENTATION.md) for supported parameters and numerical differences from CAT-Surface.

## Python API

```python
from cat_surface_gpu import (
    GiftiSurface,
    Surf2SphereParameters,
    load_gifti_surface,
    save_gifti_surface,
    surf_to_sphere_cupy,
)

surface = load_gifti_surface("white.gii")
sphere_vertices, timings = surf_to_sphere_cupy(
    surface.vertices, surface.faces, Surf2SphereParameters(stop_at=10)
)
save_gifti_surface(
    GiftiSurface(sphere_vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

`SurfWarpParameters` and `surfwarp_cupy` provide the registration API. See [implementation details](docs/IMPLEMENTATION.md) for the full call and supported inputs.

## Validation and development

The package includes synthetic fixtures and CPU/CUDA tests. A skipped CUDA test does not establish GPU correctness. Numerical comparisons should use the same inputs, parameters, and upstream revision, check triangle arrays exactly, and report coordinate differences. [Reproducibility](docs/REPRODUCIBILITY.md) describes the procedure; [contributing](docs/CONTRIBUTING.md) covers development.

## Credits and license

This implementation was adapted from FlashCHARM's CAT module. It builds on [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface), developed by Christian Gaser, and the surface-registration workflow used by [SimNIBS](https://github.com/simnibs/simnibs). DARTEL originates from John Ashburner's work.

Project code is distributed under [GPL-3.0-or-later](LICENSE). Upstream components retain their applicable notices. See [third-party notices](THIRD_PARTY_NOTICES.md) for source revisions and attribution.
