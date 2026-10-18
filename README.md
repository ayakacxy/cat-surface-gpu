<div align="center">

# CAT-Surface GPU

Python implementations of `CAT_Surf2Sphere` and `CAT_SurfWarp` for CPU, CUDA and Metal.

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![CI](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA 12.6 tested](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](docs/RELEASE_v0.2.0.md)
[![Linux / Windows / macOS](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](docs/IMPLEMENTATION.md)

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Benchmarks](docs/RELEASE_v0.3.0.md) · [Build from source](docs/BUILDING.md) · [Upstream credits](THIRD_PARTY_NOTICES.md)

⚡ **GPU-accelerated** · 🧠 **Surface registration** · 🧪 **CUDA-tested** · 📦 **Python package**

</div>

CAT-Surface GPU provides two components for the cortical-surface registration workflow used by CAT-Surface and SimNIBS:

- `CAT_Surf2Sphere`: cortical surface inflation and projection onto a sphere.
- `CAT_SurfWarp`: depth features, initial rotation, spherical resampling, DARTEL registration, averaging, and registered-sphere output.

**v0.3.0** supports spherical inflation and registration with CuPy CUDA on Linux and Windows, MLX Metal on Apple Silicon macOS, and NumPy/Numba CPU execution on all three platforms. CUDA uses tiled smoothing, FP32 feature solves and DARTEL; final exponentiation remains FP64.

See the [v0.3.0 release notes](docs/RELEASE_v0.3.0.md) for installation, backend options, benchmarks and numerical comparisons.

## ⚡ v0.3.0 performance

Measured on RTX 2080 Ti, Xeon Gold 6230, Python 3.11 and 8 threads, using identical 163,842-vertex fsaverage inputs and 40 PCG iterations. These are medians of three interleaved fresh-process runs, including input reading, validation, populated-cache loading, computation and compressed output. Each number is for one surface.

| Complete command | v0.2.0 | v0.3.0 | Speedup |
| --- | ---: | ---: | ---: |
| Surf2Sphere | 7.968 s | 2.612 s | 3.05× |
| SurfWarp | 20.479 s | 4.983 s | 4.11× |

The bilateral CAT workflow takes **9.731 seconds** with preloaded inputs and all four compressed files written; internal calculation timers total **5.771 seconds**. See [v0.3.0 notes](docs/RELEASE_v0.3.0.md) for mesh sizes, fixture provenance, first calls and numerical comparisons.

## ⚡ Historical v0.2.0 validation and performance

Measured on an NVIDIA GeForce RTX 2080 Ti with Python 3.11 and CUDA 12.6. Each v0.2.0 input has 163,842 vertices and 327,680 triangles:

| Component | CPU median | GPU median | CPU/GPU time ratio | Max coordinate difference |
| --- | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | 216.722 s | 10.340 s | 20.96x | 0.057757 |
| `CAT_SurfWarp`, PCG (default) | 182.182 s | 32.550 s | 5.60x | 0.166157 |
| `CAT_SurfWarp`, FMG | 182.182 s | 164.697 s | 1.11x | 0.165802 |

Medians come from three complete command runs in fresh processes with populated disk caches, including startup, input/output, and computation. The CPU reference is CAT-Surface `b0c2aa23` using FMG. The default GPU PCG comparison includes solver differences; the outputs are not numerically identical. Error values use the GIFTI coordinate units. See [v0.2.0 benchmarks](docs/BENCHMARKS_v0.2.0.md) for ranges, first-run costs, all numerical differences, and reproduction commands.

The release also passed these checks:

| Check | Result |
| --- | --- |
| Test suite | **57 passed**, including 8 CUDA tests |
| Installed-wheel commands | Full fsaverage sphere generation followed by registration completed |
| Mesh size | 163,842 vertices and 327,680 triangles per input |
| Output checks | Finite coordinates and unchanged triangle arrays |

These checks establish that the released package runs the tested workflows. See the [v0.2.0 release notes](docs/RELEASE_v0.2.0.md) and [validation record](docs/validation-v0.2.0.json).

The following measurements belong to **v0.1.0**, which used PyTorch/Triton and CPU helper programs:

| Component | v0.1.0 workload | CPU reference | GPU | Speedup |
| --- | --- | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | LH, 61,442 vertices, `stop_at=10` | 54.84 s | 10.8119 s | 5.07x |
| `CAT_SurfWarp` | LH, `steps=2`, `runs=2`, `avg` | 119.9142 s | 45.0848 s | 2.66x |
| `CAT_SurfWarp` | RH, same settings | 119.4056 s | 46.9862 s | 2.54x |

The v0.1.0 bilateral CAT registration chain took **55.2573 s** on that setup. It did not include the complete CHARM workflow. Hardware, parameters, timing scope, and numerical results remain in [BENCHMARKS.md](docs/BENCHMARKS.md); these numbers do not measure v0.2.0.

## 🧩 Requirements

- Linux, Windows, or Apple Silicon macOS
- CUDA: NVIDIA GPU and a matching CuPy/driver; Metal: Apple GPU and MLX
- Python 3.11 or newer
- NumPy, SciPy, Numba, and nibabel
- CuPy for the complete CUDA workflow

The `cuda` extra installs `cupy-cuda12x` for CUDA 12.x. Other CUDA configurations require a matching CuPy package. The wheel uses `py3-none-any` because it contains no compiled project extension; the GPU workflow still needs the runtime above.

Install CPU support with `python -m pip install .`, Mac Metal with `python -m pip install ".[metal]"`, or Linux/Windows CUDA 12 with `python -m pip install ".[cuda]"`. Both commands accept `--device cpu`, `--device metal`, `--device cuda` and `--device auto`. An unavailable explicitly requested backend raises an error. The examples below use CUDA.

<a id="install"></a>

## 🐍 Conda installation

Clone the repository, set local cache directories, and create an environment:

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu

mkdir -p .work/setup/{tmp,cache/pip,cache/conda,cache/numba,cache/cupy}
export TMPDIR="$PWD/.work/setup/tmp" TMP="$PWD/.work/setup/tmp" TEMP="$PWD/.work/setup/tmp"
export XDG_CACHE_HOME="$PWD/.work/setup/cache"
export PIP_CACHE_DIR="$PWD/.work/setup/cache/pip"
export CONDA_PKGS_DIRS="$PWD/.work/setup/cache/conda"
export NUMBA_CACHE_DIR="$PWD/.work/setup/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/setup/cache/cupy"

conda env create -f environment.yml --prefix "$PWD/.work/setup/env"
conda activate "$PWD/.work/setup/env"
python -m pip install '.[cuda,test]'
```

An existing Python environment can use the same `pip` command without creating a Conda environment. Wheel and source-archive instructions are in [BUILDING.md](docs/BUILDING.md).

Verify the environment and run the tests:

```bash
python - <<'PY'
import cupy

print("CuPy:", cupy.__version__)
print("CUDA runtime:", cupy.cuda.runtime.runtimeGetVersion())
print("CUDA devices:", cupy.cuda.runtime.getDeviceCount())
PY

python -m pytest -q --basetemp .work/setup/tmp/pytest
```

CPU tests require no GPU. CUDA tests require an NVIDIA GPU; Metal tests require Apple Silicon macOS. Tests for unavailable backends are skipped.

<a id="run"></a>

## 🌐 Quick start: `CAT_Surf2Sphere`

Inputs and outputs are GIFTI surfaces. The input must be a closed triangular mesh suitable for spherical mapping. Template and subject files are supplied by the user.

Set writable work and kernel-cache directories inside the project:

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
  --stop-at 10 \
  --device cuda \
  --threads 8
```

The default `stop_at=10` retains 4,999 area-smoothing sweeps. Stage progress is printed during execution; the first run also includes kernel compilation.

## 🧭 Quick start: `CAT_SurfWarp`

Register the sphere to a template:

```bash
cat-surfwarp-gpu \
  --source-surface /path/to/white.gii \
  --source-sphere .work/example/output/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --output .work/example/output/sphere.reg.gii \
  --steps 2 \
  --runs 2 \
  --average \
  --solver-backend pcg \
  --solver-iterations 40 \
  --device cuda \
  --threads 8
```

Each anatomical surface must share vertex ordering and triangles with its corresponding sphere. The source and template may use different meshes. An explicitly requested unavailable backend raises an error. Use `--help` for the full option list.

## 📦 Python package

| Version | Implementation | CAT-Surface reference |
| --- | --- | --- |
| [v0.3.0](docs/RELEASE_v0.3.0.md) | CPU, CUDA and Metal; tiled smoothing, FP32 registration and topology reuse | `b0c2aa23` |
| [v0.2.0](docs/RELEASE_v0.2.0.md) | Python, NumPy/Numba, and CuPy; no bundled CAT programs | [`b0c2aa23`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d) |
| [v0.1.0](docs/RELEASE_v0.1.0.md) | PyTorch/Triton with bundled CPU reference and helper programs | [`628b6851`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) |

The v0.3.0 wheel contains the Python implementation and inline CUDA and Metal kernel source. It does not include anatomical data or templates. The release also provides a source archive, the pinned CAT-Surface source archive, a dependency inventory, and checksums.

v0.3.0 retains the v0.2.0 commands and CUDA APIs and adds CPU and Metal backends. See [MIGRATION.md](docs/MIGRATION.md) for version-specific upgrade steps.

## 🎯 Numerical checks and Python API

- Triangle arrays retain their source indexing.
- CUDA/Metal use FP32 DARTEL with 40 PCG iterations per linear solve; CPU and final exponentiation use FP64.
- PCG follows a different numerical path from upstream FMG; comparisons must record the solver and reference revision.
- The supported registration path uses type-5 features for one or two steps. See [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for available settings.
- The v0.1.0 numerical thresholds should not be assumed to apply to v0.2.0.

The public Python namespace remains `cat_surface_gpu`:

```python
from cat_surface_gpu import (
    GiftiSurface,
    Surf2SphereParameters,
    load_gifti_surface,
    save_gifti_surface,
    surf_to_sphere,
)

surface = load_gifti_surface("white.gii")
sphere_vertices, timings = surf_to_sphere(
    surface.vertices, surface.faces, Surf2SphereParameters(stop_at=10), device="cuda"
)
save_gifti_surface(
    GiftiSurface(sphere_vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

`SurfWarpParameters` and `surfwarp_cupy` provide the registration API. See [implementation details](docs/IMPLEMENTATION.md) for the full call and input requirements.

## 🗂️ Repository layout

```text
src/cat_surface_gpu/          Python API, CLI, and CPU/CUDA/Metal backends
tools/                        Synthetic GIFTI fixture generator
scripts/                      Build and release-verification tools
tests/                        CPU, CUDA and Metal tests
upstream/                     Source records and upstream license
docs/                         Usage, validation, benchmarks, and release notes
```

## 🧪 Synthetic smoke fixture

The repository does not distribute anatomical or subject data. Generate deterministic, non-anatomical GIFTI files for file I/O and installation checks:

```bash
python tools/generate_synthetic_fixture.py --output .work/example/smoke
python scripts/verify_release.py
```

This small fixture is for structural checks. It does not establish anatomical registration accuracy or replace the full fsaverage workflow checks reported for v0.2.0.

## 🛠️ Development and release integrity

- [BUILDING.md](docs/BUILDING.md) describes wheel, source, and release-file assembly.
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) explains numerical comparisons and timing.
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) covers development and required evidence.
- [SECURITY.md](docs/SECURITY.md) explains vulnerability reporting.
- [CHANGELOG.md](docs/CHANGELOG.md) records release-visible changes.

## 🙏 Credits and license

This project builds on [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface), developed by Christian Gaser, and the CAT surface-registration workflow used by [SimNIBS](https://github.com/simnibs/simnibs). DARTEL originates from John Ashburner's work. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source revisions and attribution.

Project code is distributed under [GPL-3.0-or-later](LICENSE). Upstream components retain their applicable notices.
