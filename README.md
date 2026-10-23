<div align="center">

# CAT-Surface GPU

Python implementations of `CAT_Surf2Sphere` and `CAT_SurfWarp` for CPU, CUDA and Metal.

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![CI](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA / Metal](https://img.shields.io/badge/GPU-CUDA%20%7C%20Metal-76B900.svg)](docs/IMPLEMENTATION.md)
[![Linux / Windows / macOS](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](docs/IMPLEMENTATION.md)

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Benchmarks](docs/BENCHMARKS_v0.3.0.md) · [Build from source](docs/BUILDING.md) · [Upstream credits](THIRD_PARTY_NOTICES.md)

⚡ **GPU-accelerated** · 🧠 **Surface registration** · 🖥️ **Cross-platform** · 📦 **Python package**

</div>

CAT-Surface GPU takes cortical surface meshes in GIFTI format and produces spheres for surface-based analysis. It provides two commands:

- **`cat-surf2sphere-gpu`** inflates a cortical surface and projects it onto a sphere.
- **`cat-surfwarp-gpu`** registers that sphere to a template using depth features, initial rotation and DARTEL.

The package runs independently. Supply your own cortical surfaces and template; MRI segmentation and cortical reconstruction are outside its scope.

## ⚡ Performance

Complete command timings on an RTX 2080 Ti and Xeon Gold 6230, using 8 CPU threads and 163,842-vertex fsaverage surfaces:

| Operation | Median time |
| --- | ---: |
| Sphere generation | 2.612 s |
| Spherical registration | 4.983 s |

Each value is for one surface and is the median of three fresh-process runs with populated caches. Timings include startup, input reading, computation and compressed output. Registration uses 40 PCG iterations. See the [benchmark report](docs/BENCHMARKS_v0.3.0.md) for input details, first-run costs and numerical comparisons.

## 🧩 Requirements

- Python 3.11 or newer; a dedicated environment is recommended.
- Linux, Windows or macOS for CPU execution.
- An NVIDIA GPU for CUDA, or Apple Silicon macOS for Metal.

<a id="install"></a>

## 🐍 Installation

Clone the repository, then install the backend you need:

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
```

| Platform | Backend | Install from the repository |
| --- | --- | --- |
| Linux or Windows with an NVIDIA GPU | CUDA / CuPy | `python -m pip install ".[cuda]"` |
| Apple Silicon macOS | Metal / MLX | `python -m pip install ".[metal]"` |
| Linux, Windows or macOS | CPU / NumPy / Numba | `python -m pip install .` |

The CUDA extra installs CuPy for CUDA 12.x and requires a compatible NVIDIA driver. For another CUDA runtime, install the base package and a matching CuPy distribution. Metal requires a native Apple Silicon Python environment.

Prebuilt wheels and source archives are available on the [Releases page](https://github.com/ayakacxy/cat-surface-gpu/releases/latest). See [building and installation](docs/BUILDING.md) for wheel installation, development environments and source builds.

<a id="run"></a>

## 🌐 Quick start: `CAT_Surf2Sphere`

The examples below use CUDA. Replace `--device cuda` with `--device metal` or `--device cpu` for another backend.

```bash
cat-surf2sphere-gpu --input-surface white.gii --output-surface sphere.gii --device cuda
```

`white.gii` must be a closed triangular surface suitable for spherical mapping. The output preserves the input triangle indices.

## 🧭 Quick start: `CAT_SurfWarp`

```bash
cat-surfwarp-gpu --source-surface white.gii --source-sphere sphere.gii --target-surface template.white.gii --target-sphere template.sphere.gii --output sphere.reg.gii --device cuda
```

Each surface and its sphere must share vertex ordering and triangles. The subject and template may have different mesh sizes. `sphere.reg.gii` contains the registered sphere on the subject's original topology.

### Choose a device

| Option | Behavior |
| --- | --- |
| `--device cuda` | Use the default NVIDIA GPU; this is the command default. |
| `--device cuda:1` | Use NVIDIA GPU 1. |
| `--device metal` | Use Apple Metal. |
| `--device cpu` | Run on the CPU. |
| `--device auto` | Use an available GPU backend, otherwise CPU. |

Both commands accept `--threads` (1–8, default 8) and `--quiet`. Progress goes to standard error and a final JSON summary goes to standard output. Run either command with `--help` for all options.

The first run includes kernel compilation. Topology and target caches are stored under `.work/cat-surface/cache` in the working directory; set `CAT_SURFACE_CACHE_ROOT` to choose another location.

## 📦 Python package

The wheel contains the Python implementation and CUDA and Metal kernel source. Cortical surfaces and templates are supplied separately. Release downloads also include the project source archive, the pinned CAT-Surface source, a dependency inventory and checksums.

## 🎯 Python API

```python
from cat_surface_gpu import GiftiSurface, load_gifti_surface, save_gifti_surface, surf_to_sphere

surface = load_gifti_surface("white.gii")
vertices, timings = surf_to_sphere(surface.vertices, surface.faces, device="cuda")
save_gifti_surface(
    GiftiSurface(vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

The [API guide](docs/IMPLEMENTATION.md) covers registration, parameters, return values and numerical precision.

## 🗂️ Repository layout

```text
src/cat_surface_gpu/          Python API, CLI, and CPU/CUDA/Metal backends
tools/                        Benchmark and synthetic GIFTI fixture tools
scripts/                      Build and release-verification tools
tests/                        CPU, CUDA and Metal tests
upstream/                     Source records and upstream license
docs/                         Usage, benchmarks and release notes
```

## 🛠️ Documentation and support

- [Documentation index](docs/README.md)
- [API and implementation](docs/IMPLEMENTATION.md)
- [Benchmarks](docs/BENCHMARKS_v0.3.0.md) and [reproducibility](docs/REPRODUCIBILITY.md)
- [Migration guide](docs/MIGRATION.md) and [changelog](docs/CHANGELOG.md)
- [Contributing](docs/CONTRIBUTING.md), [support](docs/SUPPORT.md) and [security](docs/SECURITY.md)

## 🙏 Credits and license

Use [CITATION.cff](CITATION.cff) to cite the software. The algorithms build on Christian Gaser's [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface), including DARTEL work by John Ashburner. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and source references.

CAT-Surface GPU is licensed under [GPL-3.0-or-later](LICENSE).
