# Building from source

[English](BUILDING.md) · [简体中文](BUILDING.zh-CN.md)

The package uses setuptools and requires Python 3.11 or newer. The wheel contains Python modules and embedded CUDA/Metal kernel source; backend libraries compile the kernels at runtime. Building a wheel does not require a GPU or a C/C++ compiler.

Prebuilt packages are available on the [Releases page](https://github.com/ayakacxy/cat-surface-gpu/releases/latest).

## Install a release wheel

Download the wheel from the release assets, then install it in your Python environment:

```bash
python -m pip install ./cat_surface_gpu-0.3.1-py3-none-any.whl
```

This installs CPU support. For GPU execution, also install `cupy-cuda12x>=13` for CUDA 12 or `mlx>=0.31` for Apple Silicon Metal. See the [backend requirements](../README.md#install).

## Prepare the environment

Clone the repository, enter its directory, and create a virtual environment:

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
mkdir -p .work/build/{tmp,cache,dist}
export TMPDIR="$PWD/.work/build/tmp" TMP="$PWD/.work/build/tmp" TEMP="$PWD/.work/build/tmp"
export XDG_CACHE_HOME="$PWD/.work/build/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export CUPY_CACHE_DIR="$XDG_CACHE_HOME/cupy"
export CAT_SURFACE_CACHE_ROOT="$XDG_CACHE_HOME/cat-surface"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$buildDir = Join-Path (Get-Location) ".work/build"
New-Item -ItemType Directory -Force "$buildDir/tmp", "$buildDir/cache", "$buildDir/dist" | Out-Null
$env:TMPDIR = $env:TMP = $env:TEMP = "$buildDir/tmp"
$env:XDG_CACHE_HOME = "$buildDir/cache"
$env:PIP_CACHE_DIR = "$buildDir/cache/pip"
$env:NUMBA_CACHE_DIR = "$buildDir/cache/numba"
$env:CUPY_CACHE_DIR = "$buildDir/cache/cupy"
$env:CAT_SURFACE_CACHE_ROOT = "$buildDir/cache/cat-surface"
```

Temporary files, configured caches and release archives are stored under `.work/build`. Install the development dependencies:

```bash
python -m pip install -e ".[test]"
```

The base installation supports CPU execution. Add `.[cuda]` for NVIDIA CUDA on Linux or Windows, or `.[metal]` for Apple Silicon macOS. See the [installation guide](../README.md#install) for backend requirements.

## Build and inspect the package

The following commands work on all three platforms:

```bash
python -m build --outdir .work/build/dist
python scripts/verify_release.py --wheel .work/build/dist/cat_surface_gpu-0.3.1-py3-none-any.whl
```

The build produces a wheel and a source archive. The `py3-none-any` wheel is platform-independent; NumPy, Numba, CuPy, and MLX provide their own platform-specific dependencies. The source archive also includes tests, tools, documentation, and license notices. Anatomical surfaces and templates are supplied separately by the user.

To check the installed command-line entry points:

```bash
python -m pip install --force-reinstall --no-deps .work/build/dist/cat_surface_gpu-0.3.1-py3-none-any.whl
cat-surf2sphere-gpu --help
cat-surfwarp-gpu --help
```

## Development checks

```bash
python scripts/verify_release.py
python -m ruff check src tests tools scripts setup.py
python -m ruff format --check src tests tools scripts setup.py
python -m pytest -q --basetemp .work/build/pytest -o cache_dir=.work/build/cache/pytest
```

GPU tests run when the required hardware and backend are available. Otherwise, pytest reports them as skipped. CI runs CPU tests, package checks, and command-line checks on Linux, Windows, and macOS. Numerical comparisons and timing procedures are described in [reproducibility](REPRODUCIBILITY.md).

## Assemble release assets

After building the wheel and source archive, place the CAT-Surface source archive identified by [UPSTREAM.lock](../UPSTREAM.lock) under `.work/build/source/`. Then run:

```bash
python scripts/build_release_artifacts.py --upstream-archive .work/build/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz --dist .work/build/dist
```

The script checks the archive hash and prepares:

| Asset | Contents |
| --- | --- |
| `cat_surface_gpu-0.3.1-py3-none-any.whl` | Installable Python package |
| `cat_surface_gpu-0.3.1.tar.gz` | Project source distribution |
| `CAT-Surface-<commit>.tar.gz` | Pinned algorithm reference source |
| `cat-surface-gpu-0.3.1.cdx.json` | CycloneDX dependency inventory |
| `SHA256SUMS` | SHA-256 hashes of the four assets above |

The reference source archive is distributed for source access and attribution; it is not installed by the wheel. [Third-party notices](../THIRD_PARTY_NOTICES.md) describe its license and role.

[Documentation](README.md) · [Project README](../README.md)
