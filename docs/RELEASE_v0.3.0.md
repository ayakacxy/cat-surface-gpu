# CAT-Surface GPU v0.3.0

[English](RELEASE_v0.3.0.md) · [简体中文](RELEASE_v0.3.0.zh-CN.md)

v0.3.0 adds CPU and Metal backends and accelerates CUDA spherical inflation and registration. Supported platforms are Linux, Windows and Apple Silicon macOS.

## What changed

- CUDA area smoothing uses shared-memory tiles and ordered updates. The default still performs 4,999 sweeps.
- Geometry assembly remains FP64. Feature solves and resident DARTEL use FP32; PCG still runs 40 iterations, with final exponentiation in FP64.
- Fixed topology and target features are cached by content and implementation. The two CLIs reuse validated topology instead of rebuilding it on every invocation.
- CPU and Metal share the registration workflow. Metal uses MLX for smoothing, features, spherical queries, rotation costs and DARTEL.
- Both existing commands and CUDA array APIs remain available. The package runs independently and requires no CAT executables.

## Platforms and installation

| Platform | Backend | Install |
| --- | --- | --- |
| Linux / Windows with NVIDIA GPU | CUDA / CuPy | `python -m pip install ".[cuda]"` |
| Apple Silicon macOS | Metal / MLX | `python -m pip install ".[metal]"` |
| Linux / Windows / macOS | CPU / NumPy / Numba | `python -m pip install .` |

Use Python 3.11 or newer. The CUDA extra selects CUDA 12; install a matching CuPy distribution separately for another CUDA runtime. Metal requires native Apple Silicon Python and a macOS version supported by [MLX](https://ml-explore.github.io/mlx/build/html/install.html).

Before installation or execution, keep temporary files and caches inside the project. On Linux/macOS:

```bash
mkdir -p .work/v030/{tmp,cache}
export TMPDIR="$PWD/.work/v030/tmp" TMP="$PWD/.work/v030/tmp" TEMP="$PWD/.work/v030/tmp"
export XDG_CACHE_HOME="$PWD/.work/v030/cache"
export PIP_CACHE_DIR="$PWD/.work/v030/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/v030/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/v030/cache/cupy"
export CUDA_CACHE_PATH="$PWD/.work/v030/cache/cuda"
export CAT_SURFACE_CACHE_ROOT="$PWD/.work/v030/cache/targets"
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .work/v030/tmp, .work/v030/cache | Out-Null
$env:TMPDIR = "$PWD/.work/v030/tmp"
$env:TMP = $env:TMPDIR
$env:TEMP = $env:TMPDIR
$env:XDG_CACHE_HOME = "$PWD/.work/v030/cache"
$env:PIP_CACHE_DIR = "$env:XDG_CACHE_HOME/pip"
$env:NUMBA_CACHE_DIR = "$env:XDG_CACHE_HOME/numba"
$env:CUPY_CACHE_DIR = "$env:XDG_CACHE_HOME/cupy"
$env:CUDA_CACHE_PATH = "$env:XDG_CACHE_HOME/cuda"
$env:CAT_SURFACE_CACHE_ROOT = "$env:XDG_CACHE_HOME/targets"
```

Select `--device cuda`, `--device cuda:1`, `--device metal`, or `--device cpu` on either command. `--device auto` chooses an available backend; the default remains `cuda` for compatibility. An explicit unavailable device raises an error.

```bash
cat-surf2sphere-gpu --input-surface white.gii --output-surface sphere.gii --device metal
cat-surfwarp-gpu --source-surface white.gii --source-sphere sphere.gii --target-surface template.white.gii --target-sphere template.sphere.gii --output sphere.reg.gii --device metal
```

For arrays, use `surf_to_sphere(..., device="cpu" | "cuda" | "metal")`. The existing `surfwarp_cupy(...)` accepts `backend="cpu" | "cupy" | "metal"`; the CUDA name is retained for compatibility. Metal currently uses fixed-budget PCG (`pcg_check_interval=0`). CUDA/CPU retain explicit PCG/FMG options.

## Performance

Measured on RTX 2080 Ti, Xeon Gold 6230, Python 3.11 and 8 threads, using identical 163,842-vertex fsaverage inputs and 40 PCG iterations. These are medians of three interleaved fresh-process runs, including input reading, validation, populated-cache loading, computation and compressed output. Each number is for one surface.

| Complete command | v0.2.0 | v0.3.0 | Speedup |
| --- | ---: | ---: | ---: |
| Surf2Sphere | 7.968 s | 2.612 s | 3.05× |
| SurfWarp | 20.479 s | 4.983 s | 4.11× |

The bilateral test uses resampled public fsaverage surfaces: sphere generation at 61,442 vertices, followed by upsampling and registration at 245,762 vertices. With inputs and fixed topology preloaded, serial hemispheres and all four compressed GIFTI files written, it takes **9.731 seconds**. Summed internal generation/registration timers account for **5.771 seconds**.

See the [v0.3.0 benchmark record](benchmark-v0.3.0.json) for input hashes, settings, cold calls, repeated measurements and numerical comparisons. Single-surface CLI times and a bilateral CAT workflow have different timing boundaries and are reported separately.

## Numerical checks

Regression tests on Linux completed with **60 passed, 1 skipped**, covering CPU and CUDA calculations. The skipped test requires macOS and Metal. The CPU CI matrix is configured for Linux, Windows and macOS.

The new FP32 path preserves settings and topology but changes floating-point results. Against v0.2.0 on the public fsaverage test, the maximum angular difference was approximately 0.00023° for generated spheres and 0.094° for registered spheres. These are output comparisons for this input, not anatomical-accuracy measurements. The topology-cache change is checked separately for identical output coordinates and triangle arrays.

`CAT_SURFACE_CUDA_OPTIMIZED=0` selects the earlier CUDA arithmetic for diagnostics. It is an explicit option, not an automatic fallback. The independent package also preserves v0.2.0 parameter validation, GIFTI checks, and zero-residual handling in fixed-iteration PCG.

Source revisions and file hashes are recorded in the [source manifest](../upstream/SOURCE_MANIFEST.json).
