<div align="center">

# CAT-Surface GPU

GPU-accelerated `CAT_Surf2Sphere` and `CAT_SurfWarp`-compatible surface registration.

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ayakacxy/cat-surface-gpu?display_name=tag&sort=semver)](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)
[![CI](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/cat-surface-gpu/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA 12.6 tested](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](docs/BENCHMARKS.md)
[![Linux x86-64](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](bin/linux-x86_64)

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Benchmarks](docs/BENCHMARKS.md) · [Build from source](docs/BUILDING.md) · [Upstream credits](THIRD_PARTY_NOTICES.md)

⚡ **GPU-accelerated** · 🧠 **Surface registration** · 🧪 **Reference-checked** · 📦 **Ready-to-run binaries**

</div>

CAT-Surface GPU provides two optimized components for the cortical-surface registration path used by CAT-Surface and SimNIBS:

- `CAT_Surf2Sphere`: topology-aware Triton kernels accelerate the area-preserving spherical inflation loop while preserving the upstream iteration count and dependency order.
- `CAT_SurfWarp`: a PyTorch/Triton pipeline accelerates initial rotation, surface resampling, DARTEL optimization, averaging, and final warp application. It is compatible with the `CAT_WarpSurf` role in the SimNIBS surface-registration workflow.

The CPU reference path remains explicit. Requesting CUDA without a usable GPU or requesting Triton without a working kernel raises an error; the implementation does not silently fall back to CPU.

## ⚡ Performance at a glance

Measured on an NVIDIA GeForce RTX 2080 Ti with Python 3.11, PyTorch `2.6.0+cu126`, and Triton `3.2.0`. Every comparison uses the same input, parameters, output topology, and an explicit vertex-error contract.

| Component | Workload | CPU reference | GPU | Speedup | Max vertex error |
| --- | --- | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | LH, 61,442 vertices, `stop_at=10` | 54.84 s | 10.8119 s | 5.07x | 1.14e-5 |
| `CAT_SurfWarp` | LH, `steps=2`, `runs=2`, `avg` | 119.9142 s | 45.0848 s | 2.66x | 2.85e-5 |
| `CAT_SurfWarp` | RH, `steps=2`, `runs=2`, `avg` | 119.4056 s | 46.9862 s | 2.54x | 3.41e-4 |

With the SimNIBS default two-hemisphere schedule, the measured GPU wall time was:

| Stage | Bilateral wall time |
| --- | ---: |
| `CAT_Surf2Sphere` | 15.7609 s |
| Topology upsampling | 4.5208 s |
| `CAT_SurfWarp` | 34.9756 s |
| **Total CAT registration chain** | **55.2573 s** |

Final triangle arrays were exact. Final LH/RH `sphere.reg` maximum vertex errors were `1.38e-6` and `1.81e-4`, both below the `1e-3` acceptance threshold. See [BENCHMARKS.md](docs/BENCHMARKS.md) for hardware, shapes, parameters, timing boundaries, stage profiles, and numerical results.

## 🧩 Requirements

- Linux x86-64
- NVIDIA GPU with a CUDA-compatible driver
- Conda or Miniconda
- Python 3.11
- PyTorch 2.6 and Triton 3.2 for the validated environment

The repository includes stripped Linux x86-64 CPU reference/helper binaries requiring glibc 2.29 or newer. Other platforms can use the Python implementation but must rebuild the CAT-Surface reference and helper programs from source.

## 🐍 Conda installation

Clone the repository and create the base environment:

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu

conda env create -f environment.yml
conda activate cat-surface-gpu
```

Install the CUDA 12.6 build used for the reported benchmarks, then install this project:

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install '.[cuda,test]'
```

Verify the environment:

```bash
python - <<'PY'
import torch
import triton

print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pytest -q
```

CPU-only test collection is supported, but CUDA tests are skipped when no GPU is visible. GPU performance and numerical claims require a real CUDA run.

## 🌐 Quick start: `CAT_Surf2Sphere`

The optimized path uses the bundled upstream CPU reference for the first five stages and runs the 4,999 area-smoothing sweeps on CUDA. No iteration is removed.

```bash
cat-surf2sphere-gpu \
  --input-surface /path/to/coarse.white.gii \
  --output-surface /path/to/sphere.gii \
  --stop-at 10 \
  --device cuda \
  --dtype float32 \
  --kernel triton \
  --preprocess-kernel cpu \
  --areal-schedule ordered \
  --areal-arithmetic cat
```

Run a CPU/GPU A/B with topology and error checks:

```bash
python tools/benchmark_cat_surf2sphere_gpu.py \
  --reference-cli bin/linux-x86_64/CAT_Surf2Sphere \
  --input-surface /path/to/coarse.white.gii \
  --reference-output /tmp/sphere.cpu.gii \
  --gpu-output /tmp/sphere.gpu.gii \
  --device cuda \
  --kernel triton \
  --areal-schedule ordered \
  --areal-arithmetic cat \
  --max-vertex-error 1e-3 \
  --max-mean-error 1e-3 \
  --max-p99-error 1e-3
```

## 🧭 Quick start: `CAT_SurfWarp`

The full registration runner includes input/output, initial rotation, DARTEL, `-avg`, and GIFTI writing in its reported wall time.

```bash
cat-surfwarp-gpu \
  --source-surface /path/to/white.gii \
  --source-sphere /path/to/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --output /path/to/sphere.reg.gii \
  --rotation-feature-backend cuda-official-depth \
  --stencil-threads 8 \
  --device cuda \
  --kernel triton \
  --dartel-dtype float64 \
  --squaring-kernel triton \
  --steps 2 \
  --runs 2 \
  --avg \
  --cuda-graph
```

Use `tools/benchmark_cat_surface_end_to_end.py` to compare this path against the bundled `bin/linux-x86_64/CAT_SurfWarp` CPU reference on the same inputs.

## 📦 Bundled Linux binaries

| File | Purpose |
| --- | --- |
| `CAT_Surf2Sphere` | Upstream CPU reference used by the hybrid spherical-inflation runner |
| `CAT_SurfWarp` | Upstream CPU reference for end-to-end A/B |
| `cat_surface_rotation_depth` | Exports upstream-compatible raw depth features for initial rotation |
| `cat_surface_stencil_builder` | Builds deterministic surface-resampling stencils; uses 8 CPU workers by default and accepts `--threads N` |

All four files are stripped x86-64 ELF executables built from [CAT-Surface commit `628b6851`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) with `-O2 -fPIC`. They link only to glibc, libm, and pthread at runtime. Checksums are in [`bin/linux-x86_64/SHA256SUMS`](bin/linux-x86_64/SHA256SUMS). Source files for the two helpers are available in [`native/`](native/), and the complete build procedure is in [BUILDING.md](docs/BUILDING.md).

The Python runners expose the same setting as `--stencil-threads`. The default of 8 avoids excessive CPU oversubscription when source/target work and both hemispheres run concurrently; high-core-count systems can select 16 or 32 explicitly.

## 🎯 Numerical contract

- Mesh topology must match the CPU reference exactly.
- `CAT_Surf2Sphere` ordered scheduling preserves upstream vertex dependencies and executes 4,999 sweeps for `stop_at=10`.
- The recommended warp path uses FP64 DARTEL and retains the upstream `steps`, `runs`, `loop`, `cycles`, `nit`, `its`, `code`, and `avg` semantics.
- The default public acceptance threshold is `max/mean/p99 <= 1e-3` for final vertex coordinates.
- Experimental all-GPU feature paths remain opt-in and are not used by the recommended commands above.

The installed public Python namespace is `cat_surface_gpu`:

```python
from cat_surface_gpu import run_cat_surf2sphere_gpu, run_cat_surface_gpu_pipeline
```

## 🗂️ Repository layout

```text
src/cat_surface_gpu/          Public API, GPU kernels, and registration implementation
tools/                        CLI runners, benchmarks, and A/B utilities
native/                       Source for the two bundled helper programs
scripts/                      Build and release-verification automation
tests/                        CPU contracts and CUDA tests
bin/linux-x86_64/             Bundled reference/helper executables
docs/                         Benchmarks, build, reproducibility, and community guides
```

## 🧪 Synthetic smoke fixture

The repository does not distribute anatomical or subject data. Generate deterministic, non-anatomical GIFTI files for installation and I/O smoke tests:

```bash
python tools/generate_synthetic_fixture.py --output /tmp/cat-surface-gpu-smoke
python scripts/verify_release.py
```

This fixture verifies packaging and structural contracts; it is not a substitute for the real-input numerical and performance A/B in [BENCHMARKS.md](docs/BENCHMARKS.md).

## 🛠️ Development and release integrity

- [BUILDING.md](docs/BUILDING.md) documents the pinned upstream source and native rebuild.
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) defines provenance and validation layers.
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) defines the scientific contract for changes.
- [SECURITY.md](docs/SECURITY.md) explains private vulnerability reporting and medical-data safety.
- [CHANGELOG.md](docs/CHANGELOG.md) records release-visible changes.

## 🙏 Credits and license

This project builds on the original [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface) implementation by Christian Gaser and on the CAT surface-registration workflow used by [SimNIBS](https://github.com/simnibs/simnibs). Their work, including the upstream DARTEL implementation and surface I/O stack, made this acceleration possible. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact source and license information.

The new code in this repository is licensed under [GPL-3.0-or-later](LICENSE). Bundled upstream-derived binaries and upstream components retain their applicable licenses.
