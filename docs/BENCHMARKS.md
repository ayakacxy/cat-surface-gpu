# Benchmarks: v0.1.0

[English](BENCHMARKS.md) · [简体中文](BENCHMARKS.zh-CN.md)

This report covers the PyTorch/Triton implementation with bundled CAT CPU programs. Measurements use one anonymized cortical surface pair and the SimNIBS fsaverage template. Subject data is not distributed with the project.

## Environment and timing

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GiB |
| Python / PyTorch | 3.11 / 2.6.0+cu126 |
| CUDA runtime / Triton | 12.6 / 3.2.0 |
| CAT-Surface reference | `628b6851d8638f3ab773cd25c0ec406d0ec61ede` |
| Reference build | GCC, `-O2 -fPIC`, without `-ffast-math` |
| Registration precision | FP64 DARTEL |
| Sphere precision | FP32 storage with double-precision intermediates |

CPU and GPU receive the same GIFTI inputs and parameters. Wall times include parsing, CPU helper work, device setup, synchronized computation, output transfer and GIFTI writing. Bilateral measurements use two hemisphere processes sharing one GPU. The reported workflow ends at spherical registration; it does not include segmentation or cortical surface reconstruction.

## Sphere generation

### Coarse mesh

Each hemisphere has 61,442 vertices and 122,880 triangles. The first five stages run on the bundled CPU reference, followed by ordered Triton area smoothing. With `stop_at=10`, the smoothing loop executes all 4,999 sweeps.

| Hemisphere | CPU reference | GPU total | Area smoothing | Absolute error: max / mean / p99 |
| --- | ---: | ---: | ---: | ---: |
| LH | 54.84 s | 10.8119 s | 3.7084 s | 1.1444e-5 / 1.7699e-6 / 7.6294e-6 |
| RH | Not measured in this run | 10.6667 s | 3.5142 s | 1.5259e-5 / 1.6752e-6 / 7.6294e-6 |

Triangle arrays match exactly. The LH CPU/GPU time ratio is 5.07×. When both hemispheres run concurrently, each process takes approximately 13.81 seconds and the bilateral stage completes in 15.7609 seconds.

### Full-resolution mesh

These earlier measurements use 245,762 vertices and 491,520 triangles per hemisphere, with the explicit FP32/Triton implementation. The intermediate-sphere error limits were `5e-3 / 1e-3 / 3e-3` for maximum, mean and p99 error.

| Hemisphere | CPU reference | GPU | CPU/GPU ratio | Absolute error: max / mean / p99 |
| --- | ---: | ---: | ---: | ---: |
| LH | 234.00 s | 37.8887 s | 6.18× | 3.073e-3 / 6.20e-4 / 2.407e-3 |
| RH | 236.35 s | 37.4264 s | 6.32× | 3.170e-3 / 6.12e-4 / 2.398e-3 |

Triangle arrays match exactly. The coarse ordered implementation used for the bilateral workflow meets the tighter `1e-3` maximum-error limit.

## Registration

Each source has 245,762 vertices and 491,520 triangles; each target has 163,842 vertices and 327,680 triangles. Settings are `steps=2`, `runs=2`, `avg=true`, `loop=6`, `cycles=3`, `nit=3`, `its=3` and `code=1`, with a `512 × 256` DARTEL map.

Depth features use the upstream helper plus CUDA heat filtering and normalization. DARTEL and squaring use FP64 Triton with CUDA Graph enabled. The paired runs below use 32 stencil workers.

| Hemisphere | CPU reference | GPU total | CPU/GPU ratio | Absolute error: max / mean / p99 |
| --- | ---: | ---: | ---: | ---: |
| LH | 119.914245 s | 45.084841 s | 2.659746× | 2.8491e-5 / 7.3001e-8 / 9.4995e-7 |
| RH | 119.405634 s | 46.986172 s | 2.541293× | 3.4094e-4 / 5.8125e-6 / 4.9412e-5 |

Both outputs preserve triangle arrays exactly. These are paired single measurements, rather than medians of repeated runs.

A representative RH profile attributes 19.380 seconds to the source runs and averaging, 9.155 seconds to the initial stencil and features, and 4.442 seconds to the rotated stencil. Output writing accounts for 0.823 seconds. Repeated solves and stencil construction make up most of the measured work.

### Stencil thread count

The helper accepts `--threads`; the Python command exposes `--stencil-threads`, defaulting to 8. Three fresh helper runs on the same full-resolution LH input produced:

| Workers | Median wall time | Median CPU time |
| ---: | ---: | ---: |
| 8 | 4.28 s | 7.59 s |
| 16 | 4.13 s | 8.39 s |
| 32 | 4.03 s | 10.21 s |

All nine output files were byte-identical, with SHA-256 `bffbb212d752d4c12f4eb77d03944edc4797ca449d8e36e6e3a35db925b9dd56`.

A complete paired LH run with the 8-worker default took 110.8468 seconds on CPU and 37.0151 seconds on GPU, a 2.9946× ratio. Triangle arrays matched, with maximum/mean/p99 errors of `7.06e-6 / 4.75e-8 / 4.02e-7`. Separate GPU-only runs took 31.5073 seconds at 32 workers and 31.6381 seconds at 8 workers; this small difference did not justify increasing the default thread count.

## Checks after rebuilding the release

After rebuilding the four bundled ELF programs using the [release build instructions](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BUILDING.md), a single paired LH check produced:

| Component | CPU | GPU | CPU/GPU ratio | Absolute error: max / mean / p99 |
| --- | ---: | ---: | ---: | ---: |
| Sphere generation, 61,442 vertices | 54.9849 s | 10.6890 s | 5.1440× | 1.1444e-5 / 1.6734e-6 / 7.6294e-6 |
| Registration | 126.6581 s | 35.0607 s | 3.6125× | 5.4806e-5 / 2.1548e-7 / 3.4571e-6 |

Triangle arrays matched exactly and all 4,999 sphere-smoothing sweeps ran. The CUDA host test suite completed with 34 passes and one no-GPU-test skip; the no-GPU suite completed with 33 passes and two skips.

## Bilateral workflow

This workflow runs sphere generation concurrently for both hemispheres, upsamples the topology, then runs both registrations concurrently.

| Stage | Wall time |
| --- | ---: |
| Sphere generation, both hemispheres | 15.7609 s |
| Topology upsampling | 4.5208 s |
| Registration, both hemispheres | 34.9756 s |
| **Total** | **55.2573 s** |

Final comparisons with the CPU reference:

| Hemisphere | Absolute error: max / mean / p99 | Triangle indices |
| --- | ---: | --- |
| LH | 1.3821e-6 / 7.0854e-8 / 5.5507e-7 | Exact |
| RH | 1.8090e-4 / 6.7359e-6 / 5.7042e-5 | Exact |

## Reproduce the comparison

Use a separate checkout of the [v0.1.0 tag](https://github.com/ayakacxy/cat-surface-gpu/tree/v0.1.0) and follow its installation instructions. The example below uses the scripts and binaries from that tag and stores working files under the project directory.

```bash
mkdir -p .work/v0.1-comparison/{tmp,cache/triton,cache/torch,output}
export TMPDIR="$PWD/.work/v0.1-comparison/tmp" TMP="$PWD/.work/v0.1-comparison/tmp" TEMP="$PWD/.work/v0.1-comparison/tmp"
export XDG_CACHE_HOME="$PWD/.work/v0.1-comparison/cache"
export TRITON_CACHE_DIR="$PWD/.work/v0.1-comparison/cache/triton"
export TORCH_HOME="$PWD/.work/v0.1-comparison/cache/torch"

python tools/benchmark_cat_surface_end_to_end.py \
  --reference-cli bin/linux-x86_64/CAT_SurfWarp \
  --source-surface /path/to/white.gii \
  --source-sphere /path/to/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --cpu-output .work/v0.1-comparison/output/sphere.cpu.gii \
  --gpu-output .work/v0.1-comparison/output/sphere.gpu.gii \
  --rotation-depth-probe bin/linux-x86_64/cat_surface_rotation_depth \
  --rotation-feature-backend cuda-official-depth \
  --stencil-builder bin/linux-x86_64/cat_surface_stencil_builder \
  --rotated-stencil-builder bin/linux-x86_64/cat_surface_stencil_builder \
  --stencil-threads 8 \
  --device cuda \
  --kernel triton \
  --dartel-dtype float64 \
  --squaring-kernel triton \
  --steps 2 \
  --runs 2 \
  --avg \
  --max-vertex-error 1e-3
```

The command exits with an error if triangle indices differ or the maximum error exceeds the specified limit.
