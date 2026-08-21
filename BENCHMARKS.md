# Benchmarks

This document records the measurement boundary behind the performance numbers in the README. Results are specific to the listed hardware, software, input shapes, and parameters.

## Validated environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GiB |
| Python | 3.11 |
| PyTorch | 2.6.0+cu126 |
| CUDA runtime | 12.6 |
| Triton | 3.2.0 |
| CAT-Surface reference | `628b6851d8638f3ab773cd25c0ec406d0ec61ede` |
| Reference build | GCC, `-O2 -fPIC`, no `-ffast-math` |
| GPU DARTEL precision | FP64 |
| `CAT_Surf2Sphere` precision | FP32 storage with CAT-compatible double intermediates |

The test fixture is one anonymized cortical surface pair with the SimNIBS fsaverage template. No subject data is distributed in this repository.

## Measurement rules

- CPU and GPU use the same GIFTI inputs, mesh topology, CAT-Surface parameters, and output checks.
- Full-pipeline wall times use `time.perf_counter()` and include input parsing, CPU helper work, host-to-device setup, synchronized CUDA work, output transfer, and GIFTI writing.
- CUDA stages synchronize before the timer is read.
- Bilateral wall time measures two independent hemisphere processes sharing one RTX 2080 Ti, matching the SimNIBS process-pool schedule.
- `CAT_Surf2Sphere stop_at=10` still requests 5,000 smoothing iterations and executes the upstream loop's 4,999 sweeps (`k=1..4999`).
- No result below is extrapolated to the complete SimNIBS CHARM pipeline.

## `CAT_Surf2Sphere`

### SimNIBS default coarse sphere

Input per hemisphere: 61,442 vertices and 122,880 triangles. The recommended backend keeps the first five upstream stages on the bundled CPU reference and runs area smoothing with the ordered Triton schedule.

| Hemisphere | Latest CPU reference | GPU total | Area smoothing only | Max / mean / p99 absolute error | Faces |
| --- | ---: | ---: | ---: | ---: | --- |
| LH | 54.84 s | 10.8119 s | 3.7084 s | 1.1444e-5 / 1.7699e-6 / 7.6294e-6 | exact |
| RH | not recorded in the same standalone run | 10.6667 s | 3.5142 s | 1.5259e-5 / 1.6752e-6 / 7.6294e-6 | exact |

The LH speedup against the latest CPU reference is 5.07x. The RH table intentionally does not infer a speedup from a different CPU run.

When LH and RH run concurrently on one GPU, each process reports about 13.81 seconds because the kernels share the device; the observed bilateral stage wall time is 15.7609 seconds.

### Full-resolution profile

Input per hemisphere: 245,762 vertices and 491,520 triangles. This earlier profile used the explicit FP32/Triton path and the wider `5e-3 / 1e-3 / 3e-3` intermediate-sphere contract.

| Hemisphere | Latest CPU reference | GPU | Speedup | Max / mean / p99 absolute error | Faces |
| --- | ---: | ---: | ---: | ---: | --- |
| LH | 234.00 s | 37.8887 s | 6.18x | 3.073e-3 / 6.20e-4 / 2.407e-3 | exact |
| RH | 236.35 s | 37.4264 s | 6.32x | 3.170e-3 / 6.12e-4 / 2.398e-3 | exact |

The coarse ordered schedule above is the recommended SimNIBS-default profile because it provides the tighter `1e-3` contract.

## `CAT_SurfWarp`

Input per hemisphere:

- source vertices/faces: `(245762, 3)` / `(491520, 3)`;
- target vertices/faces: `(163842, 3)` / `(327680, 3)`;
- DARTEL map: `512 x 256`;
- parameters: `steps=2`, `runs=2`, `avg=true`, `loop=6`, `cycles=3`, `nit=3`, `its=3`, `code=1`;
- initial feature backend: upstream raw depth plus CUDA heat/normalization;
- stencil helper: the historical headline run used 32 CPU workers; the current runtime-configurable helper defaults to 8;
- DARTEL and squaring: FP64 Triton with CUDA Graph enabled.

### Paired end-to-end A/B

| Hemisphere | CPU reference | GPU total | Speedup | Max / mean / p99 absolute error | Faces |
| --- | ---: | ---: | ---: | ---: | --- |
| LH | 119.914245 s | 45.084841 s | 2.659746x | 2.8491e-5 / 7.3001e-8 / 9.4995e-7 | exact |
| RH | 119.405634 s | 46.986172 s | 2.541293x | 3.4094e-4 / 5.8125e-6 / 4.9412e-5 | exact |

These are the conservative paired results used in the README. Later single-run exploratory profiles reached 38.15 seconds for LH and 37.26 seconds for RH, but those runs were not repeated as an interleaved benchmark and are therefore not used as the headline claim.

### Representative RH GPU timeline

| Stage | Seconds | Share |
| --- | ---: | ---: |
| Input | 0.337 | 0.72% |
| Initial stencil + upstream feature | 9.155 | 19.48% |
| Rotation index upload | 2.535 | 5.39% |
| Rotation search | 3.475 | 7.40% |
| Rotated stencil | 4.442 | 9.45% |
| Warm-up | 1.472 | 3.13% |
| Device preparation / upload remainder | 5.368 | 11.42% |
| Source runs + average DARTEL / warp | 19.380 | 41.25% |
| Output | 0.823 | 1.75% |

The remaining hotspot is the repeated solve/stencil lifecycle, not GIFTI writing or average combination.

### Runtime-configurable stencil workers

The bundled helper now accepts `--threads N`, and the Python runners expose the same value as `--stencil-threads`. The default is 8; thread count is no longer fixed when the binary is compiled.

On the same full-resolution LH sphere, three fresh helper runs per setting produced:

| CPU workers | Median wall time | Median CPU time | Output |
| ---: | ---: | ---: | --- |
| 8 | 4.28 s | 7.59 s | byte-identical |
| 16 | 4.13 s | 8.39 s | byte-identical |
| 32 | 4.03 s | 10.21 s | byte-identical |

All nine stencil files had SHA-256 `bffbb212d752d4c12f4eb77d03944edc4797ca449d8e36e6e3a35db925b9dd56`. Moving from 8 to 32 workers saved about 0.25 seconds in this isolated helper while consuming substantially more aggregate CPU time.

A fresh full LH CPU/GPU A/B with the runtime-configurable helper at its 8-worker default measured 110.8468 seconds for the CPU reference and 37.0151 seconds for the GPU path, or 2.9946x. Faces were exact and the final maximum/mean/p99 vertex errors were `7.06e-6 / 4.75e-8 / 4.02e-7`.

Follow-up GPU-only runs measured 31.5073 seconds with 32 workers and 31.6381 seconds with 8 workers. The 0.13-second difference is within whole-pipeline run-to-run variation; it is not treated as a stable end-to-end speedup. The 8-worker default therefore retains essentially the same observed end-to-end performance while reducing oversubscription when helpers and hemispheres overlap.

A final release-candidate smoke run after consolidating the package under the `cat_surface_gpu` namespace completed in 30.6743 seconds with the bundled helpers and the 8-worker default. Against the same-input CPU output, faces were exact and the maximum/mean/p99 vertex errors were `7.06e-6 / 4.74e-8 / 4.02e-7`. This single GPU-only run validates the packaged path; it is not used as the headline speedup.

## Bilateral SimNIBS-default CAT chain

This benchmark runs both hemisphere processes concurrently for `CAT_Surf2Sphere`, performs the same topology upsampling used by SimNIBS, and then runs both `CAT_SurfWarp` processes concurrently.

| Stage | Wall time |
| --- | ---: |
| Bilateral `CAT_Surf2Sphere` | 15.7609 s |
| Topology upsampling | 4.5208 s |
| Bilateral `CAT_SurfWarp` | 34.9756 s |
| **Total** | **55.2573 s** |

Final output validation against the latest CPU reference:

| Hemisphere | Max / mean / p99 absolute error | Faces |
| --- | ---: | --- |
| LH | 1.3821e-6 / 7.0854e-8 / 5.5507e-7 | exact |
| RH | 1.8090e-4 / 6.7359e-6 / 5.7042e-5 | exact |

This 55.26-second value covers only the CAT spherical-registration chain. It does not include cortical surface estimation, segmentation updates, or the other work included in the complete SimNIBS `Surface creation` stage.

## Reproducing the measurements

After creating the Conda environment from the README, run a single-hemisphere warp A/B:

```bash
python tools/benchmark_cat_surface_end_to_end.py \
  --reference-cli bin/linux-x86_64/CAT_SurfWarp \
  --source-surface /path/to/white.gii \
  --source-sphere /path/to/sphere.gii \
  --target-surface /path/to/template.white.gii \
  --target-sphere /path/to/template.sphere.gii \
  --cpu-output /tmp/sphere.cpu.gii \
  --gpu-output /tmp/sphere.gpu.gii \
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

The benchmark exits non-zero if the topology differs or the maximum vertex error exceeds the requested limit.
