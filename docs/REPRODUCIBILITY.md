# Reproducibility and Provenance Guide

[English](REPRODUCIBILITY.md) · [简体中文](REPRODUCIBILITY.zh-CN.md)

## v0.2.0 release checks

On 2026-10-13, all 57 tests passed on an RTX 2080 Ti, including eight CUDA tests. The installed wheel completed both command-line workflows on full fsaverage templates, passing the generated sphere directly to registration. Each input had 163,842 vertices and 327,680 triangles. Both commands used their default settings, including `rotation_point_chunk=16384` for registration. Outputs retained the input triangles and contained finite coordinates with no zero-area triangles.

The [machine-readable record](validation-v0.2.0.json) includes input and source hashes, dependency versions, parameters, and results. These checks cover the installed workflow on the stated templates; they do not establish accuracy across subjects or a speedup over the official C implementation. Separate zero-residual regression tests cover the PCG division-by-zero fix.

## Record the version and environment

For v0.2.0, record:

- package version and source revision;
- CAT-Surface reference revision `b0c2aa2395ea4957a331b9444fb54c7716b6572d`;
- Python, NumPy, SciPy, Numba, nibabel, CuPy, CUDA, and driver versions;
- GPU model, CPU thread count, and peak host/GPU memory;
- input hashes, vertex/triangle counts, parameters, solver, and iteration count.

The fixed upstream revision is also recorded in [UPSTREAM.lock](../UPSTREAM.lock). Version 0.1.0 used a different revision and implementation; its evidence is archived in [BENCHMARKS.md](BENCHMARKS.md).

## Validation layers

1. Inspect the built wheel and source archive for required code and notices, and confirm that no CAT executables or project-specific extensions are included.
2. Run CPU tests for geometry, indexing, topology, interpolation, and input validation.
3. Run CUDA tests on an actual NVIDIA GPU. Record skipped tests separately.
4. Compare complete sphere and registered-sphere outputs against an independently run reference using the same inputs and parameters.

For surface comparisons, report exact triangle equality and coordinate differences using clearly stated units. Include the maximum, mean, and an upper percentile; state whether errors are per-coordinate absolute differences or per-vertex Euclidean distances. Check finite coordinates and vertex counts before computing statistics.

Version 0.2.0 defaults to a PCG linear solver. Upstream CAT-Surface uses FMG, so equal inputs and iteration labels do not imply identical solver trajectories. Set numerical tolerances before evaluating results and record which solver was used. The v0.1.0 `1e-3` threshold is not automatically a v0.2.0 acceptance threshold.

## Measure the complete operation

Measure input reading, preprocessing, transfers, GPU computation, output transfer, and file writing together when reporting end-to-end time. Synchronize the GPU before stopping timers. Report whether interpreter startup, Numba/CuPy compilation, and cache loading are included. Separate first-run measurements from repeated runs with established caches.

Use repeated same-input runs and report the run count and spread. For bilateral runs, state whether hemispheres run serially or share the GPU concurrently. An isolated kernel speedup does not measure the complete registration workflow or the complete CHARM pipeline.

## Synthetic and anatomical inputs

The repository distributes no anatomical or subject data. Synthetic fixtures test installation, file I/O, and small numerical operations. Very coarse synthetic meshes can collapse during the full inflation schedule; use suitable anatomical meshes to validate the complete sphere-to-registration workflow. These fixtures do not reproduce anatomical performance or establish anatomical registration accuracy.

Keep benchmark inputs read-only. Store caches, logs, reference outputs, and candidate outputs under separate paths within a task directory such as `.work/comparison/`. Set `TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `NUMBA_CACHE_DIR`, and `CUPY_CACHE_DIR` before running, as shown in the [README](../README.md#run).

The upstream reference may be built and run separately for comparison. It is neither a dependency of the candidate workflow nor an automatic fallback.
