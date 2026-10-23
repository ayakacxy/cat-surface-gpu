# Reproducing results

[English](REPRODUCIBILITY.md) · [简体中文](REPRODUCIBILITY.zh-CN.md)

Use the same input surfaces, template, parameters, and solver when comparing runs. The [benchmark report](BENCHMARKS_v0.3.0.md) describes the published measurements; [benchmark-v0.3.0.json](benchmark-v0.3.0.json) contains the individual timings, input hashes, and numerical results.

## Record a run

Keep the command and its JSON output together with:

- package version or Git commit;
- operating system, Python version, and backend dependency versions;
- CPU/GPU model, CPU thread count, and selected device;
- SHA-256 hashes and vertex/triangle counts for each input;
- registration parameters, solver, and iteration settings.

Both commands send progress to standard error and their final JSON summary to standard output. For example:

```bash
cat-surf2sphere-gpu --input-surface white.gii --output-surface sphere.gii --device cuda --threads 8 > sphere.json 2> sphere.log
cat-surfwarp-gpu --source-surface white.gii --source-sphere sphere.gii --target-surface template.white.gii --target-sphere template.sphere.gii --output sphere.reg.gii --device cuda --threads 8 > registration.json 2> registration.log
```

Use the `device` field to identify the selected device. The JSON `backend` field is a fixed implementation label; it does not identify whether a particular command used CPU, CUDA, or Metal.

Input files should remain unchanged between runs. Save outputs from each configuration separately. For version comparisons, use separate source checkouts or environments so that package imports and compilation caches can be identified unambiguously.

## Compare surfaces

First check vertex counts, exact triangle equality, finite coordinates, and nonzero triangle areas. Then compare matching vertices. Useful summaries are mean, maximum, and 95th-percentile Euclidean distance. For spherical outputs, angular differences in degrees also make the scale of the difference clear.

State the metric and units with the result: a per-coordinate absolute error is different from a per-vertex Euclidean distance. Coordinate agreement measures numerical reproducibility; anatomical registration quality requires an anatomical evaluation, such as landmark or region correspondence.

The algorithm reference is the CAT-Surface revision pinned in [UPSTREAM.lock](../UPSTREAM.lock). The reference uses full multigrid (FMG), while this package defaults to preconditioned conjugate gradients (PCG). CPU and CUDA also provide an FMG option. Record the actual solver, since matching input files alone does not make two numerical methods equivalent.

## Measure runtime

A complete command measurement starts before the process launches and ends after it exits. It includes Python startup, imports, file reading, backend initialization, calculation, and file writing. The CLI's `wall_seconds` begins after argument parsing; the API's stage timings cover a smaller part of the operation. Keep these three measurements distinct.

Report first-run and warm-cache results separately. The first run may include Numba/CuPy/MLX compilation and topology/template preparation. For repeated runs, state whether they use fresh processes with existing disk caches or reuse one Python process. Use at least three repetitions and report the median and range.

For bilateral work, specify whether a measurement includes one or both hemispheres, whether they run serially or concurrently, and whether all output files have finished writing. Synchronize GPU work before stopping a manually placed timer.

The repository includes `tools/benchmark_versions.py` for comparing complete command executions between two source trees. It records first-run and repeated timings, source and input hashes, and output differences. Its required input paths and options are listed by:

```bash
python tools/benchmark_versions.py --help
```

## Caches and test data

`CAT_SURFACE_CACHE_ROOT` controls the package's topology and template cache; its default is `.work/cat-surface/cache` relative to the working directory. Numba and CuPy use `NUMBA_CACHE_DIR` and `CUPY_CACHE_DIR` for their compilation caches. Separate cache directories are useful when measuring first-run cost. The [build guide](BUILDING.md#prepare-the-environment) shows a project-local setup for Linux, macOS, and Windows.

Synthetic meshes are useful for installation, I/O, and small numerical tests. Full inflation and registration should use meshes representative of the intended anatomical workflow: very coarse meshes can collapse during inflation, and their runtime does not represent a cortical mesh. The repository does not distribute subject data or anatomical templates.

[Documentation](README.md) · [Project README](../README.md)
