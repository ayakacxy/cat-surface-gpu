# Implementation and Python API

[English](IMPLEMENTATION.md) · [简体中文](IMPLEMENTATION.zh-CN.md)

Version 0.3.0 implements spherical inflation and cortical surface registration in `cat_surface_gpu`. Its algorithm reference is CAT-Surface revision [`b0c2aa2395ea4957a331b9444fb54c7716b6572d`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d).

## Execution

NumPy and Numba handle CPU geometry, topology, and supporting calculations. CuPy RawKernel/RawModule kernels perform CUDA work, including spherical inflation and registration. The package runs independently, without CAT command-line programs or project-specific native extensions. CUDA kernel source is embedded in Python modules.

The complete workflow can use explicit CPU, CUDA or Metal backends. CPU execution uses NumPy/Numba. Metal uses MLX on Apple Silicon. Numba and CuPy kernel compilation can make the first invocation slower.

## Surfaces and coordinates

- Input GIFTI files must each contain one point-set array and one triangle array.
- Vertex arrays have shape `(N, 3)`; triangle arrays have shape `(M, 3)` with zero-based indices.
- Each source surface/sphere pair must use identical triangles and vertex ordering. The same applies to the template pair.
- The source and template may have different vertex and triangle counts.
- Surfaces must be closed triangular meshes with nonzero triangle areas, suitable for spherical mapping. This package does not reconstruct cortex from MRI or repair arbitrary input topology.
- Output vertices use float32 and triangles use int32. The registered sphere retains source triangles.
- GIFTI writing preserves image metadata, the label table, and the point-set/triangle metadata and coordinate-system transforms when a reference image is supplied. Extra data arrays, such as per-vertex measurements, are not copied.

Input and output arrays are checked for finite coordinates and zero-area triangles. Very coarse synthetic meshes may degenerate during inflation and are rejected.

The code operates on the supplied coordinates. It does not infer an MRI affine or convert anatomical coordinate systems.

## Spherical inflation

```python
from cat_surface_gpu import (
    GiftiSurface,
    Surf2SphereParameters,
    load_gifti_surface,
    save_gifti_surface,
    surf_to_sphere,
)

surface = load_gifti_surface("white.gii")
vertices, timings = surf_to_sphere(
    surface.vertices,
    surface.faces,
    parameters=Surf2SphereParameters(stop_at=10),
    device="cuda",
)
save_gifti_surface(
    GiftiSurface(vertices, surface.faces, surface.reference),
    "sphere.gii",
)
```

`surf_to_sphere(vertices, faces, parameters=None, topology=None, *, device="cuda", progress=None)` returns the new vertices and a timing dictionary. `Surf2SphereParameters` defaults to `stop_at=10`; the parameter controls the upstream inflation stages and area-smoothing schedule.

`device` accepts `cpu`, `cuda`, `metal` and `auto`. The existing `surf_to_sphere_cupy(...)` remains available as a CUDA entry point.

## Registration

```python
from cat_surface_gpu import (
    GiftiSurface,
    SurfWarpParameters,
    load_gifti_surface,
    save_gifti_surface,
    surfwarp_cupy,
)

source = load_gifti_surface("white.gii")
source_sphere = load_gifti_surface("sphere.gii")
target = load_gifti_surface("template.white.gii")
target_sphere = load_gifti_surface("template.sphere.gii")

result = surfwarp_cupy(
    source.vertices,
    source_sphere.vertices,
    source.faces,
    target.vertices,
    target_sphere.vertices,
    target.faces,
    parameters=SurfWarpParameters(steps=2, runs=2, average=True),
    solver_iterations=40,
    solver_backend="pcg",
    backend="cupy",
)
save_gifti_surface(
    GiftiSurface(result.vertices, source.faces, source_sphere.reference),
    "sphere.reg.gii",
)
```

The result includes `vertices`, `velocity`, `deformation`, `timings`, and `objectives`. Both main Python functions accept an optional `progress` callback that receives a stage string. The Python call expects already matched surface/sphere arrays; the CLI also checks the faces in each pair.

`backend` accepts `cpu`, `cupy` and `metal`. Metal uses fixed-iteration PCG with `pcg_check_interval=0`; CPU and CUDA support PCG and FMG.

| Setting | Default |
| --- | --- |
| Steps / runs | `2 / 2` |
| Initial rotation / averaging | Enabled / enabled |
| DARTEL loops | `6` |
| Spherical map size | `512 x 256` |
| Iterations per DARTEL level | `3` |
| Linear solver | `pcg`, `40` iterations |
| PCG convergence-check interval | `0` |
| Rotation point chunk | `16384` |
| Rotation refinement | Enabled, at most `500` iterations |
| CLI CPU threads | `8` (supported range: `1–8`) |

The supported registration uses type-5 depth features for one or two steps, membrane regularization (`regularization_type=1`), `code=1`, `multires_levels=0`, `lambda_value=0`, `surface_fwhm=0`, `resampled_triangles=81920`, and `cycles=3`. Other values outside the implemented path are rejected. This release does not implement every upstream CAT command-line mode.

The default PCG solver follows a different numerical path from upstream full multigrid (FMG). `pcg_check_interval=0` disables intermediate CPU convergence checks; it does not reduce the requested 40 iterations. `solver_iterations` applies only to PCG. `solver_backend="fmg"` selects the available FMG implementation, but its name alone does not establish bitwise equivalence with CAT-Surface. Record the solver in every comparison.

Lowering `rotation_point_chunk` from its default of `16384`, for example to `12288`, reduces the size of temporary rotation batches. Memory use and timing depend on the input and GPU.

## Command-line output

Both commands support `--device cpu`, `--device cuda`, `--device cuda:N`, `--device metal` or `--device auto`, `--threads`, and `--quiet`. They write stage progress to standard error and a final JSON summary to standard output. Use `--help` for the full option list.

[Documentation index](README.md) · [Project README](../README.md)
