# Implementation and Python API

[English](IMPLEMENTATION.md) · [简体中文](IMPLEMENTATION.zh-CN.md)

`cat_surface_gpu` provides spherical inflation and surface registration as array-based Python functions. File I/O uses GIFTI; computation runs on CPU, NVIDIA CUDA, or Apple Metal. Installation and command-line examples are in the [project README](../README.md).

## Surface data

A surface consists of an `(N, 3)` vertex array and an `(M, 3)` integer triangle array with zero-based indices. Inputs must be closed triangular meshes suitable for spherical mapping, with finite coordinates and no zero-area triangles.

Registration takes an anatomical surface and its sphere for both the source and the template. Within each pair, vertex order and triangles must match. The source and template may have different mesh resolutions. The output sphere keeps the source vertex order and triangles.

`load_gifti_surface(path)` reads one point-set array and one triangle array into a `GiftiSurface`. `save_gifti_surface(surface, path)` writes float32 vertices and int32 triangles. When a reference GIFTI is provided, image metadata, labels, array metadata, and coordinate-system transforms are retained; additional data arrays are not copied. Coordinates are used as supplied, without applying an MRI affine or converting coordinate systems.

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

The function signature is:

```text
surf_to_sphere(
    vertices, faces, parameters=None, topology=None,
    *, device="cuda", progress=None,
)
```

It returns `(vertices, timings)`. `stop_at=10` runs the full inflation schedule; smaller values stop earlier, and values below 5 produce an intermediate surface. An optional precomputed `SurfaceTopology` can be passed as `topology` when reusing the same mesh connectivity.

`device` accepts `cpu`, `cuda`, `metal`, or `auto`. Automatic selection uses Metal on supported macOS systems, CUDA on other systems with an available NVIDIA GPU, and CPU otherwise. An explicitly selected backend raises an error if it is unavailable.

`surf_to_sphere_cupy(...)` is also available as a CUDA-only entry point. To select a particular NVIDIA GPU in Python, call either CUDA entry point inside a `cupy.cuda.Device(index)` context.

## Surface registration

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
    parameters=SurfWarpParameters(),
    backend="cupy",
)
save_gifti_surface(
    GiftiSurface(result.vertices, source.faces, source_sphere.reference),
    "sphere.reg.gii",
)
```

Despite its name, `surfwarp_cupy` supports three backends: `"cupy"` for CUDA, `"cpu"`, and `"metal"`. It does not accept `"auto"`. The command-line interface accepts `--device auto` and selects the corresponding backend before calling this function.

`SurfWarpResult` contains:

| Field | Contents |
| --- | --- |
| `vertices` | Registered source sphere, including pole averaging when enabled |
| `velocity` | Velocity field from the final main registration run |
| `deformation` | Deformation field from that run |
| `timings` | Stage timings and rotation settings |
| `objectives` | Objective values recorded during registration |

Use `vertices` as the final registered sphere. With multiple runs or pole averaging, `velocity` and `deformation` do not describe the entire transformation from the original input sphere.

Both inflation and registration accept `progress`, a callback that receives stage messages as strings. File reading and writing are separate from the array API, so their cost is not included in its timing dictionaries.

### Registration parameters

The main options are defined by [`SurfWarpParameters`](../src/cat_surface_gpu/parameters.py):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `steps` | `2` | Number of type-5 depth feature scales; 1 or 2 |
| `runs` | `2` | Successive registration runs |
| `rotate` | `True` | Initial rigid rotation search |
| `average` | `True` | Additional registration with rotated poles, followed by averaging |
| `map_size` | `(512, 256)` | Width and height of the spherical map |
| `loop` | `6` | DARTEL schedule levels |
| `iterations` | `3` | DARTEL updates per level |
| `curvature_fwhm` | `5.0` | Initial feature smoothing scale |
| `mu` / `mu_rate` / `mu_change` | `0.125` / `1.25` / `4` | Membrane regularization schedule |
| `lm_regularization` | `0.001` | Levenberg–Marquardt regularization |

Additional keyword arguments to `surfwarp_cupy` control the solver and rotation:

| Argument | Default | Meaning |
| --- | --- | --- |
| `solver_backend` | `"pcg"` | PCG or FMG; Metal supports PCG |
| `solver_iterations` | `40` | PCG iteration limit; unused by FMG |
| `pcg_check_interval` | `0` | Interval between convergence checks; 0 disables intermediate checks |
| `rotation_point_chunk` | `16384` | Points processed per rotation batch |
| `rotation_refine` | `True` | Refine the initial rotation estimate |
| `rotation_max_iter` | `500` | Maximum rotation refinement iterations |

Metal requires `solver_backend="pcg"` and `pcg_check_interval=0`. Reducing `rotation_point_chunk` reduces temporary rotation memory. It does not change the number of input vertices.

The supported model uses type-5 depth features and membrane regularization. The following `SurfWarpParameters` values are fixed: `code=1`, `regularization_type=1`, `multires_levels=0`, `resampled_triangles=81920`, `lambda_value=0`, `surface_fwhm=0`, and `cycles=3`. Unsupported settings raise `ValueError`.

## Implementation overview

Inflation smooths the anatomical surface, projects it onto a sphere, and applies area smoothing. Registration estimates an initial rotation, resamples depth features onto spherical maps, optimizes a DARTEL velocity field, and maps the deformation back to the source vertices. Repeated runs and pole averaging are enabled by default.

NumPy and Numba provide CPU geometry and numerical routines. CUDA kernels are embedded in Python modules and compiled by CuPy; the Metal backend uses MLX. The default CUDA and Metal registration paths use FP32 feature solves and DARTEL updates, with FP64 final velocity exponentiation. Kernel compilation and reusable topology/template caches affect first-run time.

The algorithm reference is [CAT-Surface revision `b0c2aa23`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d). Its reference solver is FMG; this package defaults to PCG. Comparisons should record solver settings alongside backend and precision. See [reproducibility](REPRODUCIBILITY.md) for the comparison procedure and [benchmarks](BENCHMARKS_v0.3.0.md) for measured results.

[Documentation](README.md) · [Project README](../README.md)
