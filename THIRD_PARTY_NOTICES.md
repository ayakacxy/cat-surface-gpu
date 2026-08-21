# Third-party notices

## CAT-Surface

The accelerated algorithms and the bundled Linux reference binaries are derived from [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface), developed by Christian Gaser at Jena University Hospital.

The validated source baseline is commit [`628b6851d8638f3ab773cd25c0ec406d0ec61ede`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede). In particular:

- `Progs/CAT_Surf2Sphere.c` is the reference for spherical inflation;
- `Progs/CAT_SurfWarp.c` and `Lib/CAT_SurfWarpDartel.c` are the references for non-linear surface registration;
- the upstream DARTEL implementation originates from work by John Ashburner and is distributed within CAT-Surface under its applicable GPL terms;
- CAT-Surface's bundled surface I/O and mesh-processing dependencies retain their own licenses.

CAT-Surface original code is available under GPL-2.0-or-later or a separate commercial license from its copyright holder. The copied upstream license notice is available at [`upstream/CAT-Surface-LICENSE.txt`](upstream/CAT-Surface-LICENSE.txt).

The binaries under `bin/linux-x86_64/` were built from the exact commit above with `-O2 -fPIC` and stripped for distribution. The corresponding upstream source is available through the linked commit and the release source attachment. Source for repository-specific helper programs is included under `native/`; the pinned revision and complete build procedure are recorded in `UPSTREAM.lock` and [`docs/BUILDING.md`](docs/BUILDING.md).

## SimNIBS

[SimNIBS](https://github.com/simnibs/simnibs) provides the head-modeling workflow in which `CAT_Surf2Sphere` and `CAT_WarpSurf` are used for cortical surface registration. This repository does not redistribute SimNIBS or replace its installation automatically. Integration is explicit so that the original SimNIBS path remains available.

## PyTorch and Triton

The GPU implementation uses [PyTorch](https://github.com/pytorch/pytorch) and [Triton](https://github.com/triton-lang/triton). These projects are external dependencies and remain under their own licenses.
