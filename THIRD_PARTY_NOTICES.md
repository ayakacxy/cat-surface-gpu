# Third-party notices

## CAT-Surface

The algorithms derive from [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface), developed by Christian Gaser at Jena University Hospital.

| Release | Fixed upstream revision |
| --- | --- |
| v0.2.0 | [`b0c2aa2395ea4957a331b9444fb54c7716b6572d`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d) |
| v0.1.0 | [`628b6851d8638f3ab773cd25c0ec406d0ec61ede`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) |

The relevant reference sources include `Progs/CAT_Surf2Sphere.c`, `Progs/CAT_SurfWarp.c`, and `Lib/CAT_SurfWarpDartel.c`. The DARTEL implementation originates from John Ashburner's work in SPM and is distributed within CAT-Surface under its applicable GPL terms.

CAT-Surface original code is available under GPL-2.0-or-later or a separate commercial license from its copyright holder. The copied notice is at [upstream/CAT-Surface-LICENSE.txt](upstream/CAT-Surface-LICENSE.txt). Third-party components within the upstream source retain their own licenses.

Version 0.2.0 adapts the Python CAT implementation developed in FlashCHARM. The Python package includes Python and inline CUDA kernel source; it contains no CAT-Surface executables, native helper programs, or full upstream source tree. The release separately provides the pinned upstream source archive for reference and attribution. See [UPSTREAM.lock](UPSTREAM.lock) and the [FlashCHARM source record](upstream/FLASHCHARM_CAT_PROVENANCE.json) for source revisions and file hashes.

Version 0.1.0 distributed four Linux executables built from revision `628b6851` with `-O2 -fPIC`. Their helper source and corresponding upstream source remain available with the [v0.1.0 release](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0).

## SimNIBS

[SimNIBS](https://github.com/simnibs/simnibs) provides the head-modeling workflow in which `CAT_Surf2Sphere` and `CAT_WarpSurf` are used for cortical surface registration. This package does not redistribute SimNIBS or modify a SimNIBS installation.

## Python dependencies

Version 0.2.0 uses [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), [Numba](https://github.com/numba/numba), [nibabel](https://github.com/nipy/nibabel), and [CuPy](https://github.com/cupy/cupy) as external dependencies. They retain their respective licenses.

Version 0.1.0 used [PyTorch](https://github.com/pytorch/pytorch) and [Triton](https://github.com/triton-lang/triton). They are not required by v0.2.0.

Project code is distributed under [GPL-3.0-or-later](LICENSE), with upstream copyright and license notices retained.
