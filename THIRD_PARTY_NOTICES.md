# Third-party notices

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

CAT-Surface GPU is distributed under [GPL-3.0-or-later](LICENSE). Upstream components retain their copyright and license notices.

## CAT-Surface and DARTEL

The spherical inflation and registration algorithms are based on [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface), developed by Christian Gaser at Jena University Hospital. The reference revision is [`b0c2aa2395ea4957a331b9444fb54c7716b6572d`](https://github.com/ChristianGaser/CAT-Surface/tree/b0c2aa2395ea4957a331b9444fb54c7716b6572d), recorded in [UPSTREAM.lock](UPSTREAM.lock).

The reference sources include `Progs/CAT_Surf2Sphere.c`, `Progs/CAT_SurfWarp.c`, and `Lib/CAT_SurfWarpDartel.c`. DARTEL originates from John Ashburner's work in SPM and is distributed with CAT-Surface under its applicable GPL terms.

CAT-Surface's original code is licensed under GPL-2.0-or-later, with a separate commercial license available from its copyright holder. The commercial option does not cover third-party components. The original notice, including component-specific terms, is preserved in [upstream/CAT-Surface-LICENSE.txt](upstream/CAT-Surface-LICENSE.txt).

The package contains Python code and inline GPU kernels. It does not bundle CAT-Surface executables or the full upstream source tree. The pinned upstream source archive is provided separately with release downloads. Source hashes and adaptation records are kept in [upstream/](upstream/), including the [source manifest](upstream/SOURCE_MANIFEST.json).

## External dependencies

The following libraries are installed separately and retain their respective licenses:

| Library | Use |
| --- | --- |
| [NumPy](https://github.com/numpy/numpy) | Array operations |
| [SciPy](https://github.com/scipy/scipy) | Scientific routines and spatial queries |
| [Numba](https://github.com/numba/numba) | CPU kernel compilation |
| [nibabel](https://github.com/nipy/nibabel) | GIFTI input and output |
| [CuPy](https://github.com/cupy/cupy) | NVIDIA CUDA backend |
| [MLX](https://github.com/ml-explore/mlx) | Apple Silicon Metal backend |

[SimNIBS](https://github.com/simnibs/simnibs) uses CAT-Surface tools in its cortical surface registration workflow. SimNIBS is not redistributed by this package.

## Earlier distributions

The [v0.1.0 release](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0) used PyTorch and Triton and included four Linux executables built from CAT-Surface revision [`628b6851d8638f3ab773cd25c0ec406d0ec61ede`](https://github.com/ChristianGaser/CAT-Surface/tree/628b6851d8638f3ab773cd25c0ec406d0ec61ede) with `-O2 -fPIC`. The corresponding helper and upstream sources remain available with that release. Those dependencies and executables are not part of the current package.
