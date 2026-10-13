## CAT-Surface GPU v0.1.0

[English](RELEASE_v0.1.0.md) · [简体中文](RELEASE_v0.1.0.zh-CN.md)

The first release accelerates `CAT_Surf2Sphere` and `CAT_SurfWarp` with PyTorch and Triton while retaining CAT-Surface CPU reference and helper programs. This is an independent community project, with CAT-Surface revision `628b6851d8638f3ab773cd25c0ec406d0ec61ede` as its reference.

### Highlights

- Installed CLIs: `cat-surf2sphere-gpu`, `cat-surfwarp-gpu`, and `cat-surface-rotation-gpu`
- Ordered Triton area smoothing for spherical inflation, with the initial stages run by the bundled CPU reference
- PyTorch/Triton surface registration with FP64 DARTEL, initial rotation, resampling, averaging, and output generation
- Reproducibly built Linux x86-64 reference and helper programs, with corresponding source
- Synthetic fixtures, CPU CI, CodeQL, and a manual CUDA workflow
- Platform wheel, Linux bundle, source archives, CycloneDX dependency inventory, and SHA-256 manifest

### Audited performance

The release audit used real left-hemisphere inputs on an NVIDIA GeForce RTX 2080 Ti with PyTorch 2.6.0+cu126:

| Component | CPU reference | GPU | Speedup | Max vertex error |
| --- | ---: | ---: | ---: | ---: |
| `CAT_SurfWarp` | 126.658 s | 35.061 s | 3.613x | 5.481e-5 |
| `CAT_Surf2Sphere` | 54.985 s | 10.689 s | 5.144x | 1.144e-5 |

Triangle arrays matched exactly, and spherical inflation ran all 4,999 sweeps. These are single-run release checks. The separate bilateral CAT registration measurement was **55.2573 s**, covering spherical inflation, topology upsampling, and registration. See [v0.1.0 benchmarks](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BENCHMARKS.md) for inputs, settings, timing scope, and paired comparisons.

All measurements above describe v0.1.0 and the recorded environment. They are not v0.2.0 performance results or complete CHARM runtimes.

### Verification

- CPU test suite: 33 passed, 2 skipped
- CUDA host suite: 34 passed, 1 expected skip for the no-GPU behavior test
- CI and CodeQL passed on the v0.1.0 release commit
- Wheel installation, three CLI entry points, bundled helper hashes, and release checksums verified

### Installation

Follow the [v0.1.0 README](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/README.md) to create the Conda environment and install the matching CUDA-enabled PyTorch stack and platform wheel. For the Python/Numba/CuPy implementation, see [v0.2.0](https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.2.0).

Licensed under GPL-3.0-or-later. Upstream CAT-Surface attribution and license information are included in `THIRD_PARTY_NOTICES.md` and the corresponding source archive.

This description was revised on 2026-10-13. The v0.1.0 tag and release files are unchanged.
