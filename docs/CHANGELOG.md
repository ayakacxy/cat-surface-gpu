# Project Changelog

All notable changes are documented here. This project follows semantic versioning after the initial `0.1.0` release.

## [0.1.0] - 2026-08-21

### Added

- GPU-accelerated `CAT_Surf2Sphere` area smoothing with ordered CAT-compatible arithmetic.
- GPU-accelerated `CAT_SurfWarp`-compatible registration with FP64 DARTEL, CUDA Graph support, initial rotation, averaging, and final warp output.
- Verified Linux x86-64 reference and helper binaries with corresponding helper source.
- Installed command-line interfaces, Conda setup, CPU/CUDA contracts, benchmarks, and numerical acceptance criteria.
- Reproducible native and release-artifact builders, a platform wheel, source archive, Linux bundle, upstream corresponding-source archive, CycloneDX SBOM, and SHA-256 manifests.
- CPU CI, manually triggered self-hosted GPU CI, CodeQL, Dependabot, issue forms, and contributor/security documentation.

### Compatibility

- Validated against CAT-Surface commit `628b6851d8638f3ab773cd25c0ec406d0ec61ede`.
- The project is independent and does not modify an existing SimNIBS installation automatically.

[0.1.0]: https://github.com/ayakacxy/cat-surface-gpu/releases/tag/v0.1.0
