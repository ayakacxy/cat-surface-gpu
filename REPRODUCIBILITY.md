# Reproducibility and Provenance

## Immutable inputs

- Project version: `0.1.0`
- CAT-Surface source: `628b6851d8638f3ab773cd25c0ec406d0ec61ede`
- Native optimization flags: `-O2 -fPIC`
- Prohibited flags: `-ffast-math`, `-Ofast`, implicit mixed precision, and architecture-specific `-march=native`
- Recommended GPU path: FP64 DARTEL, ordered CAT-compatible `CAT_Surf2Sphere` arithmetic, and explicit CUDA failure

The authoritative source revision is machine-readable in [`UPSTREAM.lock`](UPSTREAM.lock). Binary hashes are in [`bin/linux-x86_64/SHA256SUMS`](bin/linux-x86_64/SHA256SUMS).

## Validation layers

1. Static checks: English-only public source, Python compilation, binary hashes, packaging contents, and ELF dependency inspection.
2. CPU contracts: deterministic unit tests for indexing, interpolation, topology, scheduling, and reference/optimized operator equivalence.
3. CUDA contracts: actual CUDA execution of Triton kernels; absence of a GPU is reported as a skip and never counted as CUDA validation.
4. End-to-end A/B: identical GIFTI inputs and parameters, synchronized wall-clock timing, exact face arrays, and max/mean/p99 vertex errors.

The private anatomical benchmark fixture is not distributed. It contains no data in the repository or release. The included synthetic fixture generator is intended for installation and structural smoke testing; it does not reproduce the headline performance numbers.

## Performance claim boundary

The reported 55.2573-second bilateral number covers the CAT spherical-registration chain only. It is not a complete CHARM or SimNIBS runtime. Every headline value in [BENCHMARKS.md](BENCHMARKS.md) records hardware, software, input shapes, parameters, timing boundaries, and numerical acceptance criteria.
