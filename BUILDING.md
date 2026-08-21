# Building the Native Binaries

The four Linux executables distributed with CAT-Surface GPU are built from the exact CAT-Surface revision recorded in [`UPSTREAM.lock`](UPSTREAM.lock). The two project-specific helper sources live in [`native/`](native/). No `-ffast-math`, native-CPU tuning, reduced iteration count, or approximate algorithm is used.

## Build dependencies

On Ubuntu 22.04 or newer:

```bash
sudo apt-get update
sudo apt-get install -y \
  autoconf automake build-essential git libtool pkg-config
```

The upstream CAT-Surface repository vendors its remaining C dependencies.

## Rebuild

```bash
BUILD_JOBS=8 ./scripts/build_native.sh
```

`BUILD_JOBS` controls compilation only. Runtime stencil parallelism is controlled independently by `--stencil-threads` and defaults to 8.

The script checks out the pinned source revision, builds the upstream static library with `-O2 -fPIC`, compiles all four executables, strips nonessential symbols, and writes a SHA-256 manifest. Outputs are placed in `build/native-output/` unless `OUTPUT_DIR` is set.

To compare a fresh build with the distributed binaries:

```bash
sha256sum -c bin/linux-x86_64/SHA256SUMS
file build/native-output/*
ldd build/native-output/CAT_SurfWarp
```

Byte-identical binaries require the same compiler, binutils, glibc, Autotools, and source path normalization. A non-identical hash is not by itself a numerical failure; run the A/B validation described in [BENCHMARKS.md](BENCHMARKS.md).

## Corresponding source

The release source archive contains all code written for this project. The exact upstream CAT-Surface source is available at the pinned commit and is attached to the GitHub release as a separate source archive. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license details.
