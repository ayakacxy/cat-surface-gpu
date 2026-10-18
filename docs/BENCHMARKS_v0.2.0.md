# v0.2.0 Performance and Output Comparisons

[English](BENCHMARKS_v0.2.0.md) · [简体中文](BENCHMARKS_v0.2.0.zh-CN.md)

This page documents v0.2.0. See the [v0.3.0 notes](RELEASE_v0.3.0.md) for current platform support, installation and benchmarks.

These measurements use the released **v0.2.0** wheel and the unmodified CAT-Surface CPU reference at commit `b0c2aa2395ea4957a331b9444fb54c7716b6572d`. The results below were recorded on **2026-10-13**. Historical PyTorch/Triton measurements remain in the [v0.1.0 benchmarks](BENCHMARKS.md).

## Environment and workload

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GiB |
| CPU | Intel Xeon Gold 6230 @ 2.10 GHz, two sockets |
| Python / CuPy | 3.11.15 / 14.2.0 |
| NumPy / SciPy / Numba / nibabel | 2.3.0 / 1.17.1 / 0.64.0 / 5.3.3 |
| CUDA runtime | 12.6 |
| Numba CPU threads | 8 |
| CPU reference build | GCC 9.4.0, `-O2 -fPIC`, static libraries |
| Inputs | Public fsaverage templates; 163,842 vertices and 327,680 triangles each |
| Sphere settings | `stop_at=10`, all 4,999 area-smoothing sweeps |
| Registration settings | `steps=2`, `runs=2`, averaging and rotation enabled, `512 x 256` map, type-5 features |
| GPU registration solver | Default PCG with 40 iterations; FMG measured separately |

Sphere generation uses `lh.white.gii`. Registration maps the fixed LH white/sphere pair to the fixed RH white/sphere pair. Both implementations receive the same files. The components are measured separately, with no generated-sphere chain or bilateral concurrency.

The C reference uses its unmodified internal threading. CPU and GPU clocks were not controlled by the benchmark; retain the measured ranges when estimating runtime.

## Complete command timings

Each row reports **three fresh-process runs** after the first GPU invocation. Disk kernel caches and OS input-file caches are populated. Timings cover process startup, imports, input reading, CPU preparation, CUDA initialization, compilation or cache loading, computation, transfer, and GIFTI writing. Reference compilation, package installation, and output validation are excluded. CPU/GPU order alternates between rounds.

| Component | CPU median | GPU median | CPU/GPU time ratio | CPU range | GPU range |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CAT_Surf2Sphere` | 216.722 s | 10.340 s | 20.96x | 212.922–263.969 s | 9.907–13.642 s |
| `CAT_SurfWarp`, default PCG | 182.182 s | 32.550 s | 5.60x | 180.606–186.225 s | 28.075–35.514 s |
| `CAT_SurfWarp`, FMG | 182.182 s | 164.697 s | 1.11x | 180.606–186.225 s | 129.926–179.939 s |

The CPU reference uses FMG. The default GPU PCG row compares the complete default workflows; its ratio includes the effect of different solvers and does not isolate GPU hardware acceleration. Selecting GPU FMG does not make the Python implementation numerically identical to the C reference.

First GPU invocations took **17.766 s** for sphere generation, **48.985 s** for PCG registration, and **130.208 s** for FMG registration. A new task cache was used, with only setup imports and an integer-vector CUDA preflight before the first sphere call; the surface kernels had not been warmed. PCG and then FMG used the same cache. These first calls are excluded from the medians and are not independent cold-cache measurements for each solver. Repeated timings still include imports and any compilation required by a fresh process.

## Output differences

CPU/GPU outputs retain the input triangle arrays exactly, have finite coordinates, and contain no zero-area triangles. The table reports the largest statistic observed across the three comparisons. Coordinate units are those stored in the GIFTI files.

| Component | Max / mean / p99 absolute coordinate difference | Max / mean / p99 vertex distance |
| --- | ---: | ---: |
| `CAT_Surf2Sphere` | 0.057757 / 0.018752 / 0.050117 | 0.062416 / 0.036491 / 0.061337 |
| `CAT_SurfWarp`, PCG | 0.166157 / 0.012807 / 0.096529 | 0.166649 / 0.027569 / 0.122179 |
| `CAT_SurfWarp`, FMG | 0.165802 / 0.012483 / 0.095415 | 0.166345 / 0.026878 / 0.122087 |

Absolute coordinate differences are computed per array element. Vertex distances are Euclidean distances between corresponding 3D vertices. The GPU smoothing uses colored independent vertex groups, whereas the reference updates vertices in its serial order. The implementations and solvers can therefore produce different coordinates. These results report the observed differences; they do not establish equivalent anatomical registration accuracy or reuse the v0.1.0 error threshold.

The [JSON record](benchmark-v0.2.0.json) includes individual times, parameters, input and implementation hashes, reference-program hashes, and all output comparisons. The tested GPU source matches the released wheel. These times cover the specified CAT commands, not the complete CHARM workflow.

## Reproduce the comparison

Use the benchmark script from the current repository checkout with an installed v0.2.0 package. The script was added after the release; it is not in the original v0.2.0 source archive. Supply the public templates or your own suitable GIFTI inputs.

Set project-local temporary and cache paths, then build the pinned CPU reference. GNU Autotools, a C compiler, Make, and the GitHub CLI are needed for this example:

```bash
mkdir -p .work/performance/{tmp,cache/pip,cache/numba,cache/cupy,cache/cuda,source}
export TMPDIR="$PWD/.work/performance/tmp" TMP="$PWD/.work/performance/tmp" TEMP="$PWD/.work/performance/tmp"
export XDG_CACHE_HOME="$PWD/.work/performance/cache"
export PIP_CACHE_DIR="$PWD/.work/performance/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/performance/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/performance/cache/cupy"
export CUDA_CACHE_PATH="$PWD/.work/performance/cache/cuda"
export PYTHONPYCACHEPREFIX="$PWD/.work/performance/cache/pycache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8

gh release download v0.2.0 --repo ayakacxy/cat-surface-gpu \
  --pattern 'CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz' \
  --dir .work/performance/source

tar -xf .work/performance/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz \
  -C .work/performance/source
CAT_REFERENCE_ROOT="$PWD/.work/performance/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d"
(
  cd "$CAT_REFERENCE_ROOT"
  NOCONFIGURE=1 ./autogen.sh
  CFLAGS='-O2 -fPIC' ./configure --disable-shared --enable-static
  make -j8 libCAT.la
  make -C Progs -j8 CAT_Surf2Sphere CAT_SurfWarp
)
```

Run the complete command comparison:

```bash
python tools/benchmark_cat_surface.py \
  --reference-sphere "$CAT_REFERENCE_ROOT/Progs/CAT_Surf2Sphere" \
  --reference-warp "$CAT_REFERENCE_ROOT/Progs/CAT_SurfWarp" \
  --source-surface /path/to/fsaverage/lh.white.gii \
  --source-sphere /path/to/fsaverage/lh.sphere.gii \
  --target-surface /path/to/fsaverage/rh.white.gii \
  --target-sphere /path/to/fsaverage/rh.sphere.gii \
  --work-dir .work/performance/results \
  --repeats 3 --threads 8 --include-fmg
```

The output directory must not already exist. The script writes a checkpoint after each command and reports triangle equality, coordinate differences, and each solver's times. `--include-fmg` adds the separate FMG comparison. Input files are not modified.
