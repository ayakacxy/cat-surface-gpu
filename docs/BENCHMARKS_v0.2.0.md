# Benchmarks: v0.2.0

[English](BENCHMARKS_v0.2.0.md) · [简体中文](BENCHMARKS_v0.2.0.zh-CN.md)

Measured on 2026-10-13 using the released wheel and the unmodified CAT-Surface CPU reference at `b0c2aa2395ea4957a331b9444fb54c7716b6572d`. Individual runs, parameters, input hashes and output comparisons are available in the [JSON record](benchmark-v0.2.0.json).

## Environment and inputs

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GiB |
| CPU | Intel Xeon Gold 6230 @ 2.10 GHz, two sockets |
| Python / CuPy | 3.11.15 / 14.2.0 |
| NumPy / SciPy / Numba / nibabel | 2.3.0 / 1.17.1 / 0.64.0 / 5.3.3 |
| CUDA runtime | 12.6 |
| Numba CPU threads | 8 |
| CPU reference build | GCC 9.4.0, `-O2 -fPIC`, static libraries |
| Input surfaces | Public fsaverage templates, 163,842 vertices and 327,680 triangles each |
| Sphere settings | `stop_at=10`, all 4,999 area-smoothing sweeps |
| Registration settings | `steps=2`, `runs=2`, averaging and rotation enabled, `512 × 256` map, type-5 features |
| GPU solver | Default PCG with 40 iterations; FMG measured separately |

Sphere generation uses `lh.white.gii`. Registration maps the supplied LH white/sphere pair to the supplied RH white/sphere pair. Both implementations receive the same files. These are separate single-surface measurements, rather than a workflow that passes the generated sphere into registration.

The C reference uses its internal threading unchanged. CPU and GPU clocks were not fixed during measurement.

## Complete command times

The table reports three fresh-process runs after the first GPU invocation, with populated kernel and input-file caches. Versions alternate order between rounds. Timings cover process startup, imports, reading, preparation, CUDA initialization, cache loading or compilation, computation, transfer and GIFTI writing. Reference compilation, package installation and output comparison are excluded.

| Component | CPU median | GPU median | CPU/GPU ratio | CPU range | GPU range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sphere generation | 216.722 s | 10.340 s | 20.96× | 212.922–263.969 s | 9.907–13.642 s |
| Registration, default PCG | 182.182 s | 32.550 s | 5.60× | 180.606–186.225 s | 28.075–35.514 s |
| Registration, FMG | 182.182 s | 164.697 s | 1.11× | 180.606–186.225 s | 129.926–179.939 s |

The CPU reference uses FMG. Its comparison with default GPU PCG includes a solver change as well as the implementation change. GPU FMG follows the same solver family but does not produce numerically identical results to the C reference.

First GPU calls took **17.766 seconds** for sphere generation, **48.985 seconds** for PCG registration and **130.208 seconds** for FMG registration. They used one newly created application cache in that order, after setup imports and an integer-vector CUDA check. Later first calls could reuse earlier kernel compilation, so these are not three independent cold-cache measurements. They are excluded from the medians.

## Output comparison

Triangle arrays match exactly, all coordinates are finite and no zero-area triangles occur. Each cell below gives the largest statistic observed across the three comparisons. Coordinate differences are per array element; vertex distances are Euclidean distances between corresponding points. Both use the GIFTI coordinate units.

| Component | Absolute coordinate difference: max / mean / p99 | Vertex distance: max / mean / p99 |
| --- | ---: | ---: |
| Sphere generation | 0.057757 / 0.018752 / 0.050117 | 0.062416 / 0.036491 / 0.061337 |
| Registration, PCG | 0.166157 / 0.012807 / 0.096529 | 0.166649 / 0.027569 / 0.122179 |
| Registration, FMG | 0.165802 / 0.012483 / 0.095415 | 0.166345 / 0.026878 / 0.122087 |

GPU smoothing updates independent vertex groups, whereas the C reference updates vertices serially. This update order and the solver differences affect the coordinates. These measurements compare geometry; they do not establish equivalent anatomical registration accuracy.

## Reproduce the comparison

Install the v0.2.0 package, then use `tools/benchmark_cat_surface.py` from the current repository. The script was added after publication and is absent from the original source archive.

The following Linux example requires GNU Autotools, a C compiler, Make and the GitHub CLI. It stores the reference source, build files and caches under `.work/performance`.

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

Run the comparison with public fsaverage templates or your own suitable GIFTI files:

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

The results directory must not already exist. The script records each completed command and compares triangle indices and coordinates. `--include-fmg` adds the separate FMG measurement. Input files are not modified.
