# Building from Source

[English](BUILDING.md) · [简体中文](BUILDING.zh-CN.md)

Version 0.3.0 ships Python source. Building the package does not compile or bundle CAT-Surface executables, project-specific native extensions, or CUDA binaries. Numba, CuPy and MLX compile kernels when the selected backend runs.

## Build a wheel and source archive

Run from the repository root with Python 3.11 or newer:

```bash
mkdir -p .work/package/{tmp,cache/pip,cache/numba,cache/cupy,dist}
export TMPDIR="$PWD/.work/package/tmp" TMP="$PWD/.work/package/tmp" TEMP="$PWD/.work/package/tmp"
export XDG_CACHE_HOME="$PWD/.work/package/cache"
export PIP_CACHE_DIR="$PWD/.work/package/cache/pip"
export NUMBA_CACHE_DIR="$PWD/.work/package/cache/numba"
export CUPY_CACHE_DIR="$PWD/.work/package/cache/cupy"

python -m pip install build
python -m build --outdir .work/package/dist
```

The wheel uses the `py3-none-any` tag because this project ships no compiled extension. Install the backend needed by your platform: CuPy and an NVIDIA driver for CUDA, or MLX for Apple Metal. CPU execution uses the base dependencies.

Install the wheel and CUDA 12.x dependency:

```bash
python -m pip install .work/package/dist/cat_surface_gpu-0.3.0-py3-none-any.whl
python -m pip install 'cupy-cuda12x>=13'
```

The source distribution includes the Python implementation, inline CUDA and Metal kernel source, tests, documentation, and attribution. It contains no anatomical inputs or templates.

## Verify the package

```bash
python scripts/verify_release.py
python -m pytest -q --basetemp .work/package/tmp/pytest
```

CPU tests and archive inspection run without a GPU. CUDA tests require an NVIDIA GPU; Metal tests require Apple Silicon macOS and MLX. Test output lists skipped cases. See [reproducibility](REPRODUCIBILITY.md) for numerical comparisons and benchmark procedures.

## Assemble release files

Keep the CAT-Surface source archive identified by `UPSTREAM.lock` under `.work/package/source/`. After building the wheel and source distribution above, run:

```bash
python scripts/build_release_artifacts.py \
  --upstream-archive .work/package/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz \
  --dist .work/package/dist
```

The script checks the upstream archive's SHA-256 against `UPSTREAM.lock` before assembling five release files:

- Python wheel;
- project source archive;
- pinned CAT-Surface source archive;
- CycloneDX dependency inventory;
- `SHA256SUMS` for the four artifacts above.

The upstream archive is provided separately for source reference. It is not installed by the wheel. There is no Linux executable bundle in v0.3.0. The [source manifest](../upstream/SOURCE_MANIFEST.json) records the hashes of the released Python modules.

## Rebuilding v0.1.0

The earlier release includes four Linux executables and a different build procedure. Use the source and [build guide at the v0.1.0 tag](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BUILDING.md) in a separate checkout if you need to reproduce those binaries. The v0.3.0 package does not use them.

[Documentation index](README.md) · [Project README](../README.md)
