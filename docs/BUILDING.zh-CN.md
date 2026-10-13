# 从源码构建

[English](BUILDING.md) · [简体中文](BUILDING.zh-CN.md)

0.2.0 发布的是 Python 源码。构建软件包时不会编译或附带 CAT-Surface 可执行文件、本项目专用的原生扩展或 CUDA 二进制文件。Numba 和 CuPy 会在相应函数运行时编译内核。

## 构建 wheel 和源码包

使用 Python 3.11 或更新版本，在仓库根目录运行：

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

wheel 使用 `py3-none-any` 标记，因为本项目没有随包发布的编译扩展。该标记不表示 CUDA 流程可以在所有平台运行；仍需合适的 CuPy 版本和 NVIDIA 驱动。

安装 wheel 和 CUDA 12.x 对应依赖：

```bash
python -m pip install .work/package/dist/cat_surface_gpu-0.2.0-py3-none-any.whl
python -m pip install 'cupy-cuda12x>=13'
```

源码包包含 Python 实现、内嵌 CUDA 内核源码、测试、文档和署名说明，不包含解剖输入数据或模板。

## 检查软件包

```bash
python scripts/verify_release.py
python -m pytest -q --basetemp .work/package/tmp/pytest
```

CUDA 测试需要可用的 GPU。CPU 测试和归档内容检查可以在没有 GPU 的环境中运行，但应单独记录跳过的 CUDA 测试。软件包检查与数值或性能验证的区别见[复现说明](REPRODUCIBILITY.zh-CN.md)。

## 整理发行文件

将 `UPSTREAM.lock` 指定的 CAT-Surface 源码归档放在 `.work/package/source/` 下。完成上述 wheel 和源码包构建后，运行：

```bash
python scripts/build_release_artifacts.py \
  --upstream-archive .work/package/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz \
  --dist .work/package/dist
```

脚本先将上游源码归档的 SHA-256 与 `UPSTREAM.lock` 对照，再整理以下五个发行文件：

- Python wheel；
- 本项目源码包；
- 固定版本的 CAT-Surface 源码包；
- CycloneDX 依赖清单；
- 记录以上四个文件校验值的 `SHA256SUMS`。

上游源码归档单独提供，用于查阅来源，不会随 wheel 安装。v0.2.0 不提供 Linux 可执行文件包。[源码清单](../upstream/SOURCE_MANIFEST.json)记录了本次发布的 Python 模块哈希。

## 重新构建 v0.1.0

旧版包含四个 Linux 可执行文件，构建方法也与本版不同。如需复现这些程序，请在单独的检出目录中使用旧版源码和 [v0.1.0 标签中的原始构建说明（英文历史文件）](https://github.com/ayakacxy/cat-surface-gpu/blob/v0.1.0/docs/BUILDING.md)。v0.2.0 软件包不使用这些程序。

[返回文档目录](README.zh-CN.md) · [返回项目首页](../README.zh-CN.md)
