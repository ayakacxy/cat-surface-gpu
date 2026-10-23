# 从源码构建

[English](BUILDING.md) · [简体中文](BUILDING.zh-CN.md)

本项目使用 setuptools 构建，要求 Python 3.11 或更新版本。wheel 包含 Python 模块和内嵌的 CUDA/Metal 内核源码，由相应后端库在运行时编译内核。构建 wheel 本身不需要 GPU 或 C/C++ 编译器。

预构建软件包可从[发布页面](https://github.com/ayakacxy/cat-surface-gpu/releases/latest)下载。

## 安装发布的 wheel

从发布附件下载 wheel 后，在 Python 环境中安装：

```bash
python -m pip install ./cat_surface_gpu-0.3.0-py3-none-any.whl
```

这会安装 CPU 支持。GPU 计算还需要相应依赖：CUDA 12 使用 `cupy-cuda12x>=13`，Apple Silicon Metal 使用 `mlx>=0.31`。具体要求见[后端安装说明](../README.zh-CN.md#install)。

## 准备环境

克隆仓库，进入项目目录并创建虚拟环境：

```bash
git clone https://github.com/ayakacxy/cat-surface-gpu.git
cd cat-surface-gpu
python -m venv .venv
```

Linux 或 macOS：

```bash
source .venv/bin/activate
mkdir -p .work/build/{tmp,cache,dist}
export TMPDIR="$PWD/.work/build/tmp" TMP="$PWD/.work/build/tmp" TEMP="$PWD/.work/build/tmp"
export XDG_CACHE_HOME="$PWD/.work/build/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export CUPY_CACHE_DIR="$XDG_CACHE_HOME/cupy"
export CAT_SURFACE_CACHE_ROOT="$XDG_CACHE_HOME/cat-surface"
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
$buildDir = Join-Path (Get-Location) ".work/build"
New-Item -ItemType Directory -Force "$buildDir/tmp", "$buildDir/cache", "$buildDir/dist" | Out-Null
$env:TMPDIR = $env:TMP = $env:TEMP = "$buildDir/tmp"
$env:XDG_CACHE_HOME = "$buildDir/cache"
$env:PIP_CACHE_DIR = "$buildDir/cache/pip"
$env:NUMBA_CACHE_DIR = "$buildDir/cache/numba"
$env:CUPY_CACHE_DIR = "$buildDir/cache/cupy"
$env:CAT_SURFACE_CACHE_ROOT = "$buildDir/cache/cat-surface"
```

以上设置将临时文件、指定的缓存和发布归档保存在 `.work/build`。接着安装开发依赖：

```bash
python -m pip install -e ".[test]"
```

基础安装支持 CPU 运行。Linux 或 Windows 上使用 NVIDIA CUDA 时增加 `.[cuda]`；Apple Silicon macOS 使用 `.[metal]`。各后端的要求见[安装指南](../README.zh-CN.md#install)。

## 构建与检查软件包

以下命令适用于三个平台：

```bash
python -m build --outdir .work/build/dist
python scripts/verify_release.py --wheel .work/build/dist/cat_surface_gpu-0.3.0-py3-none-any.whl
```

构建生成 wheel 和源码归档。`py3-none-any` wheel 本身与平台无关；NumPy、Numba、CuPy 和 MLX 各自提供平台相关依赖。源码归档还包含测试、工具、文档和许可说明。解剖表面及模板由用户另行提供。

检查安装后的命令行入口：

```bash
python -m pip install --force-reinstall --no-deps .work/build/dist/cat_surface_gpu-0.3.0-py3-none-any.whl
cat-surf2sphere-gpu --help
cat-surfwarp-gpu --help
```

## 开发检查

```bash
python scripts/verify_release.py
python -m ruff check src tests tools scripts setup.py
python -m ruff format --check src tests tools scripts setup.py
python -m pytest -q --basetemp .work/build/pytest -o cache_dir=.work/build/cache/pytest
```

具备相应硬件和后端时，GPU 测试会正常运行；否则 pytest 将其报告为跳过。CI 在 Linux、Windows 和 macOS 上执行 CPU 测试、软件包检查和命令行检查。数值比较与计时方法见[复现指南](REPRODUCIBILITY.zh-CN.md)。

## 整理发布附件

构建 wheel 和源码归档后，将 [UPSTREAM.lock](../UPSTREAM.lock) 指定的 CAT-Surface 源码归档放到 `.work/build/source/`，然后运行：

```bash
python scripts/build_release_artifacts.py --upstream-archive .work/build/source/CAT-Surface-b0c2aa2395ea4957a331b9444fb54c7716b6572d.tar.gz --dist .work/build/dist
```

脚本核对归档哈希后，整理以下文件：

| 附件 | 内容 |
| --- | --- |
| `cat_surface_gpu-0.3.0-py3-none-any.whl` | 可安装的 Python 软件包 |
| `cat_surface_gpu-0.3.0.tar.gz` | 本项目源码包 |
| `CAT-Surface-<commit>.tar.gz` | 固定版本的算法参考源码 |
| `cat-surface-gpu-0.3.0.cdx.json` | CycloneDX 依赖清单 |
| `SHA256SUMS` | 以上四个附件的 SHA-256 校验值 |

参考源码归档用于提供源码及署名信息，不会随 wheel 安装。其许可和用途见[第三方说明](../THIRD_PARTY_NOTICES.zh-CN.md)。

[文档目录](README.zh-CN.md) · [项目首页](../README.zh-CN.md)
