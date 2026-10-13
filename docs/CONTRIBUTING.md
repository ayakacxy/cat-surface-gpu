# Contributing to CAT-Surface GPU

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

Changes should preserve useful numerical results and make their effects easy to reproduce.

## Development setup

Start with the [README installation steps](../README.md#install), which put temporary files and caches inside the checkout. Then run:

```bash
python -m pip install -e '.[cuda,test]'
python -m pytest -q --basetemp .work/setup/tmp/pytest
python scripts/verify_release.py
```

CPU tests can run without CuPy or a GPU. CUDA tests require a working NVIDIA device; report skips separately.

## Algorithm changes

Describe which upstream operation changes and why. Preserve mesh indexing, supported parameters, and stopping conditions unless the change explicitly calls for a different algorithm. Keep failures visible instead of silently switching to a different implementation.

Before reporting an improvement, compare the same inputs and settings. Include triangle equality, coordinate errors, solver choice, precision, timing scope, repetitions, and peak memory. See [reproducibility](REPRODUCIBILITY.md) for the details. Never commit anatomical or identifiable medical data.

New code comments and docstrings use Chinese. Preserve third-party copyright and license notices in their original language. User documentation is available in English and Simplified Chinese.

## Pull requests

Keep changes focused and describe the resulting behavior, relevant numerical differences, and tests run. If a test needs an unavailable GPU or private fixture, say so. Identify the source of adapted code and retain the relevant author and license notices.
