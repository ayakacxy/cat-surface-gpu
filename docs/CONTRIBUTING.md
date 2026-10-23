# Contributing

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

Bug reports, documentation improvements, and code contributions are welcome. For a substantial algorithm or API change, open an issue first to discuss the approach.

## Development setup

Use Python 3.11 or newer in a virtual environment. From a source checkout, install the package in editable mode with the development tools:

```bash
python -m pip install -e ".[test]"
```

Use `.[cuda,test]` for NVIDIA CUDA or `.[metal,test]` for Apple Silicon Metal development. See [installation](../README.md#install) for backend requirements and the [build guide](BUILDING.md) for environment and cache setup.

## Making changes

Keep each pull request focused on one change. Update the English and Simplified Chinese documentation together. New code comments and docstrings use Chinese; retain third-party notices in their original language.

For algorithm changes, describe the effect on mesh indexing, solver settings, precision, and stopping conditions. Compare results using the same input surfaces and parameters. Include topology checks and numerical errors; performance claims also need hardware, timing scope, cache state, and repeated measurements. See [Reproducibility](REPRODUCIBILITY.md) for the comparison procedure.

Do not add patient data, access credentials, local caches, or generated build files to the repository. Credit adapted code and preserve its license notices.

## Checking your work

Run the checks relevant to the change. Documentation-only changes need link and example review; they do not require GPU tests.

For code changes, the standard checks are:

```bash
ruff check src tests tools scripts setup.py
ruff format --check src tests tools scripts setup.py
python -m pytest -q --basetemp .work/setup/tmp/pytest -o cache_dir=.work/setup/cache/pytest
```

Backend-specific changes also need tests on the affected hardware. If that hardware is unavailable, state which checks you could not run. For packaging changes, follow the [build guide](BUILDING.md).

## Opening a pull request

Explain the problem, the resulting behavior, and how you checked it. Link related issues and include any relevant numerical or performance results. The [pull request template](../.github/pull_request_template.md) provides a short outline.
