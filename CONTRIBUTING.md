# Contributing

Thank you for improving CAT-Surface GPU. Correctness and reproducibility take priority over isolated kernel speed.

## Development setup

```bash
conda env create -f environment.yml
conda activate cat-surface-gpu
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[cuda,test]'
pytest -q
```

## Scientific contract

- Preserve the pinned CAT-Surface algorithm, resolution, iteration counts, stopping conditions, update dependencies, reduction order, and tie-breaking semantics.
- Keep the reference path available. An optimized backend must fail explicitly instead of silently falling back.
- Do not use `-ffast-math`, `-Ofast`, implicit autocast, or unapproved floating-point reordering.
- Optimize one measurable hotspot at a time and keep it independently reviewable.
- Validate on identical inputs, shapes, precision, and parameters before reporting speed.

## Required evidence

Algorithm changes require CPU contracts and a real-CUDA A/B. Report hardware, software, timing boundary, warm-up, synchronization, repetitions, peak memory, exact face equality, and max/mean/p99 vertex errors. A CUDA skip is not a CUDA pass.

Public source comments, docstrings, CLI text, tests, and error messages must be in English. Never commit anatomical or identifiable medical data.

## Pull requests

Create a focused branch, run `python scripts/verify_release.py` and `pytest -q`, and complete the pull-request checklist. Commits materially produced with Codex should retain the project attribution trailer:

```text
Co-authored-by: Codex <codex@openai.com>
```
