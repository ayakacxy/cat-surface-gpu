# Security policy

[English](SECURITY.md) · [简体中文](SECURITY.zh-CN.md)

## Supported versions

Security fixes target the latest stable release. Update to the latest release before reporting an issue that may already have been fixed.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/ayakacxy/cat-surface-gpu/security/advisories/new). Do not disclose an unpatched vulnerability in a public issue.

Include the affected version, environment, reproduction steps, and expected impact. Use synthetic inputs and remove credentials and personal data from logs. A minimal reproduction or suggested fix is helpful.

The maintainers will review the report and coordinate a fix and public disclosure with the reporter.

## Data handling

CAT-Surface GPU processes surface files locally. It does not collect telemetry or upload input data. GIFTI files are read through nibabel; computation uses NumPy/Numba, CuPy, or MLX, depending on the selected backend.
