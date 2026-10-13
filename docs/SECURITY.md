# Security and Vulnerability Reporting

[English](SECURITY.md) · [简体中文](SECURITY.zh-CN.md)

## Supported versions

Security fixes are provided for the latest release on the default branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue for an unpatched vulnerability and do not include access tokens, private paths, medical images, or identifiable subject data.

Include the affected version, environment, reproduction steps, impact, and any proposed mitigation. You should receive an acknowledgement within seven days. Public disclosure will be coordinated after a fix is available.

## Scope

The package reads GIFTI surfaces and uses NumPy, Numba, nibabel, and CuPy to process them. Version 0.2.0 does not bundle CAT executables. The project does not collect telemetry or upload input data.
