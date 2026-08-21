"""Locate and verify the Linux helper binaries shipped with the package."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
from importlib.resources import files
import platform
from pathlib import Path


_BINARY_NAMES = {
    "CAT_Surf2Sphere",
    "CAT_SurfWarp",
    "cat_surface_rotation_depth",
    "cat_surface_stencil_builder",
}


def _binary_directory() -> Path:
    """Resolve package resources or the matching editable-checkout directory."""

    packaged = Path(str(files("cat_surface_gpu").joinpath("bin/linux-x86_64")))
    if packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "bin" / "linux-x86_64"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Bundled Linux x86-64 CAT-Surface binaries were not found. "
        "Install the official platform wheel or pass helper paths explicitly."
    )


def bundled_binary(name: str, *, verify: bool = True) -> Path:
    """Return a bundled executable after optional SHA-256 verification."""

    if name not in _BINARY_NAMES:
        raise ValueError(f"Unknown bundled binary: {name}")
    if platform.system() != "Linux" or platform.machine() not in {
        "x86_64",
        "AMD64",
    }:
        raise RuntimeError("Bundled binaries support Linux x86-64 only")
    directory = _binary_directory()
    binary = directory / name
    if not binary.is_file():
        raise FileNotFoundError(f"Bundled binary does not exist: {binary}")
    if verify:
        checksums = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            digest, filename = line.split(maxsplit=1)
            checksums[filename.strip()] = digest
        expected = checksums.get(name)
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
        if expected is None or actual != expected:
            raise RuntimeError(f"Bundled binary SHA-256 verification failed: {binary}")
    return binary


def bundled_binary_names() -> tuple[str, ...]:
    """Return bundled executable names in stable order."""

    return tuple(sorted(_BINARY_NAMES))
