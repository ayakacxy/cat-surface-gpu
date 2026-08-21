#!/usr/bin/env python3
"""Run deterministic repository and release-artifact checks."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import zipfile


REQUIRED_BINARIES = {
    "CAT_Surf2Sphere",
    "CAT_SurfWarp",
    "cat_surface_rotation_depth",
    "cat_surface_stencil_builder",
}
PUBLIC_SOURCE_GLOBS = ("src/**/*.py", "tests/**/*.py", "tools/**/*.py", "native/**/*.c")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def check_english_only(root: Path) -> None:
    """Reject CJK text from public source while allowing the Chinese README."""

    failures: list[str] = []
    for pattern in PUBLIC_SOURCE_GLOBS:
        for path in root.glob(pattern):
            if CJK_PATTERN.search(path.read_text(encoding="utf-8")):
                failures.append(str(path.relative_to(root)))
    if failures:
        raise RuntimeError("CJK text found in public source: " + ", ".join(failures))


def check_binaries(root: Path) -> None:
    """Verify the committed Linux binaries against SHA256SUMS."""

    directory = root / "bin" / "linux-x86_64"
    expected = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename.strip()] = digest
    if set(expected) != REQUIRED_BINARIES:
        raise RuntimeError(f"Unexpected checksum manifest entries: {sorted(expected)}")
    for name, digest in expected.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"SHA-256 mismatch for {name}")


def check_wheel(path: Path) -> None:
    """Verify a platform wheel contains entry points, notices, and native helpers."""

    if "none-any" in path.name or "linux" not in path.name:
        raise RuntimeError(f"Wheel must be Linux platform-specific: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for binary in REQUIRED_BINARIES:
            if not any(name.endswith(f"/bin/linux-x86_64/{binary}") for name in names):
                raise RuntimeError(f"Wheel is missing {binary}")
        required_suffixes = (
            ".dist-info/entry_points.txt",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/THIRD_PARTY_NOTICES.md",
            ".dist-info/licenses/upstream/CAT-Surface-LICENSE.txt",
        )
        for suffix in required_suffixes:
            if not any(name.endswith(suffix) for name in names):
                raise RuntimeError(f"Wheel is missing {suffix}")


def main() -> int:
    """Validate the checkout and an optional wheel."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    check_english_only(root)
    check_binaries(root)
    subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src", "tests", "tools"],
        cwd=root,
        check=True,
    )
    if args.wheel:
        check_wheel(args.wheel)
    print("Release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
