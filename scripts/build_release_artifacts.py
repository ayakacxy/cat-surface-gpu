#!/usr/bin/env python3
"""Assemble deterministic release archives, an SBOM, and checksums."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_release_tree(root: Path, destination: Path) -> None:
    """Copy the runnable Linux bundle without local caches or private fixtures."""

    files = (
        "CITATION.cff",
        "environment.yml",
        "LICENSE",
        "MANIFEST.in",
        "pyproject.toml",
        "README.md",
        "README.zh-CN.md",
        "setup.py",
        "THIRD_PARTY_NOTICES.md",
        "UPSTREAM.lock",
    )
    directories = ("bin", "docs", "native", "scripts", "src", "tools", "upstream")
    destination.mkdir(parents=True)
    for name in files:
        shutil.copy2(root / name, destination / name)
    for name in directories:
        shutil.copytree(
            root / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _write_deterministic_tar(source: Path, output: Path, epoch: int) -> None:
    """Create a gzip-compressed tar with normalized metadata."""

    with output.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=epoch
        ) as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="w") as archive:
                for path in sorted(source.rglob("*")):
                    relative = path.relative_to(source.parent)
                    info = archive.gettarinfo(str(path), arcname=str(relative))
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = epoch
                    if path.is_file():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def _write_sbom(root: Path, output: Path, version: str) -> None:
    """Write a compact CycloneDX SBOM with dependency and binary hashes."""

    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    components = []
    for dependency in project["dependencies"]:
        name = dependency.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0]
        components.append(
            {
                "type": "library",
                "name": name,
                "version": dependency[len(name) :].lstrip("=<>!~ ") or "unspecified",
                "purl": f"pkg:pypi/{name.lower().replace('_', '-')}",
            }
        )
    for binary in sorted((root / "bin" / "linux-x86_64").iterdir()):
        if binary.name == "SHA256SUMS":
            continue
        components.append(
            {
                "type": "file",
                "name": binary.name,
                "hashes": [{"alg": "SHA-256", "content": _sha256(binary)}],
            }
        )
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": project["name"],
                "version": version,
                "licenses": [{"license": {"id": project["license"]}}],
            }
        },
        "components": components,
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Build release-adjacent artifacts after Python distributions exist."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dist = (root / args.dist).resolve() if not args.dist.is_absolute() else args.dist
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]
    commit = subprocess.check_output(
        ["git", "-C", str(args.upstream_source), "rev-parse", "HEAD"], text=True
    ).strip()
    lock = dict(
        line.split("=", 1)
        for line in (root / "UPSTREAM.lock").read_text(encoding="utf-8").splitlines()
        if line
    )
    if commit != lock["CAT_SURFACE_COMMIT"]:
        raise RuntimeError(
            f"Upstream checkout is {commit}, expected {lock['CAT_SURFACE_COMMIT']}"
        )
    dist.mkdir(parents=True, exist_ok=True)
    if not list(dist.glob("*.whl")) or not list(dist.glob("*.tar.gz")):
        raise RuntimeError(
            "Build the wheel and source distribution before release assembly"
        )
    epoch = int(
        subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", "HEAD"], cwd=root, text=True
        ).strip()
    )
    with tempfile.TemporaryDirectory(prefix="cat-surface-gpu-release-") as temp_name:
        bundle = Path(temp_name) / f"cat-surface-gpu-{version}-linux-x86_64"
        _copy_release_tree(root, bundle)
        _write_deterministic_tar(
            bundle,
            dist / f"cat-surface-gpu-{version}-linux-x86_64.tar.gz",
            epoch,
        )
    upstream_archive = dist / f"CAT-Surface-{commit}.tar.gz"
    subprocess.run(
        [
            "git",
            "-C",
            str(args.upstream_source),
            "archive",
            "--format=tar.gz",
            f"--prefix=CAT-Surface-{commit}/",
            f"--output={upstream_archive}",
            commit,
        ],
        check=True,
    )
    _write_sbom(root, dist / f"cat-surface-gpu-{version}.cdx.json", version)
    checksum_path = dist / "SHA256SUMS"
    artifacts = sorted(
        path for path in dist.iterdir() if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(f"Release artifacts written to {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
