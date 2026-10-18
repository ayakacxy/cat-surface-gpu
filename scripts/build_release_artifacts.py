#!/usr/bin/env python3
"""为 Python 分发生成依赖清单、固定上游源码和 SHA-256 校验文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

from packaging.requirements import Requirement


def sha256(path: Path) -> str:
    """分块计算文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """只处理指定版本的发布文件，不混入旧版或本地临时内容。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-archive", type=Path, required=True)
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    dist = (args.dist or root / "dist" / version).resolve()
    lock = dict(
        line.split("=", 1)
        for line in (root / "UPSTREAM.lock").read_text().splitlines()
        if line
    )
    if sha256(args.upstream_archive) != lock["CAT_SURFACE_ARCHIVE_SHA256"]:
        raise RuntimeError("Upstream source archive does not match UPSTREAM.lock")
    artifacts = [
        dist / f"cat_surface_gpu-{version}-py3-none-any.whl",
        dist / f"cat_surface_gpu-{version}.tar.gz",
    ]
    if not all(path.is_file() for path in artifacts):
        raise RuntimeError("Build the current wheel and source distribution first")
    upstream = dist / f"CAT-Surface-{lock['CAT_SURFACE_COMMIT']}.tar.gz"
    shutil.copy2(args.upstream_archive, upstream)
    artifacts.append(upstream)
    components = []
    groups = {"runtime": project["dependencies"], **project["optional-dependencies"]}
    for group, dependencies in groups.items():
        for specification in dependencies:
            requirement = Requirement(specification)
            components.append(
                {
                    "type": "library",
                    "name": requirement.name,
                    "purl": f"pkg:pypi/{requirement.name.lower().replace('_', '-')}",
                    "scope": "required" if group == "runtime" else "optional",
                    "properties": [
                        {
                            "name": "dependency:specifier",
                            "value": str(requirement.specifier),
                        },
                        {"name": "dependency:group", "value": group},
                    ],
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
    sbom = dist / f"cat-surface-gpu-{version}.cdx.json"
    sbom.write_text(json.dumps(document, indent=2) + "\n")
    artifacts.append(sbom)
    (dist / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(artifacts))
    )
    print(f"Release artifacts written to {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
