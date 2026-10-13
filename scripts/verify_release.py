#!/usr/bin/env python3
"""检查源码和发布包，确保新版不依赖 CAT 外部程序。"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import tomllib
import zipfile

FORBIDDEN_IMPORTS = {"flashcharm", "simnibs", "torch", "triton", "subprocess", "ctypes"}


def check_source(root: Path) -> None:
    """静态检查导入和语法，不运行 GPU，也不产生字节码缓存。"""
    for path in sorted((root / "src" / "cat_surface_gpu").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name.split(".")[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            if FORBIDDEN_IMPORTS.intersection(names):
                raise RuntimeError(f"Unexpected runtime dependency in {path}: {names}")
    for directory in ("tests", "tools", "scripts"):
        for path in (root / directory).glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_wheel(path: Path, version: str) -> None:
    """确认 wheel 为纯 Python 分发，包含许可和两条命令。"""
    if not path.name.endswith("-py3-none-any.whl"):
        raise RuntimeError(f"Expected a pure Python wheel: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in names:
            if "/bin/" in name or name.endswith((".so", ".dll", ".exe", ".pyc")):
                raise RuntimeError(f"Unexpected compiled file: {name}")
            if not name.endswith("/") and archive.read(name).startswith(b"\x7fELF"):
                raise RuntimeError(f"Unexpected ELF file: {name}")
        for module in ("surf2sphere", "surfwarp", "dartel", "features", "cli"):
            if f"cat_surface_gpu/{module}.py" not in names:
                raise RuntimeError(f"Missing module: {module}")
        for suffix in (
            ".dist-info/entry_points.txt",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/THIRD_PARTY_NOTICES.md",
            ".dist-info/licenses/upstream/CAT-Surface-LICENSE.txt",
        ):
            if not any(name.endswith(suffix) for name in names):
                raise RuntimeError(f"Missing release file: {suffix}")
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
        if f"Version: {version}\n" not in metadata:
            raise RuntimeError("Wheel version differs from pyproject.toml")
        entry_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entries = archive.read(entry_name).decode()
        for command in ("cat-surf2sphere-gpu", "cat-surfwarp-gpu"):
            if command not in entries:
                raise RuntimeError(f"Missing command: {command}")


def main() -> int:
    """校验源码以及可选 wheel。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    check_source(root)
    if args.wheel:
        check_wheel(args.wheel, version)
    print("Release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
