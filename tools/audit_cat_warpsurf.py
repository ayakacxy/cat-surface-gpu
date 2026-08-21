#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""CAT-Surface GPU implementation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import nibabel as nib


@dataclass(frozen=True)
class GeometrySummary:
    """Represent GeometrySummary."""

    path: str
    size_bytes: int
    arrays: tuple[dict[str, Any], ...]


def summarize_geometry(path: Path) -> GeometrySummary:
    """Summarize geometry."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GIFTI file does not exist: {path}")
    image = nib.load(str(path))
    arrays = tuple(
        {
            "shape": list(array.data.shape),
            "dtype": str(array.data.dtype),
            "intent": int(array.intent),
        }
        for array in image.darrays
    )
    return GeometrySummary(
        path=str(path),
        size_bytes=path.stat().st_size,
        arrays=arrays,
    )


def _children_usage() -> tuple[float, float, int]:
    """Children usage."""

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sys.platform == "darwin":
        peak_rss = int(usage.ru_maxrss)
    else:
        peak_rss = int(usage.ru_maxrss * 1024)
    return usage.ru_utime + usage.ru_stime, peak_rss, int(usage.ru_maxrss)


def _read_version(binary: Path) -> str:
    """Read version."""

    completed = subprocess.run(
        [str(binary), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return f"unavailable (exit {completed.returncode}): {detail}"
    return (completed.stdout + completed.stderr).strip()


def run_one(
    binary: Path,
    surface_root: Path,
    template_root: Path,
    output_root: Path,
    hemisphere: str,
    steps: int,
) -> dict[str, Any]:
    """Run one."""

    surface_root = Path(surface_root).expanduser().resolve()
    template_root = Path(template_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{hemisphere}.sphere.reg.gii"
    if output.exists():
        raise FileExistsError(f"CAT output : {output}")

    white = surface_root / f"{hemisphere}.white.gii"
    sphere = surface_root / f"{hemisphere}.sphere.gii"
    template_white = template_root / f"{hemisphere}.white.gii"
    template_sphere = template_root / f"{hemisphere}.sphere.gii"
    command = [
        str(binary),
        "-steps",
        str(steps),
        "-avg",
        "-i",
        str(white),
        "-is",
        str(sphere),
        "-t",
        str(template_white),
        "-ts",
        str(template_sphere),
        "-ws",
        str(output),
    ]
    inputs = {
        "white": summarize_geometry(white),
        "sphere": summarize_geometry(sphere),
        "template_white": summarize_geometry(template_white),
        "template_sphere": summarize_geometry(template_sphere),
    }
    child_cpu_before, _, _ = _children_usage()
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    child_cpu_after, child_peak_rss, _ = _children_usage()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"CAT_WarpSurf {hemisphere} return {completed.returncode}: {detail[-4000:]}"
        )
    if not output.is_file():
        raise RuntimeError(f"CAT_WarpSurf did not produce output: {output}")
    return {
        "hemisphere": hemisphere,
        "steps": steps,
        "runs": 2,
        "average_solutions": True,
        "command": command,
        "inputs": {key: asdict(value) for key, value in inputs.items()},
        "output": asdict(summarize_geometry(output)),
        "timing_seconds": {
            "wall": wall_seconds,
            "child_cpu_delta": child_cpu_after - child_cpu_before,
            "child_peak_rss_cumulative": child_peak_rss,
        },
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args."""

    parser = argparse.ArgumentParser(description="CAT_WarpSurf CPU")
    parser.add_argument("--surface-root", required=True, type=Path)
    parser.add_argument(
        "--template-root",
        type=Path,
        required=True,
        help="SimNIBS/CAT surface , use",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        required=True,
        help="CAT_WarpSurf reference",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hemisphere", choices=("lh", "rh", "both"), default="both")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main."""

    args = parse_args(argv)
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"CAT_WarpSurf does not exist: {binary}")
    hemispheres = ("lh", "rh") if args.hemisphere == "both" else (args.hemisphere,)
    result = {
        "binary": str(binary),
        "version": _read_version(binary),
        "hemispheres": [
            run_one(
                binary,
                args.surface_root,
                args.template_root,
                args.output_root,
                hemisphere,
                args.steps,
            )
            for hemisphere in hemispheres
        ],
    }
    output_json = args.output_root.expanduser().resolve() / "cat_warpsurf_audit.json"
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
