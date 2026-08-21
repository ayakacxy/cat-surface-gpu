#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Benchmark CAT_Surf2Sphere on identical CPU and GPU inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from cat_surface_gpu import (
    read_gifti_mesh,
    run_cat_surf2sphere_gpu,
)


def _read_elapsed(path: Path) -> float:
    """Read GNU time's elapsed wall-clock field."""

    for line in path.read_text(encoding="utf-8").splitlines():
        if "Elapsed (wall clock) time" not in line:
            continue
        value = line.split("): ", 1)[1].strip()
        parts = value.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60.0 + float(seconds)
        return float(value)
    raise ValueError(f"Elapsed wall-clock time was not found in: {path}")


def _run_reference(
    binary: Path,
    input_path: Path,
    output_path: Path,
    stop_at: int,
) -> float:
    """Run the CPU reference and return end-to-end wall time."""

    start = time.perf_counter()
    completed = subprocess.run(
        [str(binary), str(input_path), str(output_path), str(stop_at), "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "CAT_Surf2Sphere CPU reference failed:\n" + completed.stderr[-4000:]
        )
    return elapsed


def _compare(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    """Compare topology and vertex-coordinate error statistics."""

    reference = read_gifti_mesh(reference_path)
    candidate = read_gifti_mesh(candidate_path)
    if not np.array_equal(reference.faces, candidate.faces):
        raise RuntimeError("GPU and CPU reference face arrays do not match")
    difference = np.abs(
        reference.vertices.astype(np.float64) - candidate.vertices.astype(np.float64)
    )
    reference_radius = np.linalg.norm(reference.vertices.astype(np.float64), axis=1)
    candidate_radius = np.linalg.norm(candidate.vertices.astype(np.float64), axis=1)
    return {
        "faces_exact": True,
        "vertices_shape": list(reference.vertices.shape),
        "vertices_max_abs": float(difference.max()),
        "vertices_mean_abs": float(difference.mean()),
        "vertices_p99_abs": float(np.quantile(difference, 0.99)),
        "radius_max_abs": float(np.abs(reference_radius - candidate_radius).max()),
        "reference_radius_min": float(reference_radius.min()),
        "reference_radius_max": float(reference_radius.max()),
        "candidate_radius_min": float(candidate_radius.min()),
        "candidate_radius_max": float(candidate_radius.max()),
    }


def main() -> int:
    """Run the paired benchmark and enforce numerical thresholds."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cli", type=Path, required=True)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--gpu-output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-time-file", type=Path)
    parser.add_argument(
        "--reuse-reference",
        action="store_true",
        help="Reuse an existing reference output instead of running the CPU executable",
    )
    parser.add_argument("--stop-at", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--kernel", choices=("torch", "triton"), default="triton")
    parser.add_argument(
        "--preprocess-kernel",
        choices=("cpu", "triton"),
        default="cpu",
        help="Preprocessing backend: upstream CPU reference or CUDA Triton",
    )
    parser.add_argument(
        "--preprocess-block-size",
        type=int,
        default=131072,
        help="Maximum vertices per preprocessing launch",
    )
    parser.add_argument(
        "--areal-block-size",
        type=int,
        help="Maximum vertices per area-smoothing launch",
    )
    parser.add_argument(
        "--areal-schedule",
        choices=("ordered", "color"),
        default="ordered",
        help="Ordered preserves upstream dependencies; color is experimental",
    )
    parser.add_argument(
        "--areal-arithmetic",
        choices=("cat", "fp32"),
        default="cat",
        help="CAT preserves upstream float/double boundaries; fp32 is experimental",
    )
    parser.add_argument("--max-vertex-error", type=float, default=5.0e-3)
    parser.add_argument("--max-mean-error", type=float, default=1.0e-3)
    parser.add_argument("--max-p99-error", type=float, default=3.0e-3)
    args = parser.parse_args()

    if args.reference_time_file is not None and not args.reuse_reference:
        parser.error("--reference-time-file requires --reuse-reference")

    reference_output = args.reference_output or args.gpu_output.with_name(
        args.gpu_output.stem + ".reference.gii"
    )
    if not args.reuse_reference:
        reference_seconds = _run_reference(
            args.reference_cli,
            args.input_surface,
            reference_output,
            args.stop_at,
        )
    else:
        if not reference_output.is_file():
            raise FileNotFoundError(
                f"--reuse-reference requires an existing file: {reference_output}"
            )
        if args.reference_time_file is not None:
            reference_seconds = _read_elapsed(args.reference_time_file)
        else:
            reference_seconds = None

    result = run_cat_surf2sphere_gpu(
        args.input_surface,
        args.gpu_output,
        reference_cli=args.reference_cli,
        stop_at=args.stop_at,
        device=args.device,
        dtype=args.dtype,
        kernel=args.kernel,
        preprocess_kernel=args.preprocess_kernel,
        preprocess_block_size=args.preprocess_block_size,
        areal_schedule=args.areal_schedule,
        areal_arithmetic=args.areal_arithmetic,
        areal_block_size=args.areal_block_size,
    )
    comparison = _compare(reference_output, args.gpu_output)
    passed = (
        comparison["vertices_max_abs"] <= args.max_vertex_error
        and comparison["vertices_mean_abs"] <= args.max_mean_error
        and comparison["vertices_p99_abs"] <= args.max_p99_error
    )
    payload = {
        "reference_backend": "CAT_Surf2Sphere_latest_C_CPU",
        "gpu_backend": (
            f"surf2sphere_pre_{args.preprocess_kernel}_"
            f"{args.areal_schedule}_gauss_seidel_{args.kernel}_cuda"
        ),
        "device": args.device,
        "dtype": args.dtype,
        "areal_arithmetic": args.areal_arithmetic,
        "stop_at": args.stop_at,
        "reference_seconds": reference_seconds,
        "gpu_total_seconds": result.timings["total_seconds"],
        "speedup": (
            None
            if reference_seconds is None
            else reference_seconds / result.timings["total_seconds"]
        ),
        "timings": result.timings,
        "comparison": comparison,
        "tolerance": {
            "max_vertex_error": args.max_vertex_error,
            "max_mean_error": args.max_mean_error,
            "max_p99_error": args.max_p99_error,
        },
        "passed": bool(passed),
        "reference_output": str(reference_output),
        "gpu_output": str(args.gpu_output),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
