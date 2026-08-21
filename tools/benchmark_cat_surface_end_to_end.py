#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""CAT-Surface GPU implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from cat_surface_gpu import (
    read_gifti_mesh,
    run_cat_surface_gpu_pipeline,
)


def _run_reference(
    binary: Path,
    source_surface: Path,
    source_sphere: Path,
    target_surface: Path,
    target_sphere: Path,
    output: Path,
    *,
    steps: int,
    runs: int,
    avg: bool,
    loop: int,
    code: int,
) -> float:
    """Run reference."""

    command = [
        str(binary),
        "-steps",
        str(steps),
        "-runs",
        str(runs),
        *(["-avg"] if avg else []),
        "-loop",
        str(loop),
        "-code",
        str(code),
        "-i",
        str(source_surface),
        "-is",
        str(source_sphere),
        "-t",
        str(target_surface),
        "-ts",
        str(target_sphere),
        "-ws",
        str(output),
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "CPU reference failed with exit code "
            f"{completed.returncode}\n{completed.stderr[-3000:]}"
        )
    if not output.is_file():
        raise RuntimeError(f"CPU reference did not produce output: {output}")
    return elapsed


def _compare_meshes(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    """Compare meshes."""

    reference = read_gifti_mesh(reference_path)
    candidate = read_gifti_mesh(candidate_path)
    if reference.faces.shape != candidate.faces.shape:
        raise ValueError(
            f"Face shape does not match : {reference.faces.shape} vs {candidate.faces.shape}"
        )
    if not np.array_equal(reference.faces, candidate.faces):
        raise ValueError("GIFTI face array does not match")
    if reference.vertices.shape != candidate.vertices.shape:
        raise ValueError(
            f"Vertex shape does not match : {reference.vertices.shape} vs {candidate.vertices.shape}"
        )
    difference = np.abs(
        reference.vertices.astype(np.float64) - candidate.vertices.astype(np.float64)
    )
    return {
        "faces_exact": True,
        "vertices_shape": list(reference.vertices.shape),
        "vertices_max_abs": float(difference.max()),
        "vertices_mean_abs": float(difference.mean()),
        "vertices_p99_abs": float(np.quantile(difference, 0.99)),
    }


def main() -> int:
    """Main."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cli", type=Path, required=True)
    parser.add_argument("--source-surface", type=Path, required=True)
    parser.add_argument("--source-sphere", type=Path, required=True)
    parser.add_argument("--target-surface", type=Path, required=True)
    parser.add_argument("--target-sphere", type=Path, required=True)
    parser.add_argument("--gpu-output", type=Path, required=True)
    parser.add_argument("--cpu-output", type=Path, required=True)
    parser.add_argument("--source-stencil", type=Path)
    parser.add_argument("--target-stencil", type=Path)
    parser.add_argument("--rotation-values", type=Path)
    parser.add_argument("--rotation-values-probe", type=Path)
    parser.add_argument(
        "--rotation-geometry-probe",
        type=Path,
        help="Upstream coarse geometry helper: CUDA hybrid feature backend use",
    )
    parser.add_argument(
        "--rotation-depth-probe",
        type=Path,
        help="Upstream raw depth-potential helper: cuda-official-depth use",
    )
    parser.add_argument(
        "--rotation-feature-backend",
        choices=(
            "auto",
            "cuda",
            "cuda-official-geometry",
            "cuda-official-depth",
            "official-cpu",
        ),
        default="auto",
        help="Rotation feature backend : hybrid backend explicit upstream helper",
    )
    parser.add_argument(
        "--rotation-feature-solver",
        choices=("colored-sor", "pcg"),
        default="colored-sor",
        help="GPU depth-potential : default use SOR",
    )
    parser.add_argument("--stencil-builder", type=Path, required=True)
    parser.add_argument("--rotated-stencil-builder", type=Path, required=True)
    parser.add_argument(
        "--stencil-threads",
        type=int,
        default=8,
        help="Each stencil helper CPU worker : default 8",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kernel", default="triton")
    parser.add_argument(
        "--dartel-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="DARTEL explicit : default FP64, FP32",
    )
    parser.add_argument(
        "--squaring-kernel",
        choices=("auto", "torch", "triton"),
        default="auto",
        help="DARTEL squaring kernel: default use Triton, torch A/B",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--avg", action="store_true", help="Run upstream -avg")
    parser.add_argument("--loop", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--nit", type=int, default=3)
    parser.add_argument("--its", type=int, default=3)
    parser.add_argument("--code", type=int, default=1)
    parser.add_argument(
        "--parallel-sides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA source/target surface : default",
    )
    parser.add_argument(
        "--optimized-dartel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DARTEL : default",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Solve shape CUDA Graph: default",
    )
    parser.add_argument(
        "--point-chunk",
        type=int,
        default=4096,
        help="Rotation cost : default 4096, 512",
    )
    parser.add_argument("--rotation-grid-size", type=int, default=128)
    parser.add_argument("--rotation-margin", type=int, default=1)
    parser.add_argument("--no-rotation-refine", action="store_true")
    parser.add_argument(
        "--max-vertex-error",
        type=float,
        default=1e-3,
        help="GPU and CPU vertex",
    )
    args = parser.parse_args()

    args.cpu_output.parent.mkdir(parents=True, exist_ok=True)
    args.gpu_output.parent.mkdir(parents=True, exist_ok=True)
    print("[1/2] CPU reference ...", flush=True)
    reference_seconds = _run_reference(
        args.reference_cli,
        args.source_surface,
        args.source_sphere,
        args.target_surface,
        args.target_sphere,
        args.cpu_output,
        steps=args.steps,
        runs=args.runs,
        avg=args.avg,
        loop=args.loop,
        code=args.code,
    )
    print("[2/2] GPU rotation + DARTEL + GIFTI ...", flush=True)
    result = run_cat_surface_gpu_pipeline(
        args.source_surface,
        args.source_sphere,
        args.target_surface,
        args.target_sphere,
        args.gpu_output,
        source_stencil_path=args.source_stencil,
        target_stencil_path=args.target_stencil,
        rotation_values_path=args.rotation_values,
        rotation_values_probe=args.rotation_values_probe,
        rotation_geometry_probe=args.rotation_geometry_probe,
        rotation_depth_probe=args.rotation_depth_probe,
        rotation_feature_backend=args.rotation_feature_backend,
        rotation_feature_solver=args.rotation_feature_solver,
        stencil_builder=args.stencil_builder,
        rotated_stencil_builder=args.rotated_stencil_builder,
        stencil_threads=args.stencil_threads,
        device=args.device,
        kernel=args.kernel,
        dartel_dtype=args.dartel_dtype,
        squaring_kernel=args.squaring_kernel,
        steps=args.steps,
        runs=args.runs,
        avg=args.avg,
        loop=args.loop,
        cycles=args.cycles,
        nit=args.nit,
        its=args.its,
        code=args.code,
        point_chunk=args.point_chunk,
        rotation_grid_size=args.rotation_grid_size,
        rotation_margin=args.rotation_margin,
        rotation_refine=not args.no_rotation_refine,
        parallel_sides=args.parallel_sides,
        optimized_dartel=args.optimized_dartel,
        cuda_graph=args.cuda_graph,
    )
    comparison = _compare_meshes(args.cpu_output, args.gpu_output)
    if comparison["vertices_max_abs"] > args.max_vertex_error:
        raise RuntimeError(
            "GPU vertex :"
            f"{comparison['vertices_max_abs']:.9g} > {args.max_vertex_error:.9g}"
        )
    gpu_seconds = float(result.timings["total_seconds"])
    payload = {
        "reference_backend": "CAT_SurfWarp_C_CPU",
        "gpu_backend": "torch_triton_cat_surface_gpu_pipeline",
        "device": str(torch.device(args.device)),
        "device_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "steps": args.steps,
        "runs": args.runs,
        "avg": args.avg,
        "reference_seconds": reference_seconds,
        "gpu_total_seconds": gpu_seconds,
        "end_to_end_speedup": reference_seconds / gpu_seconds,
        "rotation_angle": result.rotation_angle.tolist(),
        "rotation_cost": result.rotation_cost,
        "gpu_timings": result.timings,
        "output_comparison": comparison,
        "cpu_output": str(args.cpu_output),
        "gpu_output": str(args.gpu_output),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
