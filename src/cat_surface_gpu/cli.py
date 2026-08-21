"""Stable command-line interfaces for CAT-Surface GPU."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dartel_grid import resolve_device
from .gpu_pipeline import run_cat_surface_gpu_pipeline
from .resources import bundled_binary
from .rotation_pipeline import (
    RotationPipeline,
    read_rotation_points,
    read_rotation_values,
)
from .surface_stencil import SurfaceStencil
from .surf2sphere import run_cat_surf2sphere_gpu


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA at timing and result-transfer boundaries."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def surf2sphere_main(argv: list[str] | None = None) -> int:
    """Run the recommended CAT_Surf2Sphere CPU/GPU hybrid pipeline."""

    parser = argparse.ArgumentParser(
        description="Run strict CAT_Surf2Sphere area smoothing on CUDA."
    )
    parser.add_argument(
        "--reference-cli",
        type=Path,
        help="CPU reference executable (default: bundled CAT_Surf2Sphere)",
    )
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--output-surface", type=Path, required=True)
    parser.add_argument("--stop-at", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--kernel", choices=("torch", "triton"), default="triton")
    parser.add_argument("--preprocess-kernel", choices=("cpu", "triton"), default="cpu")
    parser.add_argument("--preprocess-block-size", type=int, default=131072)
    parser.add_argument("--areal-block-size", type=int)
    parser.add_argument(
        "--areal-schedule", choices=("ordered", "color"), default="ordered"
    )
    parser.add_argument("--areal-arithmetic", choices=("cat", "fp32"), default="cat")
    args = parser.parse_args(argv)
    reference_cli = args.reference_cli or bundled_binary("CAT_Surf2Sphere")
    result = run_cat_surf2sphere_gpu(
        args.input_surface,
        args.output_surface,
        reference_cli=reference_cli,
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
    print(
        json.dumps(
            {
                "backend": (
                    f"surf2sphere_pre_{args.preprocess_kernel}_"
                    f"{args.areal_schedule}_gauss_seidel_{args.kernel}_cuda"
                ),
                "device": args.device,
                "dtype": args.dtype,
                "areal_arithmetic": args.areal_arithmetic,
                "stop_at": args.stop_at,
                "reference_cli": str(reference_cli),
                "output_surface": str(result.output_path),
                "timings": result.timings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def surfwarp_main(argv: list[str] | None = None) -> int:
    """Run the recommended FP64 CAT_SurfWarp-compatible GPU pipeline."""

    parser = argparse.ArgumentParser(
        description="Run CAT_SurfWarp-compatible registration with FP64 DARTEL."
    )
    parser.add_argument("--source-surface", type=Path, required=True)
    parser.add_argument("--source-sphere", type=Path, required=True)
    parser.add_argument("--target-surface", type=Path, required=True)
    parser.add_argument("--target-sphere", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-stencil", type=Path)
    parser.add_argument("--target-stencil", type=Path)
    parser.add_argument("--rotation-values", type=Path)
    parser.add_argument("--rotation-values-probe", type=Path)
    parser.add_argument("--rotation-geometry-probe", type=Path)
    parser.add_argument(
        "--rotation-depth-probe",
        type=Path,
        help="Raw-depth helper (default: bundled cat_surface_rotation_depth)",
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
        default="cuda-official-depth",
    )
    parser.add_argument(
        "--rotation-feature-solver",
        choices=("colored-sor", "pcg"),
        default="colored-sor",
    )
    parser.add_argument(
        "--stencil-builder",
        type=Path,
        help="Stencil helper (default: bundled cat_surface_stencil_builder)",
    )
    parser.add_argument("--rotated-stencil-builder", type=Path)
    parser.add_argument("--stencil-threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kernel", default="triton")
    parser.add_argument(
        "--dartel-dtype", choices=("float64", "float32"), default="float64"
    )
    parser.add_argument(
        "--squaring-kernel", choices=("auto", "torch", "triton"), default="triton"
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--avg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--nit", type=int, default=3)
    parser.add_argument("--its", type=int, default=3)
    parser.add_argument("--code", type=int, default=1)
    parser.add_argument(
        "--parallel-sides", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--optimized-dartel", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--point-chunk", type=int, default=4096)
    parser.add_argument("--rotation-grid-size", type=int, default=128)
    parser.add_argument("--rotation-margin", type=int, default=1)
    parser.add_argument("--no-rotation-refine", action="store_true")
    parser.add_argument("--rotation-max-iter", type=int, default=500)
    parser.add_argument("--rotation-tol", type=float, default=1.0e-4)
    parser.add_argument("--rotation-simplex-step", type=float, default=0.1)
    args = parser.parse_args(argv)

    stencil_builder = args.stencil_builder or bundled_binary(
        "cat_surface_stencil_builder"
    )
    rotated_stencil_builder = args.rotated_stencil_builder or stencil_builder
    rotation_depth_probe = args.rotation_depth_probe
    if (
        rotation_depth_probe is None
        and args.rotation_feature_backend == "cuda-official-depth"
    ):
        rotation_depth_probe = bundled_binary("cat_surface_rotation_depth")
    result = run_cat_surface_gpu_pipeline(
        args.source_surface,
        args.source_sphere,
        args.target_surface,
        args.target_sphere,
        args.output,
        source_stencil_path=args.source_stencil,
        target_stencil_path=args.target_stencil,
        rotation_values_path=args.rotation_values,
        rotation_values_probe=args.rotation_values_probe,
        rotation_geometry_probe=args.rotation_geometry_probe,
        rotation_depth_probe=rotation_depth_probe,
        rotation_feature_backend=args.rotation_feature_backend,
        rotation_feature_solver=args.rotation_feature_solver,
        stencil_builder=stencil_builder,
        rotated_stencil_builder=rotated_stencil_builder,
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
        rotation_max_iter=args.rotation_max_iter,
        rotation_tol=args.rotation_tol,
        rotation_simplex_step=args.rotation_simplex_step,
        parallel_sides=args.parallel_sides,
        optimized_dartel=args.optimized_dartel,
        cuda_graph=args.cuda_graph,
    )
    device = torch.device(args.device)
    print(
        json.dumps(
            {
                "backend": "torch_triton_cat_surface_gpu_pipeline",
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "torch": torch.__version__,
                "kernel": args.kernel,
                "steps": args.steps,
                "runs": args.runs,
                "avg": args.avg,
                "source_shape": list(result.vertices.shape),
                "faces_shape": list(result.faces.shape),
                "rotation_angle": result.rotation_angle.tolist(),
                "rotation_cost": result.rotation_cost,
                "timings": result.timings,
                "output": str(result.output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def rotation_main(argv: list[str] | None = None) -> int:
    """Run CAT-compatible initial-rotation search on prepared inputs."""

    parser = argparse.ArgumentParser(description="Run CAT initial-rotation search.")
    parser.add_argument("--source-points", type=Path, required=True)
    parser.add_argument("--target-stencil", type=Path, required=True)
    parser.add_argument("--rotation-values", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--point-chunk", type=int, default=4096)
    parser.add_argument(
        "--candidate-table-dtype", choices=("int32", "int64"), default="int32"
    )
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--simplex-step", type=float, default=0.1)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    source_points = read_rotation_points(args.source_points)
    source_values, target_values = read_rotation_values(args.rotation_values)
    target_stencil = SurfaceStencil.from_file(args.target_stencil)
    table_dtype = {"int32": torch.int32, "int64": torch.int64}[
        args.candidate_table_dtype
    ]
    started = time.perf_counter()
    pipeline = RotationPipeline.from_stencil(
        target_stencil,
        device=device,
        grid_size=args.grid_size,
        margin=args.margin,
        candidate_table_dtype=table_dtype,
    )
    _synchronize(device)
    build_upload_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = pipeline.search(
        source_points,
        source_values,
        target_values,
        point_chunk=args.point_chunk,
        refine=not args.no_refine,
        max_iter=args.max_iter,
        tol=args.tol,
        simplex_step=args.simplex_step,
    )
    _synchronize(device)
    seed_costs = result.seed_costs.detach().cpu().numpy()
    print(
        json.dumps(
            {
                "feature_backend": "c_probe_rotation_values",
                "cost_backend": "torch",
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "source_points": int(source_points.shape[0]),
                "target_points": int(target_stencil.sphere_points.shape[0]),
                "build_upload_seconds": build_upload_seconds,
                "search_seconds": time.perf_counter() - started,
                "seed_cost_sha256": hashlib.sha256(
                    np.ascontiguousarray(seed_costs).tobytes()
                ).hexdigest(),
                "angle": result.angle.detach().cpu().tolist(),
                "cost": float(result.cost.detach().cpu()),
                "rotation_matrix": result.rotation_matrix.detach().cpu().tolist(),
                "iterations": result.iterations,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
