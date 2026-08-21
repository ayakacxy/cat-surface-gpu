#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""CAT-Surface GPU implementation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from cat_surface_gpu import (
    SurfaceStencil,
    prepare_dartel_cuda_graph,
    solve_dartel_from_surfaces,
)
from cat_surface_gpu.dartel_grid import resolve_device


def _synchronize(device: torch.device) -> None:
    """Synchronize."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_points(path: Path) -> np.ndarray:
    """Load points."""

    values = np.asarray(nib.load(str(path)).darrays[0].data, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"Surface must have shape [points, 3], got {values.shape}: {path}"
        )
    return values


def _solve(
    source: torch.Tensor,
    target: torch.Tensor,
    source_stencil,
    target_stencil,
    args: argparse.Namespace,
    device: torch.device,
    cuda_graph: bool = False,
):
    """Solve."""

    return solve_dartel_from_surfaces(
        source,
        target,
        source_stencil,
        target_stencil,
        fwhm=tuple(args.fwhm),
        curvtypes=tuple(args.curvtypes),
        n_steps=args.steps,
        loop=args.loop,
        cycles=args.cycles,
        nit=args.nit,
        its=args.its,
        code=args.code,
        kernel=args.kernel,
        device=device,
        dtype=torch.float64,
        parallel_sides=args.parallel_sides,
        optimized=args.optimized_dartel,
        cuda_graph=cuda_graph,
    )


def main() -> None:
    """Main."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-surface", type=Path, required=True)
    parser.add_argument("--target-surface", type=Path, required=True)
    parser.add_argument("--source-stencil", type=Path, required=True)
    parser.add_argument("--target-stencil", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kernel", default="auto")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--curvtypes", type=int, nargs="+", default=[5, 5, 2])
    parser.add_argument(
        "--fwhm", type=float, nargs="+", default=[5.0, 5.0 / 3.0, 5.0 / 9.0]
    )
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
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if len(args.curvtypes) < args.steps or len(args.fwhm) < args.steps:
        raise ValueError("curvtypes and fwhm must contain at least 'steps' entries")

    device = resolve_device(args.device)
    source_np = _load_points(args.source_surface)
    target_np = _load_points(args.target_surface)
    source_stencil_cpu = SurfaceStencil.from_file(args.source_stencil)
    target_stencil_cpu = SurfaceStencil.from_file(args.target_stencil)

    upload_start = time.perf_counter()
    source_stencil = source_stencil_cpu.to(device, geometry_dtype=torch.float32)
    target_stencil = target_stencil_cpu.to(device, geometry_dtype=torch.float32)
    source = torch.as_tensor(source_np, dtype=torch.float32, device=device).contiguous()
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device).contiguous()
    _synchronize(device)
    upload_seconds = time.perf_counter() - upload_start

    warmup_result = None
    for _ in range(args.warmup):
        warmup_result = _solve(
            source,
            target,
            source_stencil,
            target_stencil,
            args,
            device,
            cuda_graph=False,
        )
    _synchronize(device)
    capture_seconds = 0.0
    graph_enabled = bool(
        args.cuda_graph
        and device.type == "cuda"
        and args.repeat >= 2
        and warmup_result is not None
    )
    if graph_enabled:
        capture_seconds = prepare_dartel_cuda_graph(
            warmup_result.source_maps,
            warmup_result.target_maps,
            loop=args.loop,
            lmreg=1e-3,
            cycles=args.cycles,
            nit=args.nit,
            its=args.its,
            code=args.code,
            kernel=args.kernel,
            device=device,
            dtype=torch.float64,
            optimized=args.optimized_dartel,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    result = None
    for _ in range(args.repeat):
        _synchronize(device)
        start = time.perf_counter()
        result = _solve(
            source,
            target,
            source_stencil,
            target_stencil,
            args,
            device,
            cuda_graph=graph_enabled,
        )
        _synchronize(device)
        timings.append(time.perf_counter() - start)

    payload = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch": torch.__version__,
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "steps": args.steps,
        "curvtypes": args.curvtypes[: args.steps],
        "fwhm": args.fwhm[: args.steps],
        "upload_seconds": upload_seconds,
        "solve_seconds": timings,
        "solve_median_seconds": float(np.median(timings)),
        "cuda_graph_enabled": graph_enabled,
        "cuda_graph_capture_seconds": capture_seconds,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "flow_device": str(result.flow.device),
        "source_maps_device": str(result.source_maps.device),
        "target_maps_device": str(result.target_maps.device),
        "metrics_shape": list(result.metrics.shape),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
