#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""运行已接入的 CAT 初始旋转 seed/Nelder–Mead pipeline。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from fast_charm.cat_surface import (
    RotationPipeline,
    SurfaceStencil,
    read_rotation_points,
    read_rotation_values,
)
from fast_charm.cat_surface.dartel_grid import resolve_device


def _synchronize(device: torch.device) -> None:
    """在 GPU 计时和结果读取边界等待设备完成。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    """读取 C probe 产物并输出机器可读的旋转结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-points", type=Path, required=True)
    parser.add_argument("--target-stencil", type=Path, required=True)
    parser.add_argument("--rotation-values", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument(
        "--point-chunk",
        type=int,
        default=4096,
        help="旋转 cost 的点块大小；默认 4096，显存不足时可降为 512",
    )
    parser.add_argument(
        "--candidate-table-dtype",
        choices=("int32", "int64"),
        default="int32",
    )
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--simplex-step", type=float, default=0.1)
    args = parser.parse_args()

    device = resolve_device(args.device)
    source_points = read_rotation_points(args.source_points)
    source_values, target_values = read_rotation_values(args.rotation_values)
    target_stencil = SurfaceStencil.from_file(args.target_stencil)
    table_dtype = {
        "int32": torch.int32,
        "int64": torch.int64,
    }[args.candidate_table_dtype]

    pipeline_start = time.perf_counter()
    pipeline = RotationPipeline.from_stencil(
        target_stencil,
        device=device,
        grid_size=args.grid_size,
        margin=args.margin,
        candidate_table_dtype=table_dtype,
    )
    _synchronize(device)
    build_upload_seconds = time.perf_counter() - pipeline_start

    search_start = time.perf_counter()
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
    search_seconds = time.perf_counter() - search_start
    seed_costs = result.seed_costs.detach().cpu().numpy()

    payload = {
        "feature_backend": "c_probe_rotation_values",
        "cost_backend": "torch",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
        "torch": torch.__version__,
        "source_points": int(source_points.shape[0]),
        "target_points": int(target_stencil.sphere_points.shape[0]),
        "target_triangles": int(target_stencil.faces.shape[0]),
        "grid_size": args.grid_size,
        "margin": args.margin,
        "grid_max_candidates": pipeline.index_cpu.max_candidates,
        "candidate_table_dtype": str(pipeline.index.candidate_table.dtype),
        "candidate_table_dense_bytes": int(
            0
            if pipeline.index_cpu.candidate_table is None
            else pipeline.index_cpu.candidate_table.size * np.dtype(np.int32).itemsize
        ),
        "candidate_offsets_bytes": int(
            0
            if pipeline.index.candidate_offsets is None
            else pipeline.index.candidate_offsets.numel()
            * pipeline.index.candidate_offsets.element_size()
        ),
        "candidate_faces_bytes": int(
            0
            if pipeline.index.candidate_faces is None
            else pipeline.index.candidate_faces.numel()
            * pipeline.index.candidate_faces.element_size()
        ),
        "candidate_storage": (
            "compressed-csr"
            if pipeline.index.candidate_offsets is not None
            else "dense-table"
        ),
        "candidate_table_bytes": int(
            (
                0
                if pipeline.index.candidate_offsets is None
                else pipeline.index.candidate_offsets.numel()
                * pipeline.index.candidate_offsets.element_size()
            )
            + (
                0
                if pipeline.index.candidate_faces is None
                else pipeline.index.candidate_faces.numel()
                * pipeline.index.candidate_faces.element_size()
            )
            + pipeline.index.candidate_table.numel()
            * pipeline.index.candidate_table.element_size()
        ),
        "point_chunk": args.point_chunk,
        "build_upload_seconds": build_upload_seconds,
        "search_seconds": search_seconds,
        "refined": not args.no_refine,
        "seed_angle": result.seed_angle.detach().cpu().tolist(),
        "seed_cost": float(result.seed_cost.detach().cpu()),
        "seed_cost_sha256": hashlib.sha256(
            np.ascontiguousarray(seed_costs).tobytes()
        ).hexdigest(),
        "angle": result.angle.detach().cpu().tolist(),
        "cost": float(result.cost.detach().cpu()),
        "rotation_matrix": result.rotation_matrix.detach().cpu().tolist(),
        "iterations": result.iterations,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
