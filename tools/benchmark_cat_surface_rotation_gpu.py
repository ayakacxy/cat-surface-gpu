#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""测量 CAT 初始旋转候选在单一设备上的批量定位代价。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from fast_charm.cat_surface import (
    RotationGridIndex,
    SurfaceStencil,
    official_seed_grid_angles,
)
from fast_charm.cat_surface.dartel_grid import resolve_device


def _synchronize(device: torch.device) -> None:
    """在 CUDA 计时边界显式等待设备完成。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read_rotation_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 rotation-values probe 写出的 source/target 双精度曲率。"""

    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if header.size != 2:
            raise ValueError("旋转曲率文件头不完整")
        source = np.fromfile(stream, dtype="<f8", count=int(header[0]))
        target = np.fromfile(stream, dtype="<f8", count=int(header[1]))
    if source.size != int(header[0]) or target.size != int(header[1]):
        raise ValueError("旋转曲率文件长度不匹配")
    return source, target


def _read_points(path: Path) -> np.ndarray:
    """读取 C map probe 写出的 float32 粗网格点。"""

    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=1)
        if header.size != 1:
            raise ValueError("source points 文件头不完整")
        points = np.fromfile(stream, dtype="<f4", count=int(header[0]) * 3)
    if points.size != int(header[0]) * 3:
        raise ValueError("source points 文件长度不匹配")
    return points.reshape(int(header[0]), 3)


def _make_angles(count: int, seed: int) -> np.ndarray:
    """生成固定首项加确定性伪随机项的候选角度，便于重复 A/B。"""

    if count < 1:
        raise ValueError("candidates 必须为正数")
    fixed = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.2, -0.1, 0.15),
            (-0.4, 0.3, -0.2),
            (0.6, -0.5, 0.4),
            (-0.8, 0.7, -0.6),
            (0.95, -0.85, 0.75),
            (0.1, 0.2, -0.3),
            (-0.2, -0.3, 0.4),
        ),
        dtype=np.float64,
    )
    if count <= fixed.shape[0]:
        return fixed[:count].copy()
    generator = np.random.default_rng(seed)
    extra = generator.uniform(-1.0, 1.0, size=(count - fixed.shape[0], 3))
    return np.concatenate((fixed, extra), axis=0)


def main() -> int:
    """运行 GPU/CPU 候选批处理 benchmark 并输出 JSON。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-points", type=Path, required=True)
    parser.add_argument("--target-stencil", type=Path, required=True)
    parser.add_argument("--rotation-values", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=46)
    parser.add_argument(
        "--angle-set",
        choices=("custom", "official-seed"),
        default="custom",
        help="候选角度集合；official-seed 保持 CAT C 的 46 候选顺序",
    )
    parser.add_argument("--seed", type=int, default=20260820)
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
        help="设备端候选表的整数类型，用于同输入 A/B",
    )
    parser.add_argument(
        "--refine-nelder-mead",
        action="store_true",
        help="在官方 seed 后执行 C 顺序 Nelder-Mead",
    )
    parser.add_argument(
        "--reuse-cost-inputs",
        action="store_true",
        help="把旋转 cost 输入固定在设备，测量常驻输入优化",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.point_chunk < 1 or args.warmup < 0 or args.repeat < 1:
        raise ValueError("point-chunk、warmup 和 repeat 参数不合法")

    device = resolve_device(args.device)
    target_stencil = SurfaceStencil.from_file(args.target_stencil)
    source_points_np = _read_points(args.source_points)
    source_values, target_values = _read_rotation_values(args.rotation_values)
    if source_values.size != source_points_np.shape[0]:
        raise ValueError("source 曲率长度与 source points 不一致")
    if target_values.size != target_stencil.sphere_points.shape[0]:
        raise ValueError("target 曲率长度与 target stencil 不一致")

    build_start = time.perf_counter()
    index_cpu = RotationGridIndex.from_stencil(
        target_stencil, grid_size=args.grid_size, margin=args.margin
    )
    build_seconds = time.perf_counter() - build_start

    upload_start = time.perf_counter()
    candidate_table_dtype = {
        "int32": torch.int32,
        "int64": torch.int64,
    }[args.candidate_table_dtype]
    index = index_cpu.to(device, candidate_table_dtype=candidate_table_dtype)
    source_points = torch.as_tensor(
        source_points_np, dtype=torch.float32, device=device
    ).contiguous()
    source_tensor = torch.as_tensor(
        source_values, dtype=torch.float64, device=device
    ).contiguous()
    target_tensor = torch.as_tensor(
        target_values, dtype=torch.float64, device=device
    ).contiguous()
    prepared_inputs = (
        index.prepare_cost_inputs(source_points, source_tensor, target_tensor)
        if args.reuse_cost_inputs
        else None
    )
    if args.angle_set == "official-seed":
        angles_np = official_seed_grid_angles()
        if args.candidates != angles_np.shape[0]:
            raise ValueError("official-seed 要求 candidates=46")
    else:
        angles_np = _make_angles(args.candidates, args.seed)
    angles = torch.as_tensor(
        angles_np, dtype=torch.float64, device=device
    ).contiguous()
    _synchronize(device)
    upload_seconds = time.perf_counter() - upload_start

    def run_once() -> torch.Tensor:
        """执行一次完整候选批处理。"""

        if args.angle_set == "official-seed":
            if prepared_inputs is not None:
                return index.coarse_seed_search_prepared(
                    prepared_inputs, point_chunk=args.point_chunk
                )[2]
            return index.coarse_seed_search(
                source_points,
                source_tensor,
                target_tensor,
                point_chunk=args.point_chunk,
            )[2]
        if prepared_inputs is not None:
            return index.cost_batch_prepared(
                prepared_inputs, angles, point_chunk=args.point_chunk
            )
        return index.cost_batch(
            source_points,
            source_tensor,
            target_tensor,
            angles,
            point_chunk=args.point_chunk,
        )

    for _ in range(args.warmup):
        run_once()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    costs = None
    for _ in range(args.repeat):
        _synchronize(device)
        start = time.perf_counter()
        costs = run_once()
        _synchronize(device)
        timings.append(time.perf_counter() - start)
    assert costs is not None
    costs_cpu = costs.detach().cpu().numpy()
    best_index = int(np.argmin(costs_cpu))
    refinement = None
    if args.refine_nelder_mead:
        _synchronize(device)
        seed_start = time.perf_counter()
        if prepared_inputs is not None:
            seed_angle, seed_cost, _ = index.coarse_seed_search_prepared(
                prepared_inputs, point_chunk=args.point_chunk
            )
        else:
            seed_angle, seed_cost, _ = index.coarse_seed_search(
                source_points,
                source_tensor,
                target_tensor,
                point_chunk=args.point_chunk,
            )
        _synchronize(device)
        seed_seconds = time.perf_counter() - seed_start
        _synchronize(device)
        refine_start = time.perf_counter()
        if prepared_inputs is not None:
            refined_angle, refined_cost, iterations = (
                index.refine_nelder_mead_prepared(
                    prepared_inputs,
                    seed_angle,
                    point_chunk=args.point_chunk,
                )
            )
        else:
            refined_angle, refined_cost, iterations = index.refine_nelder_mead(
                source_points,
                source_tensor,
                target_tensor,
                seed_angle,
                point_chunk=args.point_chunk,
            )
        _synchronize(device)
        refine_seconds = time.perf_counter() - refine_start
        refinement = {
            "seed_seconds": seed_seconds,
            "seed_angle": seed_angle.detach().cpu().tolist(),
            "seed_cost": float(seed_cost.detach().cpu()),
            "refine_seconds": refine_seconds,
            "refined_angle": refined_angle.detach().cpu().tolist(),
            "refined_cost": float(refined_cost.detach().cpu()),
            "iterations": int(iterations),
        }

    payload = {
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
        "angle_set": args.angle_set,
        "reuse_cost_inputs": args.reuse_cost_inputs,
        "grid_size": args.grid_size,
        "margin": args.margin,
        "grid_max_candidates": index_cpu.max_candidates,
        "candidate_table_dtype": str(index.candidate_table.dtype),
        "candidate_table_dense_bytes": int(
            0
            if index_cpu.candidate_table is None
            else index_cpu.candidate_table.size * np.dtype(np.int32).itemsize
        ),
        "candidate_offsets_bytes": int(
            0
            if index.candidate_offsets is None
            else index.candidate_offsets.numel()
            * index.candidate_offsets.element_size()
        ),
        "candidate_faces_bytes": int(
            0
            if index.candidate_faces is None
            else index.candidate_faces.numel()
            * index.candidate_faces.element_size()
        ),
        "candidate_storage": (
            "compressed-csr"
            if index.candidate_offsets is not None
            else "dense-table"
        ),
        "candidate_table_bytes": int(
            (
                0
                if index.candidate_offsets is None
                else index.candidate_offsets.numel()
                * index.candidate_offsets.element_size()
            )
            + (
                0
                if index.candidate_faces is None
                else index.candidate_faces.numel()
                * index.candidate_faces.element_size()
            )
            + index.candidate_table.numel() * index.candidate_table.element_size()
        ),
        "candidates": args.candidates,
        "point_chunk": args.point_chunk,
        "build_seconds_cpu": build_seconds,
        "upload_seconds": upload_seconds,
        "batch_seconds": timings,
        "batch_median_seconds": float(np.median(timings)),
        "seconds_per_candidate": float(np.median(timings) / args.candidates),
        "cost_first": float(costs_cpu[0]),
        "cost_last": float(costs_cpu[-1]),
        "best_index": best_index,
        "best_angle": angles_np[best_index].tolist(),
        "best_cost": float(costs_cpu[best_index]),
        "refinement": refinement,
        "cost_sha256": hashlib.sha256(
            np.ascontiguousarray(costs_cpu).tobytes()
        ).hexdigest(),
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "large_arrays_device": str(index.target_points.device),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
