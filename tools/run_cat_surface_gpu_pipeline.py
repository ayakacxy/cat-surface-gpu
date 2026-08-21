#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""运行真实 GIFTI 输入的 CAT 曲面 GPU 旋转和 DARTEL 闭环。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cat_surface_gpu import run_cat_surface_gpu_pipeline


def main() -> int:
    """解析参数、执行 GPU pipeline 并输出机器可读计时。"""

    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--rotation-depth-probe", type=Path)
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
        help="初始旋转特征后端；auto 在未给 helper 时使用 GPU",
    )
    parser.add_argument(
        "--rotation-feature-solver",
        choices=("colored-sor", "pcg"),
        default="colored-sor",
        help="GPU depth-potential 求解器；默认使用图着色并行 SOR",
    )
    parser.add_argument("--stencil-builder", type=Path, required=True)
    parser.add_argument("--rotated-stencil-builder", type=Path, required=True)
    parser.add_argument(
        "--stencil-threads",
        type=int,
        default=8,
        help="每个 stencil helper 的 CPU worker 数；默认 8",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kernel", default="triton")
    parser.add_argument(
        "--dartel-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="DARTEL 显式计算精度；默认 FP64，FP32 为容差实验路径",
    )
    parser.add_argument(
        "--squaring-kernel",
        choices=("auto", "torch", "triton"),
        default="auto",
        help="DARTEL squaring kernel；默认使用 Triton，torch 供 A/B 对照",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--avg", action="store_true", help="执行官方 -avg 双极点平均")
    parser.add_argument("--loop", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--nit", type=int, default=3)
    parser.add_argument("--its", type=int, default=3)
    parser.add_argument("--code", type=int, default=1)
    parser.add_argument(
        "--parallel-sides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 下重叠 source/target 曲面阶段；默认开启",
    )
    parser.add_argument(
        "--optimized-dartel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 DARTEL 多通道共享采样；默认开启",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="重复 solve 时启用固定 shape 的 CUDA Graph；默认开启",
    )
    parser.add_argument(
        "--point-chunk",
        type=int,
        default=4096,
        help="旋转 cost 的点块大小；默认 4096，显存不足时可降为 512",
    )
    parser.add_argument("--rotation-grid-size", type=int, default=128)
    parser.add_argument("--rotation-margin", type=int, default=1)
    parser.add_argument("--no-rotation-refine", action="store_true")
    parser.add_argument("--rotation-max-iter", type=int, default=500)
    parser.add_argument("--rotation-tol", type=float, default=1.0e-4)
    parser.add_argument("--rotation-simplex-step", type=float, default=0.1)
    args = parser.parse_args()

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
        rotation_max_iter=args.rotation_max_iter,
        rotation_tol=args.rotation_tol,
        rotation_simplex_step=args.rotation_simplex_step,
        parallel_sides=args.parallel_sides,
        optimized_dartel=args.optimized_dartel,
        cuda_graph=args.cuda_graph,
    )
    device = torch.device(args.device)
    payload = {
        "backend": "torch_triton_cat_surface_gpu_pipeline",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
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
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
