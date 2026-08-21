#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""对同一真实 GIFTI 输入运行 CPU reference 与 GPU pipeline 端到端 A/B。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from fast_charm.cat_surface import (
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
    """运行官方 C reference 并返回包含 I/O 的墙钟时间。"""

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
            "CPU reference 失败，返回码 "
            f"{completed.returncode}\n{completed.stderr[-3000:]}"
        )
    if not output.is_file():
        raise RuntimeError(f"CPU reference 没有生成输出：{output}")
    return elapsed


def _compare_meshes(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    """比较最终 GIFTI 的拓扑和顶点误差。"""

    reference = read_gifti_mesh(reference_path)
    candidate = read_gifti_mesh(candidate_path)
    if reference.faces.shape != candidate.faces.shape:
        raise ValueError(
            f"面片形状不一致：{reference.faces.shape} vs {candidate.faces.shape}"
        )
    if not np.array_equal(reference.faces, candidate.faces):
        raise ValueError("最终 GIFTI 面片数组不一致")
    if reference.vertices.shape != candidate.vertices.shape:
        raise ValueError(
            f"顶点形状不一致：{reference.vertices.shape} vs {candidate.vertices.shape}"
        )
    difference = np.abs(
        reference.vertices.astype(np.float64)
        - candidate.vertices.astype(np.float64)
    )
    return {
        "faces_exact": True,
        "vertices_shape": list(reference.vertices.shape),
        "vertices_max_abs": float(difference.max()),
        "vertices_mean_abs": float(difference.mean()),
        "vertices_p99_abs": float(np.quantile(difference, 0.99)),
    }


def main() -> int:
    """运行同输入 CPU/GPU A/B 并输出端到端收益。"""

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
        help="官方 coarse 几何导出 helper；供 CUDA hybrid feature 后端使用",
    )
    parser.add_argument(
        "--rotation-depth-probe",
        type=Path,
        help="官方 raw depth-potential 导出 helper；供 cuda-official-depth 使用",
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
        help="初始旋转特征后端；hybrid 后端需要显式提供对应官方 helper",
    )
    parser.add_argument(
        "--rotation-feature-solver",
        choices=("colored-sor", "pcg"),
        default="colored-sor",
        help="GPU depth-potential 求解器；默认使用图着色并行 SOR",
    )
    parser.add_argument("--stencil-builder", type=Path, required=True)
    parser.add_argument("--rotated-stencil-builder", type=Path, required=True)
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
        help="至少三次正式 solve 时启用固定 shape 的 CUDA Graph；默认开启",
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
    parser.add_argument(
        "--max-vertex-error",
        type=float,
        default=1e-3,
        help="允许 GPU 与 CPU 最终顶点的最大绝对误差",
    )
    args = parser.parse_args()

    args.cpu_output.parent.mkdir(parents=True, exist_ok=True)
    args.gpu_output.parent.mkdir(parents=True, exist_ok=True)
    print("[1/2] 运行 CPU reference ...", flush=True)
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
    print("[2/2] 运行 GPU rotation + DARTEL + GIFTI ...", flush=True)
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
            "GPU 最终顶点误差超出合同："
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
