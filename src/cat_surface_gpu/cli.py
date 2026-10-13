"""独立 Python/CuPy 球面生成和配准命令行。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import __version__
from .io import GiftiSurface, load_gifti_surface, save_gifti_surface
from .parameters import Surf2SphereParameters, SurfWarpParameters
from .surf2sphere import surf_to_sphere_cupy
from .surfwarp import surfwarp_cupy
from .topology import build_surface_topology


def _common_options(parser: argparse.ArgumentParser) -> None:
    """添加两个入口共用的运行选项。"""

    parser.add_argument(
        "--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Numba CPU threads (1-8)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages"
    )
    parser.add_argument("--version", action="version", version=__version__)


def _cuda_device(name: str):
    """仅选择真实 CUDA 设备；依赖或设备缺失时明确报错。"""

    match = re.fullmatch(r"cuda(?::(\d+))?", name)
    if match is None:
        raise ValueError("完整球面流程需要 CUDA；--device 应为 cuda 或 cuda:N")
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CuPy 未安装；请安装与本机 CUDA 匹配的 CuPy 包") from error
    device_id = int(match.group(1) or 0)
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise RuntimeError(f"CUDA 无法初始化：{error}") from error
    if device_id >= count:
        raise ValueError(f"CUDA 设备 {device_id} 不存在；检测到 {count} 个设备")
    return cp.cuda.Device(device_id)


def _setup(args, input_paths: tuple[Path, ...], output: Path):
    """检查路径、线程数，并创建进度输出函数。"""

    from numba import config, set_num_threads

    if not 1 <= args.threads <= min(8, config.NUMBA_NUM_THREADS):
        raise ValueError(
            f"--threads 必须在 1 到 {min(8, config.NUMBA_NUM_THREADS)} 之间"
        )
    set_num_threads(args.threads)
    for path in input_paths:
        if not path.is_file():
            raise ValueError(f"输入文件不存在：{path}")
        if path.resolve() == output.resolve():
            raise ValueError("输出文件不能覆盖输入文件")
    if output.suffix != ".gii":
        raise ValueError("输出文件必须使用 .gii 扩展名")

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"[cat-surface-gpu] {message}", file=sys.stderr, flush=True)

    return progress


def _paired_surfaces(surface_path: Path, sphere_path: Path, name: str):
    """检查解剖表面及对应球面的顶点和三角形编号。"""

    surface = load_gifti_surface(surface_path)
    sphere = load_gifti_surface(sphere_path)
    if surface.vertices.shape != sphere.vertices.shape:
        raise ValueError(f"{name} 表面和球面的顶点数量不一致")
    if not np.array_equal(surface.faces, sphere.faces):
        raise ValueError(f"{name} 表面和球面的三角形编号或顺序不一致")
    build_surface_topology(surface.faces, len(surface.vertices))
    return surface, sphere


def surf2sphere_main(argv: list[str] | None = None) -> int:
    """从 GIFTI 解剖表面生成球面；默认执行全部阶段。"""

    parser = argparse.ArgumentParser(
        description="Generate a GIFTI sphere with Python/CuPy."
    )
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--output-surface", type=Path, required=True)
    parser.add_argument(
        "--stop-at",
        type=int,
        default=10,
        help="Stop stage; values below 5 produce an intermediate surface",
    )
    _common_options(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        parameters = Surf2SphereParameters(stop_at=args.stop_at)
        progress = _setup(args, (args.input_surface,), args.output_surface)
        progress("Reading surface and checking topology")
        surface = load_gifti_surface(args.input_surface)
        topology = build_surface_topology(surface.faces, len(surface.vertices))
        with _cuda_device(args.device):
            vertices, timings = surf_to_sphere_cupy(
                surface.vertices,
                surface.faces,
                parameters,
                topology,
                progress=progress,
            )
        save_gifti_surface(
            GiftiSurface(vertices, surface.faces, surface.reference),
            args.output_surface,
        )
        progress(f"Saved {args.output_surface}")
        print(
            json.dumps(
                {
                    "version": __version__,
                    "backend": "python-cupy",
                    "device": args.device,
                    "output_surface": str(args.output_surface),
                    "parameters": asdict(parameters),
                    "timings": timings,
                    "wall_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


def surfwarp_main(argv: list[str] | None = None) -> int:
    """将源球面配准到目标球面，并保存源顶点顺序。"""

    parser = argparse.ArgumentParser(
        description="Register GIFTI surfaces with Python/CuPy DARTEL."
    )
    for name in (
        "source-surface",
        "source-sphere",
        "target-surface",
        "target-sphere",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    defaults = SurfWarpParameters()
    parser.add_argument("--steps", type=int, choices=(1, 2), default=defaults.steps)
    parser.add_argument("--runs", type=int, default=defaults.runs)
    parser.add_argument(
        "--rotate", action=argparse.BooleanOptionalAction, default=defaults.rotate
    )
    parser.add_argument(
        "--average",
        "--avg",
        action=argparse.BooleanOptionalAction,
        default=defaults.average,
    )
    parser.add_argument("--loop", type=int, default=defaults.loop)
    parser.add_argument(
        "--map-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=defaults.map_size,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=defaults.iterations,
        help="DARTEL updates per schedule level",
    )
    parser.add_argument("--mu", type=float, default=defaults.mu)
    parser.add_argument("--mu-rate", type=float, default=defaults.mu_rate)
    parser.add_argument("--mu-change", type=int, default=defaults.mu_change)
    parser.add_argument(
        "--lm-regularization", type=float, default=defaults.lm_regularization
    )
    parser.add_argument("--curvature-fwhm", type=float, default=defaults.curvature_fwhm)
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=40,
        help="PCG maximum iterations; unused by FMG",
    )
    parser.add_argument("--solver-backend", choices=("pcg", "fmg"), default="pcg")
    parser.add_argument(
        "--pcg-check-interval",
        type=int,
        default=0,
        help="0 disables intermediate PCG convergence checks",
    )
    parser.add_argument("--rotation-point-chunk", type=int, default=16384)
    parser.add_argument("--rotation-max-iter", type=int, default=500)
    parser.add_argument("--no-rotation-refine", action="store_true")
    _common_options(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        parameters = SurfWarpParameters(
            steps=args.steps,
            runs=args.runs,
            rotate=args.rotate,
            average=args.average,
            loop=args.loop,
            map_size=tuple(args.map_size),
            iterations=args.iterations,
            mu=args.mu,
            mu_rate=args.mu_rate,
            mu_change=args.mu_change,
            lm_regularization=args.lm_regularization,
            curvature_fwhm=args.curvature_fwhm,
        )
        inputs = (
            args.source_surface,
            args.source_sphere,
            args.target_surface,
            args.target_sphere,
        )
        progress = _setup(args, inputs, args.output)
        progress("Reading surfaces and checking paired meshes")
        source, source_sphere = _paired_surfaces(
            args.source_surface, args.source_sphere, "source"
        )
        target, target_sphere = _paired_surfaces(
            args.target_surface, args.target_sphere, "target"
        )
        with _cuda_device(args.device):
            result = surfwarp_cupy(
                source.vertices,
                source_sphere.vertices,
                source.faces,
                target.vertices,
                target_sphere.vertices,
                target.faces,
                parameters,
                solver_iterations=args.solver_iterations,
                solver_backend=args.solver_backend,
                rotation_point_chunk=args.rotation_point_chunk,
                rotation_refine=not args.no_rotation_refine,
                rotation_max_iter=args.rotation_max_iter,
                pcg_check_interval=args.pcg_check_interval,
                progress=progress,
            )
        save_gifti_surface(
            GiftiSurface(result.vertices, source.faces, source_sphere.reference),
            args.output,
        )
        progress(f"Saved {args.output}")
        print(
            json.dumps(
                {
                    "version": __version__,
                    "backend": "python-cupy",
                    "device": args.device,
                    "output": str(args.output),
                    "parameters": asdict(parameters),
                    "timings": result.timings,
                    "objectives": result.objectives,
                    "wall_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0
