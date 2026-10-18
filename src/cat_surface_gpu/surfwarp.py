"基于官方 CAT 当前流程组装的球面 DARTEL 配准。"

from __future__ import annotations

from cat_surface_gpu.parameters import SurfWarpParameters
from cat_surface_gpu.dartel import (
    _bilinear,
    dartel_symmetric_step,
    exponentiate_velocity,
)
from cat_surface_gpu.features import (
    _smoothing_parameters,
    build_smoothing_topology,
    depth_potential_feature_cupy,
    smooth_positions_cupy,
    type5_depth_feature_cupy,
)
from cat_surface_gpu.backends.cuda.tiled import enabled as cuda_optimized
from cat_surface_gpu.geometry import (
    apply_sphere_sheet_mapping,
    build_sphere_sheet_mapping,
)
from cat_surface_gpu.resampling import (
    build_resampling_topology,
    prepare_spherical_resampler,
)
from cat_surface_gpu.rotation import (
    RotationGridIndex,
    average_xz_sphere,
    rotate_vertices,
    rotation_matrix_from_angles,
)

from dataclasses import dataclass, replace
from typing import Callable
from .io import GiftiSurface
from .parameters import _positive_integer
import time

import numpy as np


@dataclass(frozen=True)
class SurfWarpResult:
    "保存注册球面、最终速度场和分阶段耗时。"

    vertices: np.ndarray
    velocity: np.ndarray
    deformation: np.ndarray
    timings: dict[str, float]
    objectives: tuple[float, ...]


def _validate_inputs(
    source_vertices,
    source_sphere_vertices,
    source_faces,
    target_vertices,
    target_sphere_vertices,
    target_faces,
    solver_iterations,
    solver_backend,
    rotation_point_chunk,
    rotation_max_iter,
    pcg_check_interval,
) -> None:
    """在创建 CUDA 数据前检查网格数组和求解选项。"""

    for name, vertices, sphere, faces in (
        ("source", source_vertices, source_sphere_vertices, source_faces),
        ("target", target_vertices, target_sphere_vertices, target_faces),
    ):
        GiftiSurface(vertices, faces)
        GiftiSurface(sphere, faces)
        if np.asarray(vertices).shape != np.asarray(sphere).shape:
            raise ValueError(f"{name} 表面与球面的顶点数量必须相同")
    for name, value in (
        ("solver_iterations", solver_iterations),
        ("rotation_point_chunk", rotation_point_chunk),
        ("rotation_max_iter", rotation_max_iter),
    ):
        _positive_integer(name, value)
    if pcg_check_interval is not None:
        _positive_integer("pcg_check_interval", pcg_check_interval, minimum=0)
    if solver_backend not in ("pcg", "fmg"):
        raise ValueError("solver_backend 必须是 pcg 或 fmg")


def _normalise_sphere(vertices: np.ndarray) -> np.ndarray:
    "复现 CAT 入口的质心平移和逐点单位化。"

    points = np.ascontiguousarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Sphere vertices must have shape [N,3]")
    centre = points.astype(np.float64).mean(axis=0)
    centered = np.asarray(points.astype(np.float64) - centre[None, :], dtype=np.float32)
    lengths = np.linalg.norm(centered.astype(np.float64), axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("Sphere contains vertices that cannot be normalized")
    normalized = np.asarray(
        centered.astype(np.float64) / lengths[:, None], dtype=np.float32
    )
    return np.ascontiguousarray(normalized)


def _initial_rotation_cupy(
    source_vertices: np.ndarray,
    source_sphere_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_sphere_vertices: np.ndarray,
    target_faces: np.ndarray,
    *,
    rotation_point_chunk: int,
    rotation_refine: bool,
    rotation_max_iter: int,
    backend: str = "cupy",
) -> tuple[np.ndarray, float, int, int]:
    "执行官方首轮 15/10 mm depth-potential 刚体旋转。"

    started = time.perf_counter()
    source_sphere_vertices = _normalise_sphere(source_sphere_vertices)
    target_sphere_vertices = _normalise_sphere(target_sphere_vertices)
    source_topology = build_resampling_topology(
        source_faces,
        len(source_sphere_vertices),
        binary_cache=backend == "cupy" and cuda_optimized(),
    )
    target_topology = build_resampling_topology(
        target_faces,
        len(target_sphere_vertices),
        binary_cache=backend == "cupy" and cuda_optimized(),
    )
    source_resampler = prepare_spherical_resampler(
        source_sphere_vertices,
        source_faces,
        topology=source_topology,
        backend=backend,
    )
    target_resampler = prepare_spherical_resampler(
        target_sphere_vertices,
        target_faces,
        topology=target_topology,
        backend=backend,
    )
    resampled_source = source_resampler.apply(source_vertices)
    resampled_source_sphere = source_resampler.apply(source_sphere_vertices)
    resampled_target = target_resampler.apply(target_vertices)
    resampled_target_sphere = target_resampler.apply(target_sphere_vertices)
    common_faces = source_resampler.target_faces
    if not np.array_equal(common_faces, target_resampler.target_faces):
        raise RuntimeError("初始旋转的源和目标标准球 faces 不一致")
    topology = build_smoothing_topology(
        common_faces, source_resampler.target_vertices.shape[0]
    )
    smoothed_source = smooth_positions_cupy(
        resampled_source,
        topology,
        _smoothing_parameters(resampled_source, common_faces, 15.0)[0],
        array_module=np if backend in ("cpu", "metal") else None,
    )
    smoothed_target = smooth_positions_cupy(
        resampled_target,
        topology,
        _smoothing_parameters(resampled_target, common_faces, 10.0)[0],
        array_module=np if backend in ("cpu", "metal") else None,
    )
    if backend == "metal" or (backend == "cupy" and cuda_optimized()):
        from importlib import import_module

        feature_module = import_module(
            f"cat_surface_gpu.backends.{'metal' if backend == 'metal' else 'cuda'}.features"
        )
        depth_potential_feature = feature_module.depth_potential_feature
        from cat_surface_gpu.templates import target_features

        source_feature = depth_potential_feature(
            smoothed_source, common_faces, 50.0, topology=topology
        )
        (target_feature,) = target_features(
            "depth",
            smoothed_target,
            common_faces,
            [50.0],
            lambda: [
                depth_potential_feature(
                    smoothed_target, common_faces, 50.0, topology=topology
                )
            ],
            backend=backend,
        )
    else:
        source_feature = depth_potential_feature_cupy(
            smoothed_source,
            common_faces,
            50.0,
            topology=topology,
            array_module=np if backend in ("cpu", "metal") else None,
        )
        target_feature = depth_potential_feature_cupy(
            smoothed_target,
            common_faces,
            50.0,
            topology=topology,
            array_module=np if backend in ("cpu", "metal") else None,
        )
    # 更细空间网格将每次查询候选面从约一百二十个减少至约四十五个，
    # 代价是额外索引分配；内存不足时在下方缩小网格。
    rotation_grid_size = 256
    # 较大分块显著减少每次代价求值的 CUDA 核启动次数；内存不足时逐步缩小，
    # 保持候选角度、面顺序与 Nelder–Mead 停止条件不变。
    if backend in ("cpu", "metal"):
        cp = np
    else:
        import cupy as cp

    point_chunk = (
        min(int(rotation_point_chunk), 64)
        if backend in ("cpu", "metal")
        else int(rotation_point_chunk)
    )
    while True:
        try:
            rotation_index = RotationGridIndex.from_geometry(
                resampled_target_sphere,
                common_faces,
                grid_size=rotation_grid_size,
            )
            rotation_result = rotation_index.search(
                resampled_source_sphere,
                source_feature,
                target_feature,
                point_chunk=point_chunk,
                max_iter=rotation_max_iter,
                refine=rotation_refine,
                array_module=np
                if backend in ("cpu", "metal") or cuda_optimized()
                else None,
                cuda_costs=backend == "cupy" and cuda_optimized(),
                metal_costs=backend == "metal",
            )
            break
        except Exception as error:
            if backend in ("cpu", "metal") or not isinstance(
                error, cp.cuda.memory.OutOfMemoryError
            ):
                raise
            if point_chunk > 2048:
                point_chunk = max(2048, point_chunk // 2)
            elif rotation_grid_size > 128:
                rotation_grid_size //= 2
                point_chunk = int(rotation_point_chunk)
            else:
                raise
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    rotated = rotate_vertices(source_sphere_vertices, rotation_result.rotation_matrix)
    return rotated, time.perf_counter() - started, point_chunk, rotation_grid_size


def _point_to_uv(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    "复现官方新坐标系的单位球点到 UV 变换。"

    points = np.asarray(vertices, dtype=np.float64)
    lengths = np.linalg.norm(points, axis=1)
    unit = points / lengths[:, None]
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    u = phi / (2.0 * np.pi)
    v = np.arccos(np.clip(unit[:, 2], -1.0, 1.0)) / np.pi
    return u, v


def apply_deformation_to_sphere(
    sphere_vertices: np.ndarray,
    deformation: np.ndarray,
    *,
    inverse: bool = True,
) -> np.ndarray:
    "复现 ``apply_warp(sphere, sphere, deformation)`` 的 UV 位移路径。"

    mapping = np.ascontiguousarray(deformation, dtype=np.float64)
    if mapping.ndim != 3 or mapping.shape[0] != 2:
        raise ValueError("Deformation must have shape (2, H, W)")
    height, width = mapping.shape[1:]
    grid_y, grid_x = np.meshgrid(
        np.arange(1, height + 1, dtype=np.float64),
        np.arange(1, width + 1, dtype=np.float64),
        indexing="ij",
    )
    row_v = (np.arange(height, dtype=np.float64) + 0.5) / height
    pole_weight = np.sin(row_v * np.pi)[:, None]
    displacement_u = (mapping[0] - grid_x) / width
    displacement_v = (mapping[1] - grid_y) / height
    displacement_u = np.where(
        displacement_u >= 1.0,
        displacement_u - np.floor(displacement_u),
        displacement_u,
    )
    displacement_u = np.where(
        displacement_u <= -1.0,
        displacement_u + np.floor(-displacement_u),
        displacement_u,
    )
    displacement_u = np.where(
        displacement_u >= 0.5, displacement_u - 1.0, displacement_u
    )
    displacement_u = np.where(
        displacement_u <= -0.5, displacement_u + 1.0, displacement_u
    )
    displacement_v = np.where(
        displacement_v >= 1.0,
        displacement_v - np.floor(displacement_v),
        displacement_v,
    )
    displacement_v = np.where(
        displacement_v <= -1.0,
        displacement_v + np.floor(-displacement_v),
        displacement_v,
    )
    # 上游先将周期位移折回主区间，再施加球面面积权重；不得调换顺序。
    displacement_u *= pole_weight
    displacement_v *= pole_weight

    u, v = _point_to_uv(sphere_vertices)
    x = u * width - 0.5
    y = v * height - 0.5
    sampled_u = _bilinear(displacement_u, x, y)
    sampled_v = _bilinear(displacement_v, x, y)
    sampled_u = np.where(sampled_u >= 1.0, sampled_u - np.floor(sampled_u), sampled_u)
    sampled_u = np.where(sampled_u <= -1.0, sampled_u + np.floor(-sampled_u), sampled_u)
    sampled_u = np.where(sampled_u < -0.5, sampled_u + 1.0, sampled_u)
    sampled_u = np.where(sampled_u > 0.5, sampled_u - 1.0, sampled_u)
    sampled_v = np.where(sampled_v >= 1.0, sampled_v - np.floor(sampled_v), sampled_v)
    sampled_v = np.where(sampled_v <= -1.0, sampled_v + np.floor(-sampled_v), sampled_v)
    if inverse:
        sampled_u = -sampled_u
        sampled_v = -sampled_v
    u = u + sampled_u
    v = v + sampled_v
    below = v < 0.0
    v = np.where(below, -v, v)
    u = np.where(below, u + 0.5, u)
    above = v > 1.0
    v = np.where(above, 2.0 - v, v)
    u = np.where(above, u + 0.5, u)
    u = np.mod(u, 1.0)
    phi = u * (2.0 * np.pi)
    theta = v * np.pi
    output = np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=1,
    )
    return np.ascontiguousarray(output, dtype=np.float32)


def surfwarp_single_run_cupy(
    source_vertices: np.ndarray,
    source_sphere_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_sphere_vertices: np.ndarray,
    target_faces: np.ndarray,
    parameters: SurfWarpParameters | None = None,
    *,
    solver_iterations: int = 40,
    solver_backend: str = "pcg",
    rotation_point_chunk: int = 16384,
    rotation_refine: bool = True,
    rotation_max_iter: int = 500,
    source_resampling_topology=None,
    target_resampling_topology=None,
    target_resampler=None,
    smoothing_topology=None,
    target_mapping=None,
    target_feature_sheets=None,
    pcg_check_interval: int | None = 0,
    backend: str = "cupy",
    progress: Callable[[str], None] | None = None,
) -> SurfWarpResult:
    """校验公开单次配准调用，再进入已验证的内部计算。"""

    _validate_inputs(
        source_vertices,
        source_sphere_vertices,
        source_faces,
        target_vertices,
        target_sphere_vertices,
        target_faces,
        solver_iterations,
        solver_backend,
        rotation_point_chunk,
        rotation_max_iter,
        pcg_check_interval,
    )
    return _surfwarp_single_run(
        source_vertices,
        source_sphere_vertices,
        source_faces,
        target_vertices,
        target_sphere_vertices,
        target_faces,
        parameters,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        rotation_point_chunk=rotation_point_chunk,
        rotation_refine=rotation_refine,
        rotation_max_iter=rotation_max_iter,
        source_resampling_topology=source_resampling_topology,
        target_resampling_topology=target_resampling_topology,
        target_resampler=target_resampler,
        smoothing_topology=smoothing_topology,
        target_mapping=target_mapping,
        target_feature_sheets=target_feature_sheets,
        pcg_check_interval=pcg_check_interval,
        backend=backend,
        progress=progress,
    )


def _surfwarp_single_run(
    source_vertices: np.ndarray,
    source_sphere_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_sphere_vertices: np.ndarray,
    target_faces: np.ndarray,
    parameters: SurfWarpParameters | None = None,
    *,
    solver_iterations: int = 40,
    solver_backend: str = "pcg",
    rotation_point_chunk: int = 16384,
    rotation_refine: bool = True,
    rotation_max_iter: int = 500,
    source_resampling_topology=None,
    target_resampling_topology=None,
    target_resampler=None,
    smoothing_topology=None,
    target_mapping=None,
    target_feature_sheets=None,
    pcg_check_interval: int | None = 0,
    backend: str = "cupy",
    progress: Callable[[str], None] | None = None,
) -> SurfWarpResult:
    "组装一次官方 DARTEL run，可选执行官方初始刚体旋转。"

    if backend not in ("cpu", "cupy", "metal"):
        raise ValueError("backend must be cpu, cupy or metal")
    if backend == "metal":
        from .backends.metal.surf2sphere import _mlx

        _mlx()
        if solver_backend != "pcg" or pcg_check_interval != 0:
            raise ValueError("Metal currently requires PCG with pcg_check_interval=0")

    if backend in ("cpu", "metal"):
        cp = np
    else:
        import cupy as cp

    options = (
        parameters
        if parameters is not None
        else SurfWarpParameters(rotate=False, average=False, runs=1)
    )
    if options.average or options.runs != 1:
        raise ValueError("当前单次组装入口要求 average=False、runs=1")
    if options.multires_levels != 0:
        raise NotImplementedError("当前单次组装入口尚未实现 multires")
    if any(kind != 5 for kind in options.curvature_types[: options.steps]):
        raise NotImplementedError("当前单次组装入口只实现 curvtype=5")

    timings: dict[str, float] = {}
    started = time.perf_counter()
    working_source_sphere = _normalise_sphere(source_sphere_vertices)
    normalized_target_sphere = _normalise_sphere(target_sphere_vertices)
    if options.rotate:
        if progress is not None:
            progress("SurfWarp: initial rotation")
        (
            working_source_sphere,
            rotation_seconds,
            rotation_chunk_used,
            rotation_grid_used,
        ) = _initial_rotation_cupy(
            source_vertices,
            working_source_sphere,
            source_faces,
            target_vertices,
            normalized_target_sphere,
            target_faces,
            rotation_point_chunk=rotation_point_chunk,
            rotation_refine=rotation_refine,
            rotation_max_iter=rotation_max_iter,
            backend=backend,
        )
        timings["initial_rotation"] = rotation_seconds
        timings["rotation_point_chunk"] = float(rotation_chunk_used)
        timings["rotation_grid_size"] = float(rotation_grid_used)
    else:
        timings["initial_rotation"] = 0.0
    source_topology = (
        source_resampling_topology
        if source_resampling_topology is not None
        else build_resampling_topology(
            source_faces,
            len(source_sphere_vertices),
            binary_cache=backend == "cupy" and cuda_optimized(),
        )
    )
    target_topology = (
        target_resampling_topology
        if target_resampling_topology is not None
        else build_resampling_topology(
            target_faces,
            len(target_sphere_vertices),
            binary_cache=backend == "cupy" and cuda_optimized(),
        )
    )
    source_resampler = prepare_spherical_resampler(
        working_source_sphere,
        source_faces,
        topology=source_topology,
        backend=backend,
    )
    if target_resampler is None:
        target_resampler = prepare_spherical_resampler(
            normalized_target_sphere,
            target_faces,
            topology=target_topology,
            backend=backend,
        )
    resampled_source = source_resampler.apply(source_vertices)
    resampled_source_sphere = source_resampler.apply(working_source_sphere)
    resampled_target = target_resampler.apply(target_vertices)
    resampled_target_sphere = target_resampler.apply(normalized_target_sphere)
    prepared = time.perf_counter()
    timings["resampling"] = prepared - started

    common_faces = source_resampler.target_faces
    if not np.array_equal(common_faces, target_resampler.target_faces):
        raise RuntimeError("源和目标标准球 faces 不一致")
    if smoothing_topology is None:
        smoothing_topology = build_smoothing_topology(
            common_faces, source_resampler.target_vertices.shape[0]
        )

    source_mapping = build_sphere_sheet_mapping(
        resampled_source_sphere,
        common_faces,
        options.map_size,
        backend=backend,
    )
    if target_mapping is None:
        target_mapping = build_sphere_sheet_mapping(
            resampled_target_sphere,
            common_faces,
            options.map_size,
            backend=backend,
        )
    mapped = time.perf_counter()
    timings["mapping_setup"] = mapped - prepared

    schedule = options.dartel_schedule()
    # 目标标量保留在 GPU，循环结束后一次取回，避免每次 DARTEL 更新
    # 在 Python 与 CUDA 之间同步。
    objective_scalars = []
    solver_steps_used: list[int] = []
    velocity = None
    feature_seconds = 0.0
    dartel_seconds = 0.0
    source_feature_scales = target_feature_scales = None
    if backend == "metal" or (backend == "cupy" and cuda_optimized()):
        from importlib import import_module

        feature_module = import_module(
            f"cat_surface_gpu.backends.{'metal' if backend == 'metal' else 'cuda'}.features"
        )
        type5_depth_features = feature_module.type5_depth_features
        from cat_surface_gpu.templates import target_features

        feature_started = time.perf_counter()
        scales = [options.curvature_fwhm / (3.0**step) for step in range(options.steps)]
        source_feature_scales = type5_depth_features(
            resampled_source, common_faces, scales, smoothing_topology
        )
        if target_feature_sheets is None:
            target_feature_scales = target_features(
                "type5",
                resampled_target,
                common_faces,
                scales,
                lambda: type5_depth_features(
                    resampled_target, common_faces, scales, smoothing_topology
                ),
                backend=backend,
            )
        feature_seconds += time.perf_counter() - feature_started
    for step in range(options.steps):
        if progress is not None:
            progress(f"SurfWarp: feature and DARTEL step {step + 1}/{options.steps}")
        feature_started = time.perf_counter()
        fwhm = options.curvature_fwhm / (3.0**step)
        source_values = (
            source_feature_scales[step]
            if source_feature_scales is not None
            else type5_depth_feature_cupy(
                resampled_source,
                common_faces,
                fwhm,
                topology=smoothing_topology,
                array_module=np if backend in ("cpu", "metal") else None,
            )
        )
        if target_feature_sheets is None:
            target_values = (
                target_feature_scales[step]
                if target_feature_scales is not None
                else type5_depth_feature_cupy(
                    resampled_target,
                    common_faces,
                    fwhm,
                    topology=smoothing_topology,
                    array_module=np if backend in ("cpu", "metal") else None,
                )
            )
            target_sheet = apply_sphere_sheet_mapping(target_mapping, target_values)
        else:
            if len(target_feature_sheets) != options.steps:
                raise ValueError("target_feature_sheets 长度必须等于 steps")
            target_sheet = target_feature_sheets[step]
        source_sheet = apply_sphere_sheet_mapping(source_mapping, source_values)
        feature_seconds += time.perf_counter() - feature_started

        # 当前上游单层实现在每个阶段清空初始流场。
        velocity = cp.zeros(
            (2, source_sheet.shape[0], source_sheet.shape[1]),
            dtype=cp.float64,
        )
        source_gpu = cp.asarray(source_sheet, dtype=cp.float64)
        target_gpu = cp.asarray(target_sheet, dtype=cp.float64)
        dartel_started = time.perf_counter()
        if backend == "metal":
            from .backends.metal.warp import run_schedule

            velocity, values = run_schedule(
                source_sheet, target_sheet, schedule, solver_iterations
            )
            objective_scalars.extend(values)
            solver_steps_used.extend([solver_iterations] * len(values))
        elif (
            backend == "cupy"
            and cuda_optimized()
            and solver_backend == "pcg"
            and pcg_check_interval == 0
        ):
            from cat_surface_gpu.backends.cuda.warp import run_schedule

            velocity, values = run_schedule(
                source_sheet, target_sheet, schedule, solver_iterations
            )
            objective_scalars.extend(values)
            solver_steps_used.extend([solver_iterations] * len(values))
        else:
            for item in schedule:
                for _ in range(int(item["iterations"])):
                    velocity, diagnostics = dartel_symmetric_step(
                        velocity,
                        source_gpu,
                        target_gpu,
                        time_step=int(item["time_step"]),
                        mu=float(item["mu"]),
                        lm_regularization=float(item["lmreg"]),
                        solver_iterations=solver_iterations,
                        solver_backend=solver_backend,
                        collect_diagnostics=False,
                        pcg_check_interval=pcg_check_interval,
                    )
                    objective_scalars.append(diagnostics["objective"])
                    solver_steps_used.append(int(diagnostics["solver_iterations"]))
        if backend == "cupy":
            cp.cuda.get_current_stream().synchronize()
        dartel_seconds += time.perf_counter() - dartel_started
    timings["features_and_sheet"] = feature_seconds
    timings["dartel"] = dartel_seconds
    objectives = (
        [
            float(value)
            for value in (
                np.stack(objective_scalars)
                if backend in ("cpu", "metal")
                else cp.asnumpy(cp.stack(objective_scalars))
            )
        ]
        if objective_scalars
        else []
    )
    if solver_steps_used:
        timings["solver_iterations_mean"] = float(np.mean(solver_steps_used))
        timings["solver_iterations_max"] = float(np.max(solver_steps_used))

    if velocity is None:
        raise RuntimeError("没有执行 DARTEL step")
    exponent_started = time.perf_counter()
    deformation_gpu, _ = exponentiate_velocity(velocity, 10)
    deformation = (
        np.asarray(deformation_gpu)
        if backend in ("cpu", "metal")
        else cp.asnumpy(deformation_gpu)
    )
    if backend == "cupy":
        cp.cuda.get_current_stream().synchronize()
    timings["exponentiation"] = time.perf_counter() - exponent_started
    apply_started = time.perf_counter()
    registered = apply_deformation_to_sphere(
        working_source_sphere, deformation, inverse=True
    )
    timings["apply_warp"] = time.perf_counter() - apply_started
    timings["total"] = time.perf_counter() - started
    GiftiSurface(registered, source_faces)
    return SurfWarpResult(
        vertices=registered,
        velocity=np.asarray(velocity)
        if backend in ("cpu", "metal")
        else cp.asnumpy(velocity),
        deformation=deformation,
        timings=timings,
        objectives=tuple(objectives),
    )


def surfwarp_cupy(
    source_vertices: np.ndarray,
    source_sphere_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_sphere_vertices: np.ndarray,
    target_faces: np.ndarray,
    parameters: SurfWarpParameters | None = None,
    *,
    solver_iterations: int = 40,
    solver_backend: str = "pcg",
    rotation_point_chunk: int = 16384,
    rotation_refine: bool = True,
    rotation_max_iter: int = 500,
    pcg_check_interval: int | None = 0,
    backend: str = "cupy",
    progress: Callable[[str], None] | None = None,
) -> SurfWarpResult:
    "执行官方初始旋转、多个 runs 以及极点平均状态机。"

    _validate_inputs(
        source_vertices,
        source_sphere_vertices,
        source_faces,
        target_vertices,
        target_sphere_vertices,
        target_faces,
        solver_iterations,
        solver_backend,
        rotation_point_chunk,
        rotation_max_iter,
        pcg_check_interval,
    )
    if backend not in ("cpu", "cupy", "metal"):
        raise ValueError("backend must be cpu, cupy or metal")
    if backend == "metal":
        from .backends.metal.surf2sphere import _mlx

        _mlx()
        if solver_backend != "pcg" or pcg_check_interval != 0:
            raise ValueError("Metal currently requires PCG with pcg_check_interval=0")

    options = parameters if parameters is not None else SurfWarpParameters()
    if options.multires_levels != 0:
        raise NotImplementedError("当前完整入口尚未实现 multires")
    if any(kind != 5 for kind in options.curvature_types[: options.steps]):
        raise NotImplementedError("当前完整入口只实现前两层 curvtype=5")

    started = time.perf_counter()
    working_source_sphere = _normalise_sphere(source_sphere_vertices)
    normalized_target_sphere = _normalise_sphere(target_sphere_vertices)
    timings: dict[str, float] = {}
    objectives: list[float] = []
    last_result: SurfWarpResult | None = None

    if options.rotate:
        if progress is not None:
            progress("SurfWarp: initial rotation")
        (
            working_source_sphere,
            timings["initial_rotation"],
            rotation_chunk_used,
            rotation_grid_used,
        ) = _initial_rotation_cupy(
            source_vertices,
            working_source_sphere,
            source_faces,
            target_vertices,
            normalized_target_sphere,
            target_faces,
            rotation_point_chunk=rotation_point_chunk,
            rotation_refine=rotation_refine,
            rotation_max_iter=rotation_max_iter,
            backend=backend,
        )
        timings["rotation_point_chunk"] = float(rotation_chunk_used)
        timings["rotation_grid_size"] = float(rotation_grid_used)
    else:
        timings["initial_rotation"] = 0.0

    # 两次主配准共享目标球重采样、平滑拓扑和二维映射；平均极点的
    # 旋转配准仍构建自己的几何相关目标状态。
    shared_source_topology = build_resampling_topology(
        source_faces,
        len(source_sphere_vertices),
        binary_cache=backend == "cupy" and cuda_optimized(),
    )
    shared_target_topology = build_resampling_topology(
        target_faces,
        len(target_sphere_vertices),
        binary_cache=backend == "cupy" and cuda_optimized(),
    )
    shared_target_resampler = prepare_spherical_resampler(
        normalized_target_sphere,
        target_faces,
        topology=shared_target_topology,
        backend=backend,
    )
    shared_common_faces = shared_target_resampler.target_faces
    shared_smoothing_topology = build_smoothing_topology(
        shared_common_faces, shared_target_resampler.target_vertices.shape[0]
    )
    shared_target_mapping = build_sphere_sheet_mapping(
        shared_target_resampler.apply(normalized_target_sphere),
        shared_common_faces,
        options.map_size,
        backend=backend,
    )
    target_feature_started = time.perf_counter()
    target_feature_sheets = []
    resampled_target_for_features = shared_target_resampler.apply(target_vertices)
    cached_target_scales = None
    if backend == "metal" or (backend == "cupy" and cuda_optimized()):
        from importlib import import_module

        feature_module = import_module(
            f"cat_surface_gpu.backends.{'metal' if backend == 'metal' else 'cuda'}.features"
        )
        type5_depth_features = feature_module.type5_depth_features
        from cat_surface_gpu.templates import target_features

        scales = [options.curvature_fwhm / (3.0**step) for step in range(options.steps)]
        cached_target_scales = target_features(
            "type5",
            resampled_target_for_features,
            shared_common_faces,
            scales,
            lambda: type5_depth_features(
                resampled_target_for_features,
                shared_common_faces,
                scales,
                shared_smoothing_topology,
            ),
            backend=backend,
        )
    for step in range(options.steps):
        if progress is not None:
            progress(f"SurfWarp: feature and DARTEL step {step + 1}/{options.steps}")
        target_values = (
            cached_target_scales[step]
            if cached_target_scales is not None
            else type5_depth_feature_cupy(
                resampled_target_for_features,
                shared_common_faces,
                options.curvature_fwhm / (3.0**step),
                topology=shared_smoothing_topology,
                array_module=np if backend in ("cpu", "metal") else None,
            )
        )
        target_feature_sheets.append(
            apply_sphere_sheet_mapping(shared_target_mapping, target_values)
        )
    timings["target_feature_setup"] = time.perf_counter() - target_feature_started
    target_feature_sheets = tuple(target_feature_sheets)
    run_options = replace(options, rotate=False, average=False, runs=1)
    for run_index in range(options.runs):
        if progress is not None:
            progress(f"SurfWarp: run {run_index + 1}/{options.runs}")
        sphere_before_run = working_source_sphere
        run_result = _surfwarp_single_run(
            source_vertices,
            sphere_before_run,
            source_faces,
            target_vertices,
            normalized_target_sphere,
            target_faces,
            run_options,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            rotation_point_chunk=rotation_point_chunk,
            rotation_refine=rotation_refine,
            rotation_max_iter=rotation_max_iter,
            source_resampling_topology=shared_source_topology,
            target_resampling_topology=shared_target_topology,
            target_resampler=shared_target_resampler,
            smoothing_topology=shared_smoothing_topology,
            target_mapping=shared_target_mapping,
            target_feature_sheets=target_feature_sheets,
            pcg_check_interval=pcg_check_interval,
            backend=backend,
        )
        last_result = run_result
        objectives.extend(run_result.objectives)
        for name, value in run_result.timings.items():
            timings[f"run{run_index + 1}_{name}"] = float(value)

        if options.average and run_index == options.runs - 1:
            plus_matrix = rotation_matrix_from_angles(
                np.asarray((0.0, np.pi / 2.0, 0.0), dtype=np.float64)
            )
            minus_matrix = rotation_matrix_from_angles(
                np.asarray((0.0, -np.pi / 2.0, 0.0), dtype=np.float64)
            )
            rotated_source = rotate_vertices(source_vertices, plus_matrix)
            rotated_source_sphere = rotate_vertices(sphere_before_run, plus_matrix)
            rotated_target = rotate_vertices(target_vertices, plus_matrix)
            rotated_target_sphere = rotate_vertices(
                normalized_target_sphere, plus_matrix
            )
            average_started = time.perf_counter()
            rotated_result = _surfwarp_single_run(
                rotated_source,
                rotated_source_sphere,
                source_faces,
                rotated_target,
                rotated_target_sphere,
                target_faces,
                run_options,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
                rotation_point_chunk=rotation_point_chunk,
                rotation_refine=rotation_refine,
                rotation_max_iter=rotation_max_iter,
                source_resampling_topology=shared_source_topology,
                target_resampling_topology=shared_target_topology,
                smoothing_topology=shared_smoothing_topology,
                pcg_check_interval=pcg_check_interval,
                backend=backend,
            )
            rotated_back = rotate_vertices(rotated_result.vertices, minus_matrix)
            averaged = average_xz_sphere(run_result.vertices, rotated_back)
            timings["average_second_run"] = time.perf_counter() - average_started
            objectives.extend(rotated_result.objectives)
            working_source_sphere = averaged
            last_result = SurfWarpResult(
                vertices=averaged,
                velocity=run_result.velocity,
                deformation=run_result.deformation,
                timings=timings,
                objectives=tuple(objectives),
            )
        else:
            working_source_sphere = run_result.vertices

    if last_result is None:
        raise RuntimeError("runs 没有产生任何 SurfWarp 结果")
    timings["total"] = time.perf_counter() - started
    GiftiSurface(last_result.vertices, source_faces)
    return SurfWarpResult(
        vertices=last_result.vertices,
        velocity=last_result.velocity,
        deformation=last_result.deformation,
        timings=timings,
        objectives=tuple(objectives),
    )
