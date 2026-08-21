"""初始旋转的 GPU depth-potential 特征计算。

该模块保留 CAT 的粗球面、cotangent Laplacian、混合 Voronoi 面积、稳定法向、
均值曲率和最终 heat-kernel 归一化逻辑，只把 depth-potential 的 SOR 线性系统
改为 GPU 上的双精度预条件共轭梯度求解。CPU/CAT helper 仍作为长期 reference。
"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .surface_stencil import SurfaceStencilDevice


@dataclass(frozen=True)
class RotationFeatureResult:
    """保存一侧初始旋转特征和线性求解诊断量。"""

    values: torch.Tensor
    iterations: int
    relative_residual: float


def _face_vectors(
    points: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按 CAT 面片角点顺序生成三条环绕边。"""

    geometry = points.to(torch.float64)
    vertices = geometry[faces]
    v1 = vertices[:, 1] - vertices[:, 0]
    v2 = vertices[:, 2] - vertices[:, 1]
    v3 = vertices[:, 0] - vertices[:, 2]
    return v1, v2, v3


def _stable_normals(points: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """复现 CAT ``stable_normals`` 的三角形加权法向。"""

    v1, v2, v3 = _face_vectors(points, faces)
    v1 = v1 / torch.linalg.vector_norm(v1, dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).eps
    )
    v2 = v2 / torch.linalg.vector_norm(v2, dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).eps
    )
    v3 = v3 / torch.linalg.vector_norm(v3, dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).eps
    )
    cross_31 = torch.linalg.cross(v3, v1, dim=-1)
    cross_23 = torch.linalg.cross(v2, v3, dim=-1)
    cross_12 = torch.linalg.cross(v1, v2, dim=-1)
    mag_31 = torch.linalg.vector_norm(cross_31, dim=-1).clamp_min(
        torch.finfo(torch.float64).eps
    )
    mag_23 = torch.linalg.vector_norm(cross_23, dim=-1).clamp_min(
        torch.finfo(torch.float64).eps
    )
    mag_12 = torch.linalg.vector_norm(cross_12, dim=-1).clamp_min(
        torch.finfo(torch.float64).eps
    )
    weight_0 = (1.0 - (v3 * v1).sum(-1)) / mag_31
    weight_1 = (1.0 - (v1 * v2).sum(-1)) / mag_12
    weight_2 = (1.0 - (v2 * v3).sum(-1)) / mag_23
    triangle_normal = cross_12 / mag_12[:, None]

    n_points = int(points.shape[0])
    result = torch.zeros((n_points, 3), dtype=torch.float64, device=points.device)
    result.index_add_(0, faces[:, 0], triangle_normal * weight_0[:, None])
    result.index_add_(0, faces[:, 1], triangle_normal * weight_1[:, None])
    result.index_add_(0, faces[:, 2], triangle_normal * weight_2[:, None])
    return result / torch.linalg.vector_norm(
        result, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float64).eps)


def _mixed_voronoi_areas(points: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """复现 CAT ``compute_areas(..., lambda=2)`` 的混合面积。"""

    v1, v2, v3 = _face_vectors(points, faces)
    cross = torch.linalg.cross(v1, v2, dim=-1)
    area = torch.linalg.vector_norm(cross, dim=-1).clamp_min(
        torch.finfo(torch.float64).eps
    )
    dot_31 = (v3 * v1).sum(-1)
    dot_12 = (v1 * v2).sum(-1)
    dot_23 = (v2 * v3).sum(-1)
    bad_0 = dot_31 > 0.0
    bad_1 = dot_12 > 0.0
    bad_2 = dot_23 > 0.0
    bad = bad_0 | bad_1 | bad_2

    lengths_1 = (v1 * v1).sum(-1)
    lengths_2 = (v2 * v2).sum(-1)
    lengths_3 = (v3 * v3).sum(-1)
    weight_0 = lengths_3 * dot_12 + lengths_1 * dot_23
    weight_1 = lengths_2 * (v1 * v3).sum(-1) + lengths_1 * dot_23
    weight_2 = lengths_2 * (v1 * v3).sum(-1) + lengths_3 * dot_12

    contribution = torch.full(
        (faces.shape[0], 3),
        0.125,
        dtype=torch.float64,
        device=points.device,
    ) * area[:, None]
    acute = torch.stack(
        (
            -0.125 * weight_0 / area,
            -0.125 * weight_1 / area,
            -0.125 * weight_2 / area,
        ),
        dim=1,
    )
    contribution = torch.where(bad[:, None], contribution, acute)
    contribution[:, 0] += torch.where(bad_0, 0.125 * area, 0.0)
    contribution[:, 1] += torch.where(bad_1, 0.125 * area, 0.0)
    contribution[:, 2] += torch.where(bad_2, 0.125 * area, 0.0)

    result = torch.zeros(
        int(points.shape[0]), dtype=torch.float64, device=points.device
    )
    for corner in range(3):
        result.index_add_(0, faces[:, corner], contribution[:, corner])
    invalid = (~torch.isfinite(result)) | (result == 0.0)
    return torch.where(invalid, torch.ones_like(result), result)


def _system_entries(
    points: torch.Tensor,
    faces: torch.Tensor,
    areas: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 ``0.5*L + alpha*diag(area)`` 的未合并矩阵条目。

    三角形贡献只负责 cotangent Laplacian；正则项必须逐顶点追加一次，
    不能随着顶点所在的三角形数量重复累加。
    """

    v1, v2, v3 = _face_vectors(points, faces)
    area = torch.linalg.vector_norm(
        torch.linalg.cross(v1, v2, dim=-1), dim=-1
    ).clamp_min(torch.finfo(torch.float64).eps)
    dot_23 = (v2 * v3).sum(-1) / area
    dot_13 = (v1 * v3).sum(-1) / area
    dot_12 = (v1 * v2).sum(-1) / area
    i0, i1, i2 = faces.unbind(dim=1)

    rows = torch.cat((i0, i1, i2, i0, i1, i1, i2, i0, i2), dim=0)
    cols = torch.cat((i0, i1, i2, i1, i0, i2, i1, i2, i0), dim=0)
    values = torch.cat(
        (
            -0.5 * (dot_23 + dot_12),
            -0.5 * (dot_13 + dot_23),
            -0.5 * (dot_12 + dot_13),
            0.5 * dot_23,
            0.5 * dot_23,
            0.5 * dot_13,
            0.5 * dot_13,
            0.5 * dot_12,
            0.5 * dot_12,
        ),
        dim=0,
    )
    if alpha != 0.0:
        vertex_indices = torch.arange(
            points.shape[0], dtype=faces.dtype, device=points.device
        )
        rows = torch.cat((rows, vertex_indices), dim=0)
        cols = torch.cat((cols, vertex_indices), dim=0)
        values = torch.cat((values, alpha * areas), dim=0)
    return rows, cols, values


def _matrix_vector_product(
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """用预先展开的 CSR-like 条目执行一次矩阵向量乘。"""

    working = vector.to(values.dtype)
    result = torch.zeros_like(working)
    shape = (values.shape[0],) + (1,) * (working.ndim - 1)
    result.index_add_(0, rows, values.reshape(shape) * working[cols])
    return result


def _solve_depth_potential_cg(
    points: torch.Tensor,
    faces: torch.Tensor,
    areas: torch.Tensor,
    normals: torch.Tensor,
    *,
    alpha: float,
    max_iter: int,
    tolerance: float,
) -> tuple[torch.Tensor, int, float]:
    """用双精度 Jacobi-PCG 求解 CAT depth-potential 线性系统。"""

    rows, cols, values = _system_entries(points, faces, areas, alpha)
    laplacian_rows, laplacian_cols, laplacian_values = _system_entries(
        points, faces, torch.zeros_like(areas), 0.0
    )
    laplacian = _matrix_vector_product(
        laplacian_rows,
        laplacian_cols,
        laplacian_values,
        points,
    ) * 2.0
    mean_curvature = 0.5 * (laplacian * normals).sum(-1) / areas
    mean_curvature = torch.where(
        torch.isfinite(mean_curvature), mean_curvature, torch.zeros_like(mean_curvature)
    )
    factor = (mean_curvature * areas).sum() / areas.sum()
    rhs = (mean_curvature - factor) * areas

    diagonal = torch.zeros_like(areas)
    diagonal.index_add_(0, rows, torch.where(rows == cols, values, 0.0))
    diagonal = diagonal.clamp_min(torch.finfo(torch.float64).eps)

    solution = torch.zeros_like(rhs)
    residual = rhs - _matrix_vector_product(rows, cols, values, solution)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(torch.float64).eps)
    relative_residual = float(torch.linalg.vector_norm(residual) / rhs_norm)
    if relative_residual <= tolerance:
        return solution, 0, relative_residual

    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    rho = torch.dot(residual, preconditioned)
    iterations = 0
    for iteration in range(1, max_iter + 1):
        applied = _matrix_vector_product(rows, cols, values, direction)
        denominator = torch.dot(direction, applied)
        step = rho / denominator
        solution = solution + step * direction
        residual = residual - step * applied
        relative_residual = float(torch.linalg.vector_norm(residual) / rhs_norm)
        iterations = iteration
        if relative_residual <= tolerance:
            break
        preconditioned = residual / diagonal
        new_rho = torch.dot(residual, preconditioned)
        direction = preconditioned + (new_rho / rho) * direction
        rho = new_rho
    return solution, iterations, relative_residual


def _solve_depth_potential_colored_sor(
    points: torch.Tensor,
    faces: torch.Tensor,
    areas: torch.Tensor,
    normals: torch.Tensor,
    color_groups: tuple[torch.Tensor, ...],
    *,
    alpha: float,
    max_iter: int,
    relaxation: float = 1.90,
) -> tuple[torch.Tensor, int, float]:
    """用图着色并行 SOR 保留 depth-potential 的迭代形态。

    同一颜色内没有相邻顶点，因此每个颜色可以并行更新；颜色之间仍按
    固定顺序推进。这样保留官方 SOR 的局部更新和松弛因子，同时避免
    顺序 Gauss-Seidel 的逐顶点 GPU 依赖。
    """

    rows, cols, values = _system_entries(points, faces, areas, alpha)
    laplacian_rows, laplacian_cols, laplacian_values = _system_entries(
        points, faces, torch.zeros_like(areas), 0.0
    )
    laplacian = _matrix_vector_product(
        laplacian_rows,
        laplacian_cols,
        laplacian_values,
        points,
    ) * 2.0
    mean_curvature = 0.5 * (laplacian * normals).sum(-1) / areas
    mean_curvature = torch.where(
        torch.isfinite(mean_curvature), mean_curvature, torch.zeros_like(mean_curvature)
    )
    factor = (mean_curvature * areas).sum() / areas.sum()
    rhs = (mean_curvature - factor) * areas

    diagonal = torch.zeros_like(areas)
    diagonal.index_add_(0, rows, torch.where(rows == cols, values, 0.0))
    diagonal = diagonal.clamp_min(torch.finfo(torch.float64).eps)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(
        torch.finfo(torch.float64).eps
    )

    color_labels = torch.empty(
        int(points.shape[0]), dtype=torch.long, device=points.device
    )
    row_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for color, vertices in enumerate(color_groups):
        color_labels[vertices] = color
    row_colors = color_labels[rows]
    for color, vertices in enumerate(color_groups):
        selected = row_colors == color
        row_parts.append(
            (rows[selected], cols[selected], values[selected], vertices)
        )

    solution = torch.zeros_like(rhs)
    for _ in range(max_iter):
        for part_rows, part_cols, part_values, vertices in row_parts:
            partial = torch.zeros_like(rhs)
            partial.index_add_(
                0,
                part_rows,
                part_values * solution[part_cols],
            )
            solution[vertices] += relaxation * (
                rhs[vertices] - partial[vertices]
            ) / diagonal[vertices]

    residual = rhs - _matrix_vector_product(rows, cols, values, solution)
    relative_residual = float(torch.linalg.vector_norm(residual) / rhs_norm)
    return solution, max_iter, relative_residual


def compute_rotation_feature(
    stencil: SurfaceStencilDevice,
    surface_vertices: torch.Tensor,
    *,
    heat_fwhm: float,
    smoothed_surface: torch.Tensor | None = None,
    depth_values: torch.Tensor | None = None,
    depth_alpha: float = 1.0 / 1000.0,
    max_iter: int = 1000,
    tolerance: float = 1.0e-9,
    solver: str = "colored-sor",
) -> RotationFeatureResult:
    """在 GPU 上计算官方初始旋转的一侧归一化特征。

    ``smoothed_surface`` 用于混合后端：官方 C helper 负责 coarse
    重采样和 heat-kernel，GPU 只接管后续 depth-potential 与特征平滑。
    ``depth_values`` 允许官方 C helper 只导出未平滑的 depth-potential，
    GPU 接管最后的 50 mm heat-kernel；两者都不提供时保持全 GPU 路径。
    """

    if solver not in {"colored-sor", "pcg"}:
        raise ValueError("rotation feature solver 必须为 colored-sor 或 pcg")

    if smoothed_surface is None:
        mapped_surface = stencil.resample_vertices(surface_vertices)
        smoothed_surface = stencil.smooth_geometry(mapped_surface, heat_fwhm)
    else:
        smoothed_surface = torch.as_tensor(
            smoothed_surface,
            dtype=torch.float32,
            device=stencil.faces.device,
        ).contiguous()
        expected_shape = (int(stencil.sphere_points.shape[0]), 3)
        if tuple(smoothed_surface.shape) != expected_shape:
            raise ValueError(
                "官方 coarse feature 几何形状必须为 "
                f"{expected_shape}，得到 {tuple(smoothed_surface.shape)}"
            )
    if depth_values is not None:
        values = torch.as_tensor(
            depth_values, dtype=torch.float64, device=stencil.faces.device
        ).reshape(-1).contiguous()
        if values.numel() != stencil.sphere_points.shape[0]:
            raise ValueError(
                "官方 depth-potential 值数量必须等于 coarse sphere 点数"
            )
        iterations = 0
        residual = 0.0
    else:
        normals = _stable_normals(smoothed_surface, stencil.faces)
        areas = _mixed_voronoi_areas(smoothed_surface, stencil.faces)
        if solver == "colored-sor":
            values, iterations, residual = _solve_depth_potential_colored_sor(
                smoothed_surface,
                stencil.faces,
                areas,
                normals,
                stencil.color_groups(),
                alpha=depth_alpha,
                max_iter=max_iter,
            )
        else:
            values, iterations, residual = _solve_depth_potential_cg(
                smoothed_surface,
                stencil.faces,
                areas,
                normals,
                alpha=depth_alpha,
                max_iter=max_iter,
                tolerance=tolerance,
            )
    values = stencil.smooth_values(values, smoothed_surface, 50.0)
    minimum = values.amin()
    maximum = values.amax()
    normalised = (values - minimum) / (maximum - minimum).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return RotationFeatureResult(
        values=normalised.contiguous(),
        iterations=iterations,
        relative_residual=residual,
    )


def compute_rotation_features(
    stencils: Sequence[SurfaceStencilDevice],
    surfaces: Sequence[torch.Tensor],
    *,
    heat_fwhms: Sequence[float] = (15.0, 10.0),
    smoothed_surfaces: Sequence[torch.Tensor] | None = None,
    depth_values: Sequence[torch.Tensor] | None = None,
    depth_alpha: float = 1.0 / 1000.0,
    max_iter: int = 1000,
    tolerance: float = 1.0e-9,
    solver: str = "colored-sor",
    parallel_sides: bool = True,
) -> tuple[RotationFeatureResult, ...]:
    """按官方 source/target 逻辑计算多侧 GPU rotation feature。

    source 和 target 的线性系统彼此独立；CUDA 下使用独立 stream 重叠两侧
    计算，但每一侧内部仍保持原有面片、邻接和迭代顺序。CPU 或显式关闭时
    保持原顺序路径。
    """

    if len(stencils) != len(surfaces) or len(stencils) != len(heat_fwhms):
        raise ValueError("stencils、surfaces 和 heat_fwhms 长度必须一致")
    if smoothed_surfaces is not None and len(smoothed_surfaces) != len(stencils):
        raise ValueError("smoothed_surfaces 长度必须与 stencils 一致")
    if depth_values is not None and len(depth_values) != len(stencils):
        raise ValueError("depth_values 长度必须与 stencils 一致")

    def compute_one(index: int) -> RotationFeatureResult:
        """计算一侧 feature，供顺序和 stream 路径复用。"""

        return compute_rotation_feature(
            stencils[index],
            surfaces[index],
            heat_fwhm=float(heat_fwhms[index]),
            smoothed_surface=(
                None if smoothed_surfaces is None else smoothed_surfaces[index]
            ),
            depth_values=(
                None if depth_values is None else depth_values[index]
            ),
            depth_alpha=depth_alpha,
            max_iter=max_iter,
            tolerance=tolerance,
            solver=solver,
        )

    if (
        not parallel_sides
        or len(stencils) < 2
        or stencils[0].faces.device.type != "cuda"
    ):
        return tuple(compute_one(index) for index in range(len(stencils)))

    device = stencils[0].faces.device
    current_stream = torch.cuda.current_stream(device=device)
    streams = [torch.cuda.Stream(device=device) for _ in stencils]
    results: list[RotationFeatureResult | None] = [None] * len(stencils)
    for index, stream in enumerate(streams):
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            results[index] = compute_one(index)
    for stream in streams:
        current_stream.wait_stream(stream)
    return tuple(result for result in results if result is not None)
