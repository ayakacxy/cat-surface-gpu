"""最新版 CAT-Surface 的 CAT_Surf2Sphere GPU 面积平滑后端。"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from .gpu_pipeline import GiftiMesh, read_gifti_mesh
from .surface_stencil import (
    _build_color_groups,
    _build_neighbour_table,
    _build_ordered_dependency_groups,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 由运行环境决定
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _areal_smoothing_kernel(
        points_ptr,
        faces_ptr,
        incident_ptr,
        degree_ptr,
        group_ptr,
        n_group,
        max_incident: tl.constexpr,
        use_fp64: tl.constexpr,
        use_cat_mixed: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """融合一个独立顶点组的入射三角形面积和坐标更新。"""

        local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = local < n_group
        vertex = tl.load(group_ptr + local, mask=active, other=0).to(tl.int64)
        degree = tl.load(degree_ptr + vertex, mask=active, other=0)

        if use_cat_mixed:
            old_x = tl.load(points_ptr + vertex * 3, mask=active, other=0.0).to(
                tl.float64
            )
            old_y = tl.load(
                points_ptr + vertex * 3 + 1, mask=active, other=0.0
            ).to(tl.float64)
            old_z = tl.load(
                points_ptr + vertex * 3 + 2, mask=active, other=0.0
            ).to(tl.float64)
            total_area = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_x = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_y = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_z = tl.zeros((BLOCK,), dtype=tl.float64)
        elif use_fp64:
            old_x = tl.load(points_ptr + vertex * 3, mask=active, other=0.0).to(
                tl.float64
            )
            old_y = tl.load(
                points_ptr + vertex * 3 + 1, mask=active, other=0.0
            ).to(tl.float64)
            old_z = tl.load(
                points_ptr + vertex * 3 + 2, mask=active, other=0.0
            ).to(tl.float64)
            total_area = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_x = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_y = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_z = tl.zeros((BLOCK,), dtype=tl.float64)
        else:
            old_x = tl.load(points_ptr + vertex * 3, mask=active, other=0.0).to(
                tl.float32
            )
            old_y = tl.load(
                points_ptr + vertex * 3 + 1, mask=active, other=0.0
            ).to(tl.float32)
            old_z = tl.load(
                points_ptr + vertex * 3 + 2, mask=active, other=0.0
            ).to(tl.float32)
            total_area = tl.zeros((BLOCK,), dtype=tl.float32)
            weighted_x = tl.zeros((BLOCK,), dtype=tl.float32)
            weighted_y = tl.zeros((BLOCK,), dtype=tl.float32)
            weighted_z = tl.zeros((BLOCK,), dtype=tl.float32)

        for slot in range(max_incident):
            valid = active & (slot < degree)
            incident_offset = vertex * max_incident + slot
            face = tl.load(
                incident_ptr + incident_offset,
                mask=valid,
                other=0,
            ).to(tl.int64)
            face_offset = face * 3
            vertex0 = tl.load(
                faces_ptr + face_offset,
                mask=valid,
                other=0,
            ).to(tl.int64)
            vertex1 = tl.load(
                faces_ptr + face_offset + 1,
                mask=valid,
                other=0,
            ).to(tl.int64)
            vertex2 = tl.load(
                faces_ptr + face_offset + 2,
                mask=valid,
                other=0,
            ).to(tl.int64)

            if use_cat_mixed:
                p0x = tl.load(
                    points_ptr + vertex0 * 3, mask=valid, other=0.0
                ).to(tl.float32)
                p0y = tl.load(
                    points_ptr + vertex0 * 3 + 1, mask=valid, other=0.0
                ).to(tl.float32)
                p0z = tl.load(
                    points_ptr + vertex0 * 3 + 2, mask=valid, other=0.0
                ).to(tl.float32)
                p1x = tl.load(
                    points_ptr + vertex1 * 3, mask=valid, other=0.0
                ).to(tl.float32)
                p1y = tl.load(
                    points_ptr + vertex1 * 3 + 1, mask=valid, other=0.0
                ).to(tl.float32)
                p1z = tl.load(
                    points_ptr + vertex1 * 3 + 2, mask=valid, other=0.0
                ).to(tl.float32)
                p2x = tl.load(
                    points_ptr + vertex2 * 3, mask=valid, other=0.0
                ).to(tl.float32)
                p2y = tl.load(
                    points_ptr + vertex2 * 3 + 1, mask=valid, other=0.0
                ).to(tl.float32)
                p2z = tl.load(
                    points_ptr + vertex2 * 3 + 2, mask=valid, other=0.0
                ).to(tl.float32)
            else:
                p0x = tl.load(points_ptr + vertex0 * 3, mask=valid, other=0.0)
                p0y = tl.load(points_ptr + vertex0 * 3 + 1, mask=valid, other=0.0)
                p0z = tl.load(points_ptr + vertex0 * 3 + 2, mask=valid, other=0.0)
                p1x = tl.load(points_ptr + vertex1 * 3, mask=valid, other=0.0)
                p1y = tl.load(points_ptr + vertex1 * 3 + 1, mask=valid, other=0.0)
                p1z = tl.load(points_ptr + vertex1 * 3 + 2, mask=valid, other=0.0)
                p2x = tl.load(points_ptr + vertex2 * 3, mask=valid, other=0.0)
                p2y = tl.load(points_ptr + vertex2 * 3 + 1, mask=valid, other=0.0)
                p2z = tl.load(points_ptr + vertex2 * 3 + 2, mask=valid, other=0.0)
            edge0x = p1x - p0x
            edge0y = p1y - p0y
            edge0z = p1z - p0z
            edge1x = p2x - p0x
            edge1y = p2y - p0y
            edge1z = p2z - p0z
            cross_x = edge0y * edge1z - edge0z * edge1y
            cross_y = edge0z * edge1x - edge0x * edge1z
            cross_z = edge0x * edge1y - edge0y * edge1x
            if use_cat_mixed:
                area_square = (
                    cross_x * cross_x
                    + cross_y * cross_y
                    + cross_z * cross_z
                ).to(tl.float64)
                area = 0.5 * tl.sqrt(area_square)
                center_x = (
                    p0x.to(tl.float64)
                    + p1x.to(tl.float64)
                    + p2x.to(tl.float64)
                ) / 3.0
                center_y = (
                    p0y.to(tl.float64)
                    + p1y.to(tl.float64)
                    + p2y.to(tl.float64)
                ) / 3.0
                center_z = (
                    p0z.to(tl.float64)
                    + p1z.to(tl.float64)
                    + p2z.to(tl.float64)
                ) / 3.0
            else:
                area = 0.5 * tl.sqrt(
                    cross_x * cross_x
                    + cross_y * cross_y
                    + cross_z * cross_z
                )
                center_x = (p0x + p1x + p2x) / 3.0
                center_y = (p0y + p1y + p2y) / 3.0
                center_z = (p0z + p1z + p2z) / 3.0
            area = tl.minimum(tl.maximum(area, 0.0), 1.0)
            area = tl.where(valid, area, 0.0)
            total_area += area
            weighted_x += area * center_x
            weighted_y += area * center_y
            weighted_z += area * center_z

        safe_area = tl.maximum(total_area, 1.0e-20)
        new_x = weighted_x / safe_area
        new_y = weighted_y / safe_area
        new_z = weighted_z / safe_area
        new_x = tl.where(total_area > 0.0, new_x, old_x)
        new_y = tl.where(total_area > 0.0, new_y, old_y)
        new_z = tl.where(total_area > 0.0, new_z, old_z)
        if use_cat_mixed:
            tl.store(points_ptr + vertex * 3, new_x.to(tl.float32), mask=active)
            tl.store(
                points_ptr + vertex * 3 + 1,
                new_y.to(tl.float32),
                mask=active,
            )
            tl.store(
                points_ptr + vertex * 3 + 2,
                new_z.to(tl.float32),
                mask=active,
            )
        else:
            tl.store(points_ptr + vertex * 3, new_x, mask=active)
            tl.store(points_ptr + vertex * 3 + 1, new_y, mask=active)
            tl.store(points_ptr + vertex * 3 + 2, new_z, mask=active)

    @triton.jit
    def _normalize_points_kernel(
        points_ptr,
        n_points,
        radius_value: tl.constexpr,
        use_fp64: tl.constexpr,
        use_cat_mixed: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """把所有顶点投影回面积平滑使用的固定半径。"""

        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = index < n_points
        if use_cat_mixed:
            x = tl.load(points_ptr + index * 3, mask=active, other=0.0).to(
                tl.float32
            )
            y = tl.load(points_ptr + index * 3 + 1, mask=active, other=0.0).to(
                tl.float32
            )
            z = tl.load(points_ptr + index * 3 + 2, mask=active, other=0.0).to(
                tl.float32
            )
            norm_square = x * x + y * y + z * z
            norm = tl.sqrt(norm_square.to(tl.float64))
            scale = radius_value / tl.maximum(norm, 1.0e-20)
            tl.store(
                points_ptr + index * 3,
                (x.to(tl.float64) * scale).to(tl.float32),
                mask=active,
            )
            tl.store(
                points_ptr + index * 3 + 1,
                (y.to(tl.float64) * scale).to(tl.float32),
                mask=active,
            )
            tl.store(
                points_ptr + index * 3 + 2,
                (z.to(tl.float64) * scale).to(tl.float32),
                mask=active,
            )
        elif use_fp64:
            x = tl.load(points_ptr + index * 3, mask=active, other=0.0).to(
                tl.float64
            )
            y = tl.load(points_ptr + index * 3 + 1, mask=active, other=0.0).to(
                tl.float64
            )
            z = tl.load(points_ptr + index * 3 + 2, mask=active, other=0.0).to(
                tl.float64
            )
        else:
            x = tl.load(points_ptr + index * 3, mask=active, other=0.0).to(
                tl.float32
            )
            y = tl.load(points_ptr + index * 3 + 1, mask=active, other=0.0).to(
                tl.float32
            )
            z = tl.load(points_ptr + index * 3 + 2, mask=active, other=0.0).to(
                tl.float32
            )
            norm = tl.sqrt(x * x + y * y + z * z)
            scale = radius_value / tl.maximum(norm, 1.0e-20)
            tl.store(points_ptr + index * 3, x * scale, mask=active)
            tl.store(points_ptr + index * 3 + 1, y * scale, mask=active)
            tl.store(points_ptr + index * 3 + 2, z * scale, mask=active)

    @triton.jit
    def _distance_smoothing_kernel(
        points_ptr,
        neighbours_ptr,
        degree_ptr,
        selected_ptr,
        group_ptr,
        n_group,
        max_degree: tl.constexpr,
        strength: tl.constexpr,
        inv_strength: tl.constexpr,
        use_subset: tl.constexpr,
        use_fp64: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """融合一个独立顶点组的 Manhattan 距离平滑更新。"""

        local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = local < n_group
        vertex = tl.load(group_ptr + local, mask=active, other=0).to(tl.int64)
        degree = tl.load(degree_ptr + vertex, mask=active, other=0)
        selected = tl.load(selected_ptr + vertex, mask=active, other=0)

        if use_fp64:
            old_x = tl.load(points_ptr + vertex * 3, mask=active, other=0.0).to(
                tl.float64
            )
            old_y = tl.load(
                points_ptr + vertex * 3 + 1, mask=active, other=0.0
            ).to(tl.float64)
            old_z = tl.load(
                points_ptr + vertex * 3 + 2, mask=active, other=0.0
            ).to(tl.float64)
            total_distance = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_x = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_y = tl.zeros((BLOCK,), dtype=tl.float64)
            weighted_z = tl.zeros((BLOCK,), dtype=tl.float64)
        else:
            old_x = tl.load(points_ptr + vertex * 3, mask=active, other=0.0).to(
                tl.float32
            )
            old_y = tl.load(
                points_ptr + vertex * 3 + 1, mask=active, other=0.0
            ).to(tl.float32)
            old_z = tl.load(
                points_ptr + vertex * 3 + 2, mask=active, other=0.0
            ).to(tl.float32)
            total_distance = tl.zeros((BLOCK,), dtype=tl.float32)
            weighted_x = tl.zeros((BLOCK,), dtype=tl.float32)
            weighted_y = tl.zeros((BLOCK,), dtype=tl.float32)
            weighted_z = tl.zeros((BLOCK,), dtype=tl.float32)

        for slot in range(max_degree):
            valid = active & (slot < degree)
            neighbour = tl.load(
                neighbours_ptr + vertex * max_degree + slot,
                mask=valid,
                other=0,
            ).to(tl.int64)
            if use_fp64:
                neighbour_x = tl.load(
                    points_ptr + neighbour * 3, mask=valid, other=0.0
                ).to(tl.float64)
                neighbour_y = tl.load(
                    points_ptr + neighbour * 3 + 1, mask=valid, other=0.0
                ).to(tl.float64)
                neighbour_z = tl.load(
                    points_ptr + neighbour * 3 + 2, mask=valid, other=0.0
                ).to(tl.float64)
            else:
                neighbour_x = tl.load(
                    points_ptr + neighbour * 3, mask=valid, other=0.0
                ).to(tl.float32)
                neighbour_y = tl.load(
                    points_ptr + neighbour * 3 + 1, mask=valid, other=0.0
                ).to(tl.float32)
                neighbour_z = tl.load(
                    points_ptr + neighbour * 3 + 2, mask=valid, other=0.0
                ).to(tl.float32)
            distance = (
                tl.abs(neighbour_x - old_x)
                + tl.abs(neighbour_y - old_y)
                + tl.abs(neighbour_z - old_z)
            )
            positive = valid & (distance > 0.0)
            total_distance += tl.where(valid, distance, 0.0)
            weighted_x += tl.where(positive, distance * neighbour_x, 0.0)
            weighted_y += tl.where(positive, distance * neighbour_y, 0.0)
            weighted_z += tl.where(positive, distance * neighbour_z, 0.0)

        safe_distance = tl.maximum(total_distance, 1.0e-30)
        candidate_x = old_x * inv_strength + (
            weighted_x / safe_distance
        ) * strength
        candidate_y = old_y * inv_strength + (
            weighted_y / safe_distance
        ) * strength
        candidate_z = old_z * inv_strength + (
            weighted_z / safe_distance
        ) * strength
        # 与上游一致：所有邻居距离为零时，距离加权和为零而不是静默保留原点。
        candidate_x = tl.where(total_distance > 0.0, candidate_x, old_x * inv_strength)
        candidate_y = tl.where(total_distance > 0.0, candidate_y, old_y * inv_strength)
        candidate_z = tl.where(total_distance > 0.0, candidate_z, old_z * inv_strength)
        if use_subset:
            should_update = active & (degree > 1) & (selected > 0)
        else:
            should_update = active & (degree > 1)
        tl.store(
            points_ptr + vertex * 3,
            tl.where(should_update, candidate_x, old_x),
            mask=active,
        )
        tl.store(
            points_ptr + vertex * 3 + 1,
            tl.where(should_update, candidate_y, old_y),
            mask=active,
        )
        tl.store(
            points_ptr + vertex * 3 + 2,
            tl.where(should_update, candidate_z, old_z),
            mask=active,
        )


@dataclass(frozen=True)
class Surf2SphereTopology:
    """保存面积平滑所需的入射三角形表和独立顶点分组。"""

    incident_faces: np.ndarray
    incident_mask: np.ndarray
    neighbours: np.ndarray
    neighbour_mask: np.ndarray
    color_groups: tuple[np.ndarray, ...]
    ordered_groups: tuple[np.ndarray, ...]

    @classmethod
    def from_mesh(cls, mesh: GiftiMesh) -> "Surf2SphereTopology":
        """根据三角形拓扑建立一次性、可复用的 GPU 索引。"""

        faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        n_points = int(mesh.vertices.shape[0])
        face_ids = np.repeat(
            np.arange(faces.shape[0], dtype=np.int64),
            3,
        )
        vertex_ids = faces.reshape(-1)
        order = np.argsort(vertex_ids, kind="stable")
        sorted_vertices = vertex_ids[order]
        sorted_faces = face_ids[order]
        counts = np.bincount(sorted_vertices, minlength=n_points)
        max_incident = max(1, int(counts.max()))
        incident_faces = np.zeros(
            (n_points, max_incident), dtype=np.int64
        )
        incident_mask = np.zeros(
            (n_points, max_incident), dtype=bool
        )
        offsets = np.concatenate(([0], np.cumsum(counts)))
        for vertex in range(n_points):
            start = int(offsets[vertex])
            stop = int(offsets[vertex + 1])
            degree = stop - start
            if degree:
                incident_faces[vertex, :degree] = sorted_faces[start:stop]
                incident_mask[vertex, :degree] = True

        neighbours, neighbour_mask = _build_neighbour_table(faces, n_points)
        color_groups = _build_color_groups(neighbours, neighbour_mask)
        ordered_groups = _build_ordered_dependency_groups(
            neighbours, neighbour_mask
        )
        return cls(
            incident_faces=incident_faces,
            incident_mask=incident_mask,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            color_groups=color_groups,
            ordered_groups=ordered_groups,
        )


@dataclass(frozen=True)
class CatSurf2SphereGpuResult:
    """保存一次 CAT_Surf2Sphere GPU 运行的输出和阶段计时。"""

    vertices: np.ndarray
    faces: np.ndarray
    timings: dict[str, float]
    output_path: Path


def surface_area(mesh: GiftiMesh) -> float:
    """用最新版 CAT 的三角形面积公式计算输入表面积。"""

    vertices = mesh.vertices.astype(np.float64, copy=False)
    faces = mesh.faces.astype(np.int64, copy=False)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return float(0.5 * np.linalg.norm(cross, axis=1).sum(dtype=np.float64))


def _face_areas_and_vertex_areas(
    points: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在设备上计算三角形面积和等分到顶点的面积。"""

    triangles = points[faces]
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    cross = torch.linalg.cross(edge0, edge1)
    areas = 0.5 * torch.sqrt(torch.sum(cross * cross, dim=1))
    # 上游只把 NaN 面积改为零；正常表面不会产生无穷面积。
    areas = torch.where(torch.isnan(areas), torch.zeros_like(areas), areas)
    vertex_areas = torch.zeros(
        points.shape[0], dtype=points.dtype, device=points.device
    )
    contribution = areas / 3.0
    for column in range(3):
        vertex_areas.index_add_(0, faces[:, column], contribution)
    return areas, vertex_areas


def _run_distance_smoothing_cuda(
    points: torch.Tensor,
    neighbours: torch.Tensor,
    degree: torch.Tensor,
    color_groups: tuple[torch.Tensor, ...],
    *,
    iterations: int,
    strength: float,
    selected: torch.Tensor | None = None,
) -> float:
    """在 CUDA 上按图着色执行最新版 CAT 的距离平滑。"""

    if triton is None:
        raise RuntimeError("请求了 CAT_Surf2Sphere Triton kernel，但 Triton 不可用")
    if iterations < 1:
        raise ValueError("距离平滑迭代次数必须大于等于 1")
    if selected is None:
        selected = torch.ones(
            points.shape[0], dtype=torch.int8, device=points.device
        )
        use_subset = False
    else:
        selected = selected.to(device=points.device, dtype=torch.int8)
        use_subset = True

    max_degree = int(neighbours.shape[1])
    start = time.perf_counter()
    for _ in range(1, iterations):
        for group in color_groups:
            if group.numel() == 0:
                continue
            grid = (triton.cdiv(group.numel(), 128),)
            _distance_smoothing_kernel[grid](
                points,
                neighbours,
                degree,
                selected,
                group,
                group.numel(),
                max_degree=max_degree,
                strength=float(strength),
                inv_strength=float(1.0 - strength),
                use_subset=use_subset,
                use_fp64=points.dtype == torch.float64,
                BLOCK=128,
                num_warps=4,
            )
    torch.cuda.synchronize(points.device)
    return time.perf_counter() - start


def _make_ordered_preprocess_groups(
    color_groups: tuple[np.ndarray, ...],
    n_points: int,
    target: torch.device,
    block_size: int | None,
) -> tuple[torch.Tensor, ...]:
    """按顶点编号分块后保留图着色，减少 Gauss–Seidel 调度重排。"""

    if block_size is None or block_size >= n_points:
        groups = color_groups
    else:
        if block_size < 1:
            raise ValueError("前处理顶点分块大小必须为正数")
        groups_list: list[np.ndarray] = []
        for start in range(0, n_points, block_size):
            stop = min(n_points, start + block_size)
            for group in color_groups:
                selected = group[(group >= start) & (group < stop)]
                if selected.size:
                    groups_list.append(selected)
        groups = tuple(groups_list)
    return tuple(
        torch.as_tensor(group, dtype=torch.int32, device=target)
        for group in groups
    )


def _compute_distortion_selection(
    points: torch.Tensor,
    reference: torch.Tensor,
    current_areas: torch.Tensor,
    reference_areas: torch.Tensor,
    neighbours: torch.Tensor,
    neighbour_mask: torch.Tensor,
    degree: torch.Tensor,
    inflated_surface_area: torch.Tensor,
    reference_surface_area: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """计算最新版 CAT 的局部压缩/拉伸判据并返回顶点选择掩码。"""

    # 上游将这些诊断数组存为 float，因此这里也在判据边界显式使用 FP32。
    current_neighbours = points[neighbours]
    reference_neighbours = reference[neighbours]
    current_delta = current_neighbours - points[:, None, :]
    reference_delta = reference_neighbours - reference[:, None, :]
    current_distance = torch.sqrt(torch.sum(current_delta * current_delta, dim=2))
    reference_distance = torch.sqrt(
        torch.sum(reference_delta * reference_delta, dim=2)
    )
    finite_reference = neighbour_mask & (reference_distance > 0.0)
    linear_ratio = torch.where(
        finite_reference,
        current_distance / reference_distance.clamp_min(1.0e-30),
        torch.zeros_like(current_distance),
    )
    max_linear = linear_ratio.to(torch.float32).amax(dim=1)

    current_positive = current_areas > 0.0
    reference_nonzero = reference_areas != 0.0
    area_ratio = torch.where(
        current_positive,
        reference_areas / current_areas.clamp_min(1.0e-30),
        torch.where(
            reference_nonzero,
            torch.full_like(reference_areas, 10000.0),
            torch.ones_like(reference_areas),
        ),
    )
    area_ratio = area_ratio.clamp_min(1.0e-8).to(torch.float32)
    surface_ratio = (
        inflated_surface_area / reference_surface_area.clamp_min(1.0e-30)
    ).to(torch.float32)
    comp_stretch = max_linear * area_ratio * surface_ratio
    neighbour_comp = comp_stretch[neighbours]
    neighbour_comp = torch.where(
        neighbour_mask, neighbour_comp, torch.zeros_like(neighbour_comp)
    )
    average_comp = (comp_stretch + neighbour_comp.sum(dim=1)) / (
        degree.to(torch.float32) + 1.0
    )
    # 度为零时上游的 compStretch 本身为零；保留同样的判定语义。
    return (average_comp > float(threshold)).to(torch.int8)


def _run_inflate_surface_stage_cuda(
    points: torch.Tensor,
    faces: torch.Tensor,
    neighbours_kernel: torch.Tensor,
    neighbours_index: torch.Tensor,
    neighbour_mask: torch.Tensor,
    degree: torch.Tensor,
    color_groups: tuple[torch.Tensor, ...],
    *,
    cycles: int,
    regular_strength: float,
    regular_iterations: int,
    inflation_factor: float,
    distortion_threshold: float,
    finger_strength: float,
    finger_iterations: int,
) -> float:
    """在 CUDA 上执行一个完整的 CAT 膨胀、诊断和手指平滑阶段。"""

    stage_start = time.perf_counter()
    reference = points.clone()
    diff_bound = reference.amax(dim=0) - reference.amin(dim=0)
    diff_bound = diff_bound.clamp_min(1.0e-20)
    points.sub_(points.mean(dim=0))

    reference_face_areas, reference_areas = _face_areas_and_vertex_areas(
        reference, faces
    )
    reference_surface_area = reference_face_areas.sum()
    delta = float(inflation_factor) - 1.0

    for cycle in range(cycles + 1):
        if cycle < cycles:
            _run_distance_smoothing_cuda(
                points,
                neighbours_kernel,
                degree,
                color_groups,
                iterations=regular_iterations,
                strength=regular_strength,
            )
            scaled = points / diff_bound[None, :]
            radius = torch.sqrt(torch.sum(scaled * scaled, dim=1))
            scale = 1.0 + delta * (1.0 - radius)
            points.mul_(scale[:, None])

        face_areas, current_areas = _face_areas_and_vertex_areas(points, faces)
        inflated_surface_area = face_areas.sum()
        selected = _compute_distortion_selection(
            points,
            reference,
            current_areas,
            reference_areas,
            neighbours_index,
            neighbour_mask,
            degree,
            inflated_surface_area,
            reference_surface_area,
            distortion_threshold,
        )

        if cycle < cycles and finger_iterations > 0:
            _run_distance_smoothing_cuda(
                points,
                neighbours_kernel,
                degree,
                color_groups,
                iterations=finger_iterations,
                strength=finger_strength,
                selected=selected,
            )

    torch.cuda.synchronize(points.device)
    return time.perf_counter() - stage_start


def run_surf2sphere_preprocess_cuda(
    vertices: np.ndarray,
    faces: np.ndarray,
    topology: Surf2SphereTopology,
    *,
    stop_at: int = 5,
    desired_surface_area: float,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = torch.float32,
    schedule_block_size: int | None = 131072,
) -> tuple[np.ndarray, dict[str, float]]:
    """把最新版 CAT_Surf2Sphere 的前五阶段显式执行在 CUDA 上。"""

    target = _resolve_device(device)
    geometry_dtype = _resolve_dtype(dtype)
    if target.type != "cuda":
        raise ValueError("当前 CAT_Surf2Sphere 前处理后端只接受 CUDA 设备")
    if triton is None:
        raise RuntimeError("请求了 CAT_Surf2Sphere Triton kernel，但 Triton 不可用")
    if stop_at < 1 or stop_at > 5:
        raise ValueError("CUDA 前处理的 stop_at 必须在 1 到 5 之间")

    with torch.no_grad():
        points = torch.as_tensor(
            vertices, dtype=geometry_dtype, device=target
        ).contiguous()
        face_table = torch.as_tensor(
            np.ascontiguousarray(faces, dtype=np.int64),
            dtype=torch.long,
            device=target,
        )
        neighbours_index = torch.as_tensor(
            np.ascontiguousarray(topology.neighbours, dtype=np.int64),
            dtype=torch.long,
            device=target,
        )
        neighbours_kernel = neighbours_index.to(torch.int32)
        neighbour_mask = torch.as_tensor(
            np.ascontiguousarray(topology.neighbour_mask, dtype=bool),
            dtype=torch.bool,
            device=target,
        )
        degree = neighbour_mask.sum(dim=1, dtype=torch.int32)
        color_groups = _make_ordered_preprocess_groups(
            topology.color_groups,
            points.shape[0],
            target,
            schedule_block_size,
        )
        n_faces = int(faces.shape[0])
        factor = (
            float(n_faces) / 350000.0 if n_faces > 350000 else 1.0
        )
        regular_iterations = lambda value: int(round(factor * value))
        timings: dict[str, float] = {}
        stages = (
            (
                1,
                "low_smooth",
                1,
                0.2,
                regular_iterations(50),
                1.0,
                3.0,
                1.0,
                0,
            ),
            (
                2,
                "inflate",
                2,
                1.0,
                regular_iterations(30),
                1.4,
                3.0,
                1.0,
                30,
            ),
            (
                3,
                "very_inflate",
                4,
                1.0,
                regular_iterations(30),
                1.1,
                3.0,
                1.0,
                0,
            ),
            (
                4,
                "high_smooth",
                6,
                1.0,
                regular_iterations(60),
                1.6,
                3.0,
                1.0,
                60,
            ),
            (
                5,
                "ellipsoid",
                6,
                1.0,
                regular_iterations(50),
                1.4,
                4.0,
                1.0,
                60,
            ),
        )
        torch.cuda.synchronize(target)
        total_start = time.perf_counter()
        for stage_index, name, cycles, strength, iterations, inflation, threshold, finger_strength, finger_iters in stages:
            if stage_index > stop_at:
                break
            timings[f"{name}_seconds"] = _run_inflate_surface_stage_cuda(
                points,
                face_table,
                neighbours_kernel,
                neighbours_index,
                neighbour_mask,
                degree,
                color_groups,
                cycles=cycles,
                regular_strength=strength,
                regular_iterations=iterations,
                inflation_factor=inflation,
                distortion_threshold=threshold,
                finger_strength=finger_strength,
                finger_iterations=finger_iters,
            )

        if stop_at >= 5:
            conversion_start = time.perf_counter()
            converted = convert_ellipsoid_to_sphere_with_surface_area(
                points.cpu().numpy(), desired_surface_area
            )
            points = torch.as_tensor(
                converted, dtype=geometry_dtype, device=target
            ).contiguous()
            torch.cuda.synchronize(target)
            timings["ellipsoid_to_sphere_seconds"] = (
                time.perf_counter() - conversion_start
            )
        torch.cuda.synchronize(target)
        timings["preprocess_total_seconds"] = time.perf_counter() - total_start
        result = points.cpu().numpy()
    return np.ascontiguousarray(result, dtype=np.float32), timings


def _resolve_device(device: str | torch.device) -> torch.device:
    """解析设备并拒绝不可用的 CUDA 后端。"""

    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CAT_Surf2Sphere CUDA 后端，但 CUDA 不可用")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
    return target


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """解析显式面积平滑精度，不启用隐式 autocast。"""

    if isinstance(dtype, torch.dtype):
        result = dtype
    else:
        normalized = str(dtype).lower().replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        try:
            result = mapping[normalized]
        except KeyError as exc:
            raise ValueError(f"不支持的 CAT_Surf2Sphere 精度: {dtype}") from exc
    if result not in (torch.float32, torch.float64):
        raise ValueError(f"CAT_Surf2Sphere 只支持 FP32/FP64: {result}")
    return result


def _write_vertices_like(
    reference_path: str | Path,
    output_path: str | Path,
    vertices: np.ndarray,
) -> None:
    """替换参考 GIFTI 的顶点数组并保留其余元数据和拓扑。"""

    import copy

    import nibabel as nib

    image = copy.deepcopy(nib.load(str(reference_path)))
    if not image.darrays:
        raise ValueError(f"参考 GIFTI 没有数据数组: {reference_path}")
    image.darrays[0].data = np.ascontiguousarray(vertices, dtype=np.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output))


def convert_ellipsoid_to_sphere_with_surface_area(
    vertices: np.ndarray,
    desired_surface_area: float,
) -> np.ndarray:
    """复现最新版 CAT 的椭球到球面投影和目标面积半径。"""

    geometry = np.ascontiguousarray(vertices, dtype=np.float32)
    radius = math.sqrt(desired_surface_area / (4.0 * math.pi))
    bounds = np.stack((geometry.min(axis=0), geometry.max(axis=0)), axis=1)
    axes = (np.abs(bounds[:, 0]) + np.abs(bounds[:, 1])) * 0.5
    axes = np.maximum(axes, 1.0e-20)
    values = geometry.astype(np.float64)
    scaled = values / axes[None, :]
    norm = np.sqrt((scaled * scaled).sum(axis=1))
    result = np.zeros_like(values)
    nonzero = norm != 0.0
    result[nonzero] = radius * values[nonzero] / norm[nonzero, None]
    result[nonzero] /= axes[None, :]
    return np.ascontiguousarray(result, dtype=np.float32)


def run_areal_smoothing_cuda(
    vertices: np.ndarray,
    faces: np.ndarray,
    topology: Surf2SphereTopology,
    *,
    iterations: int,
    project_every: int,
    radius: float,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = torch.float32,
    kernel: str = "triton",
    schedule: str = "ordered",
    arithmetic: str = "cat",
    schedule_block_size: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """用保持官方依赖顺序的并行层级执行面积平滑迭代。"""

    target = _resolve_device(device)
    geometry_dtype = _resolve_dtype(dtype)
    if target.type != "cuda":
        raise ValueError("当前 CAT_Surf2Sphere 优化后端只接受 CUDA 设备")
    if kernel not in ("torch", "triton"):
        raise ValueError(f"不支持的 CAT_Surf2Sphere kernel: {kernel}")
    if schedule not in ("ordered", "color"):
        raise ValueError(f"不支持的 CAT_Surf2Sphere schedule: {schedule}")
    if arithmetic not in ("cat", "fp32"):
        raise ValueError(f"不支持的 CAT_Surf2Sphere arithmetic: {arithmetic}")
    if kernel == "triton" and triton is None:
        raise RuntimeError("请求了 CAT_Surf2Sphere Triton kernel，但 Triton 不可用")
    use_cat_mixed = arithmetic == "cat" and geometry_dtype == torch.float32
    if iterations < 1:
        raise ValueError("面积平滑迭代次数必须大于等于 1")
    if project_every < 0:
        raise ValueError("球面投影周期不能为负数")

    with torch.no_grad():
        points = torch.as_tensor(
            vertices,
            dtype=geometry_dtype,
            device=target,
        ).contiguous()
        if kernel == "triton":
            face_table = torch.as_tensor(
                np.ascontiguousarray(faces, dtype=np.int32),
                dtype=torch.int32,
                device=target,
            )
            incident = torch.as_tensor(
                np.ascontiguousarray(topology.incident_faces, dtype=np.int32),
                dtype=torch.int32,
                device=target,
            )
            degree = torch.as_tensor(
                topology.incident_mask.sum(axis=1).astype(np.int32),
                dtype=torch.int32,
                device=target,
            )
            if schedule == "ordered":
                # 同层顶点没有相邻关系，层间顺序等价于官方原地循环。
                color_groups = tuple(
                    torch.as_tensor(group, dtype=torch.int32, device=target)
                    for group in topology.ordered_groups
                )
            else:
                color_groups = _make_ordered_preprocess_groups(
                    topology.color_groups,
                    points.shape[0],
                    target,
                    schedule_block_size,
                )
            max_incident = int(topology.incident_faces.shape[1])
        else:
            face_table = torch.as_tensor(
                np.ascontiguousarray(faces, dtype=np.int64),
                dtype=torch.long,
                device=target,
            )
            incident = torch.as_tensor(
                topology.incident_faces,
                dtype=torch.long,
                device=target,
            )
            incident_mask = torch.as_tensor(
                topology.incident_mask,
                dtype=geometry_dtype,
                device=target,
            )
            face_vertex_ids = face_table[incident]
            groups = (
                topology.ordered_groups
                if schedule == "ordered"
                else topology.color_groups
            )
            color_groups = tuple(
                torch.as_tensor(group, dtype=torch.long, device=target)
                for group in groups
            )
        radius_tensor = torch.as_tensor(
            radius, dtype=geometry_dtype, device=target
        )
        torch.cuda.synchronize(target)
        start = time.perf_counter()
        for iteration in range(1, iterations):
            for group in color_groups:
                if kernel == "triton":
                    grid = (triton.cdiv(group.numel(), 128),)
                    _areal_smoothing_kernel[grid](
                        points,
                        face_table,
                        incident,
                        degree,
                        group,
                        group.numel(),
                        max_incident=max_incident,
                        use_fp64=geometry_dtype == torch.float64,
                        use_cat_mixed=use_cat_mixed,
                        BLOCK=128,
                        num_warps=4,
                    )
                else:
                    triangle = points[face_vertex_ids[group]]
                    edge0 = triangle[:, :, 1] - triangle[:, :, 0]
                    edge1 = triangle[:, :, 2] - triangle[:, :, 0]
                    areas = 0.5 * torch.linalg.vector_norm(
                        torch.linalg.cross(edge0, edge1, dim=-1),
                        dim=-1,
                    )
                    areas = torch.clamp(areas, min=0.0, max=1.0)
                    areas = areas * incident_mask[group]
                    centers = (
                        triangle[:, :, 0]
                        + triangle[:, :, 1]
                        + triangle[:, :, 2]
                    ) / 3.0
                    total_area = areas.sum(dim=1)
                    candidate = (
                        (areas[:, :, None] * centers).sum(dim=1)
                        / total_area.clamp_min(1.0e-20)[:, None]
                    )
                    old = points[group]
                    candidate = torch.where(
                        (total_area > 0.0)[:, None], candidate, old
                    )
                    points.index_copy_(0, group, candidate)

            if project_every > 0 and iteration % project_every == 0:
                if kernel == "triton":
                    grid = (triton.cdiv(points.shape[0], 128),)
                    _normalize_points_kernel[grid](
                        points,
                        points.shape[0],
                        radius_value=float(radius),
                        use_fp64=geometry_dtype == torch.float64,
                        use_cat_mixed=use_cat_mixed,
                        BLOCK=128,
                        num_warps=4,
                    )
                else:
                    point_norm = torch.linalg.vector_norm(points, dim=1)
                    scale = radius_tensor / point_norm.clamp_min(1.0e-20)
                    points.mul_(scale[:, None])

        torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - start
        result = points.cpu().numpy()
    return np.ascontiguousarray(result, dtype=np.float32), {
        "areal_smoothing_seconds": elapsed,
        "color_groups": float(len(color_groups)),
        "ordered_schedule": float(schedule == "ordered"),
        "cat_mixed_arithmetic": float(use_cat_mixed),
        "iterations_requested": float(iterations),
        "iterations_executed": float(max(0, iterations - 1)),
    }


def run_cat_surf2sphere_gpu(
    input_surface: str | Path,
    output_surface: str | Path,
    *,
    reference_cli: str | Path,
    stop_at: int = 10,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = torch.float32,
    kernel: str = "triton",
    preprocess_kernel: str = "cpu",
    preprocess_block_size: int | None = 131072,
    areal_schedule: str = "ordered",
    areal_arithmetic: str = "cat",
    areal_block_size: int | None = None,
) -> CatSurf2SphereGpuResult:
    """运行最新版 CAT_Surf2Sphere 的显式 CPU/GPU 组合路径。"""

    if stop_at < 1:
        raise ValueError("stop_at 必须大于等于 1")
    input_path = Path(input_surface)
    output_path = Path(output_surface)
    reference_path = Path(reference_cli)
    if preprocess_kernel not in ("cpu", "triton"):
        raise ValueError(
            f"不支持的 CAT_Surf2Sphere preprocess kernel: {preprocess_kernel}"
        )
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    if preprocess_kernel == "triton":
        input_mesh = read_gifti_mesh(input_path)
        topology_start = time.perf_counter()
        topology = Surf2SphereTopology.from_mesh(input_mesh)
        timings["topology_build_seconds"] = time.perf_counter() - topology_start
        pre_vertices, pre_timings = run_surf2sphere_preprocess_cuda(
            input_mesh.vertices,
            input_mesh.faces,
            topology,
            stop_at=min(stop_at, 5),
            desired_surface_area=surface_area(input_mesh),
            device=device,
            dtype=dtype,
            schedule_block_size=preprocess_block_size,
        )
        timings.update(pre_timings)
        if stop_at <= 5:
            _write_vertices_like(input_path, output_path, pre_vertices)
            result_mesh = read_gifti_mesh(output_path)
            timings["total_seconds"] = time.perf_counter() - total_start
            return CatSurf2SphereGpuResult(
                vertices=result_mesh.vertices,
                faces=result_mesh.faces,
                timings=timings,
                output_path=output_path,
            )

        radius = float(
            np.sqrt(np.max(np.sum(pre_vertices.astype(np.float64) ** 2, axis=1)))
        )
        smoothed, smooth_timings = run_areal_smoothing_cuda(
            pre_vertices,
            input_mesh.faces,
            topology,
            iterations=1000 * (stop_at - 5),
            project_every=1000,
            radius=radius,
            device=device,
            dtype=dtype,
            kernel=kernel,
            schedule=areal_schedule,
            arithmetic=areal_arithmetic,
            schedule_block_size=areal_block_size,
        )
        timings.update(smooth_timings)
        conversion_start = time.perf_counter()
        converted = convert_ellipsoid_to_sphere_with_surface_area(
            smoothed,
            surface_area(input_mesh),
        )
        timings["ellipsoid_to_sphere_seconds"] = (
            time.perf_counter() - conversion_start
        )
        write_start = time.perf_counter()
        _write_vertices_like(input_path, output_path, converted)
        timings["output_write_seconds"] = time.perf_counter() - write_start
        result_mesh = read_gifti_mesh(output_path)
        timings["total_seconds"] = time.perf_counter() - total_start
        return CatSurf2SphereGpuResult(
            vertices=result_mesh.vertices,
            faces=result_mesh.faces,
            timings=timings,
            output_path=output_path,
        )

    if stop_at <= 5:
        start = time.perf_counter()
        completed = subprocess.run(
            [
                str(reference_path),
                str(input_path),
                str(output_path),
                str(stop_at),
                "0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        timings["reference_prefix_seconds"] = time.perf_counter() - start
        if completed.returncode != 0:
            raise RuntimeError(
                "CAT_Surf2Sphere CPU 前置失败:\n"
                + completed.stderr[-4000:]
            )
        mesh = read_gifti_mesh(output_path)
        timings["total_seconds"] = time.perf_counter() - total_start
        return CatSurf2SphereGpuResult(
            vertices=mesh.vertices,
            faces=mesh.faces,
            timings=timings,
            output_path=output_path,
        )

    with tempfile.TemporaryDirectory(prefix="cat_surface_gpu_surf2sphere_") as temp_dir:
        prefix_path = Path(temp_dir) / "stage5.gii"
        start = time.perf_counter()
        completed = subprocess.run(
            [
                str(reference_path),
                str(input_path),
                str(prefix_path),
                "5",
                "0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        timings["reference_prefix_seconds"] = time.perf_counter() - start
        if completed.returncode != 0:
            raise RuntimeError(
                "CAT_Surf2Sphere CPU 前置失败:\n"
                + completed.stderr[-4000:]
            )

        prefix_mesh = read_gifti_mesh(prefix_path)
        topology_start = time.perf_counter()
        topology = Surf2SphereTopology.from_mesh(prefix_mesh)
        timings["topology_build_seconds"] = time.perf_counter() - topology_start
        radius = float(
            np.sqrt(
                np.max(
                    np.sum(
                        prefix_mesh.vertices.astype(np.float64) ** 2,
                        axis=1,
                    )
                )
            )
        )
        smoothed, smooth_timings = run_areal_smoothing_cuda(
            prefix_mesh.vertices,
            prefix_mesh.faces,
            topology,
            iterations=1000 * (stop_at - 5),
            project_every=1000,
            radius=radius,
            device=device,
            dtype=dtype,
            kernel=kernel,
            schedule=areal_schedule,
            arithmetic=areal_arithmetic,
            schedule_block_size=areal_block_size,
        )
        timings.update(smooth_timings)
        conversion_start = time.perf_counter()
        source_mesh = read_gifti_mesh(input_path)
        converted = convert_ellipsoid_to_sphere_with_surface_area(
            smoothed,
            surface_area(source_mesh),
        )
        timings["ellipsoid_to_sphere_seconds"] = (
            time.perf_counter() - conversion_start
        )
        write_start = time.perf_counter()
        _write_vertices_like(prefix_path, output_path, converted)
        timings["output_write_seconds"] = time.perf_counter() - write_start

    result_mesh = read_gifti_mesh(output_path)
    timings["total_seconds"] = time.perf_counter() - total_start
    return CatSurf2SphereGpuResult(
        vertices=result_mesh.vertices,
        faces=result_mesh.faces,
        timings=timings,
        output_path=output_path,
    )
