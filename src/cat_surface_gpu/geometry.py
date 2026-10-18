"直接对应官方 CAT_Surf.c 的基础几何算子。"

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resampling import (
    ResamplingStencil,
    apply_resampling_stencil,
    build_resampling_stencil,
    nearest_vertices_cpu,
)


def _validated_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.ascontiguousarray(vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Triangles must have shape (M, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Vertices contain nonfinite values")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= points.shape[0]):
        raise ValueError("Triangle index is outside the vertex range")
    return points, triangles


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    "使用官方三角形叉积公式计算每个面的面积。"

    points, triangles = _validated_arrays(vertices, faces)
    tiles = points[triangles]
    cross = np.cross(tiles[:, 1] - tiles[:, 0], tiles[:, 2] - tiles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    "计算完整网格面积，并使用 FP64 汇总。"

    return float(face_areas(vertices, faces).sum(dtype=np.float64))


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    "把每个三角形面积的三分之一累加到三个顶点。"

    points, triangles = _validated_arrays(vertices, faces)
    areas = face_areas(points, triangles) / 3.0
    result = np.zeros(points.shape[0], dtype=np.float64)
    for column in range(3):
        np.add.at(result, triangles[:, column], areas)
    return result


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    "按官方质心平移语义把表面移到原点。"

    points = np.ascontiguousarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3)")
    return points - points.mean(axis=0, dtype=np.float64)


def normalize_to_radius(vertices: np.ndarray, radius: float) -> np.ndarray:
    "把所有非零顶点沿径向投影到指定半径。"

    points = np.ascontiguousarray(vertices, dtype=np.float64).copy()
    lengths = np.linalg.norm(points, axis=1)
    nonzero = lengths > 0.0
    points[nonzero] *= (float(radius) / lengths[nonzero])[:, None]
    return points


def ellipsoid_to_sphere(
    vertices: np.ndarray,
    desired_surface_area: float,
) -> np.ndarray:
    "移植官方 ``convert_ellipsoid_to_sphere_with_surface_area``。"

    if desired_surface_area <= 0.0:
        raise ValueError("Target surface area must be positive")
    points = np.ascontiguousarray(vertices, dtype=np.float64).copy()
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    axes = (np.abs(bounds_min) + np.abs(bounds_max)) * 0.5
    axes = np.maximum(axes, 1.0e-20)
    scaled_square = np.square(points / axes[None, :])
    factors = np.sqrt(scaled_square.sum(axis=1))
    nonzero = factors != 0.0
    points[nonzero] /= factors[nonzero, None]
    radius = np.sqrt(float(desired_surface_area) / (4.0 * np.pi))
    points *= radius / axes[None, :]
    return points


@dataclass(frozen=True)
class SphereSheetMapping:
    "保存固定球面到固定二维网格像素中心的可复用插值表。"

    faces: np.ndarray
    stencil: ResamplingStencil
    shape: tuple[int, int]


def sheet_unit_points(shape: tuple[int, int] = (512, 256)) -> np.ndarray:
    "按上游像素中心 UV 坐标生成单位球查询点。"

    width, height = map(int, shape)
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    u, v = np.meshgrid(x, y, indexing="xy")
    phi = u * (2.0 * np.pi)
    theta = v * np.pi
    points = np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )
    return np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float32)


def build_sphere_sheet_mapping(
    sphere_vertices: np.ndarray,
    faces: np.ndarray,
    shape: tuple[int, int] = (512, 256),
    *,
    backend: str = "cpu",
) -> SphereSheetMapping:
    "为固定球面一次性计算最近三角面及重心权重。"

    sphere = np.asarray(sphere_vertices, dtype=np.float64)
    lengths = np.linalg.norm(sphere, axis=1)
    unit_sphere = np.asarray(sphere / lengths[:, None], dtype=np.float32)
    targets = sheet_unit_points(shape)
    from .backends.cuda.tiled import enabled as cuda_optimized

    if backend == "cupy" and cuda_optimized():
        from .backends.cuda.resampling import build_stencil

        stencil = build_stencil(unit_sphere, faces, targets)
    elif backend == "metal":
        from .backends.metal.resampling import build_stencil

        stencil = build_stencil(unit_sphere, faces, targets)
    else:
        nearest = nearest_vertices_cpu(unit_sphere, targets, neighbours=4)
        stencil = build_resampling_stencil(
            unit_sphere,
            faces,
            targets,
            nearest_vertices=nearest,
        )
    return SphereSheetMapping(
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        stencil=stencil,
        shape=tuple(map(int, shape)),
    )


def apply_sphere_sheet_mapping(
    mapping: SphereSheetMapping,
    values: np.ndarray,
) -> np.ndarray:
    "使用准备好的映射插值，并按上游方式归一化二维网格。"

    mapped = apply_resampling_stencil(values, mapping.faces, mapping.stencil)
    minimum = float(mapped.min())
    maximum = float(mapped.max())
    width, height = mapping.shape
    return ((mapped - minimum) / (maximum - minimum)).reshape(height, width)


def map_sphere_values_to_sheet(
    sphere_vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int] = (512, 256),
) -> np.ndarray:
    "构建映射，将一组顶点值写入二维网格。"

    mapping = build_sphere_sheet_mapping(sphere_vertices, faces, shape)
    return apply_sphere_sheet_mapping(mapping, values)
