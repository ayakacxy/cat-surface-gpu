"""直接对应官方 CAT_Surf.c 的基础几何算子。"""

from __future__ import annotations

import numpy as np


def _validated_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.ascontiguousarray(vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("顶点数组必须是 (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("三角形数组必须是 (M, 3)")
    if not np.isfinite(points).all():
        raise ValueError("顶点数组包含非有限值")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= points.shape[0]):
        raise ValueError("三角形索引超出顶点范围")
    return points, triangles


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """使用官方三角形叉积公式计算每个面的面积。"""

    points, triangles = _validated_arrays(vertices, faces)
    tiles = points[triangles]
    cross = np.cross(tiles[:, 1] - tiles[:, 0], tiles[:, 2] - tiles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """计算完整网格面积，并使用 FP64 汇总。"""

    return float(face_areas(vertices, faces).sum(dtype=np.float64))


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """把每个三角形面积的三分之一累加到三个顶点。"""

    points, triangles = _validated_arrays(vertices, faces)
    areas = face_areas(points, triangles) / 3.0
    result = np.zeros(points.shape[0], dtype=np.float64)
    for column in range(3):
        np.add.at(result, triangles[:, column], areas)
    return result


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    """按官方质心平移语义把表面移到原点。"""

    points = np.ascontiguousarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("顶点数组必须是 (N, 3)")
    return points - points.mean(axis=0, dtype=np.float64)


def normalize_to_radius(vertices: np.ndarray, radius: float) -> np.ndarray:
    """把所有非零顶点沿径向投影到指定半径。"""

    points = np.ascontiguousarray(vertices, dtype=np.float64).copy()
    lengths = np.linalg.norm(points, axis=1)
    nonzero = lengths > 0.0
    points[nonzero] *= (float(radius) / lengths[nonzero])[:, None]
    return points


def ellipsoid_to_sphere(
    vertices: np.ndarray,
    desired_surface_area: float,
) -> np.ndarray:
    """移植官方 ``convert_ellipsoid_to_sphere_with_surface_area``。"""

    if desired_surface_area <= 0.0:
        raise ValueError("目标表面积必须为正数")
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
