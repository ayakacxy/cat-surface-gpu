"""球面特征与官方 512×256 DARTEL sheet 之间的映射。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resampling import (
    ResamplingStencil,
    apply_resampling_stencil,
    build_resampling_stencil,
    nearest_vertices_cpu,
)


@dataclass(frozen=True)
class SphereSheetMapping:
    """保存固定球面到固定 sheet 像素中心的可复用插值表。"""

    faces: np.ndarray
    stencil: ResamplingStencil
    shape: tuple[int, int]


def sheet_unit_points(shape: tuple[int, int] = (512, 256)) -> np.ndarray:
    """按官方像素中心 UV 约定生成单位球查询点。"""

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
) -> SphereSheetMapping:
    """为固定球面一次性建立最近三角形和重心权重。"""

    sphere = np.asarray(sphere_vertices, dtype=np.float64)
    lengths = np.linalg.norm(sphere, axis=1)
    unit_sphere = np.asarray(sphere / lengths[:, None], dtype=np.float32)
    targets = sheet_unit_points(shape)
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
    """使用预构建映射插值并按官方方式归一化 sheet。"""

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
    """构建映射后把单组顶点值写入二维 sheet。"""

    mapping = build_sphere_sheet_mapping(sphere_vertices, faces, shape)
    return apply_sphere_sheet_mapping(mapping, values)
