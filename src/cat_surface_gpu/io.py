"""不依赖 CAT 原生工具的 GIFTI 表面读写。"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class GiftiSurface:
    """保存表面顶点、三角形以及可选的参考 GIFTI。"""

    vertices: np.ndarray
    faces: np.ndarray
    reference: nib.gifti.GiftiImage | None = None

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("顶点数组必须是 (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("三角形数组必须是 (M, 3)")
        if len(vertices) < 4 or len(faces) < 4:
            raise ValueError("闭合表面至少需要四个顶点和四个三角形")
        if not np.isfinite(vertices).all():
            raise ValueError("顶点数组包含非有限值")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("三角形索引必须是整数")
        if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
            raise ValueError("三角形索引超出顶点范围")
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise ValueError("三角形包含重复顶点索引")
        triangles = np.asarray(vertices, dtype=np.float64)[faces]
        with np.errstate(over="ignore", invalid="ignore"):
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
        zero_count = int(np.count_nonzero(np.all(normals == 0.0, axis=1)))
        nonfinite_count = int(np.count_nonzero(~np.isfinite(normals).all(axis=1)))
        if zero_count or nonfinite_count:
            raise ValueError(
                f"网格包含退化三角形：{zero_count} 个面积为零，"
                f"{nonfinite_count} 个面积非有限；请检查输入几何或球面生成结果"
            )


def load_gifti_surface(path: str | Path) -> GiftiSurface:
    """按 NIFTI intent 读取 GIFTI 点集和三角形。"""

    image = nib.load(str(path))
    if not isinstance(image, nib.gifti.GiftiImage):
        raise ValueError(f"仅支持 GIFTI 表面文件：{path}")
    pointset = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    triangles = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if len(pointset) != 1 or len(triangles) != 1:
        raise ValueError(f"GIFTI 必须各含一个点集和三角形数组：{path}")
    return GiftiSurface(
        vertices=np.ascontiguousarray(pointset[0].data, dtype=np.float32),
        faces=np.ascontiguousarray(triangles[0].data),
        reference=image,
    )


def save_gifti_surface(
    surface: GiftiSurface,
    path: str | Path,
    *,
    reference: nib.gifti.GiftiImage | None = None,
) -> None:
    """用 nibabel 写出标准 pointset/triangle GIFTI。"""

    template = reference if reference is not None else surface.reference
    metadata = nib.gifti.GiftiMetaData()
    label_table = nib.gifti.GiftiLabelTable()
    if template is not None:
        metadata = template.meta
        label_table = template.labeltable
    point_array = nib.gifti.GiftiDataArray(
        np.ascontiguousarray(surface.vertices, dtype=np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    triangle_array = nib.gifti.GiftiDataArray(
        np.ascontiguousarray(surface.faces, dtype=np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    )
    if template is not None:
        points = template.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        triangles = template.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
        if len(points) == 1:
            point_array.coordsys = copy.deepcopy(points[0].coordsys)
            point_array.meta = copy.deepcopy(points[0].meta)
        if len(triangles) == 1:
            triangle_array.coordsys = copy.deepcopy(triangles[0].coordsys)
            triangle_array.meta = copy.deepcopy(triangles[0].meta)
    output = nib.gifti.GiftiImage(
        meta=metadata,
        labeltable=label_table,
        darrays=[point_array, triangle_array],
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(destination))
