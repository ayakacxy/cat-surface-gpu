"官方 CAT 球面重采样所需的标准球和重心插值。"

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
from numba import njit, prange


_ICOSAHEDRON_FACES = np.asarray(
    [
        [2, 11, 3],
        [2, 5, 11],
        [11, 5, 10],
        [5, 0, 10],
        [0, 5, 4],
        [4, 5, 2],
        [4, 2, 9],
        [9, 2, 3],
        [9, 3, 6],
        [9, 6, 8],
        [4, 9, 8],
        [4, 8, 0],
        [0, 8, 1],
        [1, 8, 6],
        [1, 6, 7],
        [7, 6, 3],
        [11, 7, 3],
        [11, 10, 7],
        [10, 1, 7],
        [10, 0, 1],
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class ResamplingStencil:
    "保存每个目标点对应的源三角形和三个重心权重。"

    face_indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class ResamplingTopology:
    "保存只由 faces 决定、可跨多轮球面坐标复用的入射面表。"

    incident_faces: np.ndarray
    incident_counts: np.ndarray


@dataclass(frozen=True)
class SphericalResampler:
    "保存一次定位后可反复应用的完整球面重采样状态。"

    source_faces: np.ndarray
    target_vertices: np.ndarray
    target_faces: np.ndarray
    stencil: ResamplingStencil

    def apply(self, values: np.ndarray) -> np.ndarray:
        "只执行三个顶点的重心插值，不再重复空间搜索。"

        return apply_resampling_stencil(values, self.source_faces, self.stencil)


@lru_cache(maxsize=8)
def _icosphere_subdivision_topology(level):
    "缓存不可变整数拓扑；为每组半径和中心重建坐标。"
    faces = (
        _ICOSAHEDRON_FACES
        if level == 0
        else _icosphere_subdivision_topology(level - 1)[1]
    )
    old_count = 10 * (4**level) + 2
    edge_pairs = np.stack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=1,
    ).reshape(-1, 2)
    edge_pairs.sort(axis=1)
    edge_keys = edge_pairs[:, 0].astype(np.int64) * old_count + edge_pairs[:, 1].astype(
        np.int64
    )
    unique_keys, first_indices, inverse_sorted = np.unique(
        edge_keys,
        return_index=True,
        return_inverse=True,
    )
    first_order = np.argsort(first_indices, kind="stable")
    sorted_to_first = np.empty(unique_keys.shape[0], dtype=np.int32)
    sorted_to_first[first_order] = np.arange(unique_keys.shape[0], dtype=np.int32)
    midpoint_indices = (old_count + sorted_to_first[inverse_sorted]).reshape(-1, 3)
    ordered_edges = edge_pairs[first_indices[first_order]]
    p0, p1, p2 = faces.T
    m01, m12, m20 = midpoint_indices.T
    new_faces = np.empty((faces.shape[0], 4, 3), dtype=np.int32)
    new_faces[:, 0] = np.stack((p0, m01, m20), axis=1)
    new_faces[:, 1] = np.stack((m01, p1, m12), axis=1)
    new_faces[:, 2] = np.stack((m20, m01, m12), axis=1)
    new_faces[:, 3] = np.stack((m20, m12, p2), axis=1)
    faces = new_faces.reshape(-1, 3)
    ordered_edges.flags.writeable = False
    faces.flags.writeable = False
    return ordered_edges, faces


def create_icosphere(
    subdivisions: int = 6,
    *,
    radius: float = 1.0,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    "按 BICPL 的顶点编号和逐面边首次出现顺序生成二十面体球。"

    if subdivisions < 0:
        raise ValueError("Subdivision count must not be negative")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Sphere radius must be finite and positive")
    a = np.sqrt((5.0 + np.sqrt(5.0)) / 10.0)
    b = np.sqrt((5.0 - np.sqrt(5.0)) / 10.0)
    vertices = np.asarray(
        [
            [a, b, 0.0],
            [a, -b, 0.0],
            [-a, b, 0.0],
            [-a, -b, 0.0],
            [0.0, a, b],
            [0.0, a, -b],
            [0.0, -a, b],
            [0.0, -a, -b],
            [b, 0.0, a],
            [-b, 0.0, a],
            [b, 0.0, -a],
            [-b, 0.0, -a],
        ],
        dtype=np.float32,
    )
    faces = _ICOSAHEDRON_FACES.copy()
    origin = np.zeros(3, dtype=np.float64)
    if center is not None:
        origin = np.asarray(center, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Sphere center must contain three finite coordinates")
    # BICPL Point 坐标为 float，每次顶点写回均舍入到 FP32。
    vertices = np.asarray(
        origin[None, :] + float(radius) * vertices.astype(np.float64),
        dtype=np.float32,
    )
    for level in range(subdivisions):
        ordered_edges, new_faces = _icosphere_subdivision_topology(level)
        midpoint_values = np.asarray(
            (
                vertices[ordered_edges[:, 0]].astype(np.float64)
                + vertices[ordered_edges[:, 1]].astype(np.float64)
            )
            * 0.5,
            dtype=np.float32,
        )
        vertices = np.concatenate((vertices, midpoint_values), axis=0)
        displacement = vertices.astype(np.float64) - origin[None, :]
        scale = float(radius) / np.linalg.norm(displacement, axis=1)
        vertices = np.asarray(
            origin[None, :] + displacement * scale[:, None], dtype=np.float32
        )
        faces = new_faces
    return vertices, faces.copy()


def cat_resampling_sphere(
    source_sphere_vertices: np.ndarray,
    subdivisions: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    "复现官方重采样球的平均半径、0.975 缩放和边界球心语义。"

    points = np.ascontiguousarray(source_sphere_vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Source sphere vertices must be a finite (N, 3) array")
    radius = float(np.linalg.norm(points, axis=1).mean(dtype=np.float64)) * 0.975
    # 当前上游直接使用边界之和而不除以二；保留此版本特有行为。
    center = points.min(axis=0) + points.max(axis=0)
    return create_icosphere(subdivisions, radius=radius, center=center)


@njit(cache=True)
def _incident_faces_ordered(faces, n_vertices):
    "按稳定面顺序计数和散布整数索引，无需排序。"
    counts = np.zeros(n_vertices, dtype=np.int32)
    for face in range(len(faces)):
        for corner in range(3):
            counts[faces[face, corner]] += 1
    width = int(counts.max()) if n_vertices else 0
    table = np.full((n_vertices, width), -1, dtype=np.int32)
    cursors = np.zeros(n_vertices, dtype=np.int32)
    for face in range(len(faces)):
        for corner in range(3):
            vertex = faces[face, corner]
            table[vertex, cursors[vertex]] = face
            cursors[vertex] += 1
    return table, counts


def _vertex_incident_faces(
    faces: np.ndarray, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    "生成定宽的顶点入射面表。"
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    if n_vertices < 0 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Incident topology requires triangles and a nonnegative vertex count"
        )
    if faces.size and (faces.min() < 0 or faces.max() >= n_vertices):
        raise ValueError("Incident topology vertex index is out of range")
    return _incident_faces_ordered(faces, n_vertices)


def nearest_vertices_cpu(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    *,
    neighbours: int = 4,
    workers: int = 8,
) -> np.ndarray:
    "使用 SciPy KD-tree 并行寻找多个最近源顶点。"

    from scipy.spatial import cKDTree

    if neighbours < 1:
        raise ValueError("Neighbor count must be positive")
    tree = cKDTree(np.asarray(source_vertices, dtype=np.float64))
    _, indices = tree.query(
        np.asarray(target_vertices, dtype=np.float64),
        k=int(neighbours),
        workers=min(max(int(workers), 1), 8),
    )
    result = np.asarray(indices, dtype=np.int32)
    if result.ndim == 1:
        result = result[:, None]
    return np.ascontiguousarray(result)


def nearest_vertices_cupy(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    "用分块矩阵乘法在 GPU 上寻找每个目标点最近的源顶点。"

    import cupy as cp

    source = cp.asarray(source_vertices, dtype=cp.float32)
    target = cp.asarray(target_vertices, dtype=cp.float32)
    source_norm = cp.sum(source * source, axis=1)
    result = np.empty(target.shape[0], dtype=np.int32)
    for start in range(0, target.shape[0], batch_size):
        stop = min(start + batch_size, target.shape[0])
        query = target[start:stop]
        distances = (
            cp.sum(query * query, axis=1)[:, None]
            + source_norm[None, :]
            - 2.0 * (query @ source.T)
        )
        result[start:stop] = cp.asnumpy(cp.argmin(distances, axis=1)).astype(
            np.int32, copy=False
        )
    return result


@njit(cache=True)
def _closest_triangle_weights(point, a, b, c):
    "返回点到三角形最近点的平方距离和重心权重。"

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.dot(ap, ap), 1.0, 0.0, 0.0
    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.dot(bp, bp), 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        closest = a + v * ab
        delta = point - closest
        return np.dot(delta, delta), 1.0 - v, v, 0.0
    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.dot(cp, cp), 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        closest = a + w * ac
        delta = point - closest
        return np.dot(delta, delta), 1.0 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = b + w * (c - b)
        delta = point - closest
        return np.dot(delta, delta), 0.0, 1.0 - w, w
    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    closest = a + ab * v + ac * w
    delta = point - closest
    return np.dot(delta, delta), 1.0 - v - w, v, w


# 保留独立编译函数；ARM 强制内联会改变距离舍入及最近面平局选择。
@njit(cache=True, inline="never", fastmath=False)
def _closest_triangle_weights_scratch(point, a, b, c, scratch):
    "复用六个三维向量，保留 BLAS 点积、区域判断和运算顺序。"
    ab, ac, ap, bp, cp, delta = (
        scratch[0],
        scratch[1],
        scratch[2],
        scratch[3],
        scratch[4],
        scratch[5],
    )
    for axis in range(3):
        ab[axis] = b[axis] - a[axis]
        ac[axis] = c[axis] - a[axis]
        ap[axis] = point[axis] - a[axis]
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.dot(ap, ap), 1.0, 0.0, 0.0
    for axis in range(3):
        bp[axis] = point[axis] - b[axis]
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.dot(bp, bp), 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        for axis in range(3):
            closest = a[axis] + v * ab[axis]
            delta[axis] = point[axis] - closest
        return np.dot(delta, delta), 1.0 - v, v, 0.0
    for axis in range(3):
        cp[axis] = point[axis] - c[axis]
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.dot(cp, cp), 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        for axis in range(3):
            closest = a[axis] + w * ac[axis]
            delta[axis] = point[axis] - closest
        return np.dot(delta, delta), 1.0 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        for axis in range(3):
            closest = b[axis] + w * (c[axis] - b[axis])
            delta[axis] = point[axis] - closest
        return np.dot(delta, delta), 0.0, 1.0 - w, w
    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    for axis in range(3):
        closest = a[axis] + ab[axis] * v + ac[axis] * w
        delta[axis] = point[axis] - closest
    return np.dot(delta, delta), 1.0 - v - w, v, w


@njit(cache=True, parallel=True)
def _stencils_from_nearest(
    source_vertices,
    faces,
    target_vertices,
    nearest_vertices,
    incident_faces,
    incident_counts,
    reuse_scratch=False,
):
    n_targets = target_vertices.shape[0]
    selected = np.empty(n_targets, dtype=np.int32)
    weights = np.empty((n_targets, 3), dtype=np.float64)
    for index in prange(n_targets):
        # 每个查询使用独立临时缓冲，并在候选三角形间复用，避免并行写入。
        scratch = np.empty((6, 3), dtype=np.float64)
        best_distance = np.inf
        best_face = -1
        best_w0 = 0.0
        best_w1 = 0.0
        best_w2 = 0.0
        for neighbour_index in range(nearest_vertices.shape[1]):
            vertex = nearest_vertices[index, neighbour_index]
            for candidate_index in range(incident_counts[vertex]):
                face_index = incident_faces[vertex, candidate_index]
                face = faces[face_index]
                if reuse_scratch:
                    distance, w0, w1, w2 = _closest_triangle_weights_scratch(
                        target_vertices[index],
                        source_vertices[face[0]],
                        source_vertices[face[1]],
                        source_vertices[face[2]],
                        scratch,
                    )
                else:
                    distance, w0, w1, w2 = _closest_triangle_weights(
                        target_vertices[index],
                        source_vertices[face[0]],
                        source_vertices[face[1]],
                        source_vertices[face[2]],
                    )
                if distance < best_distance:
                    best_distance = distance
                    best_face = face_index
                    best_w0, best_w1, best_w2 = w0, w1, w2
        selected[index] = best_face
        weights[index, 0] = best_w0
        weights[index, 1] = best_w1
        weights[index, 2] = best_w2
    return selected, weights


def build_resampling_stencil(
    source_sphere_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    *,
    nearest_vertices: np.ndarray | None = None,
    topology: ResamplingTopology | None = None,
) -> ResamplingStencil:
    "从最近源顶点的入射面中建立官方语义的最近面插值表。"

    source = np.ascontiguousarray(source_sphere_vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(source_faces, dtype=np.int32)
    target = np.ascontiguousarray(target_vertices, dtype=np.float64)
    if nearest_vertices is None:
        differences = target[:, None, :] - source[None, :, :]
        nearest = np.argmin(np.sum(differences * differences, axis=2), axis=1)[:, None]
    else:
        nearest = np.ascontiguousarray(nearest_vertices, dtype=np.int32)
        if nearest.ndim == 1:
            nearest = nearest[:, None]
    if nearest.ndim != 2 or nearest.shape[0] != target.shape[0]:
        raise ValueError("Nearest-vertex array length differs from target point count")
    tables = (
        topology
        if topology is not None
        else build_resampling_topology(triangles, source.shape[0])
    )
    face_indices, weights = _stencils_from_nearest(
        source,
        triangles,
        target,
        nearest,
        tables.incident_faces,
        tables.incident_counts,
        os.environ.get("CAT_SURFACE_STENCIL_SCRATCH", "1") == "1",
    )
    return ResamplingStencil(face_indices=face_indices, weights=weights)


def build_resampling_topology(
    source_faces: np.ndarray,
    n_vertices: int,
    *,
    binary_cache: bool = False,
) -> ResamplingTopology:
    "预构建可跨多轮变形球复用的顶点入射面表。"

    if binary_cache:
        from .templates import fixed_resampling_topology

        return fixed_resampling_topology(source_faces, n_vertices)
    incident_faces, incident_counts = _vertex_incident_faces(
        np.ascontiguousarray(source_faces, dtype=np.int32), int(n_vertices)
    )
    return ResamplingTopology(incident_faces, incident_counts)


def prepare_spherical_resampler(
    source_sphere_vertices: np.ndarray,
    source_faces: np.ndarray,
    *,
    subdivisions: int = 6,
    nearest_neighbours: int = 4,
    topology: ResamplingTopology | None = None,
    backend: str = "cpu",
) -> SphericalResampler:
    "一次性生成目标球、最近面和权重，供多个表面及标量复用。"

    target_vertices, target_faces = cat_resampling_sphere(
        source_sphere_vertices, subdivisions
    )
    from cat_surface_gpu.backends.cuda.tiled import enabled

    if backend == "cupy" and enabled() and nearest_neighbours == 4:
        from cat_surface_gpu.backends.cuda.resampling import build_stencil

        stencil = build_stencil(
            source_sphere_vertices, source_faces, target_vertices, topology
        )
    elif backend == "metal" and nearest_neighbours == 4:
        from .backends.metal.resampling import build_stencil

        stencil = build_stencil(
            source_sphere_vertices, source_faces, target_vertices, topology
        )
    else:
        nearest = nearest_vertices_cpu(
            source_sphere_vertices, target_vertices, neighbours=nearest_neighbours
        )
        stencil = build_resampling_stencil(
            source_sphere_vertices,
            source_faces,
            target_vertices,
            nearest_vertices=nearest,
            topology=topology,
        )
    return SphericalResampler(
        source_faces=np.ascontiguousarray(source_faces, dtype=np.int32),
        target_vertices=target_vertices,
        target_faces=target_faces,
        stencil=stencil,
    )


def apply_resampling_stencil(
    values: np.ndarray,
    faces: np.ndarray,
    stencil: ResamplingStencil,
) -> np.ndarray:
    "把源顶点标量或向量按给定 stencil 插值到目标球。"

    source_values = np.asarray(values)
    triangles = np.asarray(faces, dtype=np.int64)
    selected = triangles[np.asarray(stencil.face_indices, dtype=np.int64)]
    gathered = source_values[selected]
    weights = np.asarray(stencil.weights, dtype=np.float64)
    if source_values.ndim == 1:
        return np.sum(gathered * weights, axis=1)
    return np.sum(gathered * weights[:, :, None], axis=1)
