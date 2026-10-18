"直接对应官方 CAT type-5 深度特征的表面平滑算子。"

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .topology import SurfaceTopology


def build_smoothing_topology(faces: np.ndarray, n_vertices: int) -> SurfaceTopology:
    "用 CSR 稀疏邻接生成平滑所需的无序邻点表。"

    from scipy.sparse import csr_matrix

    triangles = np.ascontiguousarray(faces, dtype=np.int32)
    rows = np.concatenate((triangles[:, 0], triangles[:, 1], triangles[:, 2]))
    columns = np.concatenate((triangles[:, 1], triangles[:, 2], triangles[:, 0]))
    directed_rows = np.concatenate((rows, columns))
    directed_columns = np.concatenate((columns, rows))
    adjacency = csr_matrix(
        (
            np.ones(directed_rows.shape[0], dtype=np.uint8),
            (directed_rows, directed_columns),
        ),
        shape=(n_vertices, n_vertices),
    )
    adjacency.sum_duplicates()
    counts = np.diff(adjacency.indptr)
    width = int(counts.max())
    neighbours = np.zeros((n_vertices, width), dtype=np.int32)
    repeated_rows = np.repeat(np.arange(n_vertices, dtype=np.int32), counts)
    starts = np.repeat(adjacency.indptr[:-1], counts)
    ranks = np.arange(adjacency.indices.size, dtype=np.int64) - starts
    neighbours[repeated_rows, ranks] = adjacency.indices
    mask = np.arange(width)[None, :] < counts[:, None]
    return SurfaceTopology(
        neighbours=neighbours,
        neighbour_mask=mask,
        incident_faces=np.empty((n_vertices, 0), dtype=np.int32),
        incident_mask=np.empty((n_vertices, 0), dtype=bool),
        color_groups=(),
    )


def _smoothing_parameters(
    vertices: np.ndarray, faces: np.ndarray, fwhm: float
) -> tuple[int, float]:
    "按官方平均面边长度公式计算迭代次数和高斯带宽。"

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.stack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=1,
    ).reshape(-1, 2)
    mean_distance = float(
        np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).mean()
    )
    iterations = max(1, math.ceil(float(fwhm) ** 2 * 0.541011 / mean_distance**2))
    sigma = float(fwhm) * 0.7355345 / math.sqrt(iterations)
    return iterations, sigma


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    "由有向面法向累加并单位化得到顶点法向。"

    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int64)
    tiles = points[triangles]
    face_normals = np.cross(tiles[:, 1] - tiles[:, 0], tiles[:, 2] - tiles[:, 0])
    normals = np.zeros_like(points)
    for column in range(3):
        np.add.at(normals, triangles[:, column], face_normals)
    lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    nonzero = lengths > 0.0
    normals[nonzero] = np.asarray(
        normals[nonzero].astype(np.float64) / lengths[nonzero, None],
        dtype=np.float32,
    )
    return normals


@njit(cache=True, parallel=True, fastmath=False)
def _smooth_positions_cpu_fused(points, neighbours, mask, denominator, iterations):
    "使用双缓冲同步平滑坐标，避免每轮构造完整邻域数组。"

    current = points.copy()
    following = np.empty_like(current)
    count = current.shape[0]
    slots = neighbours.shape[1]
    three = np.float32(3.0)
    for _ in range(iterations):
        for vertex in prange(count):
            total_x = np.float32(0.0)
            total_y = np.float32(0.0)
            total_z = np.float32(0.0)
            for slot in range(slots):
                other = neighbours[vertex, slot]
                weight = mask[vertex, slot]
                total_x += current[other, 0] * weight
                total_y += current[other, 1] * weight
                total_z += current[other, 2] * weight
            following[vertex, 0] = (current[vertex, 0] + total_x / three) / denominator[
                vertex
            ]
            following[vertex, 1] = (current[vertex, 1] + total_y / three) / denominator[
                vertex
            ]
            following[vertex, 2] = (current[vertex, 2] + total_z / three) / denominator[
                vertex
            ]
        current, following = following, current
    return current


@njit(cache=True, parallel=True, fastmath=False)
def _smooth_values_cpu_fused(values, neighbours, weights, denominator, iterations):
    "使用固定原始权重和可复用缓冲同步平滑标量。"

    current = values.copy()
    following = np.empty_like(current)
    count = current.size
    slots = neighbours.shape[1]
    for _ in range(iterations):
        for vertex in prange(count):
            neighbour_sum = 0.0
            for slot in range(slots):
                neighbour_sum += (
                    current[neighbours[vertex, slot]] * weights[vertex, slot]
                )
            following[vertex] = (current[vertex] + neighbour_sum) / denominator[vertex]
        current, following = following, current
    return current


def smooth_positions_cupy(
    vertices: np.ndarray,
    topology: SurfaceTopology,
    iterations: int,
    *,
    array_module=None,
) -> np.ndarray:
    "复现中心权重 1、邻点权重 1/3 的同步位置平滑。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    if cp is np:
        current = np.ascontiguousarray(vertices, dtype=np.float32)
        neighbours = np.ascontiguousarray(topology.neighbours, dtype=np.int32)
        mask = np.ascontiguousarray(topology.neighbour_mask, dtype=np.float32)
        denominator = np.ascontiguousarray(
            1.0 + np.sum(mask, axis=1) / 3.0, dtype=np.float32
        )
        return _smooth_positions_cpu_fused(
            current, neighbours, mask, denominator, int(iterations)
        )
    current = cp.asarray(vertices, dtype=cp.float32)
    neighbours = cp.asarray(topology.neighbours, dtype=cp.int32)
    mask = cp.asarray(topology.neighbour_mask, dtype=cp.float32)
    denominator = 1.0 + cp.sum(mask, axis=1) / 3.0
    for _ in range(int(iterations)):
        neighbour_sum = cp.sum(current[neighbours] * mask[:, :, None], axis=1)
        current = (current + neighbour_sum / 3.0) / denominator[:, None]
    return np.asarray(current) if cp is np else cp.asnumpy(current)


def _smooth_values_cupy(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    topology: SurfaceTopology,
    fwhm: float,
    *,
    array_module=None,
) -> np.ndarray:
    "按固定原表面边长执行官方高斯热核标量平滑。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    iterations, sigma = _smoothing_parameters(vertices, faces, fwhm)
    points = cp.asarray(vertices, dtype=cp.float64)
    current = cp.asarray(values, dtype=cp.float64)
    neighbours = cp.asarray(topology.neighbours, dtype=cp.int32)
    mask = cp.asarray(topology.neighbour_mask)
    distances = cp.linalg.norm(points[neighbours] - points[:, None, :], axis=2)
    weights = cp.exp(-(distances * distances) / (2.0 * sigma * sigma))
    weights = cp.where(mask, weights, 0.0)
    denominator = 1.0 + cp.sum(weights, axis=1)
    if cp is np:
        return _smooth_values_cpu_fused(
            np.ascontiguousarray(current, dtype=np.float64),
            np.ascontiguousarray(neighbours, dtype=np.int32),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(denominator, dtype=np.float64),
            int(iterations),
        )
    for _ in range(iterations):
        current = (
            current + cp.sum(current[neighbours] * weights, axis=1)
        ) / denominator
    return np.asarray(current) if cp is np else cp.asnumpy(current)


def type5_depth_feature_cupy(
    vertices: np.ndarray,
    faces: np.ndarray,
    fwhm: float,
    *,
    topology: SurfaceTopology | None = None,
    array_module=None,
) -> np.ndarray:
    "计算官方 curvtype=5 的 50 mm 深度、最终平滑和归一化。"

    points = np.ascontiguousarray(vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(faces, dtype=np.int32)
    tables = (
        topology
        if topology is not None
        else build_smoothing_topology(triangles, points.shape[0])
    )
    position_iterations, _ = _smoothing_parameters(points, triangles, 50.0)
    smoothed = smooth_positions_cupy(
        points, tables, position_iterations, array_module=array_module
    )
    normals = _vertex_normals(smoothed, triangles)
    depth = np.sum((smoothed - points) * normals, axis=1, dtype=np.float32).astype(
        np.float64
    )
    values = _smooth_values_cupy(
        points, triangles, depth, tables, float(fwhm), array_module=array_module
    )
    minimum = float(values.min())
    maximum = float(values.max())
    return (values - minimum) / (maximum - minimum)


def _face_vectors_cupy(points, faces):
    "按 CAT 面片顺序生成三条有向边。"

    triangles = points[faces]
    return (
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 1],
        triangles[:, 0] - triangles[:, 2],
    )


def _stable_normals_cupy(points, faces, array_module=None):
    "复现官方 ``stable_normals`` 的角度加权顶点法向。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    v1, v2, v3 = _face_vectors_cupy(points, faces)
    eps = cp.finfo(cp.float64).eps
    v1 = v1 / cp.maximum(cp.linalg.norm(v1, axis=1, keepdims=True), eps)
    v2 = v2 / cp.maximum(cp.linalg.norm(v2, axis=1, keepdims=True), eps)
    v3 = v3 / cp.maximum(cp.linalg.norm(v3, axis=1, keepdims=True), eps)
    cross31 = cp.cross(v3, v1)
    cross23 = cp.cross(v2, v3)
    cross12 = cp.cross(v1, v2)
    mag31 = cp.maximum(cp.linalg.norm(cross31, axis=1), eps)
    mag23 = cp.maximum(cp.linalg.norm(cross23, axis=1), eps)
    mag12 = cp.maximum(cp.linalg.norm(cross12, axis=1), eps)
    weight0 = (1.0 - cp.sum(v3 * v1, axis=1)) / mag31
    weight1 = (1.0 - cp.sum(v1 * v2, axis=1)) / mag12
    weight2 = (1.0 - cp.sum(v2 * v3, axis=1)) / mag23
    triangle_normal = cross12 / mag12[:, None]
    result = cp.zeros((points.shape[0], 3), dtype=cp.float64)
    cp.add.at(result, faces[:, 0], triangle_normal * weight0[:, None])
    cp.add.at(result, faces[:, 1], triangle_normal * weight1[:, None])
    cp.add.at(result, faces[:, 2], triangle_normal * weight2[:, None])
    return result / cp.maximum(cp.linalg.norm(result, axis=1, keepdims=True), eps)


def _mixed_voronoi_areas_cupy(points, faces, array_module=None):
    "复现官方 ``compute_areas(..., lambda=2)`` 的混合面积。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    v1, v2, v3 = _face_vectors_cupy(points, faces)
    eps = cp.finfo(cp.float64).eps
    area = cp.maximum(cp.linalg.norm(cp.cross(v1, v2), axis=1), eps)
    dot31 = cp.sum(v3 * v1, axis=1)
    dot12 = cp.sum(v1 * v2, axis=1)
    dot23 = cp.sum(v2 * v3, axis=1)
    bad0 = dot31 > 0.0
    bad1 = dot12 > 0.0
    bad2 = dot23 > 0.0
    bad = bad0 | bad1 | bad2
    len1 = cp.sum(v1 * v1, axis=1)
    len2 = cp.sum(v2 * v2, axis=1)
    len3 = cp.sum(v3 * v3, axis=1)
    weight0 = len3 * dot12 + len1 * dot23
    weight1 = len2 * cp.sum(v1 * v3, axis=1) + len1 * dot23
    weight2 = len2 * cp.sum(v1 * v3, axis=1) + len3 * dot12
    contribution = cp.full((faces.shape[0], 3), 0.125, dtype=cp.float64)
    contribution *= area[:, None]
    acute = cp.stack(
        (-0.125 * weight0 / area, -0.125 * weight1 / area, -0.125 * weight2 / area),
        axis=1,
    )
    contribution = cp.where(bad[:, None], contribution, acute)
    contribution[:, 0] += cp.where(bad0, 0.125 * area, 0.0)
    contribution[:, 1] += cp.where(bad1, 0.125 * area, 0.0)
    contribution[:, 2] += cp.where(bad2, 0.125 * area, 0.0)
    result = cp.zeros(points.shape[0], dtype=cp.float64)
    for corner in range(3):
        cp.add.at(result, faces[:, corner], contribution[:, corner])
    return cp.where((~cp.isfinite(result)) | (result == 0.0), 1.0, result)


def _depth_system_entries_cupy(points, faces, areas, alpha, array_module=None):
    "生成官方 depth-potential 的 ``0.5*L + alpha*diag(area)`` 条目。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    v1, v2, v3 = _face_vectors_cupy(points, faces)
    area = cp.maximum(
        cp.linalg.norm(cp.cross(v1, v2), axis=1), cp.finfo(cp.float64).eps
    )
    dot23 = cp.sum(v2 * v3, axis=1) / area
    dot13 = cp.sum(v1 * v3, axis=1) / area
    dot12 = cp.sum(v1 * v2, axis=1) / area
    i0, i1, i2 = faces.T
    rows = cp.concatenate((i0, i1, i2, i0, i1, i1, i2, i0, i2))
    cols = cp.concatenate((i0, i1, i2, i1, i0, i2, i1, i2, i0))
    values = cp.concatenate(
        (
            -0.5 * (dot23 + dot12),
            -0.5 * (dot13 + dot23),
            -0.5 * (dot12 + dot13),
            0.5 * dot23,
            0.5 * dot23,
            0.5 * dot13,
            0.5 * dot13,
            0.5 * dot12,
            0.5 * dot12,
        )
    )
    if alpha != 0.0:
        indices = cp.arange(points.shape[0], dtype=faces.dtype)
        rows = cp.concatenate((rows, indices))
        cols = cp.concatenate((cols, indices))
        values = cp.concatenate((values, float(alpha) * areas))
    return rows, cols, values


@njit(cache=True, fastmath=False)
def _depth_matvec_cpu_ordered(rows, cols, values, vector, n_points):
    "按原始顺序累加稀疏项，避免重复索引的 NumPy 调度开销。"

    output = np.zeros(n_points, dtype=np.float64)
    for index in range(rows.size):
        output[rows[index]] += values[index] * vector[cols[index]]
    return output


def _depth_csr_cpu_ordered(rows, cols, values, n_points):
    "按行稳定分组，保留每行浮点累加顺序。"

    order = np.argsort(rows, kind="stable")
    offsets = np.empty(n_points + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.bincount(rows, minlength=n_points), out=offsets[1:])
    return (
        offsets,
        np.ascontiguousarray(cols[order], dtype=np.int32),
        np.ascontiguousarray(values[order], dtype=np.float64),
    )


@njit(cache=True, parallel=True, fastmath=False)
def _depth_matvec_cpu_csr(offsets, cols, values, vector):
    "各行独立执行保序的 FP64 稀疏乘法。"

    n_points = offsets.size - 1
    output = np.empty(n_points, dtype=np.float64)
    for row in prange(n_points):
        total = 0.0
        for index in range(offsets[row], offsets[row + 1]):
            total += values[index] * vector[cols[index]]
        output[row] = total
    return output


def _depth_matvec_cupy(rows, cols, values, vector, n_points, array_module=None):
    "执行带重复条目的 GPU 矩阵向量乘。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    if cp is np:
        return _depth_matvec_cpu_ordered(rows, cols, values, vector, n_points)
    output = cp.zeros(n_points, dtype=cp.float64)
    cp.add.at(output, rows, values * vector[cols])
    return output


def depth_potential_feature_cupy(
    vertices: np.ndarray,
    faces: np.ndarray,
    fwhm: float = 50.0,
    *,
    alpha: float = 1.0 / 1000.0,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-9,
    topology: SurfaceTopology | None = None,
    return_raw: bool = False,
    array_module=None,
) -> np.ndarray:
    "计算官方 ``curvtype=1000`` 的 depth-potential 旋转特征。\n\n线性系统仍使用官方 cotangent Laplacian 和混合 Voronoi 面积；求解器\n采用双精度 Jacobi-PCG，保留用户已经确认的 GPU PCG 路线，最后执行\n官方 50 mm heat-kernel 和 min/max 归一化。"

    if array_module is None:
        import cupy as cp
    else:
        cp = array_module

    points = cp.asarray(np.asarray(vertices, dtype=np.float64))
    triangles = cp.asarray(np.asarray(faces, dtype=np.int32))
    n_points = int(points.shape[0])
    areas = _mixed_voronoi_areas_cupy(points, triangles, cp)
    normals = _stable_normals_cupy(points, triangles, cp)
    rows, cols, values = _depth_system_entries_cupy(
        points, triangles, areas, float(alpha), cp
    )
    lap_rows, lap_cols, lap_values = _depth_system_entries_cupy(
        points, triangles, cp.zeros_like(areas), 0.0, cp
    )
    if cp is np:
        system_csr = _depth_csr_cpu_ordered(rows, cols, values, n_points)
        laplacian_csr = _depth_csr_cpu_ordered(lap_rows, lap_cols, lap_values, n_points)
    laplacian = (
        cp.stack(
            tuple(
                _depth_matvec_cpu_csr(*laplacian_csr, points[:, axis])
                if cp is np
                else _depth_matvec_cupy(
                    lap_rows, lap_cols, lap_values, points[:, axis], n_points, cp
                )
                for axis in range(3)
            ),
            axis=1,
        )
        * 2.0
    )
    mean_curvature = 0.5 * cp.sum(laplacian * normals, axis=1) / areas
    mean_curvature = cp.where(cp.isfinite(mean_curvature), mean_curvature, 0.0)
    factor = cp.sum(mean_curvature * areas) / cp.sum(areas)
    rhs = (mean_curvature - factor) * areas

    diagonal = cp.bincount(
        rows,
        weights=cp.where(rows == cols, values, 0.0),
        minlength=n_points,
    )
    eps = cp.finfo(cp.float64).eps
    diagonal = cp.maximum(diagonal, eps)
    solution = cp.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = cp.maximum(cp.linalg.norm(rhs), eps)
    rho = cp.sum(residual * (residual / diagonal))
    direction = residual / diagonal
    for _ in range(int(max_iterations)):
        applied = (
            _depth_matvec_cpu_csr(*system_csr, direction)
            if cp is np
            else _depth_matvec_cupy(rows, cols, values, direction, n_points, cp)
        )
        denominator = cp.sum(direction * applied)
        step = cp.where(denominator != 0.0, rho / denominator, 0.0)
        solution += step * direction
        residual -= step * applied
        relative = cp.linalg.norm(residual) / rhs_norm
        if float(relative.item()) <= float(tolerance):
            break
        preconditioned = residual / diagonal
        new_rho = cp.sum(residual * preconditioned)
        direction = preconditioned + (new_rho / rho) * direction
        rho = new_rho

    values_cpu = np.asarray(solution) if cp is np else cp.asnumpy(solution)
    if return_raw:
        return np.ascontiguousarray(values_cpu, dtype=np.float64)
    tables = (
        topology
        if topology is not None
        else build_smoothing_topology(np.asarray(faces), n_points)
    )
    smoothed = _smooth_values_cupy(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        values_cpu,
        tables,
        float(fwhm),
        array_module=cp,
    )
    minimum = float(np.min(smoothed))
    maximum = float(np.max(smoothed))
    return np.ascontiguousarray(
        (smoothed - minimum) / max(maximum - minimum, np.finfo(np.float64).eps),
        dtype=np.float64,
    )
