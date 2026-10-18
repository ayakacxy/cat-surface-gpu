"CAT 表面的批量 FP32 特征平滑。"

from functools import lru_cache
import numpy as np
from .surf2sphere import _mlx

_SOURCE = r"""
uint i=thread_position_in_grid.x,n=uint(current_shape[0]),channels=uint(current_shape[1]);
if(i>=n*channels)return;uint row=i/channels,channel=i%channels,slots=uint(neighbours_shape[1]);
float center=current[i],change=0.0f;
for(uint k=0;k<slots;++k){
 uint j=row*slots+k;float weight=weights[j];
 change+=(current[uint(neighbours[j])*channels+channel]-center)*weight;
}
// 增量形式精确保留常量特征，即使扩散迭代很长。
output[i]=center+change/denominator[row];
"""


@lru_cache(maxsize=1)
def _kernel():
    mx = _mlx()
    return mx.fast.metal_kernel(
        name="cat_feature_smooth_fp32",
        input_names=["current", "neighbours", "weights", "denominator"],
        output_names=["output"],
        source=_SOURCE,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=8)
def _batch(iterations):
    mx = _mlx()
    kernel = _kernel()

    @mx.compile
    def run(current, neighbours, weights, denominator):
        for _ in range(iterations):
            current = kernel(
                inputs=[current, neighbours, weights, denominator],
                output_shapes=[current.shape],
                output_dtypes=[mx.float32],
                grid=(current.size, 1, 1),
                threadgroup=(128, 1, 1),
                stream=mx.gpu,
            )[0]
        return current

    return run


def smooth(values, neighbours, weights, iterations):
    "使用固定边权和有界 GPU 图，对一个或多个通道进行扩散。"
    mx = _mlx()
    values = np.ascontiguousarray(values, np.float32)
    original_shape = values.shape
    weights = np.ascontiguousarray(weights, np.float32)
    neighbours = np.ascontiguousarray(neighbours, np.int32)
    if (
        values.ndim not in (1, 2)
        or len(values) < 1
        or neighbours.ndim != 2
        or weights.shape != neighbours.shape
        or len(neighbours) != len(values)
        or iterations < 0
    ):
        raise ValueError(
            "CAT smoothing requires matching nonempty vertex and neighbour arrays"
        )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.any(neighbours < 0)
        or np.any(neighbours >= len(values))
    ):
        raise ValueError(
            "CAT smoothing requires finite values, nonnegative weights and valid indices"
        )
    with mx.stream(mx.gpu):
        current = mx.array(values.reshape(len(values), -1))
        adjacency, coefficients = mx.array(neighbours), mx.array(weights)
        denominator = 1 + mx.sum(coefficients, axis=1)
        for start in range(0, int(iterations), 50):
            current = _batch(min(50, int(iterations) - start))(
                current, adjacency, coefficients, denominator
            )
            mx.eval(current)
        output = np.array(current).reshape(original_shape)
    return output


def smooth_values(vertices, faces, values, topology, fwhm):
    from ...features import _smoothing_parameters

    iterations, sigma = _smoothing_parameters(vertices, faces, fwhm)
    points = np.asarray(vertices, np.float64)
    distance = points[topology.neighbours] - points[:, None, :]
    weights = np.exp(-np.sum(distance * distance, axis=2) / (2 * sigma * sigma))
    weights = np.where(topology.neighbour_mask, weights, 0)
    return smooth(values, topology.neighbours, weights, iterations).astype(np.float64)


def type5_depth_features(vertices, faces, fwhms, topology):
    "跨请求的特征尺度复用依赖几何的深度场。"
    from ...features import _smoothing_parameters, _vertex_normals

    points = np.ascontiguousarray(vertices, np.float32)
    triangles = np.ascontiguousarray(faces, np.int32)
    iterations, _ = _smoothing_parameters(points, triangles, 50.0)
    positions = smooth(
        points,
        topology.neighbours,
        topology.neighbour_mask.astype(np.float32) / 3,
        iterations,
    )
    normals = _vertex_normals(positions, triangles)
    depth = np.sum((positions - points) * normals, axis=1, dtype=np.float32).astype(
        np.float64
    )
    result = []
    for fwhm in fwhms:
        values = smooth_values(points, triangles, depth, topology, fwhm)
        minimum, maximum = values.min(), values.max()
        result.append(
            (values - minimum) / max(maximum - minimum, np.finfo(np.float64).eps)
        )
    return result


_CSR = r"""
uint i=thread_position_in_grid.x,n=uint(vector_shape[0]);if(i>=n)return;
float value=0.0f;for(int k=offsets[i];k<offsets[i+1];++k)value+=entries[k]*vector[columns[k]];
output[i]=value;
"""


@lru_cache(maxsize=1)
def _csr_kernel():
    mx = _mlx()
    return mx.fast.metal_kernel(
        name="cat_depth_csr_fp32",
        input_names=["offsets", "columns", "entries", "vector"],
        output_names=["output"],
        source=_CSR,
        compile_options={"math_mode": "safe"},
    )


def _csr_apply(offsets, columns, entries, vector):
    mx = _mlx()
    return _csr_kernel()(
        inputs=[offsets, columns, entries, vector],
        output_shapes=[vector.shape],
        output_dtypes=[mx.float32],
        grid=(vector.size, 1, 1),
        threadgroup=(128, 1, 1),
        stream=mx.gpu,
    )[0]


@lru_cache(maxsize=4)
def _depth_batch(iterations):
    mx = _mlx()

    @mx.compile
    def run(offsets, columns, entries, solution, residual, direction, rho):
        for _ in range(iterations):
            applied = _csr_apply(offsets, columns, entries, direction)
            alpha = rho / mx.maximum(mx.sum(direction * applied), 1e-30)
            solution = solution + alpha * direction
            residual = residual - alpha * applied
            next_rho = mx.sum(residual * residual)
            direction = residual + next_rho / mx.maximum(rho, 1e-30) * direction
            rho = next_rho
        return solution, residual, direction, rho

    return run


def solve_depth(system, rhs, max_iterations=1000, *, diagnostics=None):
    "求解对称平衡化的稀疏系统，并检查真实残差。"
    mx = _mlx()
    if max_iterations < 1:
        raise ValueError("CAT depth solve requires a positive iteration budget")
    if not np.isfinite(rhs).all() or not np.isfinite(system.data).all():
        raise ValueError("CAT depth system must be finite")
    if not np.any(rhs):
        if diagnostics is not None:
            diagnostics.update(iterations=0, relative_residual=0.0)
        return np.zeros_like(rhs, dtype=np.float64)
    diagonal = system.diagonal()
    if np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
        raise ValueError("CAT depth system requires finite positive diagonal entries")
    scale = 1 / np.sqrt(diagonal)
    rows = np.repeat(np.arange(len(rhs)), np.diff(system.indptr))
    values = system.data * scale[rows] * scale[system.indices]
    scaled_rhs = rhs * scale
    magnitude = max(float(np.max(np.abs(scaled_rhs))), 1e-30)
    with mx.stream(mx.gpu):
        offsets, columns = (
            mx.array(system.indptr.astype(np.int32)),
            mx.array(system.indices.astype(np.int32)),
        )
        entries = mx.array(values.astype(np.float32))
        b = mx.array((scaled_rhs / magnitude).astype(np.float32))
        solution = mx.zeros_like(b)
        residual = b
        direction = b
        rho = mx.sum(b * b)
        initial = rho
        for start in range(0, max_iterations, 50):
            solution, residual, direction, rho = _depth_batch(
                min(50, max_iterations - start)
            )(offsets, columns, entries, solution, residual, direction, rho)
            # 重算真实残差，防止 FP32 假收敛。
            true_residual = b - _csr_apply(offsets, columns, entries, solution)
            relative = mx.sqrt(
                mx.sum(true_residual * true_residual) / mx.maximum(initial, 1e-30)
            )
            mx.eval(solution, residual, direction, rho, relative)
            if float(relative.item()) <= 3e-6:
                break
            if float((rho / mx.maximum(initial, 1e-30)).item()) < 1e-12:
                residual = true_residual
                direction = residual
                rho = mx.sum(residual * residual)
        result = np.array(solution).astype(np.float64) * magnitude * scale
    relative_residual = float(
        np.linalg.norm(system @ result - rhs) / max(np.linalg.norm(rhs), 1e-30)
    )
    if (
        not np.isfinite(result).all()
        or not np.isfinite(relative_residual)
        or relative_residual > 1e-4
    ):
        raise FloatingPointError("CAT FP32 depth solve failed its true-residual check")
    if diagnostics is not None:
        diagnostics.update(
            iterations=min(start + 50, max_iterations),
            relative_residual=relative_residual,
        )
    return result


def depth_potential_feature(vertices, faces, fwhm=50.0, *, topology, alpha=0.001):
    "使用 FP64 装配几何，以 Metal FP32 求解并扩散特征。"
    from scipy.sparse import csr_matrix
    from ...features import (
        _mixed_voronoi_areas_cupy,
        _stable_normals_cupy,
        _depth_system_entries_cupy,
    )

    points = np.asarray(vertices, np.float64)
    triangles = np.asarray(faces, np.int32)
    n = len(points)
    areas = _mixed_voronoi_areas_cupy(points, triangles, np)
    normals = _stable_normals_cupy(points, triangles, np)
    rows, cols, values = _depth_system_entries_cupy(points, triangles, areas, 0.0, np)
    laplacian = csr_matrix((values, (rows, cols)), shape=(n, n))
    curvature = np.sum(laplacian @ points * normals, axis=1) / areas
    curvature = np.where(np.isfinite(curvature), curvature, 0)
    factor = np.sum(curvature * areas) / np.sum(areas)
    rhs = (curvature - factor) * areas
    system = laplacian.copy()
    system.setdiag(system.diagonal() + alpha * areas)
    raw = solve_depth(system, rhs)
    values = smooth_values(points, triangles, raw, topology, fwhm)
    minimum, maximum = values.min(), values.max()
    return (values - minimum) / max(maximum - minimum, np.finfo(np.float64).eps)
