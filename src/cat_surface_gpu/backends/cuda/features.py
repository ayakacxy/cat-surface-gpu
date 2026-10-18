"保留 FP64 CAT 几何装配，采用平衡化 FP32 求解和批量扩散。"

from functools import lru_cache
import numpy as np

_SOURCE = r"""
extern "C" __global__ void smooth(const float* current,const int* neighbours,
 const float* weights,const float* denominator,float* output,int n,int channels,int slots){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*channels)return;
 int row=i/channels,channel=i%channels;float center=current[i],change=0;
 for(int k=0;k<slots;++k){int j=row*slots+k;change+=(current[neighbours[j]*channels+channel]-center)*weights[j];}
 output[i]=center+change/denominator[row];
}
extern "C" __global__ void csr(const int* offsets,const int* columns,const float* entries,
 const float* vector,float* output,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float value=0;for(int k=offsets[i];k<offsets[i+1];++k)value+=entries[k]*vector[columns[k]];
 output[i]=value;
}
extern "C" __global__ void cg_update(float* solution,float* residual,const float* direction,
 const float* applied,const float* rho,const float* denominator,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float step=rho[0]/fmaxf(denominator[0],1e-30f);
 solution[i]+=step*direction[i];residual[i]-=step*applied[i];
}
extern "C" __global__ void cg_direction(float* direction,const float* residual,
 const float* rho,const float* next_rho,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 direction[i]=residual[i]+next_rho[0]/fmaxf(rho[0],1e-30f)*direction[i];
}
"""


@lru_cache(maxsize=8)
def _kernels(device):
    "捕获计算图前编译内核。"
    import cupy as cp

    module = cp.RawModule(code=_SOURCE, options=("--std=c++11", "--fmad=false"))
    return {
        name: module.get_function(name)
        for name in ("smooth", "csr", "cg_update", "cg_direction")
    }


def smooth(values, neighbours, weights, iterations):
    "通过增量扩散和双缓冲图重放精确保留常量场。"
    import cupy as cp

    values = np.ascontiguousarray(values, np.float32)
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
        raise ValueError("Invalid CAT smoothing dimensions or iteration count")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.any(neighbours < 0)
        or np.any(neighbours >= len(values))
    ):
        raise ValueError("Invalid CAT smoothing values or indices")
    shape = values.shape
    current = cp.asarray(values.reshape(len(values), -1))
    scratch = cp.empty_like(current)
    adjacency, coefficients = cp.asarray(neighbours), cp.asarray(weights)
    denominator = 1 + cp.sum(coefficients, axis=1)
    kernel = _kernels(cp.cuda.Device().id)["smooth"]

    def step(first, second):
        kernel(
            ((current.size + 127) // 128,),
            (128,),
            (
                first,
                adjacency,
                coefficients,
                denominator,
                second,
                np.int32(len(values)),
                np.int32(current.shape[1]),
                np.int32(neighbours.shape[1]),
            ),
        )

    stream = cp.cuda.Stream(non_blocking=True)
    stream.wait_event(cp.cuda.get_current_stream().record())
    with stream:
        batch = min(50, int(iterations) - int(iterations) % 2)
        if batch:
            stream.begin_capture()
            for _ in range(batch // 2):
                step(current, scratch)
                step(scratch, current)
            graph = stream.end_capture()
            for _ in range(int(iterations) // batch):
                graph.launch(stream)
        remaining = int(iterations) % batch if batch else int(iterations)
        for _ in range(remaining // 2):
            step(current, scratch)
            step(scratch, current)
        if remaining % 2:
            step(current, scratch)
            current = scratch
        result = cp.asnumpy(current).reshape(shape)
    return result


def solve_depth(system, rhs, max_iterations=1000, *, diagnostics=None):
    "平衡稀疏系统，每五十步检查真实残差，并在结束时检查原始 FP64 系统。"
    import cupy as cp

    rhs = np.asarray(rhs, np.float64)
    system = system.tocsr(copy=True)
    system.sum_duplicates()
    if (
        max_iterations < 1
        or rhs.ndim != 1
        or not len(rhs)
        or system.shape != (len(rhs), len(rhs))
        or not np.isfinite(rhs).all()
        or not np.isfinite(system.data).all()
    ):
        raise ValueError("Invalid CAT depth system")
    if not np.any(rhs):
        if diagnostics is not None:
            diagnostics.update(iterations=0, relative_residual=0.0)
        return np.zeros_like(rhs)
    diagonal = system.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("CAT depth system requires a positive diagonal")
    scale = 1 / np.sqrt(diagonal)
    rows = np.repeat(np.arange(len(rhs)), np.diff(system.indptr))
    values = system.data * scale[rows] * scale[system.indices]
    scaled_rhs = rhs * scale
    magnitude = max(float(np.max(np.abs(scaled_rhs))), 1e-30)
    kernels = _kernels(cp.cuda.Device().id)
    stream = cp.cuda.Stream(non_blocking=True)
    stream.wait_event(cp.cuda.get_current_stream().record())
    with stream:
        offsets = cp.asarray(system.indptr, dtype=cp.int32)
        columns = cp.asarray(system.indices, dtype=cp.int32)
        entries = cp.asarray(values, dtype=cp.float32)
        b = cp.asarray(scaled_rhs / magnitude, dtype=cp.float32)
        solution = cp.zeros_like(b)
        residual, direction = b.copy(), b.copy()
        applied, products = cp.empty_like(b), cp.empty_like(b)
        rho = cp.sum(b * b)
        denominator, next_rho = cp.empty_like(rho), cp.empty_like(rho)
        initial = float(rho.item())
        n = np.int32(len(rhs))
        grid = ((len(rhs) + 127) // 128,)

        def apply(vector):
            kernels["csr"](
                grid, (128,), (offsets, columns, entries, vector, applied, n)
            )

        def step():
            apply(direction)
            cp.multiply(direction, applied, out=products)
            cp.sum(products, out=denominator)
            kernels["cg_update"](
                grid,
                (128,),
                (solution, residual, direction, applied, rho, denominator, n),
            )
            cp.multiply(residual, residual, out=products)
            cp.sum(products, out=next_rho)
            kernels["cg_direction"](
                grid, (128,), (direction, residual, rho, next_rho, n)
            )
            cp.copyto(rho, next_rho)

        # 先预热归约核，再在捕获前恢复输入。
        step()
        solution.fill(0)
        cp.copyto(residual, b)
        cp.copyto(direction, b)
        rho.fill(initial)
        stream.synchronize()
        batch = min(50, max_iterations)
        stream.begin_capture()
        for _ in range(batch):
            step()
        graph = stream.end_capture()
        completed = 0
        while completed < max_iterations:
            count = min(batch, max_iterations - completed)
            if count == batch:
                graph.launch(stream)
            else:
                for _ in range(count):
                    step()
            completed += count
            apply(solution)
            true_residual = b - applied
            relative = float(
                cp.sqrt(cp.sum(true_residual * true_residual) / initial).item()
            )
            if relative <= 3e-6:
                break
            if float(rho.item()) / initial < 1e-12:
                cp.copyto(residual, true_residual)
                cp.copyto(direction, residual)
                cp.multiply(residual, residual, out=products)
                cp.sum(products, out=rho)
        result = cp.asnumpy(solution).astype(np.float64) * magnitude * scale
    error = float(
        np.linalg.norm(system @ result - rhs) / max(np.linalg.norm(rhs), 1e-30)
    )
    if not np.isfinite(result).all() or not np.isfinite(error) or error > 1e-4:
        raise FloatingPointError(
            f"CAT FP32 depth solve failed true-residual check: {error}"
        )
    if diagnostics is not None:
        diagnostics.update(iterations=completed, relative_residual=error)
    return result


def smooth_values(vertices, faces, values, topology, fwhm):
    from cat_surface_gpu.features import _smoothing_parameters

    iterations, sigma = _smoothing_parameters(vertices, faces, fwhm)
    points = np.asarray(vertices, np.float64)
    distance = points[topology.neighbours] - points[:, None, :]
    weights = np.exp(-np.sum(distance * distance, axis=2) / (2 * sigma * sigma))
    weights = np.where(topology.neighbour_mask, weights, 0)
    return smooth(values, topology.neighbours, weights, iterations).astype(np.float64)


def type5_depth_features(vertices, faces, fwhms, topology):
    "仅构建一次原始深度场并跨平滑尺度复用。"
    from cat_surface_gpu.features import _smoothing_parameters, _vertex_normals

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


def depth_potential_feature(vertices, faces, fwhm=50.0, *, topology, alpha=0.001):
    "保留 FP64 几何装配，针对原始系统验证 CUDA FP32 解。"
    from scipy.sparse import csr_matrix
    from cat_surface_gpu.features import (
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
