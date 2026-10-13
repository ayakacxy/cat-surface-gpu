"""官方 CAT ``surf_to_sphere`` 的独立 CuPy 实现。"""

from __future__ import annotations

from functools import lru_cache
import time
from typing import Callable

import numpy as np

from .parameters import Surf2SphereParameters
from .geometry import surface_area
from .io import GiftiSurface
from .topology import SurfaceTopology, build_surface_topology


_CUDA_SOURCE = r"""
extern "C" __global__
void distance_smoothing(
    float* points,
    const int* neighbours,
    const int* degree,
    const signed char* selected,
    const int use_selected,
    const int* group,
    const int group_size,
    const int max_degree,
    const double strength)
{
    const int local = blockDim.x * blockIdx.x + threadIdx.x;
    if (local >= group_size) return;
    const int vertex = group[local];
    if (degree[vertex] <= 1) return;
    if (use_selected && selected[vertex] == 0) return;

    const double old_x = (double)points[vertex * 3];
    const double old_y = (double)points[vertex * 3 + 1];
    const double old_z = (double)points[vertex * 3 + 2];
    double total = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    double weighted_z = 0.0;
    for (int slot = 0; slot < degree[vertex]; ++slot) {
        const int neighbour = neighbours[vertex * max_degree + slot];
        const double x = (double)points[neighbour * 3];
        const double y = (double)points[neighbour * 3 + 1];
        const double z = (double)points[neighbour * 3 + 2];
        const double distance = fabs(x - old_x) + fabs(y - old_y) + fabs(z - old_z);
        total += distance;
        if (distance > 0.0) {
            weighted_x += distance * x;
            weighted_y += distance * y;
            weighted_z += distance * z;
        }
    }
    if (total <= 0.0) return;
    const double inverse = 1.0 - strength;
    points[vertex * 3] = (float)(old_x * inverse + weighted_x / total * strength);
    points[vertex * 3 + 1] = (float)(old_y * inverse + weighted_y / total * strength);
    points[vertex * 3 + 2] = (float)(old_z * inverse + weighted_z / total * strength);
}

extern "C" __global__
void inflate_points(
    float* points,
    const int n_points,
    const double span_x,
    const double span_y,
    const double span_z,
    const double inflation_delta)
{
    const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex >= n_points) return;
    const double x = (double)points[vertex * 3];
    const double y = (double)points[vertex * 3 + 1];
    const double z = (double)points[vertex * 3 + 2];
    const double scaled_x = x / span_x;
    const double scaled_y = y / span_y;
    const double scaled_z = z / span_z;
    const double radius = sqrt(
        scaled_x * scaled_x + scaled_y * scaled_y + scaled_z * scaled_z
    );
    const double scale = 1.0 + inflation_delta * (1.0 - radius);
    points[vertex * 3] = (float)(x * scale);
    points[vertex * 3 + 1] = (float)(y * scale);
    points[vertex * 3 + 2] = (float)(z * scale);
}

extern "C" __global__
void areal_smoothing(
    float* points,
    const int* faces,
    const int* incident_faces,
    const int* incident_degree,
    const int* group,
    const int group_size,
    const int max_incident)
{
    const int local = blockDim.x * blockIdx.x + threadIdx.x;
    if (local >= group_size) return;
    const int vertex = group[local];
    if (incident_degree[vertex] <= 1) return;

    double total_area = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    double weighted_z = 0.0;
    for (int slot = 0; slot < incident_degree[vertex]; ++slot) {
        const int face = incident_faces[vertex * max_incident + slot];
        const int vertex0 = faces[face * 3];
        const int vertex1 = faces[face * 3 + 1];
        const int vertex2 = faces[face * 3 + 2];
        const double p0x = (double)points[vertex0 * 3];
        const double p0y = (double)points[vertex0 * 3 + 1];
        const double p0z = (double)points[vertex0 * 3 + 2];
        const double p1x = (double)points[vertex1 * 3];
        const double p1y = (double)points[vertex1 * 3 + 1];
        const double p1z = (double)points[vertex1 * 3 + 2];
        const double p2x = (double)points[vertex2 * 3];
        const double p2y = (double)points[vertex2 * 3 + 1];
        const double p2z = (double)points[vertex2 * 3 + 2];
        const double edge0x = p1x - p0x;
        const double edge0y = p1y - p0y;
        const double edge0z = p1z - p0z;
        const double edge1x = p2x - p0x;
        const double edge1y = p2y - p0y;
        const double edge1z = p2z - p0z;
        const double cross_x = edge0y * edge1z - edge0z * edge1y;
        const double cross_y = edge0z * edge1x - edge0x * edge1z;
        const double cross_z = edge0x * edge1y - edge0y * edge1x;
        double area = 0.5 * sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
        );
        if (!isfinite(area)) area = 0.0;
        area = fmin(fmax(area, 0.0), 1.0);
        total_area += area;
        weighted_x += area * (p0x + p1x + p2x) / 3.0;
        weighted_y += area * (p0y + p1y + p2y) / 3.0;
        weighted_z += area * (p0z + p1z + p2z) / 3.0;
    }
    if (total_area <= 0.0) return;
    points[vertex * 3] = (float)(weighted_x / total_area);
    points[vertex * 3 + 1] = (float)(weighted_y / total_area);
    points[vertex * 3 + 2] = (float)(weighted_z / total_area);
}

extern "C" __global__
void normalize_radius(float* points, const int n_points, const double radius)
{
    const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex >= n_points) return;
    const double x = (double)points[vertex * 3];
    const double y = (double)points[vertex * 3 + 1];
    const double z = (double)points[vertex * 3 + 2];
    const double length = sqrt(x * x + y * y + z * z);
    if (length <= 0.0) return;
    const double scale = radius / length;
    points[vertex * 3] = (float)(x * scale);
    points[vertex * 3 + 1] = (float)(y * scale);
    points[vertex * 3 + 2] = (float)(z * scale);
}
"""


def _cupy():
    """延迟导入 CuPy，使 CPU 数值工具测试不要求 CUDA。"""

    import cupy as cp

    return cp


@lru_cache(maxsize=1)
def _kernels():
    """编译并缓存本模块的官方公式 CUDA 核。"""

    cp = _cupy()
    module = cp.RawModule(
        code=_CUDA_SOURCE,
        options=("--std=c++11",),
        name_expressions=(
            "distance_smoothing",
            "inflate_points",
            "areal_smoothing",
            "normalize_radius",
        ),
    )
    return tuple(
        module.get_function(name)
        for name in (
            "distance_smoothing",
            "inflate_points",
            "areal_smoothing",
            "normalize_radius",
        )
    )


def _device_topology(topology: SurfaceTopology):
    """把一次运行复用的拓扑表上传到当前 CUDA 设备。"""

    cp = _cupy()
    return {
        "neighbours": cp.asarray(topology.neighbours, dtype=cp.int32),
        "degree": cp.asarray(topology.degree, dtype=cp.int32),
        "incident_faces": cp.asarray(topology.incident_faces, dtype=cp.int32),
        "incident_degree": cp.asarray(
            topology.incident_mask.sum(axis=1), dtype=cp.int32
        ),
        "groups": tuple(
            cp.asarray(group, dtype=cp.int32) for group in topology.color_groups
        ),
    }


def _launch_groups(
    kernel,
    groups,
    prefix_arguments: tuple,
    suffix_arguments: tuple = (),
) -> None:
    """按无邻接顶点颜色组执行原地更新。"""

    threads = 256
    for group in groups:
        size = int(group.size)
        if size == 0:
            continue
        kernel(
            ((size + threads - 1) // threads,),
            (threads,),
            (*prefix_arguments, group, np.int32(size), *suffix_arguments),
        )


def _distance_smoothing(
    points,
    device_topology: dict,
    *,
    iterations: int,
    strength: float,
    selected=None,
) -> None:
    """执行官方曼哈顿距离加权原地平滑。"""

    cp = _cupy()
    distance_kernel, _, _, _ = _kernels()
    if selected is None:
        selected = cp.zeros(points.shape[0], dtype=cp.int8)
        use_selected = 0
    else:
        selected = cp.asarray(selected, dtype=cp.int8)
        use_selected = 1
    for _ in range(1, iterations):
        _launch_groups(
            distance_kernel,
            device_topology["groups"],
            (
                points,
                device_topology["neighbours"],
                device_topology["degree"],
                selected,
                np.int32(use_selected),
            ),
            (
                np.int32(device_topology["neighbours"].shape[1]),
                np.float64(strength),
            ),
        )


def _face_and_vertex_areas(points, faces):
    """在设备端计算官方面面积及三分之一顶点面积。"""

    cp = _cupy()
    triangles = points[faces]
    cross = cp.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * cp.sqrt(cp.sum(cross.astype(cp.float64) ** 2, axis=1))
    areas = cp.where(cp.isfinite(areas), areas, 0.0)
    vertex = cp.zeros(points.shape[0], dtype=cp.float64)
    contribution = areas / 3.0
    for column in range(3):
        cp.add.at(vertex, faces[:, column], contribution)
    return areas, vertex


def _distortion_selection(
    points,
    reference,
    current_areas,
    reference_areas,
    device_topology: dict,
    current_surface_area,
    reference_surface_area,
    threshold: float,
):
    """按官方压缩/拉伸指标选择手指状伪影顶点。"""

    cp = _cupy()
    neighbours = device_topology["neighbours"]
    degree = device_topology["degree"]
    slots = cp.arange(neighbours.shape[1], dtype=cp.int32)[None, :]
    mask = slots < degree[:, None]
    current_delta = points[neighbours] - points[:, None, :]
    reference_delta = reference[neighbours] - reference[:, None, :]
    current_distance = cp.sqrt(cp.sum(current_delta.astype(cp.float64) ** 2, axis=2))
    reference_distance = cp.sqrt(
        cp.sum(reference_delta.astype(cp.float64) ** 2, axis=2)
    )
    valid = mask & (reference_distance > 0.0)
    ratios = cp.where(
        valid, current_distance / cp.maximum(reference_distance, 1.0e-30), 0.0
    )
    maximum_linear = ratios.max(axis=1).astype(cp.float32)
    area_ratio = cp.where(
        current_areas > 0.0,
        reference_areas / cp.maximum(current_areas, 1.0e-30),
        cp.where(reference_areas != 0.0, 10000.0, 1.0),
    )
    area_ratio = cp.maximum(area_ratio, 1.0e-8).astype(cp.float32)
    surface_ratio = (current_surface_area / reference_surface_area).astype(cp.float32)
    compression = maximum_linear * area_ratio * surface_ratio
    neighbour_values = cp.where(mask, compression[neighbours], 0.0)
    average = (compression + neighbour_values.sum(axis=1)) / (
        degree.astype(cp.float32) + 1.0
    )
    return (average > cp.float32(threshold)).astype(cp.int8)


def _inflate_stage(
    points,
    faces,
    device_topology: dict,
    *,
    cycles: int,
    regular_strength: float,
    regular_iterations: int,
    inflation_factor: float,
    distortion_threshold: float,
    finger_strength: float,
    finger_iterations: int,
) -> None:
    """逐步执行官方 inflate_surface_and_smooth_fingers。"""

    cp = _cupy()
    _, inflate_kernel, _, _ = _kernels()
    reference = points.copy()
    spans = cp.maximum(reference.max(axis=0) - reference.min(axis=0), 1.0e-20)
    points -= points.mean(axis=0, dtype=cp.float64).astype(cp.float32)
    reference_face_areas, reference_areas = _face_and_vertex_areas(reference, faces)
    reference_surface_area = reference_face_areas.sum(dtype=cp.float64)
    threads = 256
    blocks = (points.shape[0] + threads - 1) // threads
    for cycle in range(cycles + 1):
        if cycle < cycles:
            _distance_smoothing(
                points,
                device_topology,
                iterations=regular_iterations,
                strength=regular_strength,
            )
            inflate_kernel(
                (blocks,),
                (threads,),
                (
                    points,
                    np.int32(points.shape[0]),
                    np.float64(spans[0].item()),
                    np.float64(spans[1].item()),
                    np.float64(spans[2].item()),
                    np.float64(inflation_factor - 1.0),
                ),
            )
        current_face_areas, current_areas = _face_and_vertex_areas(points, faces)
        selected = _distortion_selection(
            points,
            reference,
            current_areas,
            reference_areas,
            device_topology,
            current_face_areas.sum(dtype=cp.float64),
            reference_surface_area,
            distortion_threshold,
        )
        if cycle < cycles and finger_iterations > 0:
            _distance_smoothing(
                points,
                device_topology,
                iterations=finger_iterations,
                strength=finger_strength,
                selected=selected,
            )


def _ellipsoid_to_sphere_device(points, desired_surface_area: float) -> None:
    """在设备端执行官方椭球到等面积球面的投影。"""

    cp = _cupy()
    axes = (cp.abs(points.min(axis=0)) + cp.abs(points.max(axis=0))) * 0.5
    axes = cp.maximum(axes, 1.0e-20)
    factors = cp.sqrt(cp.sum((points.astype(cp.float64) / axes[None, :]) ** 2, axis=1))
    factors = cp.where(factors != 0.0, factors, 1.0)
    radius = np.sqrt(float(desired_surface_area) / (4.0 * np.pi))
    converted = points.astype(cp.float64) / factors[:, None]
    converted *= radius / axes[None, :]
    points[...] = converted.astype(cp.float32)


def _areal_smoothing(points, faces, device_topology: dict, iterations: int) -> None:
    """执行官方面积权重平滑，并每 1000 次恢复初始最大半径。"""

    cp = _cupy()
    _, _, areal_kernel, normalize_kernel = _kernels()
    radius = cp.sqrt(cp.sum(points.astype(cp.float64) ** 2, axis=1)).max().item()
    threads = 256
    blocks = (points.shape[0] + threads - 1) // threads
    max_incident = np.int32(device_topology["incident_faces"].shape[1])
    for iteration in range(1, iterations):
        _launch_groups(
            areal_kernel,
            device_topology["groups"],
            (
                points,
                faces,
                device_topology["incident_faces"],
                device_topology["incident_degree"],
            ),
            (max_incident,),
        )
        if iteration % 1000 == 0:
            normalize_kernel(
                (blocks,),
                (threads,),
                (points, np.int32(points.shape[0]), np.float64(radius)),
            )


def surf_to_sphere_cupy(
    vertices: np.ndarray,
    faces: np.ndarray,
    parameters: Surf2SphereParameters | None = None,
    topology: SurfaceTopology | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """以官方阶段、参数和公式运行独立 CuPy 球面生成。"""

    parameters = parameters or Surf2SphereParameters()
    GiftiSurface(vertices, faces)
    cp = _cupy()
    points_host = np.ascontiguousarray(vertices, dtype=np.float32)
    faces_host = np.ascontiguousarray(faces, dtype=np.int32)
    topology = topology or build_surface_topology(faces_host, points_host.shape[0])
    desired_area = surface_area(points_host, faces_host)
    points = cp.asarray(points_host)
    faces_device = cp.asarray(faces_host)
    device_topology = _device_topology(topology)
    factor = faces_host.shape[0] / 350000.0 if faces_host.shape[0] > 350000 else 1.0
    stages = (
        (1, "low_smooth", 1, 0.2, round(factor * 50), 1.0, 3.0, 1.0, 0),
        (2, "inflate", 2, 1.0, round(factor * 30), 1.4, 3.0, 1.0, 30),
        (3, "very_inflate", 4, 1.0, round(factor * 30), 1.1, 3.0, 1.0, 0),
        (4, "high_smooth", 6, 1.0, round(factor * 60), 1.6, 3.0, 1.0, 60),
        (5, "ellipsoid", 6, 1.0, round(factor * 50), 1.4, 4.0, 1.0, 60),
    )
    timings: dict[str, float] = {}
    cp.cuda.get_current_stream().synchronize()
    total_started = time.perf_counter()
    for (
        stage_index,
        name,
        cycles,
        regular_strength,
        regular_iterations,
        inflation_factor,
        distortion_threshold,
        finger_strength,
        finger_iterations,
    ) in stages:
        if stage_index > parameters.stop_at:
            break
        if progress is not None:
            progress(f"Surf2Sphere: {name}")
        started = time.perf_counter()
        _inflate_stage(
            points,
            faces_device,
            device_topology,
            cycles=cycles,
            regular_strength=regular_strength,
            regular_iterations=regular_iterations,
            inflation_factor=inflation_factor,
            distortion_threshold=distortion_threshold,
            finger_strength=finger_strength,
            finger_iterations=finger_iterations,
        )
        cp.cuda.get_current_stream().synchronize()
        timings[name] = time.perf_counter() - started
    if parameters.stop_at > 4:
        if progress is not None:
            progress("Surf2Sphere: ellipsoid projection")
        started = time.perf_counter()
        _ellipsoid_to_sphere_device(points, desired_area)
        cp.cuda.get_current_stream().synchronize()
        timings["ellipsoid_projection"] = time.perf_counter() - started
    if parameters.areal_iterations:
        if progress is not None:
            progress(
                f"Surf2Sphere: area smoothing ({parameters.areal_iterations - 1} sweeps)"
            )
        started = time.perf_counter()
        _areal_smoothing(
            points,
            faces_device,
            device_topology,
            parameters.areal_iterations,
        )
        cp.cuda.get_current_stream().synchronize()
        timings["areal_smoothing"] = time.perf_counter() - started
        started = time.perf_counter()
        _ellipsoid_to_sphere_device(points, desired_area)
        cp.cuda.get_current_stream().synchronize()
        timings["final_projection"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_started
    output = np.ascontiguousarray(cp.asnumpy(points), dtype=np.float32)
    GiftiSurface(output, faces_host)
    return output, timings
