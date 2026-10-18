"与 CuPy 共用阶段和参数的 FP64/FP32 CPU 球面生成。"

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from cat_surface_gpu.parameters import Surf2SphereParameters
from cat_surface_gpu.geometry import surface_area
from cat_surface_gpu.topology import SurfaceTopology, build_surface_topology


@njit(cache=True, fastmath=False)
def _smooth(
    points, neighbours, degree, groups, offsets, iterations, strength, selected
):
    for _ in range(1, iterations):
        for color in range(len(offsets) - 1):
            for item in range(offsets[color], offsets[color + 1]):
                vertex = groups[item]
                if degree[vertex] <= 1 or (selected.size and not selected[vertex]):
                    continue
                x0, y0, z0 = points[vertex]
                total = wx = wy = wz = 0.0
                for slot in range(degree[vertex]):
                    other = neighbours[vertex, slot]
                    x, y, z = points[other]
                    distance = (
                        abs(float(x) - float(x0))
                        + abs(float(y) - float(y0))
                        + abs(float(z) - float(z0))
                    )
                    total += distance
                    if distance > 0:
                        wx += distance * x
                        wy += distance * y
                        wz += distance * z
                if total > 0:
                    points[vertex, 0] = x0 * (1.0 - strength) + wx / total * strength
                    points[vertex, 1] = y0 * (1.0 - strength) + wy / total * strength
                    points[vertex, 2] = z0 * (1.0 - strength) + wz / total * strength


@njit(cache=True, fastmath=False)
def _inflate(points, spans, delta):
    for vertex in range(len(points)):
        x, y, z = points[vertex]
        radius = np.sqrt(
            (x / spans[0]) ** 2 + (y / spans[1]) ** 2 + (z / spans[2]) ** 2
        )
        scale = 1.0 + delta * (1.0 - radius)
        points[vertex, 0] = x * scale
        points[vertex, 1] = y * scale
        points[vertex, 2] = z * scale


@njit(cache=True, parallel=True, fastmath=False)
def _areal(points, faces, incident, incident_count, groups, offsets, iterations):
    radius = 0.0
    for vertex in range(len(points)):
        x, y, z = points[vertex]
        radius = max(radius, np.sqrt(float(x) * x + float(y) * y + float(z) * z))
    for iteration in range(1, iterations):
        for color in range(len(offsets) - 1):
            for item in prange(offsets[color], offsets[color + 1]):
                vertex = groups[item]
                if incident_count[vertex] <= 1:
                    continue
                total = wx = wy = wz = 0.0
                for slot in range(incident_count[vertex]):
                    face = faces[incident[vertex, slot]]
                    a, b, c = face[0], face[1], face[2]
                    e0x = float(points[b, 0]) - points[a, 0]
                    e0y = float(points[b, 1]) - points[a, 1]
                    e0z = float(points[b, 2]) - points[a, 2]
                    e1x = float(points[c, 0]) - points[a, 0]
                    e1y = float(points[c, 1]) - points[a, 1]
                    e1z = float(points[c, 2]) - points[a, 2]
                    cx = e0y * e1z - e0z * e1y
                    cy = e0z * e1x - e0x * e1z
                    cz = e0x * e1y - e0y * e1x
                    area = 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz)
                    if not np.isfinite(area):
                        area = 0.0
                    area = min(max(area, 0.0), 1.0)
                    total += area
                    wx += (
                        area * (float(points[a, 0]) + points[b, 0] + points[c, 0]) / 3.0
                    )
                    wy += (
                        area * (float(points[a, 1]) + points[b, 1] + points[c, 1]) / 3.0
                    )
                    wz += (
                        area * (float(points[a, 2]) + points[b, 2] + points[c, 2]) / 3.0
                    )
                if total > 0:
                    points[vertex, 0] = wx / total
                    points[vertex, 1] = wy / total
                    points[vertex, 2] = wz / total
        if iteration % 1000 == 0:
            for vertex in range(len(points)):
                x, y, z = points[vertex]
                length = np.sqrt(float(x) * x + float(y) * y + float(z) * z)
                if length > 0:
                    points[vertex, 0] = x * radius / length
                    points[vertex, 1] = y * radius / length
                    points[vertex, 2] = z * radius / length


def _areas(points, faces):
    triangles = points[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.sqrt(np.sum(cross.astype(np.float64) ** 2, axis=1))
    areas[~np.isfinite(areas)] = 0.0
    vertex_areas = np.zeros(len(points), dtype=np.float64)
    for column in range(3):
        np.add.at(vertex_areas, faces[:, column], areas / 3.0)
    return areas, vertex_areas


def _selection(
    points,
    reference,
    current_areas,
    reference_areas,
    topology,
    area,
    reference_area,
    threshold,
):
    neighbours = topology.neighbours
    degree = topology.degree
    mask = topology.neighbour_mask
    current = points[neighbours] - points[:, None, :]
    original = reference[neighbours] - reference[:, None, :]
    current_distance = np.sqrt(np.sum(current.astype(np.float64) ** 2, axis=2))
    original_distance = np.sqrt(np.sum(original.astype(np.float64) ** 2, axis=2))
    ratios = np.where(
        mask & (original_distance > 0),
        current_distance / np.maximum(original_distance, 1e-30),
        0.0,
    )
    maximum = ratios.max(axis=1).astype(np.float32)
    area_ratio = np.where(
        current_areas > 0,
        reference_areas / np.maximum(current_areas, 1e-30),
        np.where(reference_areas != 0, 10000.0, 1.0),
    )
    area_ratio = np.maximum(area_ratio, 1e-8).astype(np.float32)
    compression = maximum * area_ratio * np.float32(area / reference_area)
    average = (compression + np.where(mask, compression[neighbours], 0).sum(axis=1)) / (
        degree.astype(np.float32) + 1.0
    )
    return np.asarray(average > np.float32(threshold), dtype=np.int8)


def _project(points, desired_area):
    axes = np.maximum(
        (np.abs(points.min(axis=0)) + np.abs(points.max(axis=0))) * 0.5, 1e-20
    )
    factors = np.sqrt(np.sum((points.astype(np.float64) / axes[None, :]) ** 2, axis=1))
    factors = np.where(factors != 0, factors, 1.0)
    radius = np.sqrt(float(desired_area) / (4.0 * np.pi))
    points[:] = (
        points.astype(np.float64) / factors[:, None] * (radius / axes)[None, :]
    ).astype(np.float32)


def surf_to_sphere_cpu(
    vertices, faces, parameters=None, topology: SurfaceTopology | None = None
):
    "保留 CuPy 的颜色组原地更新、阶段顺序和默认迭代数。"
    options = parameters or Surf2SphereParameters()
    points = np.ascontiguousarray(vertices, dtype=np.float32).copy()
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    topology = topology or build_surface_topology(faces, len(points))
    desired_area = surface_area(points, faces)
    groups = np.concatenate(topology.color_groups).astype(np.int32)
    offsets = np.concatenate(
        ([0], np.cumsum([len(group) for group in topology.color_groups]))
    ).astype(np.int64)
    degree = np.asarray(topology.degree, dtype=np.int32)
    incident_degree = np.asarray(topology.incident_mask.sum(axis=1), dtype=np.int32)
    factor = len(faces) / 350000.0 if len(faces) > 350000 else 1.0
    stages = (
        (1, "low_smooth", 1, 0.2, round(factor * 50), 1.0, 3.0, 1.0, 0),
        (2, "inflate", 2, 1.0, round(factor * 30), 1.4, 3.0, 1.0, 30),
        (3, "very_inflate", 4, 1.0, round(factor * 30), 1.1, 3.0, 1.0, 0),
        (4, "high_smooth", 6, 1.0, round(factor * 60), 1.6, 3.0, 1.0, 60),
        (5, "ellipsoid", 6, 1.0, round(factor * 50), 1.4, 4.0, 1.0, 60),
    )
    timings = {}
    total_started = time.perf_counter()
    for (
        index,
        name,
        cycles,
        strength,
        iterations,
        inflation,
        threshold,
        finger_strength,
        finger_iterations,
    ) in stages:
        if index > options.stop_at:
            break
        started = time.perf_counter()
        reference = points.copy()
        spans = np.maximum(reference.max(axis=0) - reference.min(axis=0), 1e-20).astype(
            np.float64
        )
        points -= points.mean(axis=0, dtype=np.float64).astype(np.float32)
        reference_face_areas, reference_areas = _areas(reference, faces)
        reference_area = reference_face_areas.sum(dtype=np.float64)
        for cycle in range(cycles + 1):
            if cycle < cycles:
                _smooth(
                    points,
                    topology.neighbours,
                    degree,
                    groups,
                    offsets,
                    iterations,
                    strength,
                    np.empty(0, dtype=np.int8),
                )
                _inflate(points, spans, inflation - 1.0)
            current_face_areas, current_areas = _areas(points, faces)
            selected = _selection(
                points,
                reference,
                current_areas,
                reference_areas,
                topology,
                current_face_areas.sum(dtype=np.float64),
                reference_area,
                threshold,
            )
            if cycle < cycles and finger_iterations > 0:
                _smooth(
                    points,
                    topology.neighbours,
                    degree,
                    groups,
                    offsets,
                    finger_iterations,
                    finger_strength,
                    selected,
                )
        timings[name] = time.perf_counter() - started
    if options.stop_at > 4:
        started = time.perf_counter()
        _project(points, desired_area)
        timings["ellipsoid_projection"] = time.perf_counter() - started
    if options.areal_iterations:
        started = time.perf_counter()
        _areal(
            points,
            faces,
            topology.incident_faces,
            incident_degree,
            groups,
            offsets,
            options.areal_iterations,
        )
        timings["areal_smoothing"] = time.perf_counter() - started
        started = time.perf_counter()
        _project(points, desired_area)
        timings["final_projection"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_started
    return points, timings
