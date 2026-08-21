#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""CAT-Surface GPU implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import time

import numpy as np

from cat_surface_gpu.surface_stencil import SurfaceStencil


_MAP_HEADER = struct.Struct("<i")
_MAP_RECORD = np.dtype([("face", "<i4"), ("weights", "<f8", (3,))])


class UniformTriangleGrid:
    """Represent UniformTriangleGrid."""

    def __init__(
        self,
        points: np.ndarray,
        faces: np.ndarray,
        *,
        grid_size: int = 128,
        margin: int = 1,
    ) -> None:
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [n, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [m, 3]")
        if grid_size < 4 or margin < 0:
            raise ValueError("Grid_size or margin valid")
        self.points = points
        self.faces = faces
        self.grid_size = int(grid_size)
        self.margin = int(margin)
        self.grid_min = -1.0
        self.cell_size = 2.0 / float(self.grid_size)
        self.cell_count = self.grid_size**3
        self._build()

    def _build(self) -> None:
        """Build."""

        triangles = self.points[self.faces]
        lower = np.floor(
            (triangles.min(axis=1) - self.grid_min) / self.cell_size
        ).astype(np.int64)
        upper = np.floor(
            (triangles.max(axis=1) - self.grid_min) / self.cell_size
        ).astype(np.int64)
        lower = np.clip(lower - self.margin, 0, self.grid_size - 1)
        upper = np.clip(upper + self.margin, 0, self.grid_size - 1)

        cells: dict[int, list[int]] = {}
        for face, (lo, hi) in enumerate(zip(lower, upper)):
            for z in range(int(lo[2]), int(hi[2]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    base = self.grid_size * (y + self.grid_size * z)
                    for x in range(int(lo[0]), int(hi[0]) + 1):
                        cells.setdefault(base + x, []).append(face)

        max_candidates = max(len(value) for value in cells.values())
        padded = np.full((self.cell_count, max_candidates), -1, dtype=np.int32)
        for cell, values in cells.items():
            padded[cell, : len(values)] = values
        self.candidates = padded
        self.active_cells = np.fromiter(cells, dtype=np.int64)

    def query(self, query_points: np.ndarray) -> np.ndarray:
        """Query."""

        query_points = np.asarray(query_points, dtype=np.float64)
        cell = np.floor((query_points - self.grid_min) / self.cell_size).astype(
            np.int64
        )
        cell = np.clip(cell, 0, self.grid_size - 1)
        cell_id = cell[:, 0] + self.grid_size * (
            cell[:, 1] + self.grid_size * cell[:, 2]
        )
        return self.candidates[cell_id]


def _segment_distance(
    point: np.ndarray, first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Segment distance."""

    direction = second - first
    offset = point - first
    denominator = np.sum(direction * direction, axis=-1)
    alpha = np.divide(
        np.sum(offset * direction, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = first + alpha[..., None] * direction
    distance = np.sum((point - closest) ** 2, axis=-1)
    return distance, closest


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Barycentric."""

    weights: list[np.ndarray] = []
    for first, second, third in (
        (triangle[..., 1, :], triangle[..., 2, :], triangle[..., 0, :]),
        (triangle[..., 2, :], triangle[..., 0, :], triangle[..., 1, :]),
        (triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]),
    ):
        point_offset = point - first
        horizontal = second - first
        upward = third - first
        normal = np.cross(horizontal, upward)
        vertical = np.cross(normal, horizontal)
        numerator = np.sum(point_offset * vertical, axis=-1)
        denominator = np.sum(upward * vertical, axis=-1)
        weights.append(numerator / denominator)
    return np.stack(weights, axis=-1)


def closest_faces(
    query_points: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    candidates: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest faces."""

    query_points = np.asarray(query_points, dtype=np.float64)
    triangles_all = np.asarray(points, dtype=np.float64)[faces]
    n_points = query_points.shape[0]
    selected_faces = np.full(n_points, -1, dtype=np.int32)
    selected_weights = np.zeros((n_points, 3), dtype=np.float64)
    for start in range(0, n_points, batch_size):
        stop = min(start + batch_size, n_points)
        query = query_points[start:stop]
        candidate_faces = candidates[start:stop]
        valid = candidate_faces >= 0
        safe_faces = np.maximum(candidate_faces, 0)
        triangles = triangles_all[safe_faces]
        first = triangles[..., 0, :]
        second = triangles[..., 1, :]
        third = triangles[..., 2, :]
        edge_a = second - first
        edge_b = third - first
        normal = np.cross(edge_a, edge_b)
        normal_sq = np.sum(normal * normal, axis=-1)
        offset = first - query[:, None, :]
        plane_t = np.divide(
            np.sum(offset * normal, axis=-1),
            normal_sq,
            out=np.zeros_like(normal_sq),
            where=normal_sq != 0.0,
        )
        projected = query[:, None, :] + plane_t[..., None] * normal

        point_offset = projected - first
        xx = np.sum(edge_a * edge_a, axis=-1)
        xy = np.sum(edge_a * edge_b, axis=-1)
        xv = np.sum(edge_a * point_offset, axis=-1)
        yy = np.sum(edge_b * edge_b, axis=-1)
        yv = np.sum(edge_b * point_offset, axis=-1)
        denominator = xx * yy - xy * xy
        xpos = np.divide(
            xv * yy - yv * xy,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0.0,
        )
        ypos = np.divide(
            yv * xx - xv * xy,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0.0,
        )
        inside = (
            (denominator != 0.0)
            & (xpos >= -1.0e-3)
            & (xpos <= 1.0 + 1.0e-3)
            & (ypos >= -1.0e-3)
            & (ypos <= 1.0 + 1.0e-3)
            & (xpos + ypos >= -1.0e-3)
            & (xpos + ypos <= 1.0 + 1.0e-3)
        )
        plane_distance = plane_t * plane_t * normal_sq

        vertex_distance = np.stack(
            (
                np.sum((query[:, None, :] - first) ** 2, axis=-1),
                np.sum((query[:, None, :] - second) ** 2, axis=-1),
                np.sum((query[:, None, :] - third) ** 2, axis=-1),
            ),
            axis=-1,
        )
        closest_vertex = np.argmin(vertex_distance, axis=-1)
        vertex_closest = np.take_along_axis(
            np.stack((first, second, third), axis=-2),
            closest_vertex[..., None, None],
            axis=-2,
        )[..., 0, :]
        prev_index = (closest_vertex - 1) % 3
        next_index = (closest_vertex + 1) % 3
        vertices = np.stack((first, second, third), axis=-2)
        prev_vertex = np.take_along_axis(
            vertices, prev_index[..., None, None], axis=-2
        )[..., 0, :]
        next_vertex = np.take_along_axis(
            vertices, next_index[..., None, None], axis=-2
        )[..., 0, :]
        edge_one_distance, edge_one_point = _segment_distance(
            query[:, None, :], prev_vertex, vertex_closest
        )
        edge_two_distance, edge_two_point = _segment_distance(
            query[:, None, :], vertex_closest, next_vertex
        )
        outside_distance = np.where(
            edge_one_distance < edge_two_distance,
            edge_one_distance,
            edge_two_distance,
        )
        outside_point = np.where(
            (edge_one_distance < edge_two_distance)[..., None],
            edge_one_point,
            edge_two_point,
        )
        candidate_distance = np.where(inside, plane_distance, outside_distance)
        candidate_point = np.where(inside[..., None], projected, outside_point)
        candidate_distance = np.where(valid, candidate_distance, np.inf)

        best_index = np.argmin(candidate_distance, axis=-1)
        row = np.arange(stop - start)
        selected = candidate_faces[row, best_index]
        point = candidate_point[row, best_index]
        triangle = triangles[row, best_index]
        selected_faces[start:stop] = selected
        selected_weights[start:stop] = _barycentric(point, triangle)
    return selected_faces, selected_weights


def _read_c_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read c map."""

    with path.open("rb") as stream:
        raw = stream.read(_MAP_HEADER.size)
        if len(raw) != _MAP_HEADER.size:
            raise ValueError("C map file")
        (n_points,) = _MAP_HEADER.unpack(raw)
        records = np.fromfile(stream, dtype=_MAP_RECORD, count=n_points)
    if records.size != n_points:
        raise ValueError("C map file")
    return records["face"].copy(), records["weights"].copy()


def _read_rotated_points(path: Path) -> np.ndarray:
    """Read rotated points."""

    with path.open("rb") as stream:
        raw = stream.read(_MAP_HEADER.size)
        (n_points,) = _MAP_HEADER.unpack(raw)
        points = np.fromfile(stream, dtype="<f4", count=3 * n_points)
    if points.size != 3 * n_points:
        raise ValueError("Rotation file length")
    return points.reshape(n_points, 3)


def _read_rotation_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read rotation values."""

    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if header.size != 2:
            raise ValueError("Rotation file")
        source = np.fromfile(stream, dtype="<f8", count=int(header[0]))
        target = np.fromfile(stream, dtype="<f8", count=int(header[1]))
    if source.size != header[0] or target.size != header[1]:
        raise ValueError("Rotation file length")
    return source, target


def main() -> int:
    """Main."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-stencil", type=Path, required=True)
    parser.add_argument("--rotated-points", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--rotation-values", type=Path)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=1)
    args = parser.parse_args()

    stencil = SurfaceStencil.from_file(args.target_stencil)
    target_points = stencil.sphere_points
    target_faces = stencil.faces
    rotated_points = _read_rotated_points(args.rotated_points)
    reference_faces, reference_weights = _read_c_map(args.reference_map)

    start = time.perf_counter()
    grid = UniformTriangleGrid(
        target_points, target_faces, grid_size=args.grid_size, margin=args.margin
    )
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    candidates = grid.query(rotated_points)
    query_seconds = time.perf_counter() - start
    start = time.perf_counter()
    faces, weights = closest_faces(
        rotated_points, target_points, target_faces, candidates
    )
    locate_seconds = time.perf_counter() - start
    covered = np.any(candidates == reference_faces[:, None], axis=1)
    matching = faces == reference_faces
    weight_delta = np.abs(weights - reference_weights)
    result = {
        "grid_size": args.grid_size,
        "margin": args.margin,
        "max_candidates": int(candidates.shape[1]),
        "candidate_entries": int(np.count_nonzero(candidates >= 0)),
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
        "locate_seconds": locate_seconds,
        "reference_coverage": float(np.mean(covered)),
        "face_match": float(np.mean(matching)),
        "weight_max_abs_all": float(weight_delta.max()),
        "weight_max_abs_matching": float(weight_delta[matching].max())
        if np.any(matching)
        else None,
        "weight_mean_abs_matching": float(weight_delta[matching].mean())
        if np.any(matching)
        else None,
    }
    if args.rotation_values is not None:
        source_values, target_values = _read_rotation_values(args.rotation_values)
        if source_values.size != rotated_points.shape[0]:
            raise ValueError("Source length and rotation does not match")
        c_sampled = (
            target_values[target_faces[reference_faces]] * reference_weights
        ).sum(axis=1)
        grid_sampled = (target_values[target_faces[faces]] * weights).sum(axis=1)
        sample_delta = np.abs(c_sampled - grid_sampled)
        result.update(
            {
                "sample_max_abs": float(sample_delta.max()),
                "sample_mean_abs": float(sample_delta.mean()),
                "sample_p99_abs": float(np.percentile(sample_delta, 99)),
                "cost_reference": float(np.sum((source_values - c_sampled) ** 2)),
                "cost_grid": float(np.sum((source_values - grid_sampled) ** 2)),
                "cost_abs_delta": float(
                    abs(
                        np.sum((source_values - c_sampled) ** 2)
                        - np.sum((source_values - grid_sampled) ** 2)
                    )
                ),
            }
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
