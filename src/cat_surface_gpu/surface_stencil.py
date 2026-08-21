"""Precomputed spherical resampling stencils and device-resident operators."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Literal

import numpy as np
import torch


_HEADER = struct.Struct("<8i")
_MAGIC = 0x46534354

_TOPOLOGY_DEVICE_CACHE: dict[
    tuple[object, ...],
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]],
] = {}


def _topology_cache_key(faces: np.ndarray) -> tuple[object, ...]:
    """Topology cache key."""

    contiguous = np.ascontiguousarray(faces)
    return (
        tuple(int(item) for item in contiguous.shape),
        contiguous.dtype.str,
        contiguous.tobytes(),
    )


def _build_neighbour_table(
    faces: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build neighbour table."""

    directed = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 0]],
            faces[:, [1, 2]],
            faces[:, [2, 1]],
            faces[:, [2, 0]],
            faces[:, [0, 2]],
        ),
        axis=0,
    )
    directed = np.unique(directed, axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]
    counts = np.bincount(directed[:, 0], minlength=n_points)
    max_degree = int(counts.max())
    neighbours = np.zeros((counts.size, max_degree), dtype=np.int64)
    mask = np.zeros((counts.size, max_degree), dtype=bool)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    for vertex in range(counts.size):
        start, stop = offsets[vertex], offsets[vertex + 1]
        degree = stop - start
        neighbours[vertex, :degree] = directed[start:stop, 1]
        mask[vertex, :degree] = True
    return neighbours, mask


def _build_color_groups(
    neighbours: np.ndarray, neighbour_mask: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Build color groups."""

    n_points = int(neighbours.shape[0])
    colours = np.full(n_points, -1, dtype=np.int32)
    max_colour = -1
    for vertex in range(n_points):
        adjacent_colours = {
            int(colours[neighbour])
            for neighbour in neighbours[vertex, neighbour_mask[vertex]]
            if colours[neighbour] >= 0
        }
        colour = 0
        while colour in adjacent_colours:
            colour += 1
        colours[vertex] = colour
        max_colour = max(max_colour, colour)
    return tuple(
        np.flatnonzero(colours == colour).astype(np.int64, copy=False)
        for colour in range(max_colour + 1)
    )


def _build_ordered_dependency_groups(
    neighbours: np.ndarray, neighbour_mask: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Build ordered dependency groups."""

    n_points = int(neighbours.shape[0])
    levels = np.zeros(n_points, dtype=np.int32)
    for vertex in range(n_points):
        lower = neighbours[vertex, neighbour_mask[vertex]]
        lower = lower[lower < vertex]
        if lower.size:
            levels[vertex] = int(levels[lower].max()) + 1
    return tuple(
        np.flatnonzero(levels == level).astype(np.int64, copy=False)
        for level in range(int(levels.max()) + 1)
    )


@dataclass(frozen=True)
class SurfaceStencil:
    """Store CPU indices, barycentric weights, topology, and unit-sphere points."""

    sphere_points: np.ndarray
    faces: np.ndarray
    surface_indices: np.ndarray
    surface_weights: np.ndarray
    sheet_indices: np.ndarray
    sheet_weights: np.ndarray
    nx: int
    ny: int
    source_points: int
    unit_sphere_points: np.ndarray | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "SurfaceStencil":
        """Read a versioned stencil produced by the bundled native helper."""

        with Path(path).open("rb") as stream:
            header_data = stream.read(_HEADER.size)
            if len(header_data) != _HEADER.size:
                raise ValueError("Stencil file")
            magic, version, n_points, n_triangles, n_sheet, nx, ny, source_points = (
                _HEADER.unpack(header_data)
            )
            if magic != _MAGIC or version not in (1, 2):
                raise ValueError("Stencil file magic or")
            sphere_points = np.fromfile(stream, dtype="<f8", count=3 * n_points)
            faces = np.fromfile(stream, dtype="<i4", count=3 * n_triangles)
            surface_indices = np.fromfile(stream, dtype="<i4", count=3 * n_points)
            surface_weights = np.fromfile(stream, dtype="<f8", count=3 * n_points)
            sheet_indices = np.fromfile(stream, dtype="<i4", count=3 * n_sheet)
            sheet_weights = np.fromfile(stream, dtype="<f8", count=3 * n_sheet)
            if version == 2:
                unit_sphere_points = np.fromfile(
                    stream, dtype="<f4", count=3 * source_points
                )
            else:
                unit_sphere_points = None

        arrays = (
            sphere_points,
            faces,
            surface_indices,
            surface_weights,
            sheet_indices,
            sheet_weights,
        )
        expected = (
            3 * n_points,
            3 * n_triangles,
            3 * n_points,
            3 * n_points,
            3 * n_sheet,
            3 * n_sheet,
        )
        if any(array.size != size for array, size in zip(arrays, expected)):
            raise ValueError("Stencil file length")
        if (
            unit_sphere_points is not None
            and unit_sphere_points.size != 3 * source_points
        ):
            raise ValueError("Stencil unit sphere length")
        return cls(
            sphere_points=sphere_points.reshape(n_points, 3),
            faces=faces.reshape(n_triangles, 3),
            surface_indices=surface_indices.reshape(n_points, 3),
            surface_weights=surface_weights.reshape(n_points, 3),
            sheet_indices=sheet_indices.reshape(ny, nx, 3),
            sheet_weights=sheet_weights.reshape(ny, nx, 3),
            nx=nx,
            ny=ny,
            source_points=source_points,
            unit_sphere_points=(
                None
                if unit_sphere_points is None
                else unit_sphere_points.reshape(source_points, 3)
            ),
        )

    def to(
        self,
        device: str | torch.device,
        *,
        geometry_dtype: torch.dtype = torch.float32,
        weight_dtype: torch.dtype = torch.float64,
    ) -> "SurfaceStencilDevice":
        """Upload a stencil and its reusable topology tables to one device."""

        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        faces = self.faces.astype(np.int64, copy=False)
        topology_key = _topology_cache_key(faces)
        device_key = (target.type, target.index, *topology_key)
        cached_topology = _TOPOLOGY_DEVICE_CACHE.get(device_key)
        if cached_topology is None:
            neighbours, neighbour_mask = _build_neighbour_table(
                faces, self.sphere_points.shape[0]
            )
            color_groups = _build_color_groups(neighbours, neighbour_mask)
            cached_topology = (
                torch.as_tensor(neighbours, dtype=torch.long, device=target),
                torch.as_tensor(neighbour_mask, dtype=torch.bool, device=target),
                tuple(
                    torch.as_tensor(group, dtype=torch.long, device=target)
                    for group in color_groups
                ),
            )
            _TOPOLOGY_DEVICE_CACHE[device_key] = cached_topology
        neighbours, neighbour_mask, color_groups = cached_topology
        return SurfaceStencilDevice(
            sphere_points=torch.as_tensor(
                self.sphere_points, dtype=geometry_dtype, device=target
            ).contiguous(),
            faces=torch.as_tensor(self.faces, dtype=torch.long, device=target),
            surface_indices=torch.as_tensor(
                self.surface_indices, dtype=torch.long, device=target
            ),
            surface_weights=torch.as_tensor(
                self.surface_weights, dtype=weight_dtype, device=target
            ),
            sheet_indices=torch.as_tensor(
                self.sheet_indices, dtype=torch.long, device=target
            ),
            sheet_weights=torch.as_tensor(
                self.sheet_weights, dtype=weight_dtype, device=target
            ),
            nx=self.nx,
            ny=self.ny,
            source_points=self.source_points,
            unit_sphere_points=(
                None
                if self.unit_sphere_points is None
                else torch.as_tensor(
                    self.unit_sphere_points,
                    dtype=geometry_dtype,
                    device=target,
                ).contiguous()
            ),
            _neighbours=torch.as_tensor(neighbours, dtype=torch.long, device=target),
            _neighbour_mask=torch.as_tensor(
                neighbour_mask, dtype=torch.bool, device=target
            ),
            _color_groups=tuple(
                torch.as_tensor(group, dtype=torch.long, device=target)
                for group in color_groups
            ),
        )


@dataclass
class SurfaceStencilDevice:
    """Store device-resident stencil and topology tensors for repeated use."""

    sphere_points: torch.Tensor
    faces: torch.Tensor
    surface_indices: torch.Tensor
    surface_weights: torch.Tensor
    sheet_indices: torch.Tensor
    sheet_weights: torch.Tensor
    nx: int
    ny: int
    source_points: int
    unit_sphere_points: torch.Tensor | None = None
    _neighbours: torch.Tensor | None = None
    _neighbour_mask: torch.Tensor | None = None
    _color_groups: tuple[torch.Tensor, ...] | None = None

    def resample_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Resample vertices with precomputed triangle indices and weights."""

        value = torch.as_tensor(vertices, device=self.sphere_points.device)
        if value.ndim != 2 or tuple(value.shape) != (self.source_points, 3):
            raise ValueError(
                f"vertices shape must be {(self.source_points, 3)}, got {tuple(value.shape)}"
            )
        indices = self.surface_indices
        weights = self.surface_weights
        if value.dtype == torch.float32:
            result = torch.zeros(
                (indices.shape[0], value.shape[1]),
                dtype=value.dtype,
                device=value.device,
            )
            for corner in range(3):
                weighted = (
                    value[indices[:, corner]].to(torch.float64)
                    * weights[:, corner, None]
                ).to(torch.float32)
                result = result + weighted
            return result.contiguous()
        sampled = value[indices]
        return (sampled * weights.unsqueeze(-1)).sum(dim=1).to(value.dtype)

    def map_values_to_sheet(self, values: torch.Tensor) -> torch.Tensor:
        """Map spherical vertex values onto the DARTEL sheet grid."""

        value = torch.as_tensor(values, device=self.sphere_points.device).reshape(-1)
        if value.numel() != self.sphere_points.shape[0]:
            raise ValueError(
                f"values must contain {self.sphere_points.shape[0]} elements, got {value.numel()}"
            )
        mapped = (value[self.sheet_indices] * self.sheet_weights).sum(dim=-1)
        return mapped.reshape(self.ny, self.nx).contiguous()

    def _build_neighbours(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build neighbours."""

        if self._neighbours is not None and self._neighbour_mask is not None:
            return self._neighbours, self._neighbour_mask
        faces = self.faces.detach().cpu().numpy().astype(np.int64, copy=False)
        neighbours, mask = _build_neighbour_table(
            faces, int(self.sphere_points.shape[0])
        )
        target = self.sphere_points.device
        self._neighbours = torch.as_tensor(neighbours, dtype=torch.long, device=target)
        self._neighbour_mask = torch.as_tensor(mask, dtype=torch.bool, device=target)
        return self._neighbours, self._neighbour_mask

    def color_groups(self) -> tuple[torch.Tensor, ...]:
        """Color groups."""

        if self._color_groups is not None:
            return self._color_groups
        neighbours, neighbour_mask = self._build_neighbours()
        groups = _build_color_groups(
            neighbours.detach().cpu().numpy(),
            neighbour_mask.detach().cpu().numpy(),
        )
        target = self.sphere_points.device
        self._color_groups = tuple(
            torch.as_tensor(group, dtype=torch.long, device=target) for group in groups
        )
        return self._color_groups

    def _edge_mean_distance(self, vertices: torch.Tensor) -> torch.Tensor:
        """Edge mean distance."""

        geometry = vertices.to(torch.float64)
        face_points = geometry[self.faces]
        edge_lengths = torch.linalg.vector_norm(
            face_points[:, [1, 2, 0]] - face_points, dim=-1
        )
        return edge_lengths.mean()

    def _heat_parameters(
        self, vertices: torch.Tensor, fwhm: float
    ) -> tuple[int, torch.Tensor]:
        """Heat parameters."""

        edge_mean = self._edge_mean_distance(vertices)
        raw_iterations = float(fwhm) * float(fwhm) * 0.541011 / (edge_mean * edge_mean)
        iterations = max(1, int(torch.ceil(raw_iterations).item()))
        sigma = float(fwhm) * 0.7355345 / float(iterations**0.5)
        return iterations, torch.as_tensor(
            sigma, dtype=torch.float64, device=vertices.device
        )

    def smooth_geometry(self, vertices: torch.Tensor, fwhm: float) -> torch.Tensor:
        """Apply upstream-compatible iterative heat-kernel geometry smoothing."""

        neighbours, mask = self._build_neighbours()
        iterations, _sigma = self._heat_parameters(vertices, fwhm)
        safe_neighbours = neighbours.clamp_min(0)
        mask_value = mask.to(torch.float64)
        value = vertices.to(torch.float32)
        for _ in range(iterations):
            neighbour_values = value[safe_neighbours].to(torch.float64)
            numerator = (
                value.to(torch.float64)
                + (neighbour_values * mask_value.unsqueeze(-1)).sum(1) / 3.0
            )
            denominator = 1.0 + mask_value.sum(1) / 3.0
            value = (numerator / denominator.unsqueeze(-1)).to(torch.float32)
        return value.contiguous()

    def smooth_values(
        self, values: torch.Tensor, vertices: torch.Tensor, fwhm: float
    ) -> torch.Tensor:
        """Apply upstream-compatible iterative heat-kernel value smoothing."""

        neighbours, mask = self._build_neighbours()
        iterations, sigma = self._heat_parameters(vertices, fwhm)
        safe_neighbours = neighbours.clamp_min(0)
        mask_value = mask.to(vertices.dtype)
        value = values.to(torch.float64)
        geometry = vertices.to(torch.float64)
        neighbour_points = geometry[safe_neighbours]
        distances = torch.linalg.vector_norm(
            neighbour_points - geometry[:, None, :], dim=-1
        )
        weights = torch.exp(-(distances * distances) / (2.0 * sigma * sigma))
        weights = weights * mask_value
        for _ in range(iterations):
            neighbour_values = value[safe_neighbours]
            numerator = value + (weights * neighbour_values).sum(1)
            denominator = 1.0 + weights.sum(1)
            value = numerator / denominator
        return value.contiguous()

    def smooth_values_many(
        self,
        values: torch.Tensor,
        vertices: torch.Tensor,
        fwhms: tuple[float, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Smooth values many."""

        if not fwhms:
            return ()
        base = torch.as_tensor(
            values, dtype=torch.float64, device=self.sphere_points.device
        )
        if base.ndim != 1 or base.numel() != self.sphere_points.shape[0]:
            raise ValueError(
                "smooth_values_many values must have shape [sphere_points]"
            )
        geometry = torch.as_tensor(
            vertices, dtype=torch.float64, device=self.sphere_points.device
        )
        if tuple(geometry.shape) != tuple(base.shape) + (3,):
            raise ValueError(
                "smooth_values_many vertices must have shape [sphere_points, 3]"
            )
        neighbours, mask = self._build_neighbours()
        safe_neighbours = neighbours.clamp_min(0)
        iterations: list[int] = []
        sigmas: list[float] = []
        for fwhm in fwhms:
            count, sigma = self._heat_parameters(geometry, float(fwhm))
            iterations.append(count)
            sigmas.append(float(sigma.item()))

        neighbour_points = geometry[safe_neighbours]
        distance_squared = ((neighbour_points - geometry[:, None, :]) ** 2).sum(-1)
        sigma_tensor = torch.as_tensor(
            sigmas, dtype=torch.float64, device=geometry.device
        ).view(-1, 1, 1)
        weights = torch.exp(
            -distance_squared.unsqueeze(0) / (2.0 * sigma_tensor * sigma_tensor)
        )
        weights = weights * mask.to(torch.float64).unsqueeze(0)
        value = base.unsqueeze(0).expand(len(fwhms), -1).clone()
        for iteration in range(max(iterations)):
            active = [
                index for index, count in enumerate(iterations) if iteration < count
            ]
            if not active:
                break
            active_index = torch.as_tensor(
                active, dtype=torch.long, device=value.device
            )
            active_value = value.index_select(0, active_index)
            active_weights = weights.index_select(0, active_index)
            neighbour_values = active_value[:, safe_neighbours]
            numerator = active_value + (active_weights * neighbour_values).sum(-1)
            denominator = 1.0 + active_weights.sum(-1)
            value.index_copy_(
                0,
                active_index,
                numerator / denominator,
            )
        return tuple(value[index].contiguous() for index in range(len(fwhms)))

    def vertex_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        """Vertex normals."""

        geometry = vertices.to(torch.float32)
        face_points = geometry[self.faces]
        edge_a = face_points[:, 1] - face_points[:, 0]
        edge_b = face_points[:, 2] - face_points[:, 0]
        face_normals = torch.linalg.cross(edge_a, edge_b, dim=-1)
        face_normals = face_normals / torch.linalg.vector_norm(
            face_normals, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(vertices.dtype).eps)
        accumulated = torch.zeros_like(geometry)
        for corner in range(3):
            previous = face_points[:, (corner - 1) % 3] - face_points[:, corner]
            following = face_points[:, (corner + 1) % 3] - face_points[:, corner]
            denominator = (
                torch.linalg.vector_norm(previous, dim=-1)
                * torch.linalg.vector_norm(following, dim=-1)
            ).clamp_min(torch.finfo(torch.float32).eps)
            angle = torch.acos(
                (previous * following).sum(-1).div(denominator).clamp(-1.0, 1.0)
            )
            accumulated.index_add_(
                0, self.faces[:, corner], face_normals * angle[:, None]
            )
        return accumulated / torch.linalg.vector_norm(
            accumulated, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).eps)

    def curvature_type5(self, vertices: torch.Tensor, fwhm: float) -> torch.Tensor:
        """Curvature type5."""

        geometry = vertices.to(torch.float32)
        smoothed_vertices = self.smooth_geometry(geometry, 50.0)
        normals = self.vertex_normals(smoothed_vertices)
        values = (
            ((smoothed_vertices - geometry) * normals).sum(dim=-1).to(torch.float64)
        )
        values = self.smooth_values(values, geometry, fwhm)
        minimum = values.amin()
        maximum = values.amax()
        return ((values - minimum) / (maximum - minimum)).contiguous()

    def curvature_type5_to_sheet_many(
        self, vertices: torch.Tensor, fwhms: tuple[float, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Curvature type5 to sheet many."""

        if not fwhms:
            return ()
        geometry = vertices.to(torch.float32)
        smoothed_vertices = self.smooth_geometry(geometry, 50.0)
        normals = self.vertex_normals(smoothed_vertices)
        base_values = (
            ((smoothed_vertices - geometry) * normals).sum(dim=-1).to(torch.float64)
        )
        maps: list[torch.Tensor] = []
        smoothed_values = self.smooth_values_many(
            base_values, geometry, tuple(float(item) for item in fwhms)
        )
        for values in smoothed_values:
            minimum = values.amin()
            maximum = values.amax()
            normalised = (values - minimum) / (maximum - minimum)
            mapped = self.map_values_to_sheet(normalised)
            sheet_minimum = mapped.amin()
            sheet_maximum = mapped.amax()
            maps.append(
                (
                    (mapped - sheet_minimum) / (sheet_maximum - sheet_minimum)
                ).contiguous()
            )
        return tuple(maps)

    def curvature_type2(self, vertices: torch.Tensor, fwhm: float) -> torch.Tensor:
        """Curvature type2."""

        geometry = vertices.to(torch.float32)
        neighbours, mask = self._build_neighbours()
        safe_neighbours = neighbours.clamp_min(0)
        mask_value = mask.to(torch.float64)
        counts = mask_value.sum(dim=1)
        normals = self.vertex_normals(geometry).to(torch.float64)
        points = geometry.to(torch.float64)
        neighbour_points = points[safe_neighbours]
        neighbour_normals = normals[safe_neighbours]
        delta_coord = neighbour_points - points[:, None, :]
        delta_normal = neighbour_normals - normals[:, None, :]

        basis0 = delta_coord[:, 0, :]
        basis0 = basis0 - (basis0 * normals).sum(dim=-1, keepdim=True) * normals
        basis0 = basis0 / torch.linalg.vector_norm(
            basis0, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float64).eps)
        basis1 = torch.linalg.cross(-basis0, normals, dim=-1)
        projected_coord = (
            delta_coord
            - (delta_coord * normals[:, None, :]).sum(dim=-1, keepdim=True)
            * normals[:, None, :]
        )
        projected_normal = (
            delta_normal
            - (delta_normal * normals[:, None, :]).sum(dim=-1, keepdim=True)
            * normals[:, None, :]
        )
        dc_x = (projected_coord * basis0[:, None, :]).sum(dim=-1)
        dc_y = (projected_coord * basis1[:, None, :]).sum(dim=-1)
        dn_x = (projected_normal * basis0[:, None, :]).sum(dim=-1)
        dn_y = (projected_normal * basis1[:, None, :]).sum(dim=-1)

        sum1 = (dc_x * dn_x * mask_value).sum(dim=1)
        sum2 = ((dc_x * dn_y + dc_y * dn_x) * mask_value).sum(dim=1)
        sum3 = (dc_y * dn_y * mask_value).sum(dim=1)
        wx = (dc_x * dc_x * mask_value).sum(dim=1)
        wy = (dc_y * dc_y * mask_value).sum(dim=1)
        wxy = (dc_x * dc_y * mask_value).sum(dim=1)
        wx2 = wx * wx
        wy2 = wy * wy
        wxy2 = wxy * wxy
        denominator = (wx + wy) * (-wxy2 + wx * wy)
        safe_denominator = torch.where(
            denominator != 0.0,
            denominator,
            torch.ones_like(denominator),
        )
        a = (
            sum3 * wxy2 - sum2 * wxy * wy + sum1 * (-wxy2 + wx * wy + wy2)
        ) / safe_denominator
        b = (-sum3 * wx * wxy + sum2 * wx * wy - sum1 * wxy * wy) / safe_denominator
        c = (
            -sum2 * wx * wxy + sum1 * wxy2 + sum3 * (wx2 - wxy2 + wx * wy)
        ) / safe_denominator
        trace = a + c
        determinant = a * c - b * b
        discriminant = trace * trace - 4.0 * determinant
        root = torch.sqrt(discriminant.clamp_min(0.0))
        k1 = (trace + root) / 2.0
        k2 = (trace - root) / 2.0
        values = torch.where(
            counts > 2.0, torch.sqrt((k1 * k1 + k2 * k2) / 2.0), torch.zeros_like(k1)
        )
        values = self.smooth_values(values, geometry, fwhm)
        minimum = values.amin()
        maximum = values.amax()
        return ((values - minimum) / (maximum - minimum)).contiguous()

    def curvature_to_sheet(
        self,
        vertices: torch.Tensor,
        fwhm: float,
        *,
        curvtype: Literal[2, 5] = 5,
    ) -> torch.Tensor:
        """Compute supported CAT curvature and map it to a normalized sheet."""

        if curvtype == 5:
            values = self.curvature_type5(vertices, fwhm)
        elif curvtype == 2:
            values = self.curvature_type2(vertices, fwhm)
        else:
            raise NotImplementedError(
                "The GPU surface backend supports only upstream curvtype 2 and 5"
            )
        mapped = self.map_values_to_sheet(values)
        minimum = mapped.amin()
        maximum = mapped.amax()
        return ((mapped - minimum) / (maximum - minimum)).contiguous()
