"""Spatial candidate indexing and GPU costs for CAT initial rotation."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .surface_stencil import SurfaceStencil


def official_seed_grid_angles() -> np.ndarray:
    """Official seed grid angles."""

    seed0 = (-0.6, -0.3, 0.0, 0.3, 0.6)
    seed12 = (-0.3, 0.0, 0.3)
    values = [(0.0, 0.0, 0.0)]
    values.extend(
        (alpha, beta, gamma) for alpha in seed0 for beta in seed12 for gamma in seed12
    )
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class RotationGridIndex:
    """Store a reusable CPU spatial index over target-sphere triangles."""

    points: np.ndarray
    faces: np.ndarray
    candidate_table: np.ndarray | None
    grid_size: int
    margin: int
    candidate_offsets: np.ndarray | None = None
    candidate_faces: np.ndarray | None = None

    @classmethod
    def from_stencil(
        cls,
        stencil: SurfaceStencil,
        *,
        grid_size: int = 128,
        margin: int = 1,
        include_dense_candidate_table: bool = True,
    ) -> "RotationGridIndex":
        """From stencil."""

        return cls.from_geometry(
            stencil.sphere_points,
            stencil.faces,
            grid_size=grid_size,
            margin=margin,
            include_dense_candidate_table=include_dense_candidate_table,
        )

    @classmethod
    def from_geometry(
        cls,
        points: np.ndarray,
        faces: np.ndarray,
        *,
        grid_size: int = 128,
        margin: int = 1,
        include_dense_candidate_table: bool = True,
    ) -> "RotationGridIndex":
        """From geometry."""

        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [n, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [m, 3]")
        if grid_size < 4 or margin < 0:
            raise ValueError(
                "grid_size must be at least 4 and margin must be non-negative"
            )

        cell_size = 2.0 / float(grid_size)
        triangles = points[faces]
        lower = np.floor((triangles.min(axis=1) + 1.0) / cell_size).astype(np.int64)
        upper = np.floor((triangles.max(axis=1) + 1.0) / cell_size).astype(np.int64)
        lower = np.clip(lower - margin, 0, grid_size - 1)
        upper = np.clip(upper + margin, 0, grid_size - 1)
        cells: dict[int, list[int]] = {}
        for face, (lo, hi) in enumerate(zip(lower, upper)):
            for z in range(int(lo[2]), int(hi[2]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    base = grid_size * (y + grid_size * z)
                    for x in range(int(lo[0]), int(hi[0]) + 1):
                        cells.setdefault(base + x, []).append(face)

        if not cells:
            raise ValueError(
                "At least one triangle does not cover any spatial grid cell"
            )
        max_candidates = max(len(value) for value in cells.values())
        table = None
        if include_dense_candidate_table:
            table = np.full((grid_size**3, max_candidates), -1, dtype=np.int32)
            for cell, values in cells.items():
                table[cell, : len(values)] = values
        offsets = np.zeros(grid_size**3 + 1, dtype=np.int32)
        compressed_faces: list[int] = []
        for cell in range(grid_size**3):
            values = cells.get(cell, ())
            compressed_faces.extend(values)
            offsets[cell + 1] = len(compressed_faces)
        return cls(
            points=np.ascontiguousarray(points),
            faces=np.ascontiguousarray(faces),
            candidate_table=(None if table is None else np.ascontiguousarray(table)),
            grid_size=int(grid_size),
            margin=int(margin),
            candidate_offsets=np.ascontiguousarray(offsets),
            candidate_faces=np.ascontiguousarray(
                np.asarray(compressed_faces, dtype=np.int32)
            ),
        )

    @property
    def max_candidates(self) -> int:
        """Max candidates."""

        if self.candidate_offsets is not None:
            return int(np.diff(self.candidate_offsets).max())
        if self.candidate_table is None:
            return 0
        return int(self.candidate_table.shape[1])

    def to(
        self,
        device: str | torch.device,
        *,
        geometry_dtype: torch.dtype = torch.float32,
        candidate_table_dtype: torch.dtype = torch.int32,
        compressed_candidate_table: bool = True,
    ) -> "RotationGridIndexDevice":
        """To."""

        target = torch.device(device)
        triangles = self.points[self.faces]
        if compressed_candidate_table:
            if self.candidate_offsets is None or self.candidate_faces is None:
                raise ValueError("RotationGridIndex is missing candidate index")
            candidate_table = torch.empty(0, dtype=candidate_table_dtype, device=target)
            candidate_offsets = torch.as_tensor(
                self.candidate_offsets, dtype=torch.int32, device=target
            ).contiguous()
            candidate_faces = torch.as_tensor(
                self.candidate_faces,
                dtype=candidate_table_dtype,
                device=target,
            ).contiguous()
        else:
            if self.candidate_table is None:
                raise ValueError(
                    "Dense candidates were requested, but this index contains only CSR data"
                )
            candidate_table = torch.as_tensor(
                self.candidate_table, dtype=candidate_table_dtype, device=target
            ).contiguous()
            candidate_offsets = None
            candidate_faces = None
        return RotationGridIndexDevice(
            target_points=torch.as_tensor(
                self.points, dtype=geometry_dtype, device=target
            ).contiguous(),
            target_faces=torch.as_tensor(
                self.faces, dtype=torch.long, device=target
            ).contiguous(),
            target_triangles=torch.as_tensor(
                triangles, dtype=geometry_dtype, device=target
            ).contiguous(),
            candidate_table=candidate_table,
            grid_size=self.grid_size,
            candidate_offsets=candidate_offsets,
            candidate_faces=candidate_faces,
        )


@dataclass(frozen=True)
class RotationCostInputs:
    """Store validated source/target tensors reused by every rotation candidate."""

    source_points: torch.Tensor
    source_values: torch.Tensor
    target_values: torch.Tensor


@dataclass(frozen=True)
class RotationSearchResult:
    """Store seed costs, refined angles, matrix, cost, and iteration count."""

    seed_angle: torch.Tensor
    seed_cost: torch.Tensor
    seed_costs: torch.Tensor
    angle: torch.Tensor
    cost: torch.Tensor
    rotation_matrix: torch.Tensor
    iterations: int


@dataclass
class RotationGridIndexDevice:
    """Evaluate CAT rotation candidates against a device-resident spatial index."""

    target_points: torch.Tensor
    target_faces: torch.Tensor
    target_triangles: torch.Tensor
    candidate_table: torch.Tensor
    grid_size: int
    candidate_offsets: torch.Tensor | None = None
    candidate_faces: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        """Device."""

        return self.target_points.device

    def _rotation_matrices(self, angles: torch.Tensor) -> torch.Tensor:
        """Rotation matrices."""

        angles = torch.as_tensor(
            angles, dtype=self.target_points.dtype, device=self.device
        )
        if angles.ndim == 1:
            angles = angles.reshape(1, 3)
        if angles.ndim != 2 or angles.shape[1] != 3:
            raise ValueError("angles must have shape [candidates, 3]")
        alpha, beta, gamma = angles.unbind(dim=1)
        zero = torch.zeros_like(alpha)
        one = torch.ones_like(alpha)
        rx = torch.stack(
            (
                one,
                zero,
                zero,
                zero,
                torch.cos(alpha),
                torch.sin(alpha),
                zero,
                -torch.sin(alpha),
                torch.cos(alpha),
            ),
            dim=1,
        )
        ry = torch.stack(
            (
                torch.cos(beta),
                zero,
                torch.sin(beta),
                zero,
                one,
                zero,
                -torch.sin(beta),
                zero,
                torch.cos(beta),
            ),
            dim=1,
        )
        rz = torch.stack(
            (
                torch.cos(gamma),
                torch.sin(gamma),
                zero,
                -torch.sin(gamma),
                torch.cos(gamma),
                zero,
                zero,
                zero,
                one,
            ),
            dim=1,
        )
        first = torch.empty_like(rx)
        result = torch.empty_like(rx)
        for row in range(3):
            for column in range(3):
                first[:, row + 3 * column] = sum(
                    ry[:, row + 3 * index] * rx[:, index + 3 * column]
                    for index in range(3)
                )
        for row in range(3):
            for column in range(3):
                result[:, row + 3 * column] = sum(
                    rz[:, row + 3 * index] * first[:, index + 3 * column]
                    for index in range(3)
                )
        return result.reshape(-1, 3, 3)

    def rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Rotation matrix."""

        matrices = self._rotation_matrices(angles)
        if matrices.shape[0] != 1:
            raise ValueError("rotation_matrix accepts exactly one angle vector")
        return matrices[0]

    def rotate_points(
        self, source_points: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        """Rotate points."""

        source = torch.as_tensor(
            source_points, dtype=self.target_points.dtype, device=self.device
        ).contiguous()
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source_points must have shape [n, 3]")
        matrix = self._rotation_matrices(angles)
        return torch.matmul(source.unsqueeze(0), matrix.transpose(1, 2))

    @staticmethod
    def _segment_distance(
        point: torch.Tensor, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment distance."""

        direction = second - first
        offset = point - first
        denominator = (direction * direction).sum(dim=-1)
        alpha = torch.where(
            denominator != 0.0,
            (offset * direction).sum(dim=-1) / denominator,
            torch.zeros_like(denominator),
        )
        alpha = alpha.clamp(0.0, 1.0)
        closest = first + alpha.unsqueeze(-1) * direction
        distance = ((point - closest) ** 2).sum(dim=-1)
        return distance, closest

    @staticmethod
    def _barycentric(point: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
        """Barycentric."""

        values = []
        for first, second, third in (
            (triangle[..., 1, :], triangle[..., 2, :], triangle[..., 0, :]),
            (triangle[..., 2, :], triangle[..., 0, :], triangle[..., 1, :]),
            (triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]),
        ):
            point_offset = point - first
            horizontal = second - first
            upward = third - first
            normal = torch.cross(horizontal, upward, dim=-1)
            vertical = torch.cross(normal, horizontal, dim=-1)
            numerator = (point_offset * vertical).sum(dim=-1)
            denominator = (upward * vertical).sum(dim=-1)
            values.append(numerator / denominator)
        return torch.stack(values, dim=-1)

    def _locate_chunk(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Locate chunk."""

        candidates = self._query_grid(query)
        valid = candidates >= 0
        if not bool(valid.any(dim=-1).all()):
            raise RuntimeError(
                "Rotation is not covered by target sphere candidate index"
            )
        safe = candidates.clamp_min(0)
        triangles = self.target_triangles[safe.to(torch.long)]
        first = triangles[..., 0, :]
        second = triangles[..., 1, :]
        third = triangles[..., 2, :]
        edge_a = second - first
        edge_b = third - first
        normal = torch.cross(edge_a, edge_b, dim=-1)
        normal_sq = (normal * normal).sum(dim=-1)
        offset = first - query.unsqueeze(2)
        plane_t = torch.where(
            normal_sq != 0.0,
            (offset * normal).sum(dim=-1) / normal_sq,
            torch.zeros_like(normal_sq),
        )
        projected = query.unsqueeze(2) + plane_t.unsqueeze(-1) * normal
        point_offset = projected - first
        xx = (edge_a * edge_a).sum(dim=-1)
        xy = (edge_a * edge_b).sum(dim=-1)
        xv = (edge_a * point_offset).sum(dim=-1)
        yy = (edge_b * edge_b).sum(dim=-1)
        yv = (edge_b * point_offset).sum(dim=-1)
        denominator = xx * yy - xy * xy
        xpos = torch.where(
            denominator != 0.0,
            (xv * yy - yv * xy) / denominator,
            torch.zeros_like(denominator),
        )
        ypos = torch.where(
            denominator != 0.0,
            (yv * xx - xv * xy) / denominator,
            torch.zeros_like(denominator),
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
        vertex_distance = torch.stack(
            (
                ((query.unsqueeze(2) - first) ** 2).sum(dim=-1),
                ((query.unsqueeze(2) - second) ** 2).sum(dim=-1),
                ((query.unsqueeze(2) - third) ** 2).sum(dim=-1),
            ),
            dim=-1,
        )
        closest_vertex = vertex_distance.argmin(dim=-1)
        vertices = torch.stack((first, second, third), dim=-2)
        vertex_index = (
            closest_vertex.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*closest_vertex.shape, 1, 3)
        )
        closest = torch.gather(vertices, -2, vertex_index).squeeze(-2)
        previous_index = ((closest_vertex - 1) % 3).unsqueeze(-1).unsqueeze(-1)
        next_index = ((closest_vertex + 1) % 3).unsqueeze(-1).unsqueeze(-1)
        previous = torch.gather(
            vertices, -2, previous_index.expand(*closest_vertex.shape, 1, 3)
        ).squeeze(-2)
        following = torch.gather(
            vertices, -2, next_index.expand(*closest_vertex.shape, 1, 3)
        ).squeeze(-2)
        edge_one_distance, edge_one_point = self._segment_distance(
            query.unsqueeze(2), previous, closest
        )
        edge_two_distance, edge_two_point = self._segment_distance(
            query.unsqueeze(2), closest, following
        )
        first_edge = edge_one_distance < edge_two_distance
        outside_distance = torch.where(first_edge, edge_one_distance, edge_two_distance)
        outside_point = torch.where(
            first_edge.unsqueeze(-1), edge_one_point, edge_two_point
        )
        candidate_distance = torch.where(inside, plane_distance, outside_distance)
        candidate_point = torch.where(inside.unsqueeze(-1), projected, outside_point)
        candidate_distance = torch.where(
            valid, candidate_distance, torch.full_like(candidate_distance, float("inf"))
        )
        best = candidate_distance.argmin(dim=-1)
        face = torch.gather(candidates, -1, best.unsqueeze(-1)).squeeze(-1)
        point_index = best.unsqueeze(-1).unsqueeze(-1).expand(*best.shape, 1, 3)
        point = torch.gather(candidate_point, -2, point_index).squeeze(-2)
        triangle = torch.gather(
            triangles, -3, point_index.unsqueeze(-2).expand(*best.shape, 1, 3, 3)
        ).squeeze(-3)
        return face, self._barycentric(point, triangle)

    def _query_grid(self, points: torch.Tensor) -> torch.Tensor:
        """Query grid."""

        cell = torch.floor((points + 1.0) * (self.grid_size / 2.0)).to(torch.long)
        cell = cell.clamp(0, self.grid_size - 1)
        cell_id = cell[..., 0] + self.grid_size * (
            cell[..., 1] + self.grid_size * cell[..., 2]
        )
        if self.candidate_offsets is None or self.candidate_faces is None:
            return self.candidate_table[cell_id.to(torch.long)]
        starts = self.candidate_offsets[cell_id.to(torch.long)]
        stops = self.candidate_offsets[(cell_id + 1).to(torch.long)]
        width = int((stops - starts).max().item())
        if width < 1:
            raise RuntimeError(
                "Rotation is not covered by target sphere candidate index"
            )
        local = torch.arange(width, dtype=torch.long, device=points.device)
        positions = starts.unsqueeze(-1).to(torch.long) + local
        valid = positions < stops.unsqueeze(-1).to(torch.long)
        safe = positions.clamp_max(self.candidate_faces.numel() - 1)
        candidates = self.candidate_faces[safe]
        return torch.where(valid, candidates, torch.full_like(candidates, -1))

    def prepare_cost_inputs(
        self,
        source_points: torch.Tensor,
        source_values: torch.Tensor,
        target_values: torch.Tensor,
    ) -> RotationCostInputs:
        """Prepare cost inputs."""

        source = torch.as_tensor(
            source_points, dtype=self.target_points.dtype, device=self.device
        ).contiguous()
        source_curvature = (
            torch.as_tensor(source_values, dtype=torch.float64, device=self.device)
            .reshape(-1)
            .contiguous()
        )
        target_curvature = (
            torch.as_tensor(target_values, dtype=torch.float64, device=self.device)
            .reshape(-1)
            .contiguous()
        )
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source_points must have shape [n, 3]")
        if source_curvature.numel() != source.shape[0]:
            raise ValueError("source_values length does not match source_points")
        if target_curvature.numel() != self.target_points.shape[0]:
            raise ValueError("target_values length does not match the target sphere")
        return RotationCostInputs(
            source_points=source,
            source_values=source_curvature,
            target_values=target_curvature,
        )

    def cost_batch_prepared(
        self,
        inputs: RotationCostInputs,
        angles: torch.Tensor,
        *,
        point_chunk: int = 1024,
    ) -> torch.Tensor:
        """Cost batch prepared."""

        if point_chunk < 1:
            raise ValueError("point_chunk must be positive")
        if inputs.source_points.device != self.device:
            raise ValueError(
                "RotationCostInputs and the index must be on the same device"
            )
        source = inputs.source_points
        source_curvature = inputs.source_values
        target_curvature = inputs.target_values
        angles = torch.as_tensor(angles, device=self.device)
        if angles.ndim == 1:
            angles = angles.reshape(1, 3)
        if angles.ndim != 2 or angles.shape[1] != 3:
            raise ValueError("angles must have shape [candidates, 3]")
        rotated = self.rotate_points(source, angles)
        costs = torch.zeros(angles.shape[0], dtype=torch.float64, device=self.device)
        for start in range(0, source.shape[0], point_chunk):
            stop = min(start + point_chunk, source.shape[0])
            faces, weights = self._locate_chunk(rotated[:, start:stop])
            vertices = self.target_faces[faces.to(torch.long)]
            sampled = (target_curvature[vertices] * weights.to(torch.float64)).sum(
                dim=-1
            )
            difference = source_curvature[start:stop].unsqueeze(0) - sampled
            costs += (difference * difference).sum(dim=1)
        return costs

    def cost_batch(
        self,
        source_points: torch.Tensor,
        source_values: torch.Tensor,
        target_values: torch.Tensor,
        angles: torch.Tensor,
        *,
        point_chunk: int = 1024,
    ) -> torch.Tensor:
        """Cost batch."""

        inputs = self.prepare_cost_inputs(source_points, source_values, target_values)
        return self.cost_batch_prepared(inputs, angles, point_chunk=point_chunk)

    def coarse_seed_search_prepared(
        self,
        inputs: RotationCostInputs,
        *,
        point_chunk: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate all 46 upstream seeds using already prepared device inputs."""

        angles = torch.as_tensor(
            official_seed_grid_angles(), dtype=torch.float64, device=self.device
        )
        costs = self.cost_batch_prepared(inputs, angles, point_chunk=point_chunk)
        best_index = torch.argmin(costs)
        return angles[best_index], costs[best_index], costs

    def coarse_seed_search(
        self,
        source_points: torch.Tensor,
        source_values: torch.Tensor,
        target_values: torch.Tensor,
        *,
        point_chunk: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coarse seed search."""

        inputs = self.prepare_cost_inputs(
            source_points,
            source_values,
            target_values,
        )
        return self.coarse_seed_search_prepared(inputs, point_chunk=point_chunk)

    def refine_nelder_mead_prepared(
        self,
        inputs: RotationCostInputs,
        initial_angles: torch.Tensor,
        *,
        point_chunk: int = 1024,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        simplex_step: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Run CAT-ordered Nelder-Mead using already prepared device inputs."""

        if max_iter < 1 or tol < 0.0 or simplex_step <= 0.0:
            raise ValueError("max_iter, tol, and simplex_step must be positive")
        seed = torch.as_tensor(
            initial_angles, dtype=torch.float64, device=self.device
        ).reshape(-1)
        if seed.numel() != 3:
            raise ValueError("initial_angles must contain exactly three angles")
        simplex = torch.stack(
            (
                seed,
                seed
                + torch.as_tensor(
                    (simplex_step, 0.0, 0.0), dtype=torch.float64, device=self.device
                ),
                seed
                + torch.as_tensor(
                    (0.0, simplex_step, 0.0), dtype=torch.float64, device=self.device
                ),
                seed
                + torch.as_tensor(
                    (0.0, 0.0, simplex_step), dtype=torch.float64, device=self.device
                ),
            ),
            dim=0,
        )
        f_values = self.cost_batch_prepared(
            inputs,
            simplex,
            point_chunk=point_chunk,
        )

        def evaluate_one(candidate: torch.Tensor) -> torch.Tensor:
            """Evaluate one."""

            return self.cost_batch_prepared(
                inputs,
                candidate.reshape(1, 3),
                point_chunk=point_chunk,
            )[0]

        for iteration in range(max_iter):
            highest = 0
            lowest = 0
            second_highest = 1
            for index in range(4):
                if bool(f_values[index] > f_values[highest]):
                    second_highest = highest
                    highest = index
                elif bool(
                    (f_values[index] > f_values[second_highest]) and index != highest
                ):
                    second_highest = index
                if bool(f_values[index] < f_values[lowest]):
                    lowest = index

            centroid = torch.zeros(3, dtype=torch.float64, device=self.device)
            for index in range(4):
                if index != highest:
                    centroid += simplex[index]
            centroid /= 3.0

            reflected = centroid + (centroid - simplex[highest])
            f_reflected = evaluate_one(reflected)
            if bool(f_reflected < f_values[lowest]):
                expanded = centroid + 2.0 * (reflected - centroid)
                f_expanded = evaluate_one(expanded)
                if bool(f_expanded < f_reflected):
                    simplex[highest] = expanded
                    f_values[highest] = f_expanded
                else:
                    simplex[highest] = reflected
                    f_values[highest] = f_reflected
            elif bool(f_reflected < f_values[second_highest]):
                simplex[highest] = reflected
                f_values[highest] = f_reflected
            else:
                contracted = centroid + 0.5 * (simplex[highest] - centroid)
                f_contracted = evaluate_one(contracted)
                if bool(f_contracted < f_values[highest]):
                    simplex[highest] = contracted
                    f_values[highest] = f_contracted
                else:
                    for index in range(4):
                        if index != lowest:
                            simplex[index] = simplex[lowest] + 0.5 * (
                                simplex[index] - simplex[lowest]
                            )
                            f_values[index] = evaluate_one(simplex[index])

            max_diff = torch.max(torch.abs(f_values - f_values[lowest]))
            if bool(max_diff < tol):
                break

        return simplex[lowest].clone(), f_values[lowest].clone(), iteration + 1

    def refine_nelder_mead(
        self,
        source_points: torch.Tensor,
        source_values: torch.Tensor,
        target_values: torch.Tensor,
        initial_angles: torch.Tensor,
        *,
        point_chunk: int = 1024,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        simplex_step: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Refine nelder mead."""

        inputs = self.prepare_cost_inputs(source_points, source_values, target_values)
        return self.refine_nelder_mead_prepared(
            inputs,
            initial_angles,
            point_chunk=point_chunk,
            max_iter=max_iter,
            tol=tol,
            simplex_step=simplex_step,
        )

    def search_rotation(
        self,
        source_points: torch.Tensor,
        source_values: torch.Tensor,
        target_values: torch.Tensor,
        *,
        point_chunk: int = 1024,
        refine: bool = True,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        simplex_step: float = 0.1,
    ) -> RotationSearchResult:
        """Search rotation."""

        inputs = self.prepare_cost_inputs(source_points, source_values, target_values)
        return self.search_rotation_prepared(
            inputs,
            point_chunk=point_chunk,
            refine=refine,
            max_iter=max_iter,
            tol=tol,
            simplex_step=simplex_step,
        )

    def search_rotation_prepared(
        self,
        inputs: RotationCostInputs,
        *,
        point_chunk: int = 1024,
        refine: bool = True,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        simplex_step: float = 0.1,
    ) -> RotationSearchResult:
        """Run seed search and optional refinement on prepared inputs."""

        seed_angle, seed_cost, seed_costs = self.coarse_seed_search_prepared(
            inputs, point_chunk=point_chunk
        )
        if refine:
            angle, cost, iterations = self.refine_nelder_mead_prepared(
                inputs,
                seed_angle,
                point_chunk=point_chunk,
                max_iter=max_iter,
                tol=tol,
                simplex_step=simplex_step,
            )
        else:
            angle = seed_angle.clone()
            cost = seed_cost.clone()
            iterations = 0
        return RotationSearchResult(
            seed_angle=seed_angle.clone(),
            seed_cost=seed_cost.clone(),
            seed_costs=seed_costs,
            angle=angle,
            cost=cost,
            rotation_matrix=self.rotation_matrix(angle),
            iterations=iterations,
        )
