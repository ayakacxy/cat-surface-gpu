"""High-level initial-rotation pipeline over fixed native helper outputs."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dartel_grid import resolve_device
from .surface_rotation import (
    RotationCostInputs,
    RotationGridIndex,
    RotationGridIndexDevice,
    RotationSearchResult,
)
from .surface_stencil import SurfaceStencil


def read_rotation_values(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read rotation values."""

    path = Path(path)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if header.size != 2:
            raise ValueError(f"Rotation feature file : {path}")
        source = np.fromfile(stream, dtype="<f8", count=int(header[0]))
        target = np.fromfile(stream, dtype="<f8", count=int(header[1]))
        trailing = stream.read(1)
    if source.size != int(header[0]) or target.size != int(header[1]):
        raise ValueError(f"Rotation feature file length : {path}")
    if trailing:
        raise ValueError(f"Rotation feature file : {path}")
    return np.ascontiguousarray(source), np.ascontiguousarray(target)


def read_rotation_points(path: str | Path) -> np.ndarray:
    """Read rotation points."""

    path = Path(path)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=1)
        if header.size != 1:
            raise ValueError(f"Rotation file : {path}")
        n_points = int(header[0])
        if n_points < 0:
            raise ValueError(f"Rotation file : {path}")
        payload = stream.read()

    n_values = 3 * n_points
    expected_float32_bytes = n_values * np.dtype("<f4").itemsize
    expected_float64_bytes = n_values * np.dtype("<f8").itemsize
    if len(payload) == expected_float32_bytes:
        points = np.frombuffer(payload, dtype="<f4", count=n_values)
    elif len(payload) == expected_float64_bytes:
        points = np.frombuffer(payload, dtype="<f8", count=n_values)
    else:
        raise ValueError(
            f"Rotation file length : {path}, {len(payload)} ,"
            f"{expected_float32_bytes} or {expected_float64_bytes}"
        )
    return np.ascontiguousarray(points.reshape(n_points, 3))


@dataclass
class RotationPipeline:
    """Combine a reusable target index with CAT-compatible rotation search."""

    index_cpu: RotationGridIndex
    index: RotationGridIndexDevice

    @classmethod
    def from_stencil(
        cls,
        target_stencil: SurfaceStencil,
        *,
        device: str | torch.device = "auto",
        grid_size: int = 128,
        margin: int = 1,
        candidate_table_dtype: torch.dtype = torch.int32,
        geometry_dtype: torch.dtype = torch.float32,
        compressed_candidate_table: bool = True,
    ) -> "RotationPipeline":
        """Build and upload a rotation index from a surface stencil."""

        target_device = resolve_device(device)
        index_cpu = RotationGridIndex.from_stencil(
            target_stencil,
            grid_size=grid_size,
            margin=margin,
            include_dense_candidate_table=not compressed_candidate_table,
        )
        index = index_cpu.to(
            target_device,
            geometry_dtype=geometry_dtype,
            candidate_table_dtype=candidate_table_dtype,
            compressed_candidate_table=compressed_candidate_table,
        )
        return cls(index_cpu=index_cpu, index=index)

    @classmethod
    def from_index_cpu(
        cls,
        index_cpu: RotationGridIndex,
        *,
        device: str | torch.device = "auto",
        candidate_table_dtype: torch.dtype = torch.int32,
        geometry_dtype: torch.dtype = torch.float32,
    ) -> "RotationPipeline":
        """From index cpu."""

        target_device = resolve_device(device)
        index = index_cpu.to(
            target_device,
            geometry_dtype=geometry_dtype,
            candidate_table_dtype=candidate_table_dtype,
        )
        return cls(index_cpu=index_cpu, index=index)

    @property
    def device(self) -> torch.device:
        """Device."""

        return self.index.device

    def prepare_inputs(
        self,
        source_points: np.ndarray | torch.Tensor,
        source_values: np.ndarray | torch.Tensor,
        target_values: np.ndarray | torch.Tensor,
    ) -> RotationCostInputs:
        """Validate and upload source points and source/target features once."""

        return self.index.prepare_cost_inputs(
            source_points, source_values, target_values
        )

    def search(
        self,
        source_points: np.ndarray | torch.Tensor,
        source_values: np.ndarray | torch.Tensor,
        target_values: np.ndarray | torch.Tensor,
        *,
        point_chunk: int = 1024,
        refine: bool = True,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        simplex_step: float = 0.1,
    ) -> RotationSearchResult:
        """Run the 46-seed search and optional CAT-ordered refinement."""

        inputs = self.prepare_inputs(source_points, source_values, target_values)
        return self.index.search_rotation_prepared(
            inputs,
            point_chunk=point_chunk,
            refine=refine,
            max_iter=max_iter,
            tol=tol,
            simplex_step=simplex_step,
        )


def run_rotation_pipeline(
    target_stencil: SurfaceStencil,
    source_points: np.ndarray | torch.Tensor,
    source_values: np.ndarray | torch.Tensor,
    target_values: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "auto",
    grid_size: int = 128,
    margin: int = 1,
    candidate_table_dtype: torch.dtype = torch.int32,
    geometry_dtype: torch.dtype = torch.float32,
    point_chunk: int = 1024,
    refine: bool = True,
    max_iter: int = 500,
    tol: float = 1.0e-4,
    simplex_step: float = 0.1,
) -> RotationSearchResult:
    """Run rotation pipeline."""

    pipeline = RotationPipeline.from_stencil(
        target_stencil,
        device=device,
        grid_size=grid_size,
        margin=margin,
        candidate_table_dtype=candidate_table_dtype,
        geometry_dtype=geometry_dtype,
    )
    return pipeline.search(
        source_points,
        source_values,
        target_values,
        point_chunk=point_chunk,
        refine=refine,
        max_iter=max_iter,
        tol=tol,
        simplex_step=simplex_step,
    )


def run_rotation_pipeline_from_files(
    source_points_path: str | Path,
    target_stencil_path: str | Path,
    rotation_values_path: str | Path,
    *,
    device: str | torch.device = "auto",
    grid_size: int = 128,
    margin: int = 1,
    candidate_table_dtype: torch.dtype = torch.int32,
    geometry_dtype: torch.dtype = torch.float32,
    point_chunk: int = 1024,
    refine: bool = True,
    max_iter: int = 500,
    tol: float = 1.0e-4,
    simplex_step: float = 0.1,
) -> RotationSearchResult:
    """Run rotation pipeline from files."""

    source_points = read_rotation_points(source_points_path)
    source_values, target_values = read_rotation_values(rotation_values_path)
    target_stencil = SurfaceStencil.from_file(target_stencil_path)
    return run_rotation_pipeline(
        target_stencil,
        source_points,
        source_values,
        target_values,
        device=device,
        grid_size=grid_size,
        margin=margin,
        candidate_table_dtype=candidate_table_dtype,
        geometry_dtype=geometry_dtype,
        point_chunk=point_chunk,
        refine=refine,
        max_iter=max_iter,
        tol=tol,
        simplex_step=simplex_step,
    )
