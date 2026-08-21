"""CAT-Surface GPU implementation."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import torch

from cat_surface_gpu import (
    RotationGridIndex,
    RotationPipeline,
    SurfaceStencil,
    official_seed_grid_angles,
)


def _make_tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    """Make tetrahedron."""

    points = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        ((0, 2, 3), (2, 1, 3), (1, 0, 3), (0, 1, 2)),
        dtype=np.int32,
    )
    return points, faces


def test_rotation_grid_identity_cost_is_zero_for_vertex_values():
    """Test rotation grid identity cost is zero for vertex values."""

    points, faces = _make_tetrahedron()
    index = RotationGridIndex.from_geometry(points, faces, grid_size=16, margin=16).to(
        "cpu"
    )
    assert index.candidate_table.dtype == torch.int32
    reference_index = RotationGridIndex.from_geometry(
        points, faces, grid_size=16, margin=16
    ).to("cpu", candidate_table_dtype=torch.int64)
    assert reference_index.candidate_table.dtype == torch.int64
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    costs = index.cost_batch(
        torch.as_tensor(points),
        values,
        values,
        torch.zeros((1, 3), dtype=torch.float64),
        point_chunk=2,
    )
    torch.testing.assert_close(
        costs,
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_rotation_pipeline_default_keeps_only_compressed_candidate_index():
    """Test rotation pipeline default keeps only compressed candidate index."""

    points, faces = _make_tetrahedron()
    stencil = SurfaceStencil(
        sphere_points=points.astype(np.float64),
        faces=faces,
        surface_indices=np.zeros((4, 3), dtype=np.int32),
        surface_weights=np.zeros((4, 3), dtype=np.float64),
        sheet_indices=np.zeros((1, 1, 3), dtype=np.int32),
        sheet_weights=np.zeros((1, 1, 3), dtype=np.float64),
        nx=1,
        ny=1,
        source_points=4,
    )
    pipeline = RotationPipeline.from_stencil(
        stencil, device="cpu", grid_size=16, margin=16
    )
    assert pipeline.index_cpu.candidate_table is None
    assert pipeline.index_cpu.max_candidates > 0
    assert pipeline.index.candidate_offsets is not None


def test_rotation_grid_batch_returns_finite_distinct_candidates():
    """Test rotation grid batch returns finite distinct candidates."""

    points, faces = _make_tetrahedron()
    index = RotationGridIndex.from_geometry(points, faces, grid_size=16, margin=16).to(
        "cpu"
    )
    source_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    target_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    angles = torch.tensor(((0.0, 0.0, 0.0), (0.1, -0.2, 0.3)), dtype=torch.float64)
    costs = index.cost_batch(
        torch.as_tensor(points),
        source_values,
        target_values,
        angles,
        point_chunk=2,
    )
    assert costs.shape == (2,)
    assert torch.isfinite(costs).all()
    assert torch.isfinite(costs[1])


def test_prepared_cost_inputs_preserve_cost_batch_values():
    """Test prepared cost inputs preserve cost batch values."""

    points, faces = _make_tetrahedron()
    index = RotationGridIndex.from_geometry(points, faces, grid_size=16, margin=16).to(
        "cpu"
    )
    source_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    angles = torch.tensor(((0.0, 0.0, 0.0), (0.1, -0.2, 0.3)), dtype=torch.float64)
    ordinary = index.cost_batch(
        points, source_values, source_values, angles, point_chunk=2
    )
    prepared = index.prepare_cost_inputs(points, source_values, source_values)
    resident = index.cost_batch_prepared(prepared, angles, point_chunk=2)
    torch.testing.assert_close(resident, ordinary, rtol=0.0, atol=0.0)


def test_rotation_grid_uses_c_array_rotation_convention_for_identity():
    """Test rotation grid uses c array rotation convention for identity."""

    points, faces = _make_tetrahedron()
    index = RotationGridIndex.from_geometry(points, faces, grid_size=16).to("cpu")
    rotated = index.rotate_points(
        torch.as_tensor(points), torch.zeros((1, 3), dtype=torch.float64)
    )
    torch.testing.assert_close(rotated[0], torch.as_tensor(points), rtol=0.0, atol=0.0)


def test_official_seed_grid_preserves_c_order_and_duplicate_identity():
    """Test official seed grid preserves c order and duplicate identity."""

    angles = official_seed_grid_angles()
    assert angles.shape == (46, 3)
    np.testing.assert_array_equal(angles[0], (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(angles[1], (-0.6, -0.3, -0.3))
    np.testing.assert_array_equal(angles[23], (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(angles[-1], (0.6, 0.3, 0.3))


def test_coarse_seed_search_keeps_first_equal_minimum():
    """Test coarse seed search keeps first equal minimum."""

    points, faces = _make_tetrahedron()
    index = RotationGridIndex.from_geometry(points, faces, grid_size=16, margin=16).to(
        "cpu"
    )
    zeros = torch.zeros(4, dtype=torch.float64)
    best_angle, best_cost, costs = index.coarse_seed_search(
        torch.as_tensor(points), zeros, zeros, point_chunk=2
    )
    assert costs.shape == (46,)
    torch.testing.assert_close(
        best_angle,
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        best_cost,
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_nelder_mead_keeps_first_simplex_point_for_equal_costs():
    """Test nelder mead keeps first simplex point for equal costs."""

    points, faces = _make_tetrahedron()
    index = RotationGridIndex.from_geometry(points, faces, grid_size=16, margin=16).to(
        "cpu"
    )
    zeros = torch.zeros(4, dtype=torch.float64)
    result, cost, iterations = index.refine_nelder_mead(
        torch.as_tensor(points),
        zeros,
        zeros,
        torch.zeros(3, dtype=torch.float64),
        point_chunk=2,
    )
    torch.testing.assert_close(
        result,
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cost,
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert iterations == 1


def test_rotation_pipeline_returns_seed_matrix_without_refinement():
    """Test rotation pipeline returns seed matrix without refinement."""

    points, faces = _make_tetrahedron()
    stencil = SurfaceStencil(
        sphere_points=points.astype(np.float64),
        faces=faces,
        surface_indices=np.zeros((4, 3), dtype=np.int32),
        surface_weights=np.zeros((4, 3), dtype=np.float64),
        sheet_indices=np.zeros((1, 1, 3), dtype=np.int32),
        sheet_weights=np.zeros((1, 1, 3), dtype=np.float64),
        nx=1,
        ny=1,
        source_points=4,
    )
    pipeline = RotationPipeline.from_stencil(
        stencil, device="cpu", grid_size=16, margin=16
    )
    result = pipeline.search(
        points,
        np.zeros(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        point_chunk=2,
        refine=False,
    )
    torch.testing.assert_close(
        result.angle,
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert result.rotation_matrix.shape == (3, 3)
    assert result.iterations == 0
