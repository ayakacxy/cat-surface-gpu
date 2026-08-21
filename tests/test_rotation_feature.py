"""CAT-Surface GPU implementation."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from cat_surface_gpu.rotation_feature import (
    _system_entries,
    compute_rotation_feature,
)

from test_surface_stencil import _make_small_stencil


def test_depth_potential_regularisation_is_added_once_per_vertex():
    """Test depth potential regularisation is added once per vertex."""

    stencil = _make_small_stencil()
    points = torch.as_tensor(stencil.sphere_points, dtype=torch.float64)
    faces = torch.as_tensor(stencil.faces, dtype=torch.long)
    areas = torch.ones(points.shape[0], dtype=torch.float64)
    rows0, cols0, values0 = _system_entries(points, faces, areas, 0.0)
    rows1, cols1, values1 = _system_entries(points, faces, areas, 0.25)

    def aggregate(rows, cols, values):
        result = defaultdict(float)
        for row, col, value in zip(rows.tolist(), cols.tolist(), values.tolist()):
            result[(row, col)] += value
        return result

    base = aggregate(rows0, cols0, values0)
    regularised = aggregate(rows1, cols1, values1)
    assert set(base) == set(regularised)
    for key in base:
        expected = 0.25 if key[0] == key[1] else 0.0
        assert abs((regularised[key] - base[key]) - expected) < 1e-12


def test_colored_groups_have_no_adjacent_vertices():
    """Test colored groups have no adjacent vertices."""

    device_stencil = _make_small_stencil().to("cpu")
    groups = device_stencil.color_groups()
    labels = np.full(device_stencil.faces.shape[0], -1, dtype=np.int32)
    for color, vertices in enumerate(groups):
        labels[vertices.numpy()] = color
    assert np.all(labels >= 0)
    for face in device_stencil.faces.numpy():
        assert len({int(labels[int(vertex)]) for vertex in face}) == 3


def test_official_depth_values_only_skips_the_gpu_linear_solve():
    """Test official depth values only skips the gpu linear solve."""

    stencil = _make_small_stencil().to("cpu")
    vertices = torch.as_tensor(stencil.sphere_points, dtype=torch.float32)
    depth_values = torch.tensor((0.0, 0.5, 1.5, 3.0), dtype=torch.float64)
    smoothed = stencil.smooth_values(depth_values, vertices, 50.0)
    expected = (smoothed - smoothed.amin()) / (
        smoothed.amax() - smoothed.amin()
    ).clamp_min(torch.finfo(torch.float64).eps)
    result = compute_rotation_feature(
        stencil,
        vertices,
        heat_fwhm=15.0,
        smoothed_surface=vertices,
        depth_values=depth_values,
    )
    assert result.iterations == 0
    assert result.relative_residual == 0.0
    torch.testing.assert_close(result.values, expected, rtol=0.0, atol=0.0)
