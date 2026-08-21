"""球面 stencil、设备常驻曲面算子和最终 warp 的基础合同测试。"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import struct

import numpy as np
import torch

from cat_surface_gpu import (
    SurfaceStencil,
    apply_flow_to_sphere,
    apply_flow_to_stenciled_sphere,
    default_dartel_parameters,
)


def _make_small_stencil() -> SurfaceStencil:
    """构造一个不依赖真实大曲面的最小 stencil。"""

    points = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 3), (2, 1, 3), (1, 0, 3), (0, 1, 2)), dtype=np.int32
    )
    identity_indices = np.repeat(np.arange(4, dtype=np.int32)[:, None], 3, axis=1)
    identity_weights = np.zeros((4, 3), dtype=np.float64)
    identity_weights[:, 0] = 1.0
    sheet_indices = identity_indices.reshape(2, 2, 3)
    sheet_weights = identity_weights.reshape(2, 2, 3)
    return SurfaceStencil(
        sphere_points=points,
        faces=faces,
        surface_indices=identity_indices,
        surface_weights=identity_weights,
        sheet_indices=sheet_indices,
        sheet_weights=sheet_weights,
        nx=2,
        ny=2,
        source_points=4,
        unit_sphere_points=points.astype(np.float32),
    )


def test_stencil_v2_round_trip_preserves_cached_unit_points(tmp_path):
    """v2 文件应同时保存原始 stencil 和官方单位球面点。"""

    stencil = _make_small_stencil()
    path = tmp_path / "small.stencil"
    with path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8i",
                0x46534354,
                2,
                4,
                4,
                4,
                2,
                2,
                4,
            )
        )
        stencil.sphere_points.astype("<f8").tofile(stream)
        stencil.faces.astype("<i4").tofile(stream)
        stencil.surface_indices.astype("<i4").tofile(stream)
        stencil.surface_weights.astype("<f8").tofile(stream)
        stencil.sheet_indices.astype("<i4").tofile(stream)
        stencil.sheet_weights.astype("<f8").tofile(stream)
        stencil.unit_sphere_points.astype("<f4").tofile(stream)

    loaded = SurfaceStencil.from_file(path)
    np.testing.assert_array_equal(loaded.faces, stencil.faces)
    np.testing.assert_array_equal(
        loaded.unit_sphere_points, stencil.unit_sphere_points
    )


def test_device_stencil_keeps_resampling_and_neighbours_on_cpu_device():
    """设备 stencil 应一次性生成邻接表且不改变重采样结果。"""

    stencil = _make_small_stencil().to("cpu")
    vertices = torch.as_tensor(_make_small_stencil().sphere_points)
    sampled = stencil.resample_vertices(vertices)
    torch.testing.assert_close(sampled, vertices)
    assert stencil._neighbours is not None
    assert stencil._neighbour_mask is not None
    mapped = stencil.map_values_to_sheet(torch.arange(4, dtype=torch.float64))
    torch.testing.assert_close(
        mapped,
        torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64),
    )


def test_type5_batched_sheet_maps_match_individual_maps():
    """同一几何阶段批量复用时应保持每个 type5 sheet 图不变。"""

    stencil = _make_small_stencil().to("cpu")
    vertices = torch.as_tensor(stencil.sphere_points, dtype=torch.float32)
    fwhms = (5.0, 5.0 / 3.0)
    expected = tuple(stencil.curvature_to_sheet(vertices, value) for value in fwhms)
    actual = stencil.curvature_type5_to_sheet_many(vertices, fwhms)
    assert len(actual) == len(expected)
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate, reference, rtol=0.0, atol=0.0, equal_nan=True
        )


def test_final_warp_is_finite_unit_length_and_uses_cached_points():
    """零形变和 stencil warp 都应输出单位球面上的有限点。"""

    stencil = _make_small_stencil()
    points = torch.as_tensor(stencil.sphere_points, dtype=torch.float32)
    flow = torch.stack(
        (
            torch.arange(8, dtype=torch.float64).view(1, 8).expand(4, 8) + 1.0,
            torch.arange(4, dtype=torch.float64).view(4, 1).expand(4, 8) + 1.0,
        )
    )
    direct = apply_flow_to_sphere(
        points,
        flow,
        unit_sphere_vertices=torch.as_tensor(
            stencil.unit_sphere_points, dtype=torch.float32
        ),
        device="cpu",
    )
    cached = apply_flow_to_stenciled_sphere(points, flow, stencil, device="cpu")
    assert torch.isfinite(direct).all()
    torch.testing.assert_close(direct, cached, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(cached, dim=-1),
        torch.ones(4, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_default_parameters_match_cat_mu_schedule():
    """默认正则参数应在第四个 loop 后按官方比例降低 mu。"""

    params = default_dartel_parameters()
    assert len(params) == 6
    assert [row[2] for row in params] == [0.125, 0.125, 0.125, 0.125, 0.1, 0.1]
