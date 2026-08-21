"""CAT-Surface GPU implementation."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np

from cat_surface_gpu import (
    GiftiMesh,
    Surf2SphereTopology,
    convert_ellipsoid_to_sphere_with_surface_area,
    surface_area,
)


def _tetrahedron() -> GiftiMesh:
    """Tetrahedron."""

    vertices = np.asarray(
        (
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
        dtype=np.int32,
    )
    return GiftiMesh(vertices=vertices, faces=faces)


def test_surf2sphere_topology_keeps_all_incident_faces_and_colors():
    """Test surf2sphere topology keeps all incident faces and colors."""

    mesh = _tetrahedron()
    topology = Surf2SphereTopology.from_mesh(mesh)
    assert topology.incident_faces.shape == (4, 3)
    assert topology.incident_mask.all()
    assert len(topology.color_groups) >= 4
    assert len(topology.ordered_groups) >= 1

    faces = {tuple(face) for face in mesh.faces.tolist()}
    for group in topology.color_groups:
        for left, vertex in enumerate(group):
            for other in group[left + 1 :]:
                assert not any(
                    int(vertex) in face and int(other) in face for face in faces
                )
    for group in topology.ordered_groups:
        for left, vertex in enumerate(group):
            for other in group[left + 1 :]:
                assert not any(
                    int(vertex) in face and int(other) in face for face in faces
                )


def test_surface_area_and_ellipsoid_projection_are_finite():
    """Test surface area and ellipsoid projection are finite."""

    mesh = _tetrahedron()
    area = surface_area(mesh)
    projected = convert_ellipsoid_to_sphere_with_surface_area(
        mesh.vertices,
        area,
    )
    assert np.isfinite(projected).all()
    radius = np.linalg.norm(projected.astype(np.float64), axis=1)
    np.testing.assert_allclose(
        radius,
        np.sqrt(area / (4.0 * np.pi)),
        rtol=0.0,
        atol=2.0e-6,
    )
