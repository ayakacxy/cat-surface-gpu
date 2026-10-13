"""CAT 独立实现的参数、拓扑和基础几何测试。"""

import numpy as np
import pytest

from cat_surface_gpu.parameters import (
    Surf2SphereParameters,
    SurfWarpParameters,
)
from cat_surface_gpu.geometry import (
    ellipsoid_to_sphere,
    surface_area,
    vertex_areas,
)
from cat_surface_gpu.dartel import (
    compose,
    dartel_symmetric_step,
    exponentiate_velocity,
    identity_mapping,
    membrane_operator,
    objective_terms,
    small_deformation,
    solve_membrane_system,
    solve_membrane_system_fmg,
)
from cat_surface_gpu.topology import build_surface_topology
from cat_surface_gpu.resampling import (
    apply_resampling_stencil,
    build_resampling_stencil,
    create_icosphere,
    prepare_spherical_resampler,
)
from cat_surface_gpu.mapping import sheet_unit_points
from cat_surface_gpu.features import build_smoothing_topology
from cat_surface_gpu.surfwarp import apply_deformation_to_sphere
from cat_surface_gpu.rotation import (
    average_xz_sphere,
    official_rotation_seed_angles,
    rotation_matrix_from_angles,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


def test_official_parameter_defaults_and_schedule() -> None:
    sphere = Surf2SphereParameters()
    warp = SurfWarpParameters()
    assert sphere.areal_iterations == 5000
    assert warp.map_size == (512, 256)
    assert warp.curvature_types == (5, 5, 2)
    assert warp.steps == 2
    assert warp.runs == 2
    schedule = warp.dartel_schedule()
    assert len(schedule) == 6
    assert [item["mu"] for item in schedule] == [
        0.125,
        0.125,
        0.125,
        0.125,
        0.1,
        0.1,
    ]
    assert all(item["cycles"] == 3 for item in schedule)
    assert all(item["iterations"] == 3 for item in schedule)


def test_closed_topology_tables() -> None:
    vertices, faces = _tetrahedron()
    topology = build_surface_topology(faces, vertices.shape[0])
    assert topology.neighbours.shape == (4, 3)
    assert np.all(topology.degree == 3)
    assert np.all(topology.incident_mask.sum(axis=1) == 3)
    for group in topology.color_groups:
        selected = set(map(int, group))
        for vertex in selected:
            assert selected.isdisjoint(
                map(int, topology.neighbours[vertex, topology.neighbour_mask[vertex]])
            )


def test_surface_area_vertex_area_and_sphere_projection() -> None:
    vertices, faces = _tetrahedron()
    total = surface_area(vertices, faces)
    per_vertex = vertex_areas(vertices, faces)
    assert np.isclose(per_vertex.sum(), total)
    projected = ellipsoid_to_sphere(vertices * [2.0, 3.0, 4.0], total)
    expected_radius = np.sqrt(total / (4.0 * np.pi))
    assert np.allclose(np.linalg.norm(projected, axis=1), expected_radius)


def test_official_icosphere_numbering_and_counts() -> None:
    vertices, faces = create_icosphere(2)
    assert vertices.shape == (162, 3)
    assert faces.shape == (320, 3)
    assert np.array_equal(faces[0], [2, 42, 44])
    assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)


def test_resampling_stencil_reproduces_linear_coordinates() -> None:
    vertices, faces = _tetrahedron()
    target = np.asarray(
        [0.2 * vertices[0] + 0.3 * vertices[1] + 0.5 * vertices[2]],
        dtype=np.float64,
    )
    stencil = build_resampling_stencil(
        vertices, faces, target, nearest_vertices=np.asarray([2], dtype=np.int32)
    )
    output = apply_resampling_stencil(vertices, faces, stencil)
    assert np.allclose(output, target)


def test_prepared_resampler_reuses_one_stencil_for_vectors_and_scalars() -> None:
    vertices, faces = _tetrahedron()
    resampler = prepare_spherical_resampler(vertices, faces, subdivisions=0)
    mapped_vertices = resampler.apply(vertices)
    mapped_values = resampler.apply(np.arange(vertices.shape[0], dtype=np.float64))
    assert mapped_vertices.shape == (12, 3)
    assert mapped_values.shape == (12,)
    assert np.isfinite(mapped_vertices).all()
    assert np.isfinite(mapped_values).all()


def test_sheet_unit_points_follow_pixel_center_contract() -> None:
    points = sheet_unit_points((8, 4))
    assert points.shape == (32, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    expected_theta = np.pi * 0.5 / 4.0
    expected_phi = 2.0 * np.pi * 0.5 / 8.0
    assert np.allclose(
        points[0],
        [
            np.sin(expected_theta) * np.cos(expected_phi),
            np.sin(expected_theta) * np.sin(expected_phi),
            np.cos(expected_theta),
        ],
    )


def test_fast_smoothing_topology_has_same_neighbour_sets() -> None:
    vertices, faces = _tetrahedron()
    reference = build_surface_topology(faces, vertices.shape[0])
    candidate = build_smoothing_topology(faces, vertices.shape[0])
    for vertex in range(vertices.shape[0]):
        expected = set(reference.neighbours[vertex, reference.neighbour_mask[vertex]])
        actual = set(candidate.neighbours[vertex, candidate.neighbour_mask[vertex]])
        assert actual == expected


def test_zero_deformation_only_normalizes_sphere_radius() -> None:
    vertices, _ = create_icosphere(1, radius=3.0)
    identity = identity_mapping((16, 32))
    warped = apply_deformation_to_sphere(vertices, identity, inverse=True)
    expected = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    assert np.allclose(warped, expected, atol=2.0e-6)


def test_warp_wraps_longitude_before_pole_weighting() -> None:
    height, width = 4, 4
    mapping = identity_mapping((height, width))
    mapping[0] += 0.75 * width
    u = (1.0 + 0.5) / width
    v = (0.0 + 0.5) / height
    theta = v * np.pi
    phi = u * 2.0 * np.pi
    vertex = np.asarray(
        [[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]],
        dtype=np.float64,
    )
    warped = apply_deformation_to_sphere(vertex, mapping, inverse=False)
    expected_u = u - 0.25 * np.sin(theta)
    expected = np.asarray(
        [
            [
                np.sin(theta) * np.cos(expected_u * 2.0 * np.pi),
                np.sin(theta) * np.sin(expected_u * 2.0 * np.pi),
                np.cos(theta),
            ]
        ]
    )
    assert np.allclose(warped, expected, atol=2.0e-6)


def test_initial_rotation_contract_matches_official_angles() -> None:
    seeds = official_rotation_seed_angles()
    assert seeds.shape == (46, 3)
    assert np.array_equal(seeds[0], np.zeros(3))
    assert np.any(np.all(seeds == [-0.3, 0.0, 0.0], axis=1))
    matrix = rotation_matrix_from_angles([0.0, 0.0, 0.0])
    assert np.allclose(matrix, np.eye(3))
    matrix = rotation_matrix_from_angles([0.1, -0.2, 0.3])
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=2.0e-15)


def test_average_xz_sphere_normalizes_weighted_result() -> None:
    first = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    second = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    averaged = average_xz_sphere(first, second)
    assert averaged.dtype == np.float32
    assert np.allclose(np.linalg.norm(averaged, axis=1), 1.0)
    assert np.isfinite(averaged).all()


def test_dartel_identity_and_constant_velocity() -> None:
    identity = identity_mapping((7, 11))
    assert np.allclose(compose(identity, identity), identity)
    velocity = np.zeros_like(identity)
    velocity[0] = 0.125
    deformation, jacobian = exponentiate_velocity(velocity, 3)
    assert np.allclose(deformation, identity + velocity)
    expected = np.zeros_like(jacobian)
    expected[0, 0] = 1.0
    expected[1, 1] = 1.0
    assert np.allclose(jacobian, expected)


def test_dartel_small_deformation_central_difference() -> None:
    identity = identity_mapping((6, 8))
    velocity = np.zeros_like(identity)
    velocity[0] = np.sin(np.arange(8, dtype=np.float64) * 2.0 * np.pi / 8.0)
    mapping, jacobian = small_deformation(velocity, 0.5)
    assert np.allclose(mapping, identity + velocity * 0.5)
    assert np.allclose(jacobian[0, 1], 0.0)
    assert np.allclose(jacobian[1, 0], 0.0)
    assert np.allclose(jacobian[1, 1], 1.0)


def test_fmg_membrane_solver_reduces_linear_system_residual() -> None:
    rng = np.random.default_rng(42)
    hessian = np.stack(
        (
            np.full((8, 10), 1.2),
            np.full((8, 10), 1.1),
            np.full((8, 10), 0.05),
        )
    )
    right_hand_side = rng.normal(size=(2, 8, 10))
    solution = solve_membrane_system_fmg(
        hessian,
        right_hand_side,
        mu=0.125,
        identity_weight=1.0e-3,
    )
    reference, _ = solve_membrane_system(
        hessian,
        right_hand_side,
        mu=0.125,
        identity_weight=1.0e-3,
        max_iterations=200,
    )
    assert np.all(np.isfinite(solution))
    assert np.allclose(solution, reference, atol=1.0e-4)


def test_dartel_objective_and_membrane_zero_fixed_point() -> None:
    shape = (8, 12)
    identity = identity_mapping(shape)
    image = np.sin(np.arange(shape[1])[None, :] * 2.0 * np.pi / shape[1])
    image = np.broadcast_to(image, shape).copy()
    _, jacobian = small_deformation(np.zeros_like(identity), 1.0)
    objective, gradient, hessian = objective_terms(image, image, identity, jacobian)
    assert objective == 0.0
    assert np.allclose(gradient, 0.0)
    assert np.isfinite(hessian).all()
    assert np.allclose(membrane_operator(np.zeros_like(identity), 0.125), 0.0)
    output, diagnostics = dartel_symmetric_step(
        np.zeros_like(identity),
        image,
        image,
        time_step=0,
        mu=0.125,
        lm_regularization=1.0e-3,
        solver_iterations=5,
    )
    assert np.allclose(output, 0.0)
    assert diagnostics["objective"] == 0.0


@pytest.mark.parametrize("rhs_value", [0.0, 1.0])
@pytest.mark.parametrize("check_interval", [0, 2])
def test_pcg_exact_zero_residual_stays_finite(rhs_value, check_interval):
    """零右端项或首轮精确解都应安全经过后续的固定迭代。"""

    hessian = np.zeros((3, 8, 16), dtype=np.float64)
    rhs = np.full((2, 8, 16), rhs_value, dtype=np.float64)
    with np.errstate(divide="raise", invalid="raise"):
        solution, completed = solve_membrane_system(
            hessian,
            rhs,
            mu=0.0,
            identity_weight=1.0,
            max_iterations=5,
            check_interval=check_interval,
        )
    assert np.array_equal(solution, rhs)
    assert completed == (5 if check_interval == 0 else 2)
