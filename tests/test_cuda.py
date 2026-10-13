"""真实 CUDA 上的数值一致性及完整函数基本运行检查。"""

import numpy as np
import pytest

from cat_surface_gpu import (
    Surf2SphereParameters,
    SurfWarpParameters,
    create_icosphere,
    surf_to_sphere_cupy,
    surfwarp_cupy,
)
from cat_surface_gpu.dartel import (
    dartel_symmetric_step,
    identity_mapping,
    solve_membrane_system,
)


@pytest.fixture
def cupy_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("没有可用的 CUDA 设备")
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA 不可用：{error}")
    with cp.cuda.Device(0):
        yield cp


@pytest.mark.cuda
def test_cupy_dartel_matches_numpy_one_step(cupy_device):
    cp = cupy_device
    identity = identity_mapping((8, 16))
    velocity = np.zeros_like(identity)
    y, x = np.indices((8, 16))
    source = np.sin(x * 2 * np.pi / 16) + 0.2 * np.cos(y * np.pi / 8)
    target = np.roll(source, 1, axis=1)
    options = dict(
        time_step=1,
        mu=0.125,
        lm_regularization=0.001,
        solver_iterations=10,
        pcg_check_interval=0,
    )
    expected, _ = dartel_symmetric_step(velocity, source, target, **options)
    actual, _ = dartel_symmetric_step(
        cp.asarray(velocity), cp.asarray(source), cp.asarray(target), **options
    )
    assert np.allclose(cp.asnumpy(actual), expected, atol=1e-10, rtol=1e-8)


@pytest.mark.cuda
def test_sphere_first_smoothing_stage_on_synthetic_mesh(cupy_device):
    vertices, faces = create_icosphere(1)
    vertices = vertices * np.asarray([70, 80, 90], dtype=np.float32)
    output, timings = surf_to_sphere_cupy(
        vertices, faces, Surf2SphereParameters(stop_at=1)
    )
    assert output.shape == vertices.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert "low_smooth" in timings
    assert "ellipsoid_projection" not in timings
    triangles = output.astype(np.float64)[faces]
    assert np.all(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        > 0.0
    )


@pytest.mark.cuda
def test_default_sphere_rejects_collapsed_low_resolution_mesh(cupy_device):
    """过粗的合成网格在默认平滑后坍缩时，明确拒绝无效输出。"""

    vertices, faces = create_icosphere(1)
    vertices = vertices * np.asarray([70, 80, 90], dtype=np.float32)
    progress = []
    with pytest.raises(ValueError, match="面积为零"):
        surf_to_sphere_cupy(vertices, faces, progress=progress.append)
    assert "Surf2Sphere: area smoothing (4999 sweeps)" in progress


@pytest.mark.cuda
def test_warp_single_run_on_identical_synthetic_surfaces(cupy_device):
    sphere, faces = create_icosphere(2)
    surface = sphere * (80 + 4 * sphere[:, :1] * sphere[:, 1:2])
    options = SurfWarpParameters(
        rotate=False,
        average=False,
        runs=1,
        steps=1,
        loop=1,
        iterations=1,
        map_size=(32, 16),
    )
    result = surfwarp_cupy(
        surface, sphere, faces, surface, sphere, faces, options, solver_iterations=5
    )
    assert result.vertices.shape == sphere.shape
    assert np.isfinite(result.vertices).all()
    assert np.allclose(result.vertices, sphere, atol=3e-6)
    assert np.allclose(result.velocity, 0.0, atol=1e-10)
    assert len(result.objectives) == 1


@pytest.mark.cuda
@pytest.mark.parametrize("rhs_value", [0.0, 1.0])
@pytest.mark.parametrize("check_interval", [0, 2])
def test_cupy_pcg_exact_zero_residual_matches_numpy(
    cupy_device, rhs_value, check_interval
):
    """覆盖零右端项以及首轮获得精确解后的 CUDA 后续迭代。"""

    cp = cupy_device
    hessian = np.zeros((3, 8, 16), dtype=np.float64)
    rhs = np.full((2, 8, 16), rhs_value, dtype=np.float64)
    options = dict(
        mu=0.0, identity_weight=1.0, max_iterations=5, check_interval=check_interval
    )
    expected, expected_iterations = solve_membrane_system(hessian, rhs, **options)
    result, completed = solve_membrane_system(
        cp.asarray(hessian), cp.asarray(rhs), **options
    )
    assert np.array_equal(cp.asnumpy(result), rhs)
    assert np.array_equal(cp.asnumpy(result), expected)
    assert completed == expected_iterations == (5 if check_interval == 0 else 2)
