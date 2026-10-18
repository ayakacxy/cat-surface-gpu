"""验证新增设备入口、实际 CPU 配准及 GPU 内核精度。"""

import sys

import numpy as np
import pytest

from cat_surface_gpu import Surf2SphereParameters, SurfWarpParameters, create_icosphere
from cat_surface_gpu.surf2sphere import surf_to_sphere
from cat_surface_gpu.surfwarp import surfwarp_cupy


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_platform_complete_entry_without_cupy(backend, monkeypatch):
    """完整编排在 CPU/Metal 上不能导入 CuPy；真实 Metal 运算需要 Mac。"""
    if backend == "metal" and sys.platform != "darwin":
        pytest.skip("Metal requires a real Apple GPU")
    if backend == "metal":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("Metal unavailable")
    monkeypatch.setitem(sys.modules, "cupy", None)
    sphere, faces = create_icosphere(2)
    surface = sphere * (80 + 4 * sphere[:, :1] * sphere[:, 1:2])
    output, _ = surf_to_sphere(
        surface, faces, Surf2SphereParameters(stop_at=1), device=backend
    )
    assert np.isfinite(output).all()
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
        surface,
        sphere,
        faces,
        surface,
        sphere,
        faces,
        options,
        solver_iterations=5,
        backend=backend,
    )
    np.testing.assert_allclose(result.vertices, sphere, atol=3e-5, rtol=0)
    np.testing.assert_allclose(result.velocity, 0, atol=1e-7, rtol=0)


def test_metal_unavailable_fails_explicitly():
    """显式选择不可用的 Metal 时直接报错。"""
    if sys.platform == "darwin":
        pytest.skip("Non-Mac preflight only")
    vertices, faces = create_icosphere(1)
    with pytest.raises(RuntimeError, match="macOS"):
        surf_to_sphere(
            vertices, faces, Surf2SphereParameters(stop_at=1), device="metal"
        )


@pytest.mark.cuda
def test_fp32_pcg_against_fp64():
    """小型非平凡系统检查实际 CUDA 的 FP32 解与 FP64 参考。"""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    from cat_surface_gpu.dartel import solve_membrane_system
    from cat_surface_gpu.backends.cuda.pcg_fp32 import solve_fixed_pcg

    rng = np.random.default_rng(82)
    rhs = rng.normal(size=(2, 13, 19))
    hessian = np.zeros((3, 13, 19))
    hessian[:2] = 0.1 + rng.random((2, 13, 19))
    expected, _ = solve_membrane_system(
        hessian,
        rhs,
        mu=0.125,
        identity_weight=0.01,
        max_iterations=40,
        check_interval=0,
    )
    actual, steps = solve_fixed_pcg(
        cp.asarray(hessian, dtype=cp.float32),
        cp.asarray(rhs, dtype=cp.float32),
        0.125,
        0.01,
        40,
    )
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-5, atol=2e-5)
    assert steps == 40
