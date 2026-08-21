"""DARTEL 规则网格 Torch 后端的数值和设备合同测试。"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cat_surface_gpu import dartel_triton
from cat_surface_gpu.dartel_grid import (
    DeviceUnavailable,
    GridSpec,
    apply_membrane_system,
    composition,
    composition_jacobian,
    dartel_step,
    expdef,
    fmg2_membrane,
    from_c_layout,
    jacobian_of_displacement,
    make_identity_map,
    regularization_operator,
    relax_membrane,
    resolve_device,
    resize_field,
    squaring_update,
    to_c_layout,
)
from cat_surface_gpu.dartel_grid import _bound_indices


def _bound_scalar(i: int, j: int, spec: GridSpec) -> tuple[int, int]:
    """提供与 CAT C 实现独立的标量边界参考。"""

    x = i % spec.nx
    if spec.ny == 1:
        return x, 0
    reflected = j % (2 * spec.ny)
    y = 2 * spec.ny - reflected - 1 if reflected >= spec.ny else reflected
    return x, y


def _corners_numpy(field, x, y, spec):
    ix = math.floor(float(x))
    iy = math.floor(float(y))
    dx1 = float(x) - ix
    dy1 = float(y) - iy
    dx2 = 1.0 - dx1
    dy2 = 1.0 - dy1
    values = []
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = _bound_scalar(ix + ox, iy + oy, spec)
        values.append(float(field[yy, xx]))
    return (*values, dx1, dx2, dy1, dy2)


def _bilinear_numpy(corners):
    k22, k12, k21, k11, dx1, dx2, dy1, dy2 = corners
    return (k22 * dx2 + k12 * dx1) * dy2 + (k21 * dx2 + k11 * dx1) * dy1


def _composition_numpy(a, b, spec):
    result = np.empty_like(a)
    for y in range(spec.ny):
        for x in range(spec.nx):
            xx = float(a[0, y, x]) - 1.0
            yy = float(a[1, y, x]) - 1.0
            for component, period in ((0, spec.nx), (1, spec.ny)):
                corners = list(_corners_numpy(b[component] - 1.0, xx, yy, spec))
                corners[1] -= math.floor((corners[1] - corners[0]) / period + 0.5) * period
                corners[2] -= math.floor((corners[2] - corners[0]) / period + 0.5) * period
                corners[3] -= math.floor((corners[3] - corners[0]) / period + 0.5) * period
                result[component, y, x] = _bilinear_numpy(corners) + 1.0
    return result


def test_mixed_boundary_matches_scalar_reference():
    spec = GridSpec(nx=5, ny=4)
    ix = torch.tensor([-7, -1, 0, 4, 5, 12])
    iy = torch.tensor([-9, -1, 0, 3, 4, 11])
    actual_x, actual_y = (
        index.cpu().numpy().tolist() for index in _bound_indices(ix, iy, spec)
    )
    expected = [_bound_scalar(int(x), int(y), spec) for x, y in zip(ix, iy)]
    assert list(zip(actual_x, actual_y)) == expected


def test_composition_matches_numpy_reference():
    torch.manual_seed(7)
    spec = GridSpec(nx=7, ny=5)
    a = make_identity_map(spec, device="cpu")
    a = a + torch.randn_like(a) * 1.7
    b = make_identity_map(spec, device="cpu")
    b = b + torch.randn_like(b) * 0.8
    actual = composition(a, b, device="cpu").numpy()
    expected = _composition_numpy(a.numpy(), b.numpy(), spec)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_shared_index_composition_matches_reference_path():
    """共享索引优化必须与逐分量 reference 保持数值等价。"""

    torch.manual_seed(17)
    spec = GridSpec(nx=9, ny=7)
    a = make_identity_map(spec, device="cpu") + torch.randn(
        (2, spec.ny, spec.nx), dtype=torch.float64
    ) * 1.7
    b = make_identity_map(spec, device="cpu") + torch.randn(
        (2, spec.ny, spec.nx), dtype=torch.float64
    ) * 0.8
    ja = torch.randn((4, spec.ny, spec.nx), dtype=torch.float64)
    jb = torch.randn((4, spec.ny, spec.nx), dtype=torch.float64)
    ja[[0, 3]] += 1.0
    jb[[0, 3]] += 1.0

    optimized = composition(a, b, device="cpu", optimized=True)
    reference = composition(a, b, device="cpu", optimized=False)
    torch.testing.assert_close(optimized, reference, rtol=0.0, atol=0.0)

    optimized_pair = composition_jacobian(
        a,
        ja,
        b,
        jb,
        device="cpu",
        optimized=True,
    )
    reference_pair = composition_jacobian(
        a,
        ja,
        b,
        jb,
        device="cpu",
        optimized=False,
    )
    for optimized_value, reference_value in zip(optimized_pair, reference_pair):
        torch.testing.assert_close(
            optimized_value,
            reference_value,
            rtol=0.0,
            atol=0.0,
        )


def test_multichannel_squaring_sampling_matches_reference_path():
    """squaring 的五通道共享采样必须保持 reference 结果。"""

    torch.manual_seed(29)
    spec = GridSpec(nx=9, ny=7)
    transformation = make_identity_map(spec, device="cpu")
    transformation = transformation + torch.randn_like(transformation) * 0.03
    jacobian = torch.zeros((4, spec.ny, spec.nx), dtype=torch.float64)
    jacobian[0] = 1.0
    jacobian[3] = 1.0
    right_hand = torch.randn((2, spec.ny, spec.nx), dtype=torch.float64)
    hessian = torch.rand((3, spec.ny, spec.nx), dtype=torch.float64)
    hessian[0] += 1.0
    hessian[1] += 1.0

    optimized = squaring_update(
        right_hand,
        hessian,
        transformation,
        jacobian,
        k=3,
        save_transformation=True,
        return_scratch=True,
        device="cpu",
        optimized=True,
    )
    reference = squaring_update(
        right_hand,
        hessian,
        transformation,
        jacobian,
        k=3,
        save_transformation=True,
        return_scratch=True,
        device="cpu",
        optimized=False,
    )
    for optimized_value, reference_value in zip(optimized, reference):
        torch.testing.assert_close(
            optimized_value,
            reference_value,
            rtol=0.0,
            atol=0.0,
        )


def test_expdef_and_jacobian_have_expected_identity_limit():
    spec = GridSpec(nx=6, ny=4)
    zero = torch.zeros((2, spec.ny, spec.nx), dtype=torch.float64)
    actual_map, actual_jacobian = expdef(
        zero,
        k=3,
        return_jacobian=True,
        device="cpu",
    )
    expected_map = make_identity_map(spec, device="cpu")
    expected_jacobian = torch.zeros((4, spec.ny, spec.nx), dtype=torch.float64)
    expected_jacobian[0] = 1.0
    expected_jacobian[3] = 1.0
    torch.testing.assert_close(actual_map, expected_map, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        actual_jacobian,
        expected_jacobian,
        rtol=1e-12,
        atol=1e-12,
    )


def test_jacobian_composition_matches_finite_difference_for_small_flow():
    spec = GridSpec(nx=8, ny=6)
    x = torch.arange(spec.nx, dtype=torch.float64).view(1, spec.nx)
    y = torch.arange(spec.ny, dtype=torch.float64).view(spec.ny, 1)
    displacement = torch.stack(
        (
            0.03 * torch.sin(2.0 * math.pi * x / spec.nx).expand(spec.ny, spec.nx),
            0.02 * torch.cos(2.0 * math.pi * y / (2.0 * spec.ny)).expand(spec.ny, spec.nx),
        ),
        dim=0,
    )
    analytic = jacobian_of_displacement(displacement, device="cpu")
    expected = np.empty((4, spec.ny, spec.nx), dtype=np.float64)
    values = displacement.numpy()
    for row in range(spec.ny):
        for column in range(spec.nx):
            xm, ym = _bound_scalar(column - 1, row, spec)
            xp, yp = _bound_scalar(column + 1, row, spec)
            x0, y0 = _bound_scalar(column, row - 1, spec)
            x1, y1 = _bound_scalar(column, row + 1, spec)
            expected[0, row, column] = (values[0, yp, xp] - values[0, ym, xm]) / 2.0 + 1.0
            expected[1, row, column] = (values[1, yp, xp] - values[1, ym, xm]) / 2.0
            expected[2, row, column] = (values[0, y1, x1] - values[0, y0, x0]) / 2.0
            expected[3, row, column] = (values[1, y1, x1] - values[1, y0, x0]) / 2.0 + 1.0
    np.testing.assert_allclose(analytic.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_regularization_operator_supports_all_reference_types():
    spec = GridSpec(nx=7, ny=5)
    field = torch.arange(2 * spec.ny * spec.nx, dtype=torch.float64).reshape(
        2, spec.ny, spec.nx
    )
    params = [1.3, 0.9, 0.125, 0.02, 0.001]
    for rtype in (0, 1, 2):
        result = regularization_operator(field, params, rtype=rtype, device="cpu")
        assert result.shape == field.shape
        assert torch.isfinite(result).all()


def test_dartel_step_cpu_reference_is_finite_for_both_objectives():
    torch.manual_seed(13)
    spec = GridSpec(nx=8, ny=6)
    source = torch.rand((spec.ny, spec.nx), dtype=torch.float64)
    target = torch.rand((spec.ny, spec.nx), dtype=torch.float64)
    velocity = torch.randn((2, spec.ny, spec.nx), dtype=torch.float64) * 0.01
    params = [1.0, 1.0, 0.125, 0.02, 0.001]
    for code in (0, 1):
        updated, metrics = dartel_step(
            source,
            target,
            velocity,
            k=2,
            params=params,
            lmreg=0.0007,
            cycles=1,
            nit=1,
            code=code,
            kernel="torch",
            device="cpu",
        )
        assert updated.shape == velocity.shape
        assert metrics.shape == (3,)
        assert torch.isfinite(updated).all()
        assert torch.isfinite(metrics).all()


def test_c_layout_round_trip():
    value = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    flat = to_c_layout(value)
    restored = from_c_layout(flat, nx=4, ny=3, device="cpu")
    torch.testing.assert_close(restored, value)


def test_explicit_cuda_does_not_fallback_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("当前机器有 CUDA，跳过不可用设备分支")
    with pytest.raises(DeviceUnavailable):
        resolve_device("cuda")


def test_visible_cuda_matches_cpu_with_explicit_tolerance():
    if not torch.cuda.is_available():
        pytest.skip("当前执行环境没有可见 CUDA")
    spec = GridSpec(nx=32, ny=24)
    identity = make_identity_map(spec, device="cpu")
    flow = torch.stack(
        (
            0.03 * torch.sin(identity[0]),
            0.02 * torch.cos(identity[1]),
        )
    )
    cpu_composition = composition(identity, identity + flow, device="cpu")
    cuda_composition = composition(
        identity.cuda(),
        (identity + flow).cuda(),
        device="cuda",
    ).cpu()
    cpu_expdef = expdef(flow, k=5, device="cpu")
    cuda_expdef = expdef(flow.cuda(), k=5, device="cuda").cpu()
    params = [1.0, 1.0, 0.125, 0.0, 0.001]
    cpu_regularization = regularization_operator(flow, params, device="cpu")
    cuda_regularization = regularization_operator(
        flow.cuda(), params, device="cuda"
    ).cpu()
    for cpu_value, cuda_value in (
        (cpu_composition, cuda_composition),
        (cpu_expdef, cuda_expdef),
        (cpu_regularization, cuda_regularization),
    ):
        torch.testing.assert_close(cpu_value, cuda_value, rtol=1e-10, atol=1e-10)


def test_visible_triton_relax_and_fmg_match_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("当前执行环境没有可见 CUDA")
    if not dartel_triton.available():
        pytest.skip("当前 Python 环境没有可用 Triton")
    torch.manual_seed(19)
    spec = GridSpec(nx=32, ny=24)
    hessian = torch.rand((3, spec.ny, spec.nx), dtype=torch.float64)
    hessian[0] += 2.0
    hessian[1] += 2.0
    hessian[2] *= 0.05
    right_hand = torch.randn((2, spec.ny, spec.nx), dtype=torch.float64)
    params = [1.0, 1.0, 0.125, 0.0, 0.001]

    cpu_resize = resize_field(
        right_hand,
        GridSpec(nx=16, ny=12),
        kernel="torch",
        device="cpu",
    )
    cuda_resize = resize_field(
        right_hand.cuda(),
        GridSpec(nx=16, ny=12),
        kernel="triton",
        device="cuda",
    ).cpu()
    torch.testing.assert_close(cpu_resize, cuda_resize, rtol=1e-10, atol=1e-10)

    cpu_system = apply_membrane_system(
        hessian,
        right_hand,
        params,
        kernel="torch",
        device="cpu",
    )
    cuda_system = apply_membrane_system(
        hessian.cuda(),
        right_hand.cuda(),
        params,
        kernel="triton",
        device="cuda",
    ).cpu()
    torch.testing.assert_close(cpu_system, cuda_system, rtol=1e-10, atol=1e-10)

    cpu_relax = relax_membrane(
        hessian,
        right_hand,
        params,
        nit=4,
        kernel="torch",
        device="cpu",
    )
    cuda_relax = relax_membrane(
        hessian.cuda(),
        right_hand.cuda(),
        params,
        nit=4,
        kernel="triton",
        device="cuda",
    ).cpu()
    torch.testing.assert_close(cpu_relax, cuda_relax, rtol=1e-10, atol=1e-10)

    cpu_fmg = fmg2_membrane(
        hessian,
        right_hand,
        params,
        cycles=2,
        nit=2,
        kernel="torch",
        device="cpu",
    )
    cuda_fmg = fmg2_membrane(
        hessian.cuda(),
        right_hand.cuda(),
        params,
        cycles=2,
        nit=2,
        kernel="triton",
        device="cuda",
    ).cpu()
    torch.testing.assert_close(cpu_fmg, cuda_fmg, rtol=1e-10, atol=1e-10)

    source = torch.rand((spec.ny, spec.nx), dtype=torch.float64)
    target = torch.rand((spec.ny, spec.nx), dtype=torch.float64)
    velocity = torch.randn((2, spec.ny, spec.nx), dtype=torch.float64) * 0.01
    for code in (0, 1):
        cpu_step, cpu_metrics = dartel_step(
            source,
            target,
            velocity,
            k=2,
            params=params,
            lmreg=0.0007,
            cycles=1,
            nit=1,
            code=code,
            kernel="torch",
            device="cpu",
        )
        cuda_step, cuda_metrics = dartel_step(
            source.cuda(),
            target.cuda(),
            velocity.cuda(),
            k=2,
            params=params,
            lmreg=0.0007,
            cycles=1,
            nit=1,
            code=code,
            kernel="triton",
            device="cuda",
        )
        torch.testing.assert_close(
            cpu_step,
            cuda_step.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            cpu_metrics,
            cuda_metrics.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )
