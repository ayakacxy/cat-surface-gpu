"直接移植官方 diffeosphere.c 的球面 DARTEL 网格原语。"

from __future__ import annotations

from functools import lru_cache
from typing import Any
import os

import numpy as np
from numba import njit, prange


def _array_module(array: Any):
    "按数组类型选择 NumPy 或 CuPy，避免 CPU 测试导入 CUDA。"

    if hasattr(array, "__cuda_array_interface__"):
        import cupy as cp

        return cp
    return np


def identity_mapping(
    shape: tuple[int, int],
    *,
    array_module=np,
    dtype=np.float64,
):
    "生成官方一基坐标约定下的恒等二维映射。"

    height, width = map(int, shape)
    y, x = array_module.meshgrid(
        array_module.arange(1, height + 1, dtype=dtype),
        array_module.arange(1, width + 1, dtype=dtype),
        indexing="ij",
    )
    return array_module.stack((x, y), axis=0)


def _device_cache_id(array_module) -> int:
    "返回数组模块对应的设备编号，作为缓存键的一部分。"

    if array_module is np:
        return -1
    return int(array_module.cuda.Device().id)


@lru_cache(maxsize=8)
def _cached_identity_mapping(
    height: int,
    width: int,
    dtype_key: str,
    device_id: int,
):
    "缓存 DARTEL 重复使用的恒等坐标网格。"

    array_module = np if device_id < 0 else __import__("cupy")
    dtype = np.dtype(dtype_key)
    return identity_mapping(
        (int(height), int(width)),
        array_module=array_module,
        dtype=dtype,
    )


def _bound(x, y, width: int, height: int, xp):
    "复现官方 x 循环、y Neumann 反射边界。"

    bounded_x = xp.mod(x, width)
    if height == 1:
        bounded_y = xp.zeros_like(y)
    else:
        period = 2 * height
        folded = xp.mod(y, period)
        bounded_y = xp.where(folded >= height, period - folded - 1, folded)
    return bounded_x.astype(xp.int64), bounded_y.astype(xp.int64)


@lru_cache(maxsize=8)
def _cached_grid_indices(height: int, width: int, device_id: int):
    "缓存周期 x 轴和反射 y 轴的四邻域索引。"

    array_module = np if device_id < 0 else __import__("cupy")
    x = array_module.arange(int(width), dtype=array_module.int64)[None, :]
    y = array_module.arange(int(height), dtype=array_module.int64)[:, None]
    x_minus, y_same = _bound(x - 1, y, int(width), int(height), array_module)
    x_plus, _ = _bound(x + 1, y, int(width), int(height), array_module)
    x_same, y_minus = _bound(x, y - 1, int(width), int(height), array_module)
    _, y_plus = _bound(x, y + 1, int(width), int(height), array_module)
    return x_minus, y_same, x_plus, x_same, y_minus, y_plus


def _bilinear(field, x, y, *, coordinate_period: int | None = None):
    "按官方 bound 和双线性权重采样二维数组。"

    xp = _array_module(field)
    height, width = field.shape
    floor_x = xp.floor(x).astype(xp.int64)
    floor_y = xp.floor(y).astype(xp.int64)
    fraction_x = x - floor_x
    fraction_y = y - floor_y
    x0, y0 = _bound(floor_x, floor_y, width, height, xp)
    x1, _ = _bound(floor_x + 1, floor_y, width, height, xp)
    _, y1 = _bound(floor_x, floor_y + 1, width, height, xp)
    value00 = field[y0, x0]
    value10 = field[y0, x1]
    value01 = field[y1, x0]
    value11 = field[y1, x1]
    if coordinate_period is not None:
        period = float(coordinate_period)
        value10 -= xp.floor((value10 - value00) / period + 0.5) * period
        value01 -= xp.floor((value01 - value00) / period + 0.5) * period
        value11 -= xp.floor((value11 - value00) / period + 0.5) * period
    inverse_x = 1.0 - fraction_x
    inverse_y = 1.0 - fraction_y
    return (value00 * inverse_x + value10 * fraction_x) * inverse_y + (
        value01 * inverse_x + value11 * fraction_x
    ) * fraction_y


@njit(cache=True, inline="always")
def _sample_bilinear_cpu(field, x0, x1, y0, y1, fx, fy, period, offset):
    "复用坐标索引，保留双线性插值运算顺序。"

    value00 = field[y0, x0] + offset
    value10 = field[y0, x1] + offset
    value01 = field[y1, x0] + offset
    value11 = field[y1, x1] + offset
    if period:
        p = float(period)
        value10 -= np.floor((value10 - value00) / p + 0.5) * p
        value01 -= np.floor((value01 - value00) / p + 0.5) * p
        value11 -= np.floor((value11 - value00) / p + 0.5) * p
    inverse_x = 1.0 - fx
    inverse_y = 1.0 - fy
    return (value00 * inverse_x + value10 * fx) * inverse_y + (
        value01 * inverse_x + value11 * fx
    ) * fy


@njit(cache=True, parallel=True, fastmath=False)
def _compose_jacobian_cpu_fused(
    mapping_a, jacobian_a, mapping_b, jacobian_b, composed, jacobian
):
    "每个网格点仅计算一次邻域索引，并复合映射和四个雅可比分量。"

    height, width = mapping_a.shape[1:]
    for row in prange(height):
        for col in range(width):
            x = mapping_a[0, row, col] - 1.0
            y = mapping_a[1, row, col] - 1.0
            floor_x = int(np.floor(x))
            floor_y = int(np.floor(y))
            fx = x - floor_x
            fy = y - floor_y
            x0 = floor_x % width
            x1 = (floor_x + 1) % width
            if height == 1:
                y0 = 0
                y1 = 0
            else:
                fold0 = floor_y % (2 * height)
                fold1 = (floor_y + 1) % (2 * height)
                y0 = 2 * height - fold0 - 1 if fold0 >= height else fold0
                y1 = 2 * height - fold1 - 1 if fold1 >= height else fold1

            composed[0, row, col] = (
                _sample_bilinear_cpu(mapping_b[0], x0, x1, y0, y1, fx, fy, width, -1.0)
                + 1.0
            )
            composed[1, row, col] = (
                _sample_bilinear_cpu(mapping_b[1], x0, x1, y0, y1, fx, fy, height, -1.0)
                + 1.0
            )
            b00 = _sample_bilinear_cpu(jacobian_b[0, 0], x0, x1, y0, y1, fx, fy, 0, 0.0)
            b01 = _sample_bilinear_cpu(jacobian_b[0, 1], x0, x1, y0, y1, fx, fy, 0, 0.0)
            b10 = _sample_bilinear_cpu(jacobian_b[1, 0], x0, x1, y0, y1, fx, fy, 0, 0.0)
            b11 = _sample_bilinear_cpu(jacobian_b[1, 1], x0, x1, y0, y1, fx, fy, 0, 0.0)
            a00 = jacobian_a[0, 0, row, col]
            a01 = jacobian_a[0, 1, row, col]
            a10 = jacobian_a[1, 0, row, col]
            a11 = jacobian_a[1, 1, row, col]
            jacobian[0, 0, row, col] = a00 * b00 + a01 * b10
            jacobian[0, 1, row, col] = a00 * b01 + a01 * b11
            jacobian[1, 0, row, col] = a10 * b00 + a11 * b10
            jacobian[1, 1, row, col] = a10 * b01 + a11 * b11


@njit(cache=True, parallel=True, fastmath=False)
def _sample_transport_cpu_fused(
    gradient, hessian, mapping, sampled_gradient, sampled_hessian
):
    "梯度与 Hessian 的五个采样通道共享网格索引。"

    height, width = mapping.shape[1:]
    for row in prange(height):
        for col in range(width):
            x = mapping[0, row, col] - 1.0
            y = mapping[1, row, col] - 1.0
            floor_x = int(np.floor(x))
            floor_y = int(np.floor(y))
            fx = x - floor_x
            fy = y - floor_y
            x0 = floor_x % width
            x1 = (floor_x + 1) % width
            if height == 1:
                y0 = 0
                y1 = 0
            else:
                fold0 = floor_y % (2 * height)
                fold1 = (floor_y + 1) % (2 * height)
                y0 = 2 * height - fold0 - 1 if fold0 >= height else fold0
                y1 = 2 * height - fold1 - 1 if fold1 >= height else fold1
            for component in range(2):
                sampled_gradient[component, row, col] = _sample_bilinear_cpu(
                    gradient[component], x0, x1, y0, y1, fx, fy, 0, 0.0
                )
            for component in range(3):
                sampled_hessian[component, row, col] = _sample_bilinear_cpu(
                    hessian[component], x0, x1, y0, y1, fx, fy, 0, 0.0
                )


@njit(cache=True, parallel=True, fastmath=False)
def _transport_accumulate_cpu_fused(
    gradient,
    hessian,
    sampled_gradient,
    sampled_hessian,
    jacobian,
):
    "逐像素累加缩放平方法运输后的梯度和对称 Hessian。"

    height, width = gradient.shape[1:]
    for row in prange(height):
        for col in range(width):
            j11 = jacobian[0, 0, row, col]
            j21 = jacobian[0, 1, row, col]
            j12 = jacobian[1, 0, row, col]
            j22 = jacobian[1, 1, row, col]
            determinant = j11 * j22 - j12 * j21
            first = sampled_gradient[0, row, col]
            second = sampled_gradient[1, row, col]
            gradient[0, row, col] += determinant * (first * j11 + second * j21)
            gradient[1, row, col] += determinant * (first * j12 + second * j22)
            a00 = sampled_hessian[0, row, col]
            a11 = sampled_hessian[1, row, col]
            a01 = sampled_hessian[2, row, col]
            tmp1 = a00 * j11 + a01 * j21
            tmp2 = a01 * j11 + a11 * j21
            tmp3 = a00 * j12 + a01 * j22
            tmp4 = a01 * j12 + a11 * j22
            hessian[0, row, col] += determinant * (tmp1 * j11 + tmp2 * j21)
            hessian[1, row, col] += determinant * (tmp3 * j12 + tmp4 * j22)
            hessian[2, row, col] += determinant * (tmp1 * j12 + tmp2 * j22)


@njit(cache=True, parallel=True, fastmath=False)
def _small_deformation_cpu_fused(velocity, scale, mapping, jacobian):
    "单次遍历计算 FP64 小形变映射与中心差分雅可比。"

    height, width = velocity.shape[1:]
    half_scale = scale / 2.0
    for row in prange(height):
        up = row - 1 if row > 0 else 0
        down = row + 1 if row + 1 < height else height - 1
        for col in range(width):
            left = col - 1 if col > 0 else width - 1
            right = col + 1 if col + 1 < width else 0
            mapping[0, row, col] = float(col + 1) + velocity[0, row, col] * scale
            mapping[1, row, col] = float(row + 1) + velocity[1, row, col] * scale
            jacobian[0, 0, row, col] = (
                velocity[0, row, right] - velocity[0, row, left]
            ) * half_scale + 1.0
            jacobian[0, 1, row, col] = (
                velocity[1, row, right] - velocity[1, row, left]
            ) * half_scale
            jacobian[1, 0, row, col] = (
                velocity[0, down, col] - velocity[0, up, col]
            ) * half_scale
            jacobian[1, 1, row, col] = (
                velocity[1, down, col] - velocity[1, up, col]
            ) * half_scale + 1.0


@njit(cache=True, parallel=True, fastmath=False)
def _membrane_operator_cpu_fused(field, mu, identity_weight, output):
    "计算两个膜能分量，采用 x 周期边界与 y 反射边界。"

    height, width = field.shape[1:]
    center = 4.0 * mu + identity_weight
    for row in prange(height):
        up = row - 1 if row > 0 else 0
        down = row + 1 if row + 1 < height else height - 1
        for col in range(width):
            left = col - 1 if col > 0 else width - 1
            right = col + 1 if col + 1 < width else 0
            for component in range(2):
                neighbours = (
                    field[component, row, left]
                    + field[component, row, right]
                    + field[component, up, col]
                    + field[component, down, col]
                )
                output[component, row, col] = (
                    center * field[component, row, col] - mu * neighbours
                )


@njit(cache=True, parallel=True, fastmath=False)
def _objective_terms_cpu_fused(
    source,
    target,
    mapping,
    jacobian,
    gradient,
    hessian,
    squared_difference,
):
    "融合无权目标函数的采样、梯度及 Hessian 计算。"

    height, width = source.shape
    for row in prange(height):
        for col in range(width):
            x = mapping[0, row, col] - 1.0
            y = mapping[1, row, col] - 1.0
            floor_x = int(np.floor(x))
            floor_y = int(np.floor(y))
            fx = x - floor_x
            fy = y - floor_y
            ix = 1.0 - fx
            iy = 1.0 - fy
            x0 = floor_x % width
            x1 = (floor_x + 1) % width
            if height == 1:
                y0 = 0
                y1 = 0
            else:
                fold0 = floor_y % (2 * height)
                fold1 = (floor_y + 1) % (2 * height)
                y0 = 2 * height - fold0 - 1 if fold0 >= height else fold0
                y1 = 2 * height - fold1 - 1 if fold1 >= height else fold1
            k22 = source[y0, x0]
            k12 = source[y0, x1]
            k21 = source[y1, x0]
            k11 = source[y1, x1]
            sampled = (k11 * fx + k21 * ix) * fy + (k12 * fx + k22 * ix) * iy
            difference = sampled - target[row, col]
            derivative_x = (k11 - k21) * fy + (k12 - k22) * iy
            derivative_y = (k11 * fx + k21 * ix) - (k12 * fx + k22 * ix)
            gx = (
                jacobian[0, 0, row, col] * derivative_x
                + jacobian[0, 1, row, col] * derivative_y
            )
            gy = (
                jacobian[1, 0, row, col] * derivative_x
                + jacobian[1, 1, row, col] * derivative_y
            )
            gradient[0, row, col] = gx * difference
            gradient[1, row, col] = gy * difference
            hessian[0, row, col] = gx * gx
            hessian[1, row, col] = gy * gy
            hessian[2, row, col] = gx * gy
            squared_difference[row, col] = difference * difference


def compose(mapping_a, mapping_b):
    "复现官方 ``composition``，返回 ``B(A(x))``。"

    xp = _array_module(mapping_a)
    if mapping_a.shape != mapping_b.shape or mapping_a.ndim != 3:
        raise ValueError("Both mappings must have the same shape (2, H, W)")
    if mapping_a.shape[0] != 2:
        raise ValueError("The first mapping dimension must be two")
    height, width = mapping_a.shape[1:]
    x = mapping_a[0] - 1.0
    y = mapping_a[1] - 1.0
    result_x = _bilinear(mapping_b[0] - 1.0, x, y, coordinate_period=width) + 1.0
    result_y = _bilinear(mapping_b[1] - 1.0, x, y, coordinate_period=height) + 1.0
    return xp.stack((result_x, result_y), axis=0)


def compose_with_jacobian(mapping_a, jacobian_a, mapping_b, jacobian_b):
    "复现官方 ``composition_jacobian``。"

    xp = _array_module(mapping_a)
    if jacobian_a.shape != jacobian_b.shape:
        raise ValueError("Both Jacobians must have the same shape")
    if jacobian_a.shape[:2] != (2, 2) or jacobian_a.shape[2:] != mapping_a.shape[1:]:
        raise ValueError("Jacobian must have shape (2, 2, H, W)")
    height, width = mapping_a.shape[1:]
    if xp is np and all(
        array.dtype == np.float64 and array.flags.c_contiguous
        for array in (mapping_a, jacobian_a, mapping_b, jacobian_b)
    ):
        composed = np.empty_like(mapping_a)
        jacobian = np.empty_like(jacobian_a)
        _compose_jacobian_cpu_fused(
            mapping_a, jacobian_a, mapping_b, jacobian_b, composed, jacobian
        )
        return composed, jacobian
    if (
        xp is not np
        and mapping_a.dtype == xp.float64
        and mapping_b.dtype == xp.float64
        and jacobian_a.dtype == xp.float64
        and jacobian_b.dtype == xp.float64
        and mapping_a.flags.c_contiguous
        and mapping_b.flags.c_contiguous
        and jacobian_a.flags.c_contiguous
        and jacobian_b.flags.c_contiguous
    ):
        composed = xp.empty_like(mapping_a)
        jacobian = xp.empty_like(jacobian_a)
        count = int(height * width)
        blocks = (count + 255) // 256
        _get_compose_jacobian_kernel()(
            (blocks,),
            (256,),
            (
                mapping_a[0],
                mapping_a[1],
                jacobian_a,
                mapping_b[0],
                mapping_b[1],
                jacobian_b,
                composed[0],
                composed[1],
                jacobian,
                np.int32(height),
                np.int32(width),
            ),
        )
        return composed, jacobian
    x = mapping_a[0] - 1.0
    y = mapping_a[1] - 1.0
    composed = compose(mapping_a, mapping_b)
    sampled = xp.empty_like(jacobian_b)
    for derivative in range(2):
        for component in range(2):
            sampled[derivative, component] = _bilinear(
                jacobian_b[derivative, component], x, y
            )
    jacobian = xp.einsum("achw,cbhw->abhw", jacobian_a, sampled)
    return composed, jacobian


def small_deformation(velocity, scale: float):
    "复现官方 ``smalldef_jac`` 的形变和中心差分 Jacobian。"

    xp = _array_module(velocity)
    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError("Velocity field must have shape (2, H, W)")
    height, width = velocity.shape[1:]
    if xp is np and velocity.dtype == np.float64 and velocity.flags.c_contiguous:
        mapping = np.empty_like(velocity)
        jacobian = np.empty((2, 2, height, width), dtype=np.float64)
        _small_deformation_cpu_fused(velocity, float(scale), mapping, jacobian)
        return mapping, jacobian
    if xp is not np and velocity.dtype == xp.float64:
        # 正式 SurfWarp 使用双精度；在同一核内融合映射与四个雅可比分量，
        # 避免重复构建邻域索引和中间数组。
        kernel = _get_small_deformation_kernel()
        mapping = xp.empty_like(velocity)
        jacobian = xp.empty((2, 2, height, width), dtype=velocity.dtype)
        count = int(height * width)
        blocks = (count + 255) // 256
        kernel(
            (blocks,),
            (256,),
            (
                velocity[0],
                velocity[1],
                mapping[0],
                mapping[1],
                jacobian[0, 0],
                jacobian[0, 1],
                jacobian[1, 0],
                jacobian[1, 1],
                np.int32(height),
                np.int32(width),
                np.float64(scale),
            ),
        )
        return mapping, jacobian
    device_id = _device_cache_id(xp)
    identity = _cached_identity_mapping(
        int(height),
        int(width),
        np.dtype(velocity.dtype).str,
        device_id,
    )
    mapping = identity + velocity * scale
    xm, yy, xp1, xx, ym, yp1 = _cached_grid_indices(int(height), int(width), device_id)
    half_scale = scale / 2.0
    jacobian = xp.empty((2, 2, height, width), dtype=velocity.dtype)
    jacobian[0, 0] = (velocity[0, yy, xp1] - velocity[0, yy, xm]) * half_scale + 1.0
    jacobian[0, 1] = (velocity[1, yy, xp1] - velocity[1, yy, xm]) * half_scale
    jacobian[1, 0] = (velocity[0, yp1, xx] - velocity[0, ym, xx]) * half_scale
    jacobian[1, 1] = (velocity[1, yp1, xx] - velocity[1, ym, xx]) * half_scale + 1.0
    return mapping, jacobian


def exponentiate_velocity(velocity, squarings: int):
    "复现官方 ``expdef`` 的缩放平方指数映射。"

    if squarings < 0:
        raise ValueError("squarings must not be negative")
    scale = 1.0 / float(2**squarings)
    mapping, jacobian = small_deformation(velocity, scale)
    for _ in range(squarings):
        mapping, jacobian = compose_with_jacobian(mapping, jacobian, mapping, jacobian)
    return mapping, jacobian


def _grid_neighbours(field):
    "返回与官方 bound 一致的上下左右邻域。"

    xp = _array_module(field)
    height, width = field.shape[-2:]
    xm, yy, xp1, xx, ym, yp1 = _cached_grid_indices(
        int(height), int(width), _device_cache_id(xp)
    )
    return (
        field[..., yy, xm],
        field[..., yy, xp1],
        field[..., ym, xx],
        field[..., yp1, xx],
    )


def membrane_operator(field, mu: float, identity_weight: float = 0.0):
    "复现默认 ``LtLf_me`` 膜能正则算子。"

    if (
        _array_module(field) is np
        and field.dtype == np.float64
        and field.flags.c_contiguous
    ):
        output = np.empty_like(field)
        _membrane_operator_cpu_fused(field, float(mu), float(identity_weight), output)
        return output
    left, right, up, down = _grid_neighbours(field)
    center = 4.0 * float(mu) + float(identity_weight)
    return center * field - float(mu) * (left + right + up + down)


def jacobian_divide_small_deformation(jacobian, velocity, scale: float):
    "复现官方 ``jac_div_smalldef``。"

    _, small = small_deformation(velocity, scale)
    a = small[0, 0]
    b = small[0, 1]
    c = small[1, 0]
    d = small[1, 1]
    determinant = a * d - b * c
    inverse00 = d / determinant
    inverse01 = -c / determinant
    inverse10 = -b / determinant
    inverse11 = a / determinant
    q00 = jacobian[0, 0]
    q01 = jacobian[1, 0]
    q10 = jacobian[0, 1]
    q11 = jacobian[1, 1]
    result = _array_module(jacobian).empty_like(jacobian)
    result[0, 0] = q00 * inverse00 + q01 * inverse10
    result[0, 1] = q10 * inverse00 + q11 * inverse10
    result[1, 0] = q00 * inverse01 + q01 * inverse11
    result[1, 1] = q10 * inverse01 + q11 * inverse11
    return result


_SMALL_DEFORMATION_KERNEL = None


def _get_small_deformation_kernel():
    "延迟构建双精度 smalldef 与 Jacobian 融合核。"

    global _SMALL_DEFORMATION_KERNEL
    if _SMALL_DEFORMATION_KERNEL is None:
        import cupy as cp

        module = cp.RawModule(
            code=r"""
            extern "C" __global__ void small_deformation(
                const double* velocity0, const double* velocity1,
                double* mapping0, double* mapping1,
                double* jac00, double* jac01,
                double* jac10, double* jac11,
                int height, int width, double scale) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int count = height * width;
                if (index >= count) return;
                int y = index / width;
                int x = index - y * width;
                int left_x = (x == 0) ? width - 1 : x - 1;
                int right_x = (x + 1 == width) ? 0 : x + 1;
                int up_y = (y == 0) ? 0 : y - 1;
                int down_y = (y + 1 == height) ? height - 1 : y + 1;
                int left = y * width + left_x;
                int right = y * width + right_x;
                int up = up_y * width + x;
                int down = down_y * width + x;
                double half_scale = 0.5 * scale;
                mapping0[index] = (double)(x + 1) + velocity0[index] * scale;
                mapping1[index] = (double)(y + 1) + velocity1[index] * scale;
                jac00[index] = 1.0
                    + (velocity0[right] - velocity0[left]) * half_scale;
                jac01[index] = (velocity1[right] - velocity1[left]) * half_scale;
                jac10[index] = (velocity0[down] - velocity0[up]) * half_scale;
                jac11[index] = 1.0
                    + (velocity1[down] - velocity1[up]) * half_scale;
            }
            """,
            options=("--std=c++11",),
            name_expressions=("small_deformation",),
        )
        _SMALL_DEFORMATION_KERNEL = module.get_function("small_deformation")
    return _SMALL_DEFORMATION_KERNEL


_OBJECTIVE_KERNEL = None


def _get_objective_kernel():
    "延迟构建无权重双精度目标函数、梯度和 Hessian 融合核。"

    global _OBJECTIVE_KERNEL
    if _OBJECTIVE_KERNEL is None:
        import cupy as cp

        module = cp.RawModule(
            code=r"""
            __device__ int wrap_x(int value, int width) {
                int result = value % width;
                return (result < 0) ? result + width : result;
            }

            __device__ int reflect_y(int value, int height) {
                if (height <= 1) return 0;
                int period = 2 * height;
                int result = value % period;
                if (result < 0) result += period;
                return (result >= height) ? period - result - 1 : result;
            }

            extern "C" __global__ void objective_terms(
                const double* source, const double* target,
                const double* mapping0, const double* mapping1,
                const double* jac00, const double* jac01,
                const double* jac10, const double* jac11,
                double* gradient0, double* gradient1,
                double* hessian0, double* hessian1, double* hessian2,
                double* objective, int height, int width) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int count = height * width;
                if (index >= count) return;
                double x = mapping0[index] - 1.0;
                double y = mapping1[index] - 1.0;
                int floor_x = (int)floor(x);
                int floor_y = (int)floor(y);
                double fraction_x = x - (double)floor_x;
                double fraction_y = y - (double)floor_y;
                double inverse_x = 1.0 - fraction_x;
                double inverse_y = 1.0 - fraction_y;
                int x0 = wrap_x(floor_x, width);
                int x1 = wrap_x(floor_x + 1, width);
                int y0 = reflect_y(floor_y, height);
                int y1 = reflect_y(floor_y + 1, height);
                double k22 = source[y0 * width + x0];
                double k12 = source[y0 * width + x1];
                double k21 = source[y1 * width + x0];
                double k11 = source[y1 * width + x1];
                double sampled =
                    (k11 * fraction_x + k21 * inverse_x) * fraction_y
                    + (k12 * fraction_x + k22 * inverse_x) * inverse_y;
                double difference = sampled - target[index];
                double derivative_x =
                    (k11 - k21) * fraction_y
                    + (k12 - k22) * inverse_y;
                double derivative_y =
                    (k11 * fraction_x + k21 * inverse_x)
                    - (k12 * fraction_x + k22 * inverse_x);
                double gradient_x =
                    jac00[index] * derivative_x + jac01[index] * derivative_y;
                double gradient_y =
                    jac10[index] * derivative_x + jac11[index] * derivative_y;
                gradient0[index] = gradient_x * difference;
                gradient1[index] = gradient_y * difference;
                hessian0[index] = gradient_x * gradient_x;
                hessian1[index] = gradient_y * gradient_y;
                hessian2[index] = gradient_x * gradient_y;
                objective[index] = 0.5 * difference * difference;
            }
            """,
            options=("--std=c++11",),
            name_expressions=("objective_terms",),
        )
        _OBJECTIVE_KERNEL = module.get_function("objective_terms")
    return _OBJECTIVE_KERNEL


_COMPOSE_JACOBIAN_KERNEL = None


def _get_compose_jacobian_kernel():
    "延迟构建映射与 Jacobian 复合的双精度融合核。"

    global _COMPOSE_JACOBIAN_KERNEL
    if _COMPOSE_JACOBIAN_KERNEL is None:
        import cupy as cp

        module = cp.RawModule(
            code=r"""
            __device__ int wrap_x(int value, int width) {
                int result = value % width;
                return (result < 0) ? result + width : result;
            }

            __device__ int reflect_y(int value, int height) {
                if (height <= 1) return 0;
                int period = 2 * height;
                int result = value % period;
                if (result < 0) result += period;
                return (result >= height) ? period - result - 1 : result;
            }

            __device__ double sample_field(
                const double* field, double x, double y,
                int height, int width, int period) {
                int floor_x = (int)floor(x);
                int floor_y = (int)floor(y);
                double fraction_x = x - (double)floor_x;
                double fraction_y = y - (double)floor_y;
                double inverse_x = 1.0 - fraction_x;
                double inverse_y = 1.0 - fraction_y;
                int x0 = wrap_x(floor_x, width);
                int x1 = wrap_x(floor_x + 1, width);
                int y0 = reflect_y(floor_y, height);
                int y1 = reflect_y(floor_y + 1, height);
                double value00 = field[y0 * width + x0];
                double value10 = field[y0 * width + x1];
                double value01 = field[y1 * width + x0];
                double value11 = field[y1 * width + x1];
                if (period > 0) {
                    double p = (double)period;
                    value10 -= floor((value10 - value00) / p + 0.5) * p;
                    value01 -= floor((value01 - value00) / p + 0.5) * p;
                    value11 -= floor((value11 - value00) / p + 0.5) * p;
                }
                return (value00 * inverse_x + value10 * fraction_x) * inverse_y
                    + (value01 * inverse_x + value11 * fraction_x) * fraction_y;
            }

            extern "C" __global__ void compose_with_jacobian(
                const double* mapping_a0, const double* mapping_a1,
                const double* jacobian_a,
                const double* mapping_b0, const double* mapping_b1,
                const double* jacobian_b,
                double* mapping_out0, double* mapping_out1,
                double* jacobian_out,
                int height, int width) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int count = height * width;
                if (index >= count) return;
                int y = index / width;
                double query_x = mapping_a0[index] - 1.0;
                double query_y = mapping_a1[index] - 1.0;
                double mapping_value0 = sample_field(
                    mapping_b0, query_x, query_y, height, width, width);
                double mapping_value1 = sample_field(
                    mapping_b1, query_x, query_y, height, width, height);
                mapping_out0[index] = mapping_value0;
                mapping_out1[index] = mapping_value1;

                double sampled00 = sample_field(
                    jacobian_b + 0 * count, query_x, query_y,
                    height, width, 0);
                double sampled01 = sample_field(
                    jacobian_b + 1 * count, query_x, query_y,
                    height, width, 0);
                double sampled10 = sample_field(
                    jacobian_b + 2 * count, query_x, query_y,
                    height, width, 0);
                double sampled11 = sample_field(
                    jacobian_b + 3 * count, query_x, query_y,
                    height, width, 0);
                double a00 = jacobian_a[0 * count + index];
                double a01 = jacobian_a[1 * count + index];
                double a10 = jacobian_a[2 * count + index];
                double a11 = jacobian_a[3 * count + index];
                jacobian_out[0 * count + index] = a00 * sampled00 + a01 * sampled10;
                jacobian_out[1 * count + index] = a00 * sampled01 + a01 * sampled11;
                jacobian_out[2 * count + index] = a10 * sampled00 + a11 * sampled10;
                jacobian_out[3 * count + index] = a10 * sampled01 + a11 * sampled11;
            }
            """,
            options=("--std=c++11",),
            name_expressions=("compose_with_jacobian",),
        )
        _COMPOSE_JACOBIAN_KERNEL = module.get_function("compose_with_jacobian")
    return _COMPOSE_JACOBIAN_KERNEL


_PCG_KERNELS = None


def _get_pcg_kernels():
    "延迟构建 PCG 的双精度膜算子和逐点更新核。"

    global _PCG_KERNELS
    if _PCG_KERNELS is None:
        import cupy as cp

        module = cp.RawModule(
            code=r"""
            extern "C" __global__ void apply_membrane_hessian(
                const double* field0, const double* field1,
                const double* hessian0, const double* hessian1,
                const double* hessian2, double* output0, double* output1,
                int height, int width, double mu, double identity_weight) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int count = height * width;
                if (index >= count) return;
                int y = index / width;
                int x = index - y * width;
                int left_x = (x == 0) ? width - 1 : x - 1;
                int right_x = (x + 1 == width) ? 0 : x + 1;
                int up_y = (y == 0) ? 0 : y - 1;
                int down_y = (y + 1 == height) ? height - 1 : y + 1;
                int left = y * width + left_x;
                int right = y * width + right_x;
                int up = up_y * width + x;
                int down = down_y * width + x;
                double center = 4.0 * mu + identity_weight;
                double membrane0 = center * field0[index]
                    - mu * (field0[left] + field0[right]
                            + field0[up] + field0[down]);
                double membrane1 = center * field1[index]
                    - mu * (field1[left] + field1[right]
                            + field1[up] + field1[down]);
                output0[index] = membrane0
                    + hessian0[index] * field0[index]
                    + hessian2[index] * field1[index];
                output1[index] = membrane1
                    + hessian1[index] * field1[index]
                    + hessian2[index] * field0[index];
            }

            extern "C" __global__ void apply_preconditioner(
                const double* residual0, const double* residual1,
                const double* hessian0, const double* hessian1,
                const double* hessian2, double* output0, double* output1,
                int count, double diagonal) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                if (index >= count) return;
                double diagonal0 = hessian0[index] + diagonal;
                double diagonal1 = hessian1[index] + diagonal;
                double cross = hessian2[index];
                double determinant = diagonal0 * diagonal1 - cross * cross;
                if (determinant < 1.0e-30) determinant = 1.0e-30;
                output0[index] = (diagonal1 * residual0[index]
                                  - cross * residual1[index]) / determinant;
                output1[index] = (diagonal0 * residual1[index]
                                  - cross * residual0[index]) / determinant;
            }

            extern "C" __global__ void update_solution_residual(
                double* solution0, double* solution1,
                double* residual0, double* residual1,
                const double* direction0, const double* direction1,
                const double* applied0, const double* applied1,
                const double* step, int count) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                if (index >= count) return;
                double value = *step;
                solution0[index] += value * direction0[index];
                solution1[index] += value * direction1[index];
                residual0[index] -= value * applied0[index];
                residual1[index] -= value * applied1[index];
            }

            extern "C" __global__ void update_direction(
                double* direction0, double* direction1,
                const double* preconditioned0,
                const double* preconditioned1,
                const double* beta, int count) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                if (index >= count) return;
                double value = *beta;
                direction0[index] = preconditioned0[index]
                    + value * direction0[index];
                direction1[index] = preconditioned1[index]
                    + value * direction1[index];
            }
            """,
            options=("--std=c++11",),
            name_expressions=(
                "apply_membrane_hessian",
                "apply_preconditioner",
                "update_solution_residual",
                "update_direction",
            ),
        )
        _PCG_KERNELS = tuple(
            module.get_function(name)
            for name in (
                "apply_membrane_hessian",
                "apply_preconditioner",
                "update_solution_residual",
                "update_direction",
            )
        )
    return _PCG_KERNELS


def _solve_membrane_system_cupy_fused(
    hessian,
    right_hand_side,
    *,
    mu: float,
    identity_weight: float,
    max_iterations: int,
    relative_tolerance: float,
    check_interval: int | None = None,
):
    "使用融合 CUDA kernel 执行 CuPy PCG，保留逐轮收敛检查。"

    import cupy as cp

    configured_interval = (
        int(os.environ.get("CAT_SURFACE_PCG_CHECK_INTERVAL", "1"))
        if check_interval is None
        else int(check_interval)
    )
    if (
        configured_interval == 0
        and max_iterations > 0
        and os.environ.get("CAT_SURFACE_PCG_GRAPH", "1") == "1"
        and hessian.dtype == cp.float64
        and right_hand_side.dtype == cp.float64
        and hessian.flags.c_contiguous
        and right_hand_side.flags.c_contiguous
        and right_hand_side.ndim == 3
        and right_hand_side.shape[0] == 2
        and hessian.shape == (3, *right_hand_side.shape[1:])
    ):
        from cat_surface_gpu.backends.cuda.pcg_graph import solve_fixed_pcg

        return solve_fixed_pcg(
            hessian, right_hand_side, mu, identity_weight, max_iterations
        )

    apply_kernel, precondition_kernel, update_kernel, direction_kernel = (
        _get_pcg_kernels()
    )
    height, width = map(int, right_hand_side.shape[1:])
    count = height * width
    blocks = (count + 255) // 256
    hessian0, hessian1, hessian2 = hessian
    solution = cp.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    preconditioned = cp.empty_like(right_hand_side)
    direction = cp.empty_like(right_hand_side)
    applied = cp.empty_like(right_hand_side)
    diagonal = 4.0 * float(mu) + float(identity_weight)
    precondition_arguments = (
        residual[0],
        residual[1],
        hessian0,
        hessian1,
        hessian2,
        preconditioned[0],
        preconditioned[1],
        np.int32(count),
        np.float64(diagonal),
    )
    precondition_kernel((blocks,), (256,), precondition_arguments)
    direction[...] = preconditioned
    residual_dot = cp.sum(residual * preconditioned, dtype=cp.float64)
    initial_norm = cp.sqrt(cp.sum(right_hand_side * right_hand_side, dtype=cp.float64))
    threshold = float(relative_tolerance) * cp.maximum(initial_norm, 1.0e-30)
    # 间隔为零时执行完整固定迭代预算，不在中途检查收敛；
    # 迭代数不变，仅避免每轮向 CPU 同步一个残差标量。
    check_interval = max(0, configured_interval)
    for completed in range(1, int(max_iterations) + 1):
        apply_kernel(
            (blocks,),
            (256,),
            (
                direction[0],
                direction[1],
                hessian0,
                hessian1,
                hessian2,
                applied[0],
                applied[1],
                np.int32(height),
                np.int32(width),
                np.float64(mu),
                np.float64(identity_weight),
            ),
        )
        denominator = cp.sum(direction * applied, dtype=cp.float64)
        step = residual_dot / cp.maximum(denominator, 1.0e-30)
        update_kernel(
            (blocks,),
            (256,),
            (
                solution[0],
                solution[1],
                residual[0],
                residual[1],
                direction[0],
                direction[1],
                applied[0],
                applied[1],
                step,
                np.int32(count),
            ),
        )
        if check_interval > 0 and (
            completed % check_interval == 0 or completed == int(max_iterations)
        ):
            if bool(
                (
                    cp.sqrt(cp.sum(residual * residual, dtype=cp.float64)) <= threshold
                ).item()
            ):
                return solution, completed
        if completed == int(max_iterations):
            # 最后一轮无需构建下一方向，因此省去预条件器和点积。
            break
        precondition_kernel(
            (blocks,),
            (256,),
            (
                residual[0],
                residual[1],
                hessian0,
                hessian1,
                hessian2,
                preconditioned[0],
                preconditioned[1],
                np.int32(count),
                np.float64(diagonal),
            ),
        )
        next_dot = cp.sum(residual * preconditioned, dtype=cp.float64)
        beta = next_dot / cp.where(residual_dot == 0.0, 1.0, residual_dot)
        direction_kernel(
            (blocks,),
            (256,),
            (
                direction[0],
                direction[1],
                preconditioned[0],
                preconditioned[1],
                beta,
                np.int32(count),
            ),
        )
        residual_dot = next_dot
    return solution, int(max_iterations)


def objective_terms(source, target, mapping, jacobian, weights=None):
    "复现官方单通道 ``initialise_objfun``。"

    xp = _array_module(source)
    if source.shape != target.shape or mapping.shape[1:] != source.shape:
        raise ValueError("Source, target, and mapping grid dimensions must match")
    height, width = source.shape
    if (
        xp is np
        and weights is None
        and all(
            array.dtype == np.float64 and array.flags.c_contiguous
            for array in (source, target, mapping, jacobian)
        )
    ):
        gradient = np.empty((2, height, width), dtype=np.float64)
        hessian = np.empty((3, height, width), dtype=np.float64)
        squared_difference = np.empty((height, width), dtype=np.float64)
        _objective_terms_cpu_fused(
            source,
            target,
            mapping,
            jacobian,
            gradient,
            hessian,
            squared_difference,
        )
        return (
            0.5 * np.sum(squared_difference, dtype=np.float64),
            gradient,
            hessian,
        )
    if (
        xp is not np
        and weights is None
        and source.dtype == xp.float64
        and target.dtype == xp.float64
        and mapping.dtype == xp.float64
        and jacobian.dtype == xp.float64
        and source.flags.c_contiguous
        and target.flags.c_contiguous
        and mapping.flags.c_contiguous
        and jacobian.flags.c_contiguous
    ):
        # SurfWarp 默认无逐像素权重，融合双线性采样、梯度、Hessian
        # 及目标函数项，避免每轮 DARTEL 重复分配十余个数组。
        import cupy as cp

        gradient = cp.empty((2, height, width), dtype=cp.float64)
        hessian = cp.empty((3, height, width), dtype=cp.float64)
        objective_terms_gpu = cp.empty((height, width), dtype=cp.float64)
        count = int(height * width)
        blocks = (count + 255) // 256
        _get_objective_kernel()(
            (blocks,),
            (256,),
            (
                source,
                target,
                mapping[0],
                mapping[1],
                jacobian[0, 0],
                jacobian[0, 1],
                jacobian[1, 0],
                jacobian[1, 1],
                gradient[0],
                gradient[1],
                hessian[0],
                hessian[1],
                hessian[2],
                objective_terms_gpu,
                np.int32(height),
                np.int32(width),
            ),
        )
        return (
            cp.sum(objective_terms_gpu, dtype=cp.float64),
            gradient,
            hessian,
        )
    x = mapping[0] - 1.0
    y = mapping[1] - 1.0
    floor_x = xp.floor(x).astype(xp.int64)
    floor_y = xp.floor(y).astype(xp.int64)
    fraction_x = x - floor_x
    fraction_y = y - floor_y
    inverse_x = 1.0 - fraction_x
    inverse_y = 1.0 - fraction_y
    x0, y0 = _bound(floor_x, floor_y, width, height, xp)
    x1, _ = _bound(floor_x + 1, floor_y, width, height, xp)
    _, y1 = _bound(floor_x, floor_y + 1, width, height, xp)
    k22 = source[y0, x0]
    k12 = source[y0, x1]
    k21 = source[y1, x0]
    k11 = source[y1, x1]
    sampled = (k11 * fraction_x + k21 * inverse_x) * fraction_y + (
        k12 * fraction_x + k22 * inverse_x
    ) * inverse_y
    difference = sampled - target
    derivative_x = (k11 - k21) * fraction_y + (k12 - k22) * inverse_y
    derivative_y = (k11 * fraction_x + k21 * inverse_x) - (
        k12 * fraction_x + k22 * inverse_x
    )
    gradient_x = jacobian[0, 0] * derivative_x + jacobian[0, 1] * derivative_y
    gradient_y = jacobian[1, 0] * derivative_x + jacobian[1, 1] * derivative_y
    weight = 1.0 if weights is None else weights
    hessian = xp.stack(
        (
            gradient_x * gradient_x * weight,
            gradient_y * gradient_y * weight,
            gradient_x * gradient_y * weight,
        ),
        axis=0,
    )
    gradient = xp.stack(
        (gradient_x * difference * weight, gradient_y * difference * weight),
        axis=0,
    )
    objective = 0.5 * xp.sum(difference * difference * weight, dtype=xp.float64)
    return objective, gradient, hessian


def transport_derivatives(
    gradient,
    hessian,
    mapping,
    jacobian,
    squarings: int,
    *,
    save_transform: bool,
):
    "复现官方 ``squaring`` 对梯度和 Hessian 的传输。"

    xp = _array_module(gradient)
    transported_gradient = gradient.copy()
    transported_hessian = hessian.copy()
    current_mapping = mapping
    current_jacobian = jacobian
    for index in range(squarings):
        if (
            xp is not np
            and os.environ.get("CAT_SURFACE_TRANSPORT_FUSED", "1") == "1"
            and all(
                array.dtype == xp.float64 and array.flags.c_contiguous
                for array in (
                    transported_gradient,
                    transported_hessian,
                    current_mapping,
                    current_jacobian,
                )
            )
            and transported_gradient.ndim == 3
            and transported_gradient.shape[0] == 2
            and transported_hessian.shape == (3, *transported_gradient.shape[1:])
            and current_mapping.shape == transported_gradient.shape
            and current_jacobian.shape == (2, 2, *transported_gradient.shape[1:])
        ):
            from cat_surface_gpu.backends.cuda.transport import transport_accumulate

            transported_gradient, transported_hessian = transport_accumulate(
                transported_gradient,
                transported_hessian,
                current_mapping,
                current_jacobian,
            )
            if save_transform or index < squarings - 1:
                current_mapping, current_jacobian = compose_with_jacobian(
                    current_mapping, current_jacobian, current_mapping, current_jacobian
                )
            continue
        cpu_fused = xp is np and all(
            array.dtype == np.float64 and array.flags.c_contiguous
            for array in (
                transported_gradient,
                transported_hessian,
                current_mapping,
                current_jacobian,
            )
        )
        if cpu_fused:
            sampled_gradient = np.empty_like(transported_gradient)
            sampled_hessian = np.empty_like(transported_hessian)
            _sample_transport_cpu_fused(
                transported_gradient,
                transported_hessian,
                current_mapping,
                sampled_gradient,
                sampled_hessian,
            )
        else:
            x = current_mapping[0] - 1.0
            y = current_mapping[1] - 1.0
            sampled_gradient = xp.stack(
                tuple(_bilinear(transported_gradient[item], x, y) for item in range(2)),
                axis=0,
            )
            sampled_hessian = xp.stack(
                tuple(_bilinear(transported_hessian[item], x, y) for item in range(3)),
                axis=0,
            )
        if cpu_fused:
            _transport_accumulate_cpu_fused(
                transported_gradient,
                transported_hessian,
                sampled_gradient,
                sampled_hessian,
                current_jacobian,
            )
            if save_transform or index < squarings - 1:
                current_mapping, current_jacobian = compose_with_jacobian(
                    current_mapping,
                    current_jacobian,
                    current_mapping,
                    current_jacobian,
                )
            continue
        j11 = current_jacobian[0, 0]
        j21 = current_jacobian[0, 1]
        j12 = current_jacobian[1, 0]
        j22 = current_jacobian[1, 1]
        determinant = j11 * j22 - j12 * j21
        first = sampled_gradient[0]
        second = sampled_gradient[1]
        added_gradient = xp.stack(
            (
                determinant * (first * j11 + second * j21),
                determinant * (first * j12 + second * j22),
            ),
            axis=0,
        )
        a00, a11, a01 = sampled_hessian
        tmp1 = a00 * j11 + a01 * j21
        tmp2 = a01 * j11 + a11 * j21
        tmp3 = a00 * j12 + a01 * j22
        tmp4 = a01 * j12 + a11 * j22
        added_hessian = xp.stack(
            (
                determinant * (tmp1 * j11 + tmp2 * j21),
                determinant * (tmp3 * j12 + tmp4 * j22),
                determinant * (tmp1 * j12 + tmp2 * j22),
            ),
            axis=0,
        )
        transported_gradient += added_gradient
        transported_hessian += added_hessian
        if save_transform or index < squarings - 1:
            current_mapping, current_jacobian = compose_with_jacobian(
                current_mapping,
                current_jacobian,
                current_mapping,
                current_jacobian,
            )
    return transported_gradient, transported_hessian, current_mapping, current_jacobian


@njit(cache=True, parallel=True, fastmath=False)
def _apply_membrane_cpu_fused(
    field,
    hessian,
    mu,
    identity_weight,
    output,
):
    "单次遍历计算膜能双分量和 Hessian 交叉项，并处理边界。"

    height, width = field.shape[1:]
    center = 4.0 * mu + identity_weight
    for y in prange(height):
        up = max(y - 1, 0)
        down = min(y + 1, height - 1)
        for x in range(width):
            left = (x - 1) % width
            right = (x + 1) % width
            value0 = field[0, y, x]
            value1 = field[1, y, x]
            membrane0 = center * value0 - mu * (
                field[0, y, left]
                + field[0, y, right]
                + field[0, up, x]
                + field[0, down, x]
            )
            membrane1 = center * value1 - mu * (
                field[1, y, left]
                + field[1, y, right]
                + field[1, up, x]
                + field[1, down, x]
            )
            output[0, y, x] = (
                membrane0 + hessian[0, y, x] * value0 + hessian[2, y, x] * value1
            )
            output[1, y, x] = (
                membrane1 + hessian[1, y, x] * value1 + hessian[2, y, x] * value0
            )


@njit(cache=True, parallel=True, fastmath=False)
def _precondition_cpu_fused(residual, hessian, diagonal, output):
    "每个体素求解一个 2×2 Jacobi 预条件块并复用输出缓冲。"

    height, width = residual.shape[1:]
    for y in prange(height):
        for x in range(width):
            diagonal0 = hessian[0, y, x] + diagonal
            diagonal1 = hessian[1, y, x] + diagonal
            cross = hessian[2, y, x]
            determinant = max(diagonal0 * diagonal1 - cross * cross, 1.0e-30)
            first = residual[0, y, x]
            second = residual[1, y, x]
            output[0, y, x] = (diagonal1 * first - cross * second) / determinant
            output[1, y, x] = (diagonal0 * second - cross * first) / determinant


@njit(cache=True, parallel=True, fastmath=False)
def _pcg_solution_residual_cpu_fused(solution, residual, direction, applied, step):
    "同步更新 PCG 解和残差，保留逐元素 FP64 运算顺序。"

    height, width = residual.shape[1:]
    for row in prange(height):
        for col in range(width):
            for component in range(2):
                solution[component, row, col] += step * direction[component, row, col]
                residual[component, row, col] -= step * applied[component, row, col]


@njit(cache=True, parallel=True, fastmath=False)
def _pcg_direction_cpu_fused(direction, preconditioned, beta):
    "原地更新搜索方向，避免 PCG 每轮分配完整场。"

    height, width = direction.shape[1:]
    for row in prange(height):
        for col in range(width):
            for component in range(2):
                direction[component, row, col] = (
                    preconditioned[component, row, col]
                    + beta * direction[component, row, col]
                )


def solve_membrane_system(
    hessian,
    right_hand_side,
    *,
    mu: float,
    identity_weight: float,
    max_iterations: int = 40,
    relative_tolerance: float = 1.0e-6,
    check_interval: int | None = None,
):
    "用预条件共轭梯度求解官方膜能线性系统。"

    xp = _array_module(right_hand_side)
    if xp is not np:
        return _solve_membrane_system_cupy_fused(
            hessian,
            right_hand_side,
            mu=mu,
            identity_weight=identity_weight,
            max_iterations=max_iterations,
            relative_tolerance=relative_tolerance,
            check_interval=check_interval,
        )
    diagonal = 4.0 * float(mu) + float(identity_weight)
    applied_workspace = np.empty_like(right_hand_side)
    preconditioned_workspace = np.empty_like(right_hand_side)

    def apply(field):
        _apply_membrane_cpu_fused(
            field, hessian, float(mu), float(identity_weight), applied_workspace
        )
        return applied_workspace

    def precondition(residual):
        _precondition_cpu_fused(residual, hessian, diagonal, preconditioned_workspace)
        return preconditioned_workspace

    solution = xp.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    preconditioned = precondition(residual)
    direction = preconditioned.copy()
    residual_dot = xp.sum(residual * preconditioned, dtype=xp.float64)
    initial_norm = xp.sqrt(xp.sum(right_hand_side * right_hand_side, dtype=xp.float64))
    threshold = float(relative_tolerance) * xp.maximum(initial_norm, 1.0e-30)
    configured_interval = 1 if check_interval is None else max(0, int(check_interval))
    completed = 0
    for completed in range(1, max_iterations + 1):
        applied = apply(direction)
        denominator = xp.sum(direction * applied, dtype=xp.float64)
        step = residual_dot / xp.maximum(denominator, 1.0e-30)
        _pcg_solution_residual_cpu_fused(
            solution, residual, direction, applied, float(step)
        )
        if (
            configured_interval > 0
            and (completed % configured_interval == 0 or completed == max_iterations)
            and bool(
                (
                    xp.sqrt(xp.sum(residual * residual, dtype=xp.float64)) <= threshold
                ).item()
            )
        ):
            break
        if completed == max_iterations:
            break
        next_preconditioned = precondition(residual)
        next_dot = xp.sum(residual * next_preconditioned, dtype=xp.float64)
        _pcg_direction_cpu_fused(
            direction,
            next_preconditioned,
            float(next_dot / xp.where(residual_dot == 0.0, 1.0, residual_dot)),
        )
        residual_dot = next_dot
    return solution, completed


def _fmg_wt2(value, xp):
    "返回官方 FMG ``wt2`` 三次 B 样条权重。"

    distance = xp.abs(value)
    return xp.where(
        distance < 0.5,
        0.75 - distance * distance,
        xp.where(
            distance < 1.5,
            0.5 * (1.5 - distance) * (1.5 - distance),
            0.0,
        ),
    )


def _fmg_resize(field, output_shape):
    "复现官方 ``resize`` 的 y 方向 Neumann、x 方向周期插值。"

    xp = _array_module(field)
    if field.ndim != 3:
        raise ValueError("FMG grid must have shape (channels, height, width)")
    input_height, input_width = field.shape[1:]
    output_height, output_width = map(int, output_shape)

    scale_y = input_height / float(output_height)
    location_y = (xp.arange(output_height, dtype=xp.float64) + 0.5) * scale_y - 0.5
    origin_y = xp.floor(location_y + 0.5).astype(xp.int64)
    weight_y = _fmg_wt2(origin_y - location_y, xp)
    weight_y_minus = _fmg_wt2(origin_y - 1 - location_y, xp)
    weight_y_plus = _fmg_wt2(origin_y + 1 - location_y, xp)
    zeros_y = xp.zeros_like(origin_y)
    _, index_y_minus = _bound(zeros_y, origin_y - 1, input_width, input_height, xp)
    _, index_y = _bound(zeros_y, origin_y, input_width, input_height, xp)
    _, index_y_plus = _bound(zeros_y, origin_y + 1, input_width, input_height, xp)
    intermediate = (
        field[:, index_y_minus, :] * weight_y_minus[None, :, None]
        + field[:, index_y, :] * weight_y[None, :, None]
        + field[:, index_y_plus, :] * weight_y_plus[None, :, None]
    )

    scale_x = input_width / float(output_width)
    location_x = (xp.arange(output_width, dtype=xp.float64) + 0.5) * scale_x - 0.5
    origin_x = xp.floor(location_x + 0.5).astype(xp.int64)
    weight_x = _fmg_wt2(origin_x - location_x, xp)
    weight_x_minus = _fmg_wt2(origin_x - 1 - location_x, xp)
    weight_x_plus = _fmg_wt2(origin_x + 1 - location_x, xp)
    index_x_minus, _ = _bound(
        origin_x - 1, xp.zeros_like(origin_x), input_width, input_height, xp
    )
    index_x, _ = _bound(
        origin_x, xp.zeros_like(origin_x), input_width, input_height, xp
    )
    index_x_plus, _ = _bound(
        origin_x + 1, xp.zeros_like(origin_x), input_width, input_height, xp
    )
    return (
        intermediate[:, :, index_x_minus] * weight_x_minus[None, None, :]
        + intermediate[:, :, index_x] * weight_x[None, None, :]
        + intermediate[:, :, index_x_plus] * weight_x_plus[None, None, :]
    )


def _fmg_membrane_weights(mu, identity_weight, spacing_x, spacing_y):
    "生成官方膜能 FMG 在当前层级使用的五点模板系数。"

    weight_x = float(mu) * float(spacing_x) ** 2
    weight_y = float(mu) * float(spacing_y) ** 2
    return (
        2.0 * (weight_x + weight_y) + float(identity_weight),
        -weight_y,
        -weight_x,
    )


def _fmg_apply_membrane(field, hessian, weights):
    "应用 FMG 膜能加 Hessian 的矩阵。"

    xp = _array_module(field)
    center, vertical, horizontal = weights
    left, right, up, down = _grid_neighbours(field)
    result = xp.empty_like(field)
    result[0] = (
        center * field[0]
        + vertical * (up[0] + down[0])
        + horizontal * (left[0] + right[0])
        + hessian[0] * field[0]
        + hessian[2] * field[1]
    )
    result[1] = (
        center * field[1]
        + vertical * (up[1] + down[1])
        + horizontal * (left[1] + right[1])
        + hessian[1] * field[1]
        + hessian[2] * field[0]
    )
    return result


_FMG_RELAX_KERNEL = None


def _get_fmg_relax_kernel():
    "延迟构建 CuPy FMG 棋盘格松弛核。"

    global _FMG_RELAX_KERNEL
    if _FMG_RELAX_KERNEL is None:
        import cupy as cp

        _FMG_RELAX_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__ void relax_membrane(
                double* field, const double* hessian, const double* rhs,
                int height, int width, int parity,
                double center, double vertical, double horizontal) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int count = height * width;
                if (index >= count) return;
                int y = index / width;
                int x = index - y * width;
                if (((x + y) & 1) != parity) return;
                int left_x = (x == 0) ? width - 1 : x - 1;
                int right_x = (x + 1 == width) ? 0 : x + 1;
                int up_y;
                int down_y;
                if (height == 1) {
                    up_y = 0;
                    down_y = 0;
                } else {
                    up_y = (y == 0) ? 0 : y - 1;
                    down_y = (y + 1 == height) ? height - 1 : y + 1;
                }
                int left = y * width + left_x;
                int right = y * width + right_x;
                int up = up_y * width + x;
                int down = down_y * width + x;
                double a00 = hessian[index];
                double a11 = hessian[count + index];
                double a01 = hessian[2 * count + index];
                double sux = rhs[index]
                    - vertical * (field[up] + field[down])
                    - horizontal * (field[left] + field[right]);
                double suy = rhs[count + index]
                    - vertical * (field[count + up] + field[count + down])
                    - horizontal * (field[count + left] + field[count + right]);
                double diagonal_x = a00 + center;
                double diagonal_y = a11 + center;
                double determinant = diagonal_x * diagonal_y - a01 * a01;
                if (determinant < 1.0e-30) determinant = 1.0e-30;
                field[index] = (diagonal_y * sux - a01 * suy) / determinant;
                field[count + index] = (diagonal_x * suy - a01 * sux) / determinant;
            }
            """,
            "relax_membrane",
        )
    return _FMG_RELAX_KERNEL


def _fmg_relax_membrane(field, hessian, right_hand_side, weights, iterations):
    "执行官方膜能棋盘格 Gauss–Seidel 松弛。"

    xp = _array_module(field)
    height, width = field.shape[1:]
    if xp is not np:
        kernel = _get_fmg_relax_kernel()
        count = int(height * width)
        block_size = 256
        grid_size = (count + block_size - 1) // block_size
        center, vertical, horizontal = weights
        arguments = (
            field,
            hessian,
            right_hand_side,
            np.int32(height),
            np.int32(width),
            np.int32(0),
            np.float64(center),
            np.float64(vertical),
            np.float64(horizontal),
        )
        for iteration in range(2 * int(iterations)):
            arguments = arguments[:5] + (np.int32((iteration + 1) & 1),) + arguments[6:]
            kernel((grid_size,), (block_size,), arguments)
        return field
    rows = xp.arange(height, dtype=xp.int64)[:, None]
    columns = xp.arange(width, dtype=xp.int64)[None, :]
    parity = (rows + columns) & 1
    center, vertical, horizontal = weights
    for iteration in range(2 * int(iterations)):
        left, right, up, down = _grid_neighbours(field)
        residual_x = right_hand_side[0] - vertical * (up[0] + down[0])
        residual_x -= horizontal * (left[0] + right[0])
        residual_y = right_hand_side[1] - vertical * (up[1] + down[1])
        residual_y -= horizontal * (left[1] + right[1])
        diagonal_x = hessian[0] + center
        diagonal_y = hessian[1] + center
        determinant = xp.maximum(
            diagonal_x * diagonal_y - hessian[2] * hessian[2],
            1.0e-30,
        )
        next_x = (diagonal_y * residual_x - hessian[2] * residual_y) / determinant
        next_y = (diagonal_x * residual_y - hessian[2] * residual_x) / determinant
        active = parity == ((iteration + 1) & 1)
        field[0] = xp.where(active, next_x, field[0])
        field[1] = xp.where(active, next_y, field[1])
    return field


def solve_membrane_system_fmg(
    hessian,
    right_hand_side,
    *,
    mu: float,
    identity_weight: float,
    cycles: int = 3,
    relax_iterations: int = 3,
):
    "用官方 FMG 层级和膜能棋盘格松弛求解线性系统。\n\n该函数只实现官方当前默认的膜能 ``rtype=1``，用于 CuPy/NumPy 数值与速度\nA/B；其余正则类型仍由 PCG 路径处理。"

    xp = _array_module(right_hand_side)
    if hessian.shape[0] != 3 or right_hand_side.shape[0] != 2:
        raise ValueError(
            "FMG Hessian must have shape (3,H,W) and the right-hand side (2,H,W)"
        )
    base_height, base_width = right_hand_side.shape[1:]
    dimensions = [(base_height, base_width)]
    while dimensions[-1][0] >= 2 or dimensions[-1][1] >= 2:
        current_height, current_width = dimensions[-1]
        dimensions.append(((current_height + 1) // 2, (current_width + 1) // 2))

    hessians = [hessian]
    rhs_base = [right_hand_side]
    for output_height, output_width in dimensions[1:]:
        hessians.append(_fmg_resize(hessians[-1], (output_height, output_width)))
        rhs_base.append(_fmg_resize(rhs_base[-1], (output_height, output_width)))
    rhs_levels = [value.copy() for value in rhs_base]
    fields = [xp.zeros_like(value) for value in rhs_base]

    def level_weights(level):
        height, width = dimensions[level]
        return _fmg_membrane_weights(
            mu,
            identity_weight,
            width / float(base_width),
            height / float(base_height),
        )

    coarsest = len(dimensions) - 1
    coarse_hessian = hessians[coarsest]
    coarse_rhs = rhs_levels[coarsest]
    coarse_a0 = coarse_hessian[0] + float(identity_weight)
    coarse_a1 = coarse_hessian[1] + float(identity_weight)
    coarse_determinant = xp.maximum(
        coarse_a0 * coarse_a1 * 1.0000000001 - coarse_hessian[2] * coarse_hessian[2],
        1.0e-12 * coarse_a0 * coarse_a1,
    )
    fields[coarsest][0] = (
        coarse_a1 * coarse_rhs[0] - coarse_hessian[2] * coarse_rhs[1]
    ) / coarse_determinant
    fields[coarsest][1] = (
        coarse_a0 * coarse_rhs[1] - coarse_hessian[2] * coarse_rhs[0]
    ) / coarse_determinant
    _fmg_relax_membrane(
        fields[coarsest],
        coarse_hessian,
        coarse_rhs,
        level_weights(coarsest),
        relax_iterations,
    )

    for level in range(coarsest - 1, -1, -1):
        fields[level] = _fmg_resize(fields[level + 1], dimensions[level])
        if level > 0:
            rhs_levels[level] = rhs_base[level].copy()
        for _ in range(int(cycles)):
            for inner in range(level, coarsest):
                _fmg_relax_membrane(
                    fields[inner],
                    hessians[inner],
                    rhs_levels[inner],
                    level_weights(inner),
                    relax_iterations,
                )
                residual = rhs_levels[inner] - _fmg_apply_membrane(
                    fields[inner], hessians[inner], level_weights(inner)
                )
                rhs_levels[inner + 1] = _fmg_resize(residual, dimensions[inner + 1])
                fields[inner + 1] = xp.zeros_like(rhs_levels[inner + 1])
            _fmg_relax_membrane(
                fields[coarsest],
                hessians[coarsest],
                rhs_levels[coarsest],
                level_weights(coarsest),
                relax_iterations,
            )
            for inner in range(coarsest - 1, level - 1, -1):
                correction = _fmg_resize(fields[inner + 1], dimensions[inner])
                fields[inner] += correction
                _fmg_relax_membrane(
                    fields[inner],
                    hessians[inner],
                    rhs_levels[inner],
                    level_weights(inner),
                    relax_iterations,
                )
    return fields[0]


def dartel_symmetric_step(
    velocity,
    source,
    target,
    *,
    time_step: int,
    mu: float,
    lm_regularization: float,
    weights=None,
    solver_iterations: int = 40,
    solver_backend: str = "pcg",
    fmg_cycles: int = 3,
    fmg_relax_iterations: int = 3,
    collect_diagnostics: bool = True,
    pcg_check_interval: int | None = None,
):
    "执行官方默认 code=1、rtype=1 的一次 DARTEL 更新。"

    xp = _array_module(velocity)
    scale = 1.0 / float(2**time_step)
    mapping, jacobian = exponentiate_velocity(velocity, time_step)
    adjusted = jacobian_divide_small_deformation(jacobian, velocity, scale)
    objective_forward, gradient, hessian = objective_terms(
        source, target, mapping, adjusted, weights
    )
    inverse_mapping, inverse_jacobian = small_deformation(velocity, -scale)
    gradient, hessian, mapping, jacobian = transport_derivatives(
        gradient,
        hessian,
        inverse_mapping,
        inverse_jacobian,
        time_step,
        save_transform=True,
    )

    adjusted_reverse = jacobian_divide_small_deformation(jacobian, velocity, -scale)
    objective_reverse, reverse_gradient, reverse_hessian = objective_terms(
        target, source, mapping, adjusted_reverse, weights
    )
    forward_mapping, forward_jacobian = small_deformation(velocity, scale)
    reverse_gradient, reverse_hessian, _, _ = transport_derivatives(
        reverse_gradient,
        reverse_hessian,
        forward_mapping,
        forward_jacobian,
        time_step,
        save_transform=False,
    )
    gradient -= reverse_gradient
    hessian += reverse_hessian

    regularization = membrane_operator(velocity, mu, 0.0)
    right_hand_side = gradient * scale + regularization
    hessian *= scale
    if solver_backend == "pcg":
        update, solver_steps = solve_membrane_system(
            hessian,
            right_hand_side,
            mu=mu,
            identity_weight=lm_regularization,
            max_iterations=solver_iterations,
            check_interval=pcg_check_interval,
        )
    elif solver_backend == "fmg":
        update = solve_membrane_system_fmg(
            hessian,
            right_hand_side,
            mu=mu,
            identity_weight=lm_regularization,
            cycles=fmg_cycles,
            relax_iterations=fmg_relax_iterations,
        )
        solver_steps = int(fmg_cycles * fmg_relax_iterations)
    else:
        raise ValueError("solver_backend 必须是 pcg 或 fmg")
    output = velocity - update
    diagnostics = {
        "objective": objective_forward + objective_reverse,
        "solver_iterations": solver_steps,
    }
    if collect_diagnostics:
        # SurfWarp 主循环不使用这些标量，保留选项用于独立数值诊断。
        diagnostics.update(
            {
                "regularization": 0.5
                * xp.sum(regularization * velocity, dtype=xp.float64),
                "gradient_norm": xp.sqrt(
                    xp.sum(right_hand_side * right_hand_side, dtype=xp.float64)
                ),
            }
        )
    return output, diagnostics
