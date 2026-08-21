"""DARTEL 规则网格算子的 Triton 融合 kernel。"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import torch


# 这些权重只由网格参数和正则参数决定；缓存后可避免每次 FMG 调用都发生
# 一次 host→device 的小张量创建，也使 CUDA Graph 捕获期间不再触发 H2D 拷贝。
_WEIGHTS_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _cached_weights(
    values: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """返回指定设备和 dtype 上可复用的常量权重。"""

    key = (
        device.type,
        device.index,
        dtype,
        *(float(value) for value in values),
    )
    cached = _WEIGHTS_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, device=device, dtype=dtype)
        _WEIGHTS_CACHE[key] = cached
    return cached


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 由设备环境决定
    triton = None
    tl = None


def available() -> bool:
    """返回 Triton 包是否可导入。"""

    return triton is not None and tl is not None


if available():

    @triton.jit
    def _wt2_kernel(distance):
        """计算 CAT resize 使用的三点二次插值权重。"""

        absolute = tl.abs(distance)
        inner = 0.75 - absolute * absolute
        outer = 0.5 * (1.5 - absolute) * (1.5 - absolute)
        return tl.where(absolute < 0.5, inner, tl.where(absolute < 1.5, outer, 0.0))

    @triton.jit
    def _resize_field_kernel(
        source_ptr,
        output_ptr,
        source_nx,
        source_ny,
        target_nx,
        target_ny,
        source_points,
        target_points,
        total,
        coordinate_fp32: tl.constexpr,
        offsets: tl.constexpr,
    ):
        """以一个程序块完成一个多通道场的三点二维重采样。"""

        index = tl.program_id(0) * offsets + tl.arange(0, offsets)
        mask = index < total
        channel = index // target_points
        local = index - channel * target_points
        x = local % target_nx
        y = local // target_nx

        # 坐标和权重也随显式 DARTEL dtype 选择；FP32 路径避免在每次
        # multilevel resize 中无意引入 FP64 运算，FP64 reference 保持原公式。
        if coordinate_fp32:
            x_location = (x.to(tl.float32) + 0.5) * source_nx / target_nx - 0.5
            x_origin = tl.floor(x_location + 0.5).to(tl.int32)
            x_distance = x_origin.to(tl.float32) - x_location
        else:
            x_location = (x.to(tl.float64) + 0.5) * source_nx / target_nx - 0.5
            x_origin = tl.floor(x_location + 0.5).to(tl.int32)
            x_distance = x_origin.to(tl.float64) - x_location
        x_weight = _wt2_kernel(x_distance)
        x_minus_weight = _wt2_kernel(x_distance - 1.0)
        x_plus_weight = _wt2_kernel(x_distance + 1.0)

        if coordinate_fp32:
            y_location = (y.to(tl.float32) + 0.5) * source_ny / target_ny - 0.5
            y_origin = tl.floor(y_location + 0.5).to(tl.int32)
            y_distance = y_origin.to(tl.float32) - y_location
        else:
            y_location = (y.to(tl.float64) + 0.5) * source_ny / target_ny - 0.5
            y_origin = tl.floor(y_location + 0.5).to(tl.int32)
            y_distance = y_origin.to(tl.float64) - y_location
        y_weight = _wt2_kernel(y_distance)
        y_minus_weight = _wt2_kernel(y_distance - 1.0)
        y_plus_weight = _wt2_kernel(y_distance + 1.0)

        x_minus = tl.where(x_origin == 0, source_nx - 1, x_origin - 1)
        x_center = x_origin
        x_plus = tl.where(
            x_origin == source_nx - 1,
            0,
            x_origin + 1,
        )
        y_minus = tl.where(y_origin == 0, 0, y_origin - 1)
        y_center = y_origin
        y_plus = tl.where(
            y_origin == source_ny - 1,
            source_ny - 1,
            y_origin + 1,
        )

        base = channel * source_points
        value_x_minus_y_minus = tl.load(
            source_ptr + base + y_minus * source_nx + x_minus,
            mask=mask,
            other=0.0,
        )
        value_x_minus_y_center = tl.load(
            source_ptr + base + y_center * source_nx + x_minus,
            mask=mask,
            other=0.0,
        )
        value_x_minus_y_plus = tl.load(
            source_ptr + base + y_plus * source_nx + x_minus,
            mask=mask,
            other=0.0,
        )
        value_x_center_y_minus = tl.load(
            source_ptr + base + y_minus * source_nx + x_center,
            mask=mask,
            other=0.0,
        )
        value_x_center_y_center = tl.load(
            source_ptr + base + y_center * source_nx + x_center,
            mask=mask,
            other=0.0,
        )
        value_x_center_y_plus = tl.load(
            source_ptr + base + y_plus * source_nx + x_center,
            mask=mask,
            other=0.0,
        )
        value_x_plus_y_minus = tl.load(
            source_ptr + base + y_minus * source_nx + x_plus,
            mask=mask,
            other=0.0,
        )
        value_x_plus_y_center = tl.load(
            source_ptr + base + y_center * source_nx + x_plus,
            mask=mask,
            other=0.0,
        )
        value_x_plus_y_plus = tl.load(
            source_ptr + base + y_plus * source_nx + x_plus,
            mask=mask,
            other=0.0,
        )

        interpolated_x_minus = (
            y_minus_weight * value_x_minus_y_minus
            + y_weight * value_x_minus_y_center
            + y_plus_weight * value_x_minus_y_plus
        )
        interpolated_x_center = (
            y_minus_weight * value_x_center_y_minus
            + y_weight * value_x_center_y_center
            + y_plus_weight * value_x_center_y_plus
        )
        interpolated_x_plus = (
            y_minus_weight * value_x_plus_y_minus
            + y_weight * value_x_plus_y_center
            + y_plus_weight * value_x_plus_y_plus
        )
        result = (
            x_minus_weight * interpolated_x_minus
            + x_weight * interpolated_x_center
            + x_plus_weight * interpolated_x_plus
        )
        tl.store(output_ptr + index, result, mask=mask)

    @triton.jit
    def _apply_membrane_system_kernel(
        field_ptr,
        hessian_ptr,
        output_ptr,
        weights_ptr,
        n_points,
        nx,
        ny,
        offsets: tl.constexpr,
    ):
        """融合膜正则模板和三分量 Hessian 的系统乘法。"""

        index = tl.program_id(0) * offsets + tl.arange(0, offsets)
        mask = index < n_points
        x = index % nx
        y = index // nx
        xm = y * nx + tl.where(x == 0, nx - 1, x - 1)
        xp = y * nx + tl.where(x == nx - 1, 0, x + 1)
        ym = tl.where(y == 0, y * nx + x, (y - 1) * nx + x)
        yp = tl.where(y == ny - 1, (ny - 1) * nx + x, (y + 1) * nx + x)

        w00 = tl.load(weights_ptr + 0)
        w01 = tl.load(weights_ptr + 1)
        w10 = tl.load(weights_ptr + 2)
        vx = tl.load(field_ptr + index, mask=mask, other=0.0)
        vy = tl.load(field_ptr + n_points + index, mask=mask, other=0.0)
        vxm = tl.load(field_ptr + xm, mask=mask, other=0.0)
        vxp = tl.load(field_ptr + xp, mask=mask, other=0.0)
        vxm_y = tl.load(field_ptr + ym, mask=mask, other=0.0)
        vxp_y = tl.load(field_ptr + yp, mask=mask, other=0.0)
        vym = tl.load(field_ptr + n_points + xm, mask=mask, other=0.0)
        vyp = tl.load(field_ptr + n_points + xp, mask=mask, other=0.0)
        vym_y = tl.load(field_ptr + n_points + ym, mask=mask, other=0.0)
        vyp_y = tl.load(field_ptr + n_points + yp, mask=mask, other=0.0)

        regularized_x = w00 * vx + w01 * (vxm_y + vxp_y) + w10 * (vxm + vxp)
        regularized_y = w00 * vy + w01 * (vym_y + vyp_y) + w10 * (vym + vyp)
        hessian_xx = tl.load(hessian_ptr + index, mask=mask, other=0.0)
        hessian_yy = tl.load(hessian_ptr + n_points + index, mask=mask, other=0.0)
        hessian_xy = tl.load(hessian_ptr + 2 * n_points + index, mask=mask, other=0.0)
        tl.store(
            output_ptr + index,
            regularized_x + hessian_xx * vx + hessian_xy * vy,
            mask=mask,
        )
        tl.store(
            output_ptr + n_points + index,
            regularized_y + hessian_xy * vx + hessian_yy * vy,
            mask=mask,
        )

    @triton.jit
    def _relax_membrane_kernel(
        input_ptr,
        output_ptr,
        a_ptr,
        b_ptr,
        weights_ptr,
        n_points,
        nx,
        ny,
        color,
        offsets: tl.constexpr,
    ):
        """为一个红黑颜色执行一次膜正则 Gauss-Seidel 更新。"""

        index = tl.program_id(0) * offsets + tl.arange(0, offsets)
        mask = index < n_points
        x = index % nx
        y = index // nx
        xm = y * nx + tl.where(x == 0, nx - 1, x - 1)
        xp = y * nx + tl.where(x == nx - 1, 0, x + 1)
        # y=0 的镜像邻居是当前行同一列，而不是展平数组的第0点。
        ym = tl.where(y == 0, y * nx + x, (y - 1) * nx + x)
        yp = tl.where(y == ny - 1, (ny - 1) * nx + x, (y + 1) * nx + x)
        active = mask & (((x + y) & 1) == color)
        w00 = tl.load(weights_ptr + 0)
        w01 = tl.load(weights_ptr + 1)
        w10 = tl.load(weights_ptr + 2)

        ux = tl.load(input_ptr + index, mask=mask, other=0.0)
        uy = tl.load(input_ptr + n_points + index, mask=mask, other=0.0)
        ux_m = tl.load(input_ptr + xm, mask=mask, other=0.0)
        ux_p = tl.load(input_ptr + xp, mask=mask, other=0.0)
        ux_ym = tl.load(input_ptr + ym, mask=mask, other=0.0)
        ux_yp = tl.load(input_ptr + yp, mask=mask, other=0.0)
        uy_m = tl.load(input_ptr + n_points + xm, mask=mask, other=0.0)
        uy_p = tl.load(input_ptr + n_points + xp, mask=mask, other=0.0)
        uy_ym = tl.load(input_ptr + n_points + ym, mask=mask, other=0.0)
        uy_yp = tl.load(input_ptr + n_points + yp, mask=mask, other=0.0)

        rhs_x = tl.load(b_ptr + index, mask=mask, other=0.0)
        rhs_y = tl.load(b_ptr + n_points + index, mask=mask, other=0.0)
        a_xx = tl.load(a_ptr + index, mask=mask, other=0.0) + w00
        a_yy = tl.load(a_ptr + n_points + index, mask=mask, other=0.0) + w00
        a_xy = tl.load(a_ptr + 2 * n_points + index, mask=mask, other=0.0)
        residual_x = rhs_x - w01 * (ux_ym + ux_yp) - w10 * (ux_m + ux_p)
        residual_y = rhs_y - w01 * (uy_ym + uy_yp) - w10 * (uy_m + uy_p)
        determinant = a_xx * a_yy * 1.0000000001 - a_xy * a_xy
        new_x = (a_yy * residual_x - a_xy * residual_y) / determinant
        new_y = (-a_xy * residual_x + a_xx * residual_y) / determinant
        tl.store(output_ptr + index, tl.where(active, new_x, ux), mask=mask)
        tl.store(output_ptr + n_points + index, tl.where(active, new_y, uy), mask=mask)

    @triton.jit
    def _relax_membrane_color_inplace_kernel(
        solution_ptr,
        a_ptr,
        b_ptr,
        weights_ptr,
        n_points,
        nx,
        ny,
        color,
        offsets: tl.constexpr,
    ):
        """只遍历一个红黑颜色并原地更新其半数网格点。"""

        color_points = nx // 2
        local = tl.program_id(0) * offsets + tl.arange(0, offsets)
        mask = local < (n_points // 2)
        row = local // color_points
        pair = local - row * color_points
        # nx 为偶数时，每行每种颜色恰有 nx/2 个点；按行展开后，
        # x 的奇偶由当前行和目标颜色共同决定，得到与 checker 顺序一致的点集。
        x = 2 * pair + ((row + color) & 1)
        index = row * nx + x
        xm = row * nx + tl.where(x == 0, nx - 1, x - 1)
        xp = row * nx + tl.where(x == nx - 1, 0, x + 1)
        ym = tl.where(row == 0, index, (row - 1) * nx + x)
        yp = tl.where(row == ny - 1, (ny - 1) * nx + x, (row + 1) * nx + x)

        w00 = tl.load(weights_ptr + 0)
        w01 = tl.load(weights_ptr + 1)
        w10 = tl.load(weights_ptr + 2)
        ux = tl.load(solution_ptr + index, mask=mask, other=0.0)
        uy = tl.load(solution_ptr + n_points + index, mask=mask, other=0.0)
        ux_m = tl.load(solution_ptr + xm, mask=mask, other=0.0)
        ux_p = tl.load(solution_ptr + xp, mask=mask, other=0.0)
        ux_ym = tl.load(solution_ptr + ym, mask=mask, other=0.0)
        ux_yp = tl.load(solution_ptr + yp, mask=mask, other=0.0)
        uy_m = tl.load(solution_ptr + n_points + xm, mask=mask, other=0.0)
        uy_p = tl.load(solution_ptr + n_points + xp, mask=mask, other=0.0)
        uy_ym = tl.load(solution_ptr + n_points + ym, mask=mask, other=0.0)
        uy_yp = tl.load(solution_ptr + n_points + yp, mask=mask, other=0.0)

        rhs_x = tl.load(b_ptr + index, mask=mask, other=0.0)
        rhs_y = tl.load(b_ptr + n_points + index, mask=mask, other=0.0)
        a_xx = tl.load(a_ptr + index, mask=mask, other=0.0) + w00
        a_yy = tl.load(a_ptr + n_points + index, mask=mask, other=0.0) + w00
        a_xy = tl.load(a_ptr + 2 * n_points + index, mask=mask, other=0.0)
        residual_x = rhs_x - w01 * (ux_ym + ux_yp) - w10 * (ux_m + ux_p)
        residual_y = rhs_y - w01 * (uy_ym + uy_yp) - w10 * (uy_m + uy_p)
        determinant = a_xx * a_yy * 1.0000000001 - a_xy * a_xy
        new_x = (a_yy * residual_x - a_xy * residual_y) / determinant
        new_y = (-a_xy * residual_x + a_xx * residual_y) / determinant
        tl.store(solution_ptr + index, new_x, mask=mask)
        tl.store(solution_ptr + n_points + index, new_y, mask=mask)

    @triton.jit
    def _squaring_update_kernel(
        b_input_ptr,
        a_input_ptr,
        b_output_ptr,
        a_output_ptr,
        increment_ptr,
        map_ptr,
        jacobian_ptr,
        n_points,
        nx,
        ny,
        coordinate_fp32: tl.constexpr,
        offsets: tl.constexpr,
    ):
        """融合一次 DARTEL squaring 的五通道采样和增量更新。"""

        index = tl.program_id(0) * offsets + tl.arange(0, offsets)
        mask = index < n_points
        if coordinate_fp32:
            x = tl.load(map_ptr + index, mask=mask, other=0.0).to(tl.float32) - 1.0
            y = tl.load(
                map_ptr + n_points + index, mask=mask, other=0.0
            ).to(tl.float32) - 1.0
        else:
            x = tl.load(map_ptr + index, mask=mask, other=0.0).to(tl.float64) - 1.0
            y = tl.load(
                map_ptr + n_points + index, mask=mask, other=0.0
            ).to(tl.float64) - 1.0
        ix = tl.floor(x).to(tl.int32)
        iy = tl.floor(y).to(tl.int32)
        dx1 = x - ix.to(x.dtype)
        dy1 = y - iy.to(y.dtype)
        dx2 = 1.0 - dx1
        dy2 = 1.0 - dy1

        # 用 floor 形式实现非负模；Triton 的整数 `%` 对负数的语义不适合
        # CAT 的周期边界，边界点必须与 Python ``torch.remainder`` 一致。
        if coordinate_fp32:
            x_period = tl.floor(ix.to(tl.float32) / nx).to(tl.int32)
            y_period = tl.floor(iy.to(tl.float32) / (2 * ny)).to(tl.int32)
            y_next_period = tl.floor(
                (iy + 1).to(tl.float32) / (2 * ny)
            ).to(tl.int32)
        else:
            x_period = tl.floor(ix.to(tl.float64) / nx).to(tl.int32)
            y_period = tl.floor(iy.to(tl.float64) / (2 * ny)).to(tl.int32)
            y_next_period = tl.floor(
                (iy + 1).to(tl.float64) / (2 * ny)
            ).to(tl.int32)
        x22 = ix - x_period * nx
        x12 = tl.where(x22 == nx - 1, 0, x22 + 1)
        x21 = x22
        x11 = x12
        period_y = 2 * ny
        y22_raw = iy - y_period * period_y
        y12_raw = y22_raw
        y21_raw = (iy + 1) - y_next_period * period_y
        y11_raw = y21_raw
        y22 = tl.where(y22_raw >= ny, period_y - y22_raw - 1, y22_raw)
        y12 = y22
        y21 = tl.where(y21_raw >= ny, period_y - y21_raw - 1, y21_raw)
        y11 = y21

        k22_bx = tl.load(b_input_ptr + y22 * nx + x22, mask=mask, other=0.0)
        k12_bx = tl.load(b_input_ptr + y12 * nx + x12, mask=mask, other=0.0)
        k21_bx = tl.load(b_input_ptr + y21 * nx + x21, mask=mask, other=0.0)
        k11_bx = tl.load(b_input_ptr + y11 * nx + x11, mask=mask, other=0.0)
        k22_by = tl.load(
            b_input_ptr + n_points + y22 * nx + x22, mask=mask, other=0.0
        )
        k12_by = tl.load(
            b_input_ptr + n_points + y12 * nx + x12, mask=mask, other=0.0
        )
        k21_by = tl.load(
            b_input_ptr + n_points + y21 * nx + x21, mask=mask, other=0.0
        )
        k11_by = tl.load(
            b_input_ptr + n_points + y11 * nx + x11, mask=mask, other=0.0
        )

        k22_a00 = tl.load(a_input_ptr + y22 * nx + x22, mask=mask, other=0.0)
        k12_a00 = tl.load(a_input_ptr + y12 * nx + x12, mask=mask, other=0.0)
        k21_a00 = tl.load(a_input_ptr + y21 * nx + x21, mask=mask, other=0.0)
        k11_a00 = tl.load(a_input_ptr + y11 * nx + x11, mask=mask, other=0.0)
        k22_a11 = tl.load(
            a_input_ptr + n_points + y22 * nx + x22, mask=mask, other=0.0
        )
        k12_a11 = tl.load(
            a_input_ptr + n_points + y12 * nx + x12, mask=mask, other=0.0
        )
        k21_a11 = tl.load(
            a_input_ptr + n_points + y21 * nx + x21, mask=mask, other=0.0
        )
        k11_a11 = tl.load(
            a_input_ptr + n_points + y11 * nx + x11, mask=mask, other=0.0
        )
        k22_a01 = tl.load(
            a_input_ptr + 2 * n_points + y22 * nx + x22, mask=mask, other=0.0
        )
        k12_a01 = tl.load(
            a_input_ptr + 2 * n_points + y12 * nx + x12, mask=mask, other=0.0
        )
        k21_a01 = tl.load(
            a_input_ptr + 2 * n_points + y21 * nx + x21, mask=mask, other=0.0
        )
        k11_a01 = tl.load(
            a_input_ptr + 2 * n_points + y11 * nx + x11, mask=mask, other=0.0
        )

        sampled_bx = (k22_bx * dx2 + k12_bx * dx1) * dy2 + (
            k21_bx * dx2 + k11_bx * dx1
        ) * dy1
        sampled_by = (k22_by * dx2 + k12_by * dx1) * dy2 + (
            k21_by * dx2 + k11_by * dx1
        ) * dy1
        sampled_a00 = (k22_a00 * dx2 + k12_a00 * dx1) * dy2 + (
            k21_a00 * dx2 + k11_a00 * dx1
        ) * dy1
        sampled_a11 = (k22_a11 * dx2 + k12_a11 * dx1) * dy2 + (
            k21_a11 * dx2 + k11_a11 * dx1
        ) * dy1
        sampled_a01 = (k22_a01 * dx2 + k12_a01 * dx1) * dy2 + (
            k21_a01 * dx2 + k11_a01 * dx1
        ) * dy1

        j00 = tl.load(jacobian_ptr + index, mask=mask, other=0.0)
        j10 = tl.load(jacobian_ptr + n_points + index, mask=mask, other=0.0)
        j01 = tl.load(jacobian_ptr + 2 * n_points + index, mask=mask, other=0.0)
        j11 = tl.load(jacobian_ptr + 3 * n_points + index, mask=mask, other=0.0)
        determinant = j00 * j11 - j01 * j10

        b_increment_x = determinant * (sampled_bx * j00 + sampled_by * j10)
        b_increment_y = determinant * (sampled_bx * j01 + sampled_by * j11)
        tmp1 = sampled_a00 * j00 + sampled_a01 * j10
        tmp2 = sampled_a01 * j00 + sampled_a11 * j10
        tmp3 = sampled_a00 * j01 + sampled_a01 * j11
        tmp4 = sampled_a01 * j01 + sampled_a11 * j11
        a_increment_00 = determinant * (tmp1 * j00 + tmp2 * j10)
        a_increment_11 = determinant * (tmp3 * j01 + tmp4 * j11)
        a_increment_01 = determinant * (tmp1 * j01 + tmp2 * j11)

        old_bx = tl.load(b_input_ptr + index, mask=mask, other=0.0)
        old_by = tl.load(b_input_ptr + n_points + index, mask=mask, other=0.0)
        old_a00 = tl.load(a_input_ptr + index, mask=mask, other=0.0)
        old_a11 = tl.load(a_input_ptr + n_points + index, mask=mask, other=0.0)
        old_a01 = tl.load(a_input_ptr + 2 * n_points + index, mask=mask, other=0.0)
        tl.store(b_output_ptr + index, old_bx + b_increment_x, mask=mask)
        tl.store(b_output_ptr + n_points + index, old_by + b_increment_y, mask=mask)
        tl.store(a_output_ptr + index, old_a00 + a_increment_00, mask=mask)
        tl.store(a_output_ptr + n_points + index, old_a11 + a_increment_11, mask=mask)
        tl.store(a_output_ptr + 2 * n_points + index, old_a01 + a_increment_01, mask=mask)
        tl.store(increment_ptr + index, b_increment_x, mask=mask)
        tl.store(increment_ptr + n_points + index, b_increment_y, mask=mask)


def resize_field_triton(
    value: torch.Tensor,
    target_nx: int,
    target_ny: int,
) -> torch.Tensor:
    """在 CUDA 上执行 CAT 三点二维重采样的融合实现。"""

    if not available():
        raise RuntimeError("当前 Python 环境没有可用的 Triton")
    if value.device.type != "cuda":
        raise ValueError("Triton resize 只接受 CUDA 张量")
    if value.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Triton resize 当前只接受 float32 或 float64")
    if value.ndim != 3:
        raise ValueError("Triton resize 需要 [channel, ny, nx] 张量")
    if target_nx < 1 or target_ny < 1:
        raise ValueError("Triton resize 目标尺寸必须为正数")
    value = value.contiguous()
    channels = int(value.shape[0])
    source_ny = int(value.shape[1])
    source_nx = int(value.shape[2])
    source_points = source_nx * source_ny
    target_points = int(target_nx) * int(target_ny)
    total = channels * target_points
    output = torch.empty(
        (channels, target_ny, target_nx),
        device=value.device,
        dtype=value.dtype,
    )
    block = 256
    grid = (triton.cdiv(total, block),)
    _resize_field_kernel[grid](
        value,
        output,
        source_nx,
        source_ny,
        int(target_nx),
        int(target_ny),
        source_points,
        target_points,
        total,
        coordinate_fp32=value.dtype == torch.float32,
        offsets=block,
        num_warps=4,
    )
    return output


def apply_membrane_system_triton(
    hessian: torch.Tensor,
    field: torch.Tensor,
    w00: float,
    w01: float,
    w10: float,
) -> torch.Tensor:
    """在 CUDA 上融合计算膜系统的 Hessian 加正则项。"""

    if not available():
        raise RuntimeError("当前 Python 环境没有可用的 Triton")
    if field.device.type != "cuda":
        raise ValueError("Triton 膜系统只接受 CUDA 张量")
    if field.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Triton 膜系统当前只接受 float32 或 float64")
    if field.ndim != 3 or tuple(field.shape[:1]) != (2,):
        raise ValueError("field 形状必须是 [2, ny, nx]")
    if tuple(hessian.shape) != (3, field.shape[1], field.shape[2]):
        raise ValueError("hessian 与 field 形状不匹配")
    field = field.contiguous()
    hessian = hessian.contiguous()
    nx = int(field.shape[2])
    ny = int(field.shape[1])
    n_points = nx * ny
    output = torch.empty_like(field)
    weights = _cached_weights((w00, w01, w10), field.device, field.dtype)
    block = 256
    grid = (triton.cdiv(n_points, block),)
    _apply_membrane_system_kernel[grid](
        field,
        hessian,
        output,
        weights,
        n_points,
        nx,
        ny,
        offsets=block,
        num_warps=4,
    )
    return output


def relax_membrane_triton(
    hessian: torch.Tensor,
    right_hand: torch.Tensor,
    solution: torch.Tensor,
    w00: float,
    w01: float,
    w10: float,
    nit: int,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """在 CUDA 偶数宽度网格上执行融合 Triton 膜松弛。

    ``inplace=True`` 只发射当前红黑颜色的半数点并原地更新；
    ``inplace=False`` 保留旧的全网格双缓冲实现，供数值 A/B 使用。
    """

    if not available():
        raise RuntimeError("当前 Python 环境没有可用的 Triton")
    if solution.device.type != "cuda":
        raise ValueError("Triton 膜松弛只接受 CUDA 张量")
    if solution.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Triton 膜松弛当前只接受 float32 或 float64")
    if solution.ndim != 3 or tuple(solution.shape[:1]) != (2,):
        raise ValueError("solution 形状必须是 [2, ny, nx]")
    if tuple(hessian.shape) != (3, solution.shape[1], solution.shape[2]):
        raise ValueError("hessian 与 solution 形状不匹配")
    if tuple(right_hand.shape) != tuple(solution.shape):
        raise ValueError("right_hand 与 solution 形状不匹配")
    nx = int(solution.shape[2])
    ny = int(solution.shape[1])
    if nx % 2 != 0 or nx < 2:
        raise ValueError("Triton 红黑膜松弛要求 nx 为大于1的偶数")
    if nit < 0:
        raise ValueError(f"松弛次数不能为负数，得到 {nit}")
    block = 256
    grid = (triton.cdiv(nx * ny, block),)
    weights = _cached_weights((w00, w01, w10), solution.device, solution.dtype)
    if inplace:
        color_grid = (triton.cdiv((nx * ny) // 2, block),)
        for iteration in range(2 * nit):
            _relax_membrane_color_inplace_kernel[color_grid](
                solution,
                hessian,
                right_hand,
                weights,
                nx * ny,
                nx,
                ny,
                1 if iteration % 2 == 0 else 0,
                offsets=block,
                num_warps=8,
            )
        return solution
    # 红黑更新虽然只读相反颜色，但原地写入会让不同 Triton program
    # 之间出现未定义的读写次序；用双缓冲明确表达一次颜色更新的快照。
    source = solution
    destination = torch.empty_like(solution)
    for iteration in range(2 * nit):
        _relax_membrane_kernel[grid](
            source,
            destination,
            hessian,
            right_hand,
            weights,
            nx * ny,
            nx,
            ny,
            1 if iteration % 2 == 0 else 0,
            offsets=block,
            num_warps=4,
        )
        source, destination = destination, source
    if source.data_ptr() != solution.data_ptr():
        solution.copy_(source)
    return solution


def squaring_update_triton(
    b_value: torch.Tensor,
    a_value: torch.Tensor,
    current_map: torch.Tensor,
    current_jacobian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 CUDA 上融合一次 squaring 的五通道采样和原位增量。"""

    if not available():
        raise RuntimeError("当前 Python 环境没有可用的 Triton")
    if b_value.device.type != "cuda":
        raise ValueError("Triton squaring 只接受 CUDA 张量")
    if b_value.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Triton squaring 当前只接受 float32 或 float64")
    if b_value.ndim != 3 or tuple(b_value.shape[:1]) != (2,):
        raise ValueError("b_value 形状必须是 [2, ny, nx]")
    if tuple(a_value.shape) != (3, b_value.shape[1], b_value.shape[2]):
        raise ValueError("a_value 与 b_value 形状不匹配")
    if tuple(current_map.shape) != tuple(b_value.shape):
        raise ValueError("current_map 与 b_value 形状不匹配")
    if tuple(current_jacobian.shape) != (4, b_value.shape[1], b_value.shape[2]):
        raise ValueError("current_jacobian 与 b_value 形状不匹配")
    b_value = b_value.contiguous()
    a_value = a_value.contiguous()
    current_map = current_map.contiguous()
    current_jacobian = current_jacobian.contiguous()
    increment = torch.empty_like(b_value)
    b_input = b_value.clone()
    a_input = a_value.clone()
    nx = int(b_value.shape[2])
    ny = int(b_value.shape[1])
    n_points = nx * ny
    block = 256
    grid = (triton.cdiv(n_points, block),)
    _squaring_update_kernel[grid](
        b_input,
        a_input,
        b_value,
        a_value,
        increment,
        current_map,
        current_jacobian,
        n_points,
        nx,
        ny,
        coordinate_fp32=b_value.dtype == torch.float32,
        offsets=block,
        num_warps=4,
    )
    return b_value, a_value, increment
