"""CAT-Surface DARTEL 规则网格算子的 Torch 实现。

内部张量采用 ``[component, y, x]`` 布局，最后一维对应 C 实现中连续的 x
索引。坐标图沿用 CAT-Surface 的一基坐标约定，位移场则以零为单位图上的
位移。所有算子都接收显式设备；``auto`` 只是明确请求自动选择，不会在
显式请求 CUDA 失败时静默切换到 CPU。
"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from . import dartel_triton


_GRID_PERIOD_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _grid_periods(
    spec: GridSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """返回指定网格上可复用的周期长度张量。"""

    key = (device.type, device.index, dtype, spec.nx, spec.ny)
    cached = _GRID_PERIOD_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(
            (spec.nx, spec.ny), dtype=dtype, device=device
        ).view(2, 1, 1)
        _GRID_PERIOD_CACHE[key] = cached
    return cached


class DeviceUnavailable(RuntimeError):
    """表示请求的 Torch 设备当前不可用。"""


@dataclass(frozen=True)
class GridSpec:
    """描述 DARTEL 二维网格的 x、y 尺寸。"""

    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"网格尺寸必须为正数，得到 {(self.nx, self.ny)}")

    @property
    def points(self) -> int:
        """返回网格点数。"""

        return self.nx * self.ny

    @classmethod
    def from_shape(cls, shape: Sequence[int]) -> "GridSpec":
        """从 ``[component, y, x]`` 形状创建网格描述。"""

        if len(shape) != 3 or shape[0] != 2:
            raise ValueError(f"网格形状必须是 [2, ny, nx]，得到 {tuple(shape)}")
        return cls(nx=int(shape[2]), ny=int(shape[1]))


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """解析显式 Torch 设备并检查 CUDA 是否真正可用。"""

    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise DeviceUnavailable(f"DARTEL 网格后端只支持 CPU/CUDA，得到 {resolved}")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailable("请求 CUDA，但当前 NVIDIA 驱动或设备不可用")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise DeviceUnavailable(
                f"请求 CUDA 设备 {resolved.index}，可见设备数为 {torch.cuda.device_count()}"
            )
        if resolved.index is None:
            # 统一成带显式索引的设备，避免 ``cuda`` 与 ``cuda:0`` 比较时
            # 被 Torch 视为不同对象，进而误判已上传的 stencil 所在设备。
            resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _as_grid(
    value: torch.Tensor,
    spec: GridSpec,
    device: torch.device | None,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """把输入转换为连续的二维向量场张量。"""

    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    expected = (2, spec.ny, spec.nx)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} 形状必须是 {expected}，得到 {tuple(tensor.shape)}")
    return tensor.contiguous()


def _as_jacobian(
    value: torch.Tensor,
    spec: GridSpec,
    device: torch.device | None,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """把输入转换为连续的四分量 Jacobian 张量。"""

    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    expected = (4, spec.ny, spec.nx)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} 形状必须是 {expected}，得到 {tuple(tensor.shape)}")
    return tensor.contiguous()


def _as_field(
    value: torch.Tensor,
    spec: GridSpec,
    device: torch.device | None,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """把标量或多通道场转换为连续的 ``[channel, y, x]`` 张量。"""

    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    expected = (tensor.shape[0], spec.ny, spec.nx)
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected[1:]:
        raise ValueError(
            f"{name} 形状必须是 [channel, {spec.ny}, {spec.nx}]，得到 {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def from_c_layout(
    value: torch.Tensor,
    nx: int,
    ny: int,
    components: int = 2,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """把 CAT C 的 ``[component * nx * ny]`` 布局转换为 Torch 布局。"""

    spec = GridSpec(nx=nx, ny=ny)
    target_device = resolve_device(device)
    tensor = torch.as_tensor(value, dtype=dtype, device=target_device)
    expected = components * spec.points
    if tensor.numel() != expected:
        raise ValueError(f"C 布局元素数应为 {expected}，得到 {tensor.numel()}")
    return tensor.reshape(components, spec.ny, spec.nx).contiguous()


def to_c_layout(value: torch.Tensor) -> torch.Tensor:
    """把 ``[component, y, x]`` 张量按 CAT C 的连续布局展平。"""

    if value.ndim != 3:
        raise ValueError(f"输入必须是三维张量，得到 {tuple(value.shape)}")
    return value.contiguous().reshape(-1)


def make_identity_map(
    spec: GridSpec,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """构造 CAT-Surface 使用的一基坐标 identity map。"""

    target_device = resolve_device(device)
    x = torch.arange(1, spec.nx + 1, device=target_device, dtype=dtype)
    y = torch.arange(1, spec.ny + 1, device=target_device, dtype=dtype)
    x = x.view(1, 1, spec.nx).expand(1, spec.ny, spec.nx)
    y = y.view(1, spec.ny, 1).expand(1, spec.ny, spec.nx)
    return torch.cat((x, y), dim=0).contiguous()


def _bound_indices(
    ix: torch.Tensor,
    iy: torch.Tensor,
    spec: GridSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """实现 CAT 的 x 周期边界和 y 镜像 Neumann 边界。"""

    x_index = torch.remainder(ix, spec.nx)
    if spec.ny == 1:
        y_index = torch.zeros_like(iy)
    else:
        period = 2 * spec.ny
        reflected = torch.remainder(iy, period)
        y_index = torch.where(
            reflected >= spec.ny,
            period - reflected - 1,
            reflected,
        )
    return x_index, y_index


def _shift(field: torch.Tensor, dx: int, dy: int, spec: GridSpec) -> torch.Tensor:
    """读取混合边界条件下的整数邻域。"""

    shifted_x = torch.roll(field, shifts=-dx, dims=-1)
    y = torch.arange(spec.ny, device=field.device, dtype=torch.long) + dy
    if spec.ny == 1:
        y_index = torch.zeros_like(y)
    else:
        period = 2 * spec.ny
        reflected = torch.remainder(y, period)
        y_index = torch.where(
            reflected >= spec.ny,
            period - reflected - 1,
            reflected,
        )
    return shifted_x.index_select(-2, y_index)


def _wt2(distance: torch.Tensor) -> torch.Tensor:
    """计算 CAT FMG 使用的三点二次插值权重。"""

    absolute = distance.abs()
    inner = 0.75 - absolute * absolute
    outer_distance = 1.5 - absolute
    outer = 0.5 * outer_distance * outer_distance
    return torch.where(absolute < 0.5, inner, torch.where(absolute < 1.5, outer, 0.0))


def resize_field(
    field: torch.Tensor,
    target: GridSpec,
    *,
    kernel: str = "auto",
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """按 CAT 的 separable 三点权重在两个网格间重采样。"""

    target_device = resolve_device(device)
    value = torch.as_tensor(field, dtype=dtype, device=target_device)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"resize_field 需要 [channel, y, x]，得到 {tuple(value.shape)}")
    source = GridSpec(nx=int(value.shape[2]), ny=int(value.shape[1]))
    value = value.contiguous()
    if kernel not in {"auto", "torch", "triton"}:
        raise ValueError(f"kernel 必须是 auto、torch 或 triton，得到 {kernel}")
    if kernel == "triton" and (
        target_device.type != "cuda" or not dartel_triton.available()
    ):
        raise ValueError("请求 Triton resize，但当前设备或 Python 环境不可用")
    if (
        target_device.type == "cuda"
        and kernel in {"auto", "triton"}
        and dartel_triton.available()
    ):
        return dartel_triton.resize_field_triton(value, target.nx, target.ny)

    y_target = torch.arange(target.ny, device=target_device, dtype=dtype)
    y_location = (y_target + 0.5) * source.ny / target.ny - 0.5
    y_origin = torch.floor(y_location + 0.5).to(torch.long)
    y_weight = _wt2(y_origin.to(dtype) - y_location)
    y_weight_minus = _wt2(y_origin.to(dtype) - 1.0 - y_location)
    y_weight_plus = _wt2(y_origin.to(dtype) + 1.0 - y_location)
    x_zero = torch.zeros_like(y_origin)
    _, y_minus = _bound_indices(x_zero, y_origin - 1, source)
    _, y_center = _bound_indices(x_zero, y_origin, source)
    _, y_plus = _bound_indices(x_zero, y_origin + 1, source)
    intermediate = (
        y_weight_minus.view(1, target.ny, 1) * value[:, y_minus, :]
        + y_weight.view(1, target.ny, 1) * value[:, y_center, :]
        + y_weight_plus.view(1, target.ny, 1) * value[:, y_plus, :]
    )

    x_target = torch.arange(target.nx, device=target_device, dtype=dtype)
    x_location = (x_target + 0.5) * source.nx / target.nx - 0.5
    x_origin = torch.floor(x_location + 0.5).to(torch.long)
    x_weight = _wt2(x_origin.to(dtype) - x_location)
    x_weight_minus = _wt2(x_origin.to(dtype) - 1.0 - x_location)
    x_weight_plus = _wt2(x_origin.to(dtype) + 1.0 - x_location)
    y_zero = torch.zeros_like(x_origin)
    x_minus, _ = _bound_indices(x_origin - 1, y_zero, source)
    x_center, _ = _bound_indices(x_origin, y_zero, source)
    x_plus, _ = _bound_indices(x_origin + 1, y_zero, source)
    resized = (
        x_weight_minus.view(1, 1, target.nx) * intermediate[:, :, x_minus]
        + x_weight.view(1, 1, target.nx) * intermediate[:, :, x_center]
        + x_weight_plus.view(1, 1, target.nx) * intermediate[:, :, x_plus]
    )
    return resized.contiguous()


def _corners(
    field: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: GridSpec,
) -> tuple[torch.Tensor, ...]:
    """读取一个标量场在双线性插值中的四个边界点和权重。"""

    ix = torch.floor(x).to(torch.long)
    iy = torch.floor(y).to(torch.long)
    dx1 = x - ix
    dy1 = y - iy
    dx2 = 1.0 - dx1
    dy2 = 1.0 - dy1

    x22, y22 = _bound_indices(ix, iy, spec)
    x12, y12 = _bound_indices(ix + 1, iy, spec)
    x21, y21 = _bound_indices(ix, iy + 1, spec)
    x11, y11 = _bound_indices(ix + 1, iy + 1, spec)
    return (
        field[y22, x22],
        field[y12, x12],
        field[y21, x21],
        field[y11, x11],
        dx1,
        dx2,
        dy1,
        dy2,
    )


def _bilinear(corners: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """按照 C 代码的顺序完成四点双线性插值。"""

    k22, k12, k21, k11, dx1, dx2, dy1, dy2 = corners
    return (k22 * dx2 + k12 * dx1) * dy2 + (k21 * dx2 + k11 * dx1) * dy1


def _interpolation_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    spec: GridSpec,
) -> tuple[torch.Tensor, ...]:
    """一次生成双线性插值所需的四角索引和权重。"""

    ix = torch.floor(x).to(torch.long)
    iy = torch.floor(y).to(torch.long)
    dx1 = x - ix
    dy1 = y - iy
    dx2 = 1.0 - dx1
    dy2 = 1.0 - dy1
    x22, y22 = _bound_indices(ix, iy, spec)
    x12, y12 = _bound_indices(ix + 1, iy, spec)
    x21, y21 = _bound_indices(ix, iy + 1, spec)
    x11, y11 = _bound_indices(ix + 1, iy + 1, spec)
    return x22, y22, x12, y12, x21, y21, x11, y11, dx1, dx2, dy1, dy2


def _interpolate_channels(
    field: torch.Tensor,
    indices: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """用同一组索引同时插值多个通道，保持 CAT 双线性运算顺序。"""

    (
        x22,
        y22,
        x12,
        y12,
        x21,
        y21,
        x11,
        y11,
        dx1,
        dx2,
        dy1,
        dy2,
    ) = indices
    k22 = field[:, y22, x22]
    k12 = field[:, y12, x12]
    k21 = field[:, y21, x21]
    k11 = field[:, y11, x11]
    return (k22 * dx2 + k12 * dx1) * dy2 + (k21 * dx2 + k11 * dx1) * dy1


def _sample_channels(
    field: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: GridSpec,
) -> torch.Tensor:
    """用一套边界索引同时采样多个普通标量通道。"""

    return _interpolate_channels(
        field,
        _interpolation_indices(x, y, spec),
    )


def _composition_optimized(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """共享边界索引实现坐标图组合的设备常驻路径。"""

    spec = GridSpec.from_shape(a.shape)
    a_tensor = _as_grid(a, spec, device, dtype, "A")
    b_tensor = _as_grid(b, spec, device, dtype, "B")
    x = a_tensor[0] - 1.0
    y = a_tensor[1] - 1.0
    indices = _interpolation_indices(x, y, spec)
    periods = _grid_periods(spec, device, dtype)
    # 重用角点值完成周期坐标的跨周期修正；每个通道仍使用原 CAT 公式。
    x22, y22, x12, y12, x21, y21, x11, y11 = indices[:8]
    coordinate_field = b_tensor - 1.0
    corners_22 = coordinate_field[:, y22, x22]
    corners_12 = coordinate_field[:, y12, x12]
    corners_21 = coordinate_field[:, y21, x21]
    corners_11 = coordinate_field[:, y11, x11]
    corners_12 = corners_12 - torch.floor(
        (corners_12 - corners_22) / periods + 0.5
    ) * periods
    corners_21 = corners_21 - torch.floor(
        (corners_21 - corners_22) / periods + 0.5
    ) * periods
    corners_11 = corners_11 - torch.floor(
        (corners_11 - corners_22) / periods + 0.5
    ) * periods
    _, _, _, _, _, _, _, _, dx1, dx2, dy1, dy2 = indices
    coordinates = (
        (corners_22 * dx2 + corners_12 * dx1) * dy2
        + (corners_21 * dx2 + corners_11 * dx1) * dy1
        + 1.0
    )
    return coordinates.contiguous()


def _composition_jacobian_optimized(
    a: torch.Tensor,
    ja: torch.Tensor,
    b: torch.Tensor,
    jb: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """共享索引一次采样坐标和 Jacobian 的设备常驻路径。"""

    spec = GridSpec.from_shape(a.shape)
    a_tensor = _as_grid(a, spec, device, dtype, "A")
    b_tensor = _as_grid(b, spec, device, dtype, "B")
    ja_tensor = _as_jacobian(ja, spec, device, dtype, "JA")
    jb_tensor = _as_jacobian(jb, spec, device, dtype, "JB")
    x = a_tensor[0] - 1.0
    y = a_tensor[1] - 1.0
    indices = _interpolation_indices(x, y, spec)
    # Jacobian 四个分量共享一次边界索引与一次四角 gather。
    sampled_jacobian = _interpolate_channels(jb_tensor, indices)
    periods = _grid_periods(spec, device, dtype)
    # 坐标的四角值必须按分量分别修正，不能在插值结果上再做周期校正。
    x22, y22, x12, y12, x21, y21, x11, y11 = indices[:8]
    coordinate_field = b_tensor - 1.0
    coordinate_22 = coordinate_field[:, y22, x22]
    coordinate_12 = coordinate_field[:, y12, x12]
    coordinate_21 = coordinate_field[:, y21, x21]
    coordinate_11 = coordinate_field[:, y11, x11]
    coordinate_periods = periods
    coordinate_12 = coordinate_12 - torch.floor(
        (coordinate_12 - coordinate_22) / coordinate_periods + 0.5
    ) * coordinate_periods
    coordinate_21 = coordinate_21 - torch.floor(
        (coordinate_21 - coordinate_22) / coordinate_periods + 0.5
    ) * coordinate_periods
    coordinate_11 = coordinate_11 - torch.floor(
        (coordinate_11 - coordinate_22) / coordinate_periods + 0.5
    ) * coordinate_periods
    _, _, _, _, _, _, _, _, dx1, dx2, dy1, dy2 = indices
    coordinates = (
        (coordinate_22 * dx2 + coordinate_12 * dx1) * dy2
        + (coordinate_21 * dx2 + coordinate_11 * dx1) * dy1
        + 1.0
    ).contiguous()

    jb00, jb01, jb10, jb11 = sampled_jacobian
    jacobian = torch.stack(
        (
            jb00 * ja_tensor[0] + jb10 * ja_tensor[1],
            jb01 * ja_tensor[0] + jb11 * ja_tensor[1],
            jb00 * ja_tensor[2] + jb10 * ja_tensor[3],
            jb01 * ja_tensor[2] + jb11 * ja_tensor[3],
        ),
        dim=0,
    ).contiguous()
    return coordinates, jacobian


def sample_field(
    field: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """按 CAT 的混合边界条件对标量场做双线性采样。"""

    target_device = resolve_device(device)
    field_tensor = torch.as_tensor(field, dtype=dtype, device=target_device)
    if field_tensor.ndim != 2:
        raise ValueError(f"sample_field 只接受二维标量场，得到 {tuple(field_tensor.shape)}")
    spec = GridSpec(nx=int(field_tensor.shape[1]), ny=int(field_tensor.shape[0]))
    x_tensor = torch.as_tensor(x, dtype=dtype, device=target_device)
    y_tensor = torch.as_tensor(y, dtype=dtype, device=target_device)
    if x_tensor.shape != y_tensor.shape:
        raise ValueError("sample_field 的 x/y 坐标形状必须一致")
    return _bilinear(_corners(field_tensor, x_tensor, y_tensor, spec))


def _wrapped_coordinate(
    field: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: GridSpec,
    period: int,
) -> torch.Tensor:
    """插值周期坐标，并以左下角值为参照消除跨周期跳变。"""

    corners = _corners(field - 1.0, x, y, spec)
    k22, k12, k21, k11, dx1, dx2, dy1, dy2 = corners
    k12 = k12 - torch.floor((k12 - k22) / period + 0.5) * period
    k21 = k21 - torch.floor((k21 - k22) / period + 0.5) * period
    k11 = k11 - torch.floor((k11 - k22) / period + 0.5) * period
    return _bilinear((k22, k12, k21, k11, dx1, dx2, dy1, dy2)) + 1.0


def composition(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
) -> torch.Tensor:
    """计算 CAT DARTEL 的坐标图组合 ``C = B(A)``。

    ``optimized=False`` 保留逐通道、逐次生成插值索引的 reference 路径。
    """

    target_device = resolve_device(device)
    if optimized:
        return _composition_optimized(
            a,
            b,
            device=target_device,
            dtype=dtype,
        )
    spec = GridSpec.from_shape(a.shape)
    a_tensor = _as_grid(a, spec, target_device, dtype, "A")
    b_tensor = _as_grid(b, spec, target_device, dtype, "B")
    x = a_tensor[0] - 1.0
    y = a_tensor[1] - 1.0
    cx = _wrapped_coordinate(b_tensor[0], x, y, spec, spec.nx)
    cy = _wrapped_coordinate(b_tensor[1], x, y, spec, spec.ny)
    return torch.stack((cx, cy), dim=0).contiguous()


def composition_jacobian(
    a: torch.Tensor,
    ja: torch.Tensor,
    b: torch.Tensor,
    jb: torch.Tensor,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算坐标图组合及其链式 Jacobian。

    ``optimized=False`` 保留逐分量采样的 reference 路径。
    """

    target_device = resolve_device(device)
    if optimized:
        return _composition_jacobian_optimized(
            a,
            ja,
            b,
            jb,
            device=target_device,
            dtype=dtype,
        )
    spec = GridSpec.from_shape(a.shape)
    a_tensor = _as_grid(a, spec, target_device, dtype, "A")
    b_tensor = _as_grid(b, spec, target_device, dtype, "B")
    ja_tensor = _as_jacobian(ja, spec, target_device, dtype, "JA")
    jb_tensor = _as_jacobian(jb, spec, target_device, dtype, "JB")
    x = a_tensor[0] - 1.0
    y = a_tensor[1] - 1.0

    cx = _wrapped_coordinate(b_tensor[0], x, y, spec, spec.nx)
    cy = _wrapped_coordinate(b_tensor[1], x, y, spec, spec.ny)
    sampled = [_corners(jb_tensor[index], x, y, spec) for index in range(4)]
    interpolated = [_bilinear(corners) for corners in sampled]
    jb00, jb01, jb10, jb11 = interpolated

    jc00 = jb00 * ja_tensor[0] + jb10 * ja_tensor[1]
    jc01 = jb01 * ja_tensor[0] + jb11 * ja_tensor[1]
    jc10 = jb00 * ja_tensor[2] + jb10 * ja_tensor[3]
    jc11 = jb01 * ja_tensor[2] + jb11 * ja_tensor[3]
    jacobian = torch.stack((jc00, jc01, jc10, jc11), dim=0).contiguous()
    coordinates = torch.stack((cx, cy), dim=0).contiguous()
    return coordinates, jacobian


def jacobian_of_displacement(
    displacement: torch.Tensor,
    scale: float = 1.0,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """计算 CAT DARTEL 初始化所需的中心差分 Jacobian。"""

    target_device = resolve_device(device)
    spec = GridSpec.from_shape(displacement.shape)
    field = _as_grid(displacement, spec, target_device, dtype, "displacement")
    x = torch.arange(spec.nx, device=target_device, dtype=torch.long)
    y = torch.arange(spec.ny, device=target_device, dtype=torch.long)
    x_grid = x.view(1, spec.nx).expand(spec.ny, spec.nx)
    y_grid = y.view(spec.ny, 1).expand(spec.ny, spec.nx)
    xm, ym = _bound_indices(x_grid - 1, y_grid, spec)
    xp, yp = _bound_indices(x_grid + 1, y_grid, spec)
    x0, y0 = _bound_indices(x_grid, y_grid - 1, spec)
    x1, y1 = _bound_indices(x_grid, y_grid + 1, spec)

    vx = field[0]
    vy = field[1]
    derivative_scale = float(scale) * 0.5
    j00 = (vx[ym, xm] * -1.0 + vx[yp, xp]) * derivative_scale + 1.0
    j10 = (vy[ym, xm] * -1.0 + vy[yp, xp]) * derivative_scale
    j01 = (vx[y0, x0] * -1.0 + vx[y1, x1]) * derivative_scale
    j11 = (vy[y0, x0] * -1.0 + vy[y1, x1]) * derivative_scale + 1.0
    # CAT C 按列主序保存二维 Jacobian：j00、j10、j01、j11。
    return torch.stack((j00, j10, j01, j11), dim=0).contiguous()


def smalldef_jac(
    displacement: torch.Tensor,
    scale: float,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 CAT `smalldef_jac` 的小形变坐标图和 Jacobian。"""

    target_device = resolve_device(device)
    spec = GridSpec.from_shape(displacement.shape)
    field = _as_grid(displacement, spec, target_device, dtype, "displacement")
    transformation = make_identity_map(spec, device=target_device, dtype=dtype)
    transformation = transformation + field * float(scale)
    jacobian = jacobian_of_displacement(
        field,
        scale=float(scale),
        device=target_device,
        dtype=dtype,
    )
    return transformation.contiguous(), jacobian.contiguous()


def jac_div_smalldef(
    jacobian: torch.Tensor,
    displacement: torch.Tensor,
    scale: float,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """计算 CAT `jac_div_smalldef` 的 Jacobian 右乘小形变逆。"""

    target_device = resolve_device(device)
    spec = GridSpec.from_shape(displacement.shape)
    field = _as_grid(displacement, spec, target_device, dtype, "displacement")
    old = _as_jacobian(jacobian, spec, target_device, dtype, "jacobian")
    small_jacobian = jacobian_of_displacement(
        field,
        scale=float(scale),
        device=target_device,
        dtype=dtype,
    )
    d00 = small_jacobian[0]
    d10 = small_jacobian[1]
    d01 = small_jacobian[2]
    d11 = small_jacobian[3]
    determinant = d00 * d11 - d01 * d10
    inverse00 = d11 / determinant
    inverse10 = -d10 / determinant
    inverse01 = -d01 / determinant
    inverse11 = d00 / determinant

    old00 = old[0]
    old10 = old[1]
    old01 = old[2]
    old11 = old[3]
    return torch.stack(
        (
            old00 * inverse00 + old01 * inverse10,
            old10 * inverse00 + old11 * inverse10,
            old00 * inverse01 + old01 * inverse11,
            old10 * inverse01 + old11 * inverse11,
        ),
        dim=0,
    ).contiguous()


def initialise_objfun(
    source: torch.Tensor,
    target: torch.Tensor,
    transformation: torch.Tensor,
    jacobian: torch.Tensor,
    distortion: torch.Tensor | None = None,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 CAT sum-of-squares 数据项的标量、梯度右端和 Hessian 分量。"""

    target_device = resolve_device(device)
    source_tensor = torch.as_tensor(source, dtype=dtype, device=target_device)
    target_tensor = torch.as_tensor(target, dtype=dtype, device=target_device)
    if source_tensor.ndim != 2 or target_tensor.ndim != 2:
        raise ValueError("initialise_objfun 的 source/target 必须是二维标量场")
    if tuple(source_tensor.shape) != tuple(target_tensor.shape):
        raise ValueError("initialise_objfun 的 source/target 形状必须一致")
    spec = GridSpec(nx=int(source_tensor.shape[1]), ny=int(source_tensor.shape[0]))
    map_tensor = _as_grid(transformation, spec, target_device, dtype, "transformation")
    jacobian_tensor = _as_jacobian(jacobian, spec, target_device, dtype, "jacobian")
    if distortion is None:
        weights = torch.ones_like(target_tensor)
    else:
        weights = torch.as_tensor(distortion, dtype=dtype, device=target_device)
        if tuple(weights.shape) != (spec.ny, spec.nx):
            raise ValueError(f"distortion 形状必须是 {(spec.ny, spec.nx)}")

    x = map_tensor[0] - 1.0
    y = map_tensor[1] - 1.0
    corners = _corners(source_tensor, x, y, spec)
    k22, k12, k21, k11, dx1, dx2, dy1, dy2 = corners
    sampled = _bilinear(corners)
    difference = sampled - target_tensor
    derivative_x = (k11 - k21) * dy1 + (k12 - k22) * dy2
    derivative_y = (k11 * dx1 + k21 * dx2) - (k12 * dx1 + k22 * dx2)
    gradient_x = jacobian_tensor[0] * derivative_x + jacobian_tensor[1] * derivative_y
    gradient_y = jacobian_tensor[2] * derivative_x + jacobian_tensor[3] * derivative_y
    hessian = torch.stack(
        (
            gradient_x * gradient_x * weights,
            gradient_y * gradient_y * weights,
            gradient_x * gradient_y * weights,
        ),
        dim=0,
    ).contiguous()
    right_hand = torch.stack(
        (gradient_x * difference * weights, gradient_y * difference * weights),
        dim=0,
    ).contiguous()
    objective = 0.5 * torch.sum(difference * difference * weights)
    return objective, right_hand, hessian


def squaring_update(
    right_hand: torch.Tensor,
    hessian: torch.Tensor,
    transformation: torch.Tensor,
    jacobian: torch.Tensor,
    k: int,
    *,
    save_transformation: bool = False,
    return_scratch: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
    kernel: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行 CAT DARTEL 的 b/A 累加和变换平方步骤。

    默认路径把 b 的两个分量和 A 的三个分量合并为一次多通道采样；
    ``optimized=False`` 保留逐通道调用 ``sample_field`` 的 reference 路径。
    """

    if k < 0:
        raise ValueError(f"squaring 步数 k 不能为负数，得到 {k}")
    target_device = resolve_device(device)
    if kernel not in {"auto", "torch", "triton"}:
        raise ValueError(f"kernel 必须是 auto、torch 或 triton，得到 {kernel}")
    spec = GridSpec.from_shape(transformation.shape)
    b_value = _as_grid(right_hand, spec, target_device, dtype, "right_hand").clone()
    a_tensor = torch.as_tensor(hessian, dtype=dtype, device=target_device)
    if tuple(a_tensor.shape) != (3, spec.ny, spec.nx):
        raise ValueError(f"hessian 形状必须是 {(3, spec.ny, spec.nx)}")
    a_value = a_tensor.contiguous().clone()
    current_map = _as_grid(transformation, spec, target_device, dtype, "transformation")
    current_jacobian = _as_jacobian(jacobian, spec, target_device, dtype, "jacobian")
    map_buffers = [current_map, torch.empty_like(current_map)]
    jacobian_buffers = [current_jacobian, torch.empty_like(current_jacobian)]
    current_slot = 0
    other_slot = 1
    use_fused_squaring = (
        optimized
        and target_device.type == "cuda"
        and kernel in {"auto", "triton"}
        and dartel_triton.available()
    )
    if kernel == "triton" and not use_fused_squaring and optimized:
        raise ValueError("请求 Triton squaring，但当前设备或 Python 环境不可用")

    for step in range(k):
        current_map = map_buffers[current_slot]
        current_jacobian = jacobian_buffers[current_slot]

        if use_fused_squaring:
            b_value, a_value, b_increment = dartel_triton.squaring_update_triton(
                b_value,
                a_value,
                current_map,
                current_jacobian,
            )
        else:
            x = current_map[0] - 1.0
            y = current_map[1] - 1.0
            j00 = current_jacobian[0]
            j10 = current_jacobian[1]
            j01 = current_jacobian[2]
            j11 = current_jacobian[3]
            determinant = j00 * j11 - j01 * j10
            if optimized:
                sampled = _sample_channels(
                    torch.cat((b_value, a_value), dim=0),
                    x,
                    y,
                    spec,
                )
                sampled_bx, sampled_by = sampled[0], sampled[1]
                sampled_a00, sampled_a11, sampled_a01 = sampled[2:]
            else:
                sampled_bx = sample_field(
                    b_value[0], x, y, device=target_device, dtype=dtype
                )
                sampled_by = sample_field(
                    b_value[1], x, y, device=target_device, dtype=dtype
                )
            b_increment = torch.stack(
                (
                    determinant * (sampled_bx * j00 + sampled_by * j10),
                    determinant * (sampled_bx * j01 + sampled_by * j11),
                ),
                dim=0,
            ).contiguous()
            b_value += b_increment
            if not optimized:
                sampled_a00 = sample_field(
                    a_value[0], x, y, device=target_device, dtype=dtype
                )
                sampled_a11 = sample_field(
                    a_value[1], x, y, device=target_device, dtype=dtype
                )
                sampled_a01 = sample_field(
                    a_value[2], x, y, device=target_device, dtype=dtype
                )
            tmp1 = sampled_a00 * j00 + sampled_a01 * j10
            tmp2 = sampled_a01 * j00 + sampled_a11 * j10
            tmp3 = sampled_a00 * j01 + sampled_a01 * j11
            tmp4 = sampled_a01 * j01 + sampled_a11 * j11
            a_increment = torch.stack(
                (
                    determinant * (tmp1 * j00 + tmp2 * j10),
                    determinant * (tmp3 * j01 + tmp4 * j11),
                    determinant * (tmp1 * j01 + tmp2 * j11),
                ),
                dim=0,
            ).contiguous()
            a_value += a_increment
        # C squaring 复用另一块变换缓冲区先保存 b 增量；若本轮需要平方，
        # 后续 composition 会覆盖这块缓冲区。
        map_buffers[other_slot] = b_increment

        if save_transformation or step < k - 1:
            composed_map, composed_jacobian = composition_jacobian(
                current_map,
                current_jacobian,
                current_map,
                current_jacobian,
                device=target_device,
                dtype=dtype,
            )
            map_buffers[other_slot] = composed_map
            jacobian_buffers[other_slot] = composed_jacobian
            current_slot, other_slot = other_slot, current_slot

    if save_transformation and current_slot != 0:
        # C 在 save_transformation 模式下把最终 map/J 拷回原始 t0/J0
        # 缓冲区；该缓冲区随后还会作为 FMG 的初始解工作区复用。
        map_buffers[other_slot] = map_buffers[current_slot].clone()
        jacobian_buffers[other_slot] = jacobian_buffers[current_slot].clone()
    current_map = map_buffers[current_slot]
    current_jacobian = jacobian_buffers[current_slot]
    result = (
        b_value.contiguous(),
        a_value.contiguous(),
        current_map.contiguous(),
        current_jacobian.contiguous(),
    )
    if return_scratch:
        return result + (map_buffers[0].contiguous(),)
    return result


def regularization_operator(
    field: torch.Tensor,
    params: Sequence[float],
    rtype: int = 1,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """计算 CAT DARTEL 的线性弹性、膜或弯曲正则算子。

    ``params`` 沿用 C 实现的五个参数顺序：``sx, sy, mu, lambda, identity``。
    """

    if len(params) != 5:
        raise ValueError(f"正则参数必须有5个元素，得到 {len(params)}")
    if rtype not in {0, 1, 2}:
        raise ValueError(f"正则类型必须是0、1或2，得到 {rtype}")
    target_device = resolve_device(device)
    spec = GridSpec.from_shape(field.shape)
    value = _as_grid(field, spec, target_device, dtype, "field")
    sx, sy, mu, lam, identity = (float(item) for item in params)
    sx2 = sx * sx
    sy2 = sy * sy

    xm = _shift(value, -1, 0, spec)
    xp = _shift(value, 1, 0, spec)
    ym = _shift(value, 0, -1, spec)
    yp = _shift(value, 0, 1, spec)

    if rtype == 1:
        w00 = mu * (2.0 * sx2 + 2.0 * sy2) + identity
        w01 = -mu * sy2
        w10 = -mu * sx2
        return torch.stack(
            (
                w00 * value[0] + w01 * (ym[0] + yp[0]) + w10 * (xm[0] + xp[0]),
                w00 * value[1] + w01 * (ym[1] + yp[1]) + w10 * (xm[1] + xp[1]),
            ),
            dim=0,
        ).contiguous()

    if rtype == 0:
        wx0 = mu * (2.0 * sy2 + 4.0 * sx2) + 2.0 * lam * sx2 + identity
        wy0 = mu * (2.0 * sx2 + 4.0 * sy2) + 2.0 * lam * sy2 + identity
        wx1 = -(2.0 * mu + lam) * sx2
        wy1 = -(2.0 * mu + lam) * sy2
        wx2 = -mu * sx2
        wy2 = -mu * sy2
        wxy = 0.25 * (lam + mu) * sx * sy
        xym = _shift(value, -1, -1, spec)
        xyp = _shift(value, -1, 1, spec)
        xpm = _shift(value, 1, -1, spec)
        xpp = _shift(value, 1, 1, spec)
        cross_x = xym[1] - xyp[1] - xpm[1] + xpp[1]
        cross_y = xym[0] - xyp[0] - xpm[0] + xpp[0]
        return torch.stack(
            (
                wx0 * value[0] + wy2 * (ym[0] + yp[0]) + wx1 * (xm[0] + xp[0]) + wxy * cross_x,
                wy0 * value[1] + wy1 * (ym[1] + yp[1]) + wx2 * (xm[1] + xp[1]) + wxy * cross_y,
            ),
            dim=0,
        ).contiguous()

    sx4 = sx2 * sx2
    sy4 = sy2 * sy2
    w00 = mu * (6.0 * sx4 + 6.0 * sy4 + 8.0 * sx2 * sy2) + identity
    w01 = mu * (-4.0 * sx2 * sy2 - 4.0 * sy4)
    w02 = mu * sy4
    w10 = mu * (-4.0 * sx4 - 4.0 * sx2 * sy2)
    w11 = mu * (2.0 * sx2 * sy2)
    w20 = mu * sx4
    xmm = _shift(value, -2, 0, spec)
    xpp = _shift(value, 2, 0, spec)
    ymm = _shift(value, 0, -2, spec)
    ypp = _shift(value, 0, 2, spec)
    xmy = _shift(value, -1, -1, spec)
    xpy = _shift(value, 1, -1, spec)
    xmyp = _shift(value, -1, 1, spec)
    xpyp = _shift(value, 1, 1, spec)

    def apply_bending(component: int) -> torch.Tensor:
        """对一个向量分量应用五点弯曲模板。"""

        return (
            w00 * value[component]
            + w01 * (ym[component] + yp[component])
            + w02 * (ymm[component] + ypp[component])
            + w10 * (xm[component] + xp[component])
            + w11
            * (
                xmy[component]
                + xpy[component]
                + xmyp[component]
                + xpyp[component]
            )
            + w20 * (xmm[component] + xpp[component])
        )

    return torch.stack((apply_bending(0), apply_bending(1)), dim=0).contiguous()


def apply_membrane_system(
    hessian: torch.Tensor,
    field: torch.Tensor,
    params: Sequence[float],
    *,
    kernel: str = "auto",
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """计算 FMG 膜系统的 ``(A + L'L) field``。"""

    target_device = resolve_device(device)
    spec = GridSpec(nx=int(field.shape[-1]), ny=int(field.shape[-2]))
    a_value = torch.as_tensor(hessian, dtype=dtype, device=target_device)
    if tuple(a_value.shape) != (3, spec.ny, spec.nx):
        raise ValueError(f"hessian 形状必须是 {(3, spec.ny, spec.nx)}")
    vector = _as_grid(field, spec, target_device, dtype, "field")
    if kernel not in {"auto", "torch", "triton"}:
        raise ValueError(f"kernel 必须是 auto、torch 或 triton，得到 {kernel}")
    if kernel == "triton" and (
        target_device.type != "cuda" or not dartel_triton.available()
    ):
        raise ValueError("请求 Triton 膜系统，但当前设备或 Python 环境不可用")
    sx, sy, mu, _lam, identity = (float(item) for item in params)
    w00 = mu * (2.0 * sx * sx + 2.0 * sy * sy) + identity
    w01 = -mu * sy * sy
    w10 = -mu * sx * sx
    if (
        target_device.type == "cuda"
        and kernel in {"auto", "triton"}
        and dartel_triton.available()
    ):
        return dartel_triton.apply_membrane_system_triton(
            a_value,
            vector,
            w00,
            w01,
            w10,
        )
    regularized = regularization_operator(
        vector,
        params,
        rtype=1,
        device=target_device,
        dtype=dtype,
    )
    return torch.stack(
        (
            regularized[0] + a_value[0] * vector[0] + a_value[2] * vector[1],
            regularized[1] + a_value[2] * vector[0] + a_value[1] * vector[1],
        ),
        dim=0,
    ).contiguous()


def relax_membrane(
    hessian: torch.Tensor,
    right_hand: torch.Tensor,
    params: Sequence[float],
    nit: int,
    initial: torch.Tensor | None = None,
    *,
    kernel: str = "auto",
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """执行 CAT `relax_me` 的红黑 Gauss-Seidel 松弛。"""

    if nit < 0:
        raise ValueError(f"松弛次数不能为负数，得到 {nit}")
    target_device = resolve_device(device)
    b_value = torch.as_tensor(right_hand, dtype=dtype, device=target_device)
    if b_value.ndim != 3 or b_value.shape[0] != 2:
        raise ValueError("right_hand 形状必须是 [2, ny, nx]")
    spec = GridSpec(nx=int(b_value.shape[2]), ny=int(b_value.shape[1]))
    a_value = torch.as_tensor(hessian, dtype=dtype, device=target_device)
    if tuple(a_value.shape) != (3, spec.ny, spec.nx):
        raise ValueError(f"hessian 形状必须是 {(3, spec.ny, spec.nx)}")
    if initial is None:
        solution = torch.zeros_like(b_value)
    else:
        solution = _as_grid(initial, spec, target_device, dtype, "initial").clone()

    if len(params) != 5:
        raise ValueError(f"正则参数必须有5个元素，得到 {len(params)}")
    sx, sy, mu, _lam, _identity = (float(item) for item in params)
    w00 = mu * (2.0 * sx * sx + 2.0 * sy * sy) + float(params[4])
    w01 = -mu * sy * sy
    w10 = -mu * sx * sx

    if kernel not in {"auto", "torch", "triton"}:
        raise ValueError(f"kernel 必须是 auto、torch 或 triton，得到 {kernel}")
    if kernel == "triton" and (
        target_device.type != "cuda" or not dartel_triton.available()
    ):
        raise ValueError("请求 Triton，但当前设备或 Python 环境不可用")
    if (
        target_device.type == "cuda"
        and kernel in {"auto", "triton"}
        and dartel_triton.available()
        and spec.nx % 2 == 0
        and spec.nx > 1
    ):
        return dartel_triton.relax_membrane_triton(
            a_value,
            b_value,
            solution,
            w00,
            w01,
            w10,
            nit,
        ).contiguous()

    if spec.nx % 2 == 1 and spec.nx != 1:
        if target_device.type == "cuda":
            raise ValueError("CUDA 红黑松弛要求 x 网格宽度为偶数，奇数宽度请显式使用 CPU")
        for iteration in range(2 * nit):
            for row in range(spec.ny):
                start = int((iteration % 2) == (row % 2))
                for column in range(start, spec.nx, 2):
                    def bound_scalar(ix: int, iy: int) -> tuple[int, int]:
                        """返回奇数宽度 CPU 顺序路径的混合边界索引。"""

                        xx = ix % spec.nx
                        if spec.ny == 1:
                            return xx, 0
                        reflected = iy % (2 * spec.ny)
                        yy = (
                            2 * spec.ny - reflected - 1
                            if reflected >= spec.ny
                            else reflected
                        )
                        return xx, yy

                    xm, ym = bound_scalar(column - 1, row)
                    xp, yp = bound_scalar(column + 1, row)
                    x0, y0 = bound_scalar(column, row - 1)
                    x1, y1 = bound_scalar(column, row + 1)
                    residual_x = b_value[0, row, column] - w01 * (
                        solution[0, y0, x0] + solution[0, y1, x1]
                    ) - w10 * (solution[0, ym, xm] + solution[0, yp, xp])
                    residual_y = b_value[1, row, column] - w01 * (
                        solution[1, y0, x0] + solution[1, y1, x1]
                    ) - w10 * (solution[1, ym, xm] + solution[1, yp, xp])
                    diagonal_x = a_value[0, row, column] + w00
                    diagonal_y = a_value[1, row, column] + w00
                    cross = a_value[2, row, column]
                    determinant = diagonal_x * diagonal_y * 1.0000000001 - cross * cross
                    solution[0, row, column] = (
                        diagonal_y * residual_x - cross * residual_y
                    ) / determinant
                    solution[1, row, column] = (
                        -cross * residual_x + diagonal_x * residual_y
                    ) / determinant
        return solution.contiguous()

    x = torch.arange(spec.nx, device=target_device, dtype=torch.long).view(1, spec.nx)
    y = torch.arange(spec.ny, device=target_device, dtype=torch.long).view(spec.ny, 1)
    checker = ((x + y) & 1).to(torch.bool)

    for iteration in range(2 * nit):
        xm = _shift(solution, -1, 0, spec)
        xp = _shift(solution, 1, 0, spec)
        ym = _shift(solution, 0, -1, spec)
        yp = _shift(solution, 0, 1, spec)
        residual_x = b_value[0] - w01 * (ym[0] + yp[0]) - w10 * (xm[0] + xp[0])
        residual_y = b_value[1] - w01 * (ym[1] + yp[1]) - w10 * (xm[1] + xp[1])
        diagonal_x = a_value[0] + w00
        diagonal_y = a_value[1] + w00
        cross = a_value[2]
        determinant = diagonal_x * diagonal_y * 1.0000000001 - cross * cross
        new_x = (diagonal_y * residual_x - cross * residual_y) / determinant
        new_y = (-cross * residual_x + diagonal_x * residual_y) / determinant
        # 偶数宽度的周期网格是二分图；首轮更新奇偶和为1的点，第二轮
        # 更新另一种颜色，与上游 C 的起始列规则一致。
        active = checker if iteration % 2 == 0 else ~checker
        solution = torch.stack(
            (
                torch.where(active, new_x, solution[0]),
                torch.where(active, new_y, solution[1]),
            ),
            dim=0,
        ).contiguous()
    return solution


def fmg2_membrane(
    hessian: torch.Tensor,
    right_hand: torch.Tensor,
    params: Sequence[float],
    cycles: int = 3,
    nit: int = 3,
    initial: torch.Tensor | None = None,
    *,
    kernel: str = "auto",
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    assume_initial_nonzero: bool = False,
) -> torch.Tensor:
    """执行默认膜正则 DARTEL 的 full multigrid 求解。

    ``kernel`` 传递给各层的重采样和膜松弛器；``auto`` 在可用时选择
    CUDA Triton，否则选择当前设备上的 Torch 实现。
    """

    if cycles < 0 or nit < 0:
        raise ValueError(f"FMG cycles/nit 不能为负数，得到 {(cycles, nit)}")
    target_device = resolve_device(device)
    b_value = torch.as_tensor(right_hand, dtype=dtype, device=target_device)
    if b_value.ndim != 3 or b_value.shape[0] != 2:
        raise ValueError("right_hand 形状必须是 [2, ny, nx]")
    fine = GridSpec(nx=int(b_value.shape[2]), ny=int(b_value.shape[1]))
    a_value = torch.as_tensor(hessian, dtype=dtype, device=target_device)
    if tuple(a_value.shape) != (3, fine.ny, fine.nx):
        raise ValueError(f"hessian 形状必须是 {(3, fine.ny, fine.nx)}")
    if len(params) != 5:
        raise ValueError(f"正则参数必须有5个元素，得到 {len(params)}")
    base_params = tuple(float(item) for item in params)

    specs = [fine]
    while specs[-1].nx >= 2 or specs[-1].ny >= 2:
        previous = specs[-1]
        coarse = GridSpec(nx=(previous.nx + 1) // 2, ny=(previous.ny + 1) // 2)
        specs.append(coarse)
        if coarse.nx < 2 and coarse.ny < 2:
            break

    a_levels = [a_value]
    original_b_levels = [b_value]
    for level in range(1, len(specs)):
        a_levels.append(
            resize_field(
                a_levels[-1],
                specs[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )
        )
        original_b_levels.append(
            resize_field(
                original_b_levels[-1],
                specs[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )
        )

    def level_params(level: int) -> tuple[float, float, float, float, float]:
        """按 C FMG 规则缩放每层的空间步长。"""

        return (
            base_params[0] * specs[level].nx / fine.nx,
            base_params[1] * specs[level].ny / fine.ny,
            base_params[2],
            base_params[3],
            base_params[4],
        )

    b_levels = [b_value] + [
        torch.zeros((2, spec.ny, spec.nx), device=target_device, dtype=dtype)
        for spec in specs[1:]
    ]
    u_levels = [torch.zeros_like(item) for item in b_levels]
    if initial is not None:
        u_levels[0] = _as_grid(initial, fine, target_device, dtype, "initial").clone()
        for level in range(1, len(specs)):
            u_levels[level] = resize_field(
                u_levels[level - 1],
                specs[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )

    def solve_coarse() -> None:
        """在最粗层执行 CAT 的局部二乘初值和松弛。"""

        coarse = specs[-1]
        coarse_a = a_levels[-1]
        coarse_b = original_b_levels[-1]
        diagonal_x = coarse_a[0] + base_params[4]
        diagonal_y = coarse_a[1] + base_params[4]
        cross = coarse_a[2]
        determinant = diagonal_x * diagonal_y * 1.0000000001 - cross * cross
        u_levels[-1] = torch.stack(
            (
                (diagonal_y * coarse_b[0] - cross * coarse_b[1]) / determinant,
                (-cross * coarse_b[0] + diagonal_x * coarse_b[1]) / determinant,
            ),
            dim=0,
        ).contiguous()
        u_levels[-1] = relax_membrane(
            coarse_a,
            b_levels[-1],
            level_params(len(specs) - 1),
            nit,
            u_levels[-1],
            kernel=kernel,
            device=target_device,
            dtype=dtype,
        )

    def residual(level: int) -> torch.Tensor:
        """计算一层的线性系统残差。"""

        applied = apply_membrane_system(
            a_levels[level],
            u_levels[level],
            level_params(level),
            kernel=kernel,
            device=target_device,
            dtype=dtype,
        )
        return b_levels[level] - applied

    def cycle(start_level: int) -> None:
        """执行一次从指定层向粗层再回传的 FMG cycle。"""

        for level in range(start_level, len(specs) - 1):
            u_levels[level] = relax_membrane(
                a_levels[level],
                b_levels[level],
                level_params(level),
                nit,
                u_levels[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )
            b_levels[level + 1] = resize_field(
                residual(level),
                specs[level + 1],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )
            u_levels[level + 1] = torch.zeros_like(b_levels[level + 1])
        u_levels[-1] = relax_membrane(
            a_levels[-1],
            b_levels[-1],
            level_params(len(specs) - 1),
            nit,
            u_levels[-1],
            kernel=kernel,
            device=target_device,
            dtype=dtype,
        )
        for level in range(len(specs) - 2, start_level - 1, -1):
            u_levels[level] = u_levels[level] + resize_field(
                u_levels[level + 1],
                specs[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )
            u_levels[level] = relax_membrane(
                a_levels[level],
                b_levels[level],
                level_params(level),
                nit,
                u_levels[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )

    # CUDA Graph 捕获期间不能把 CUDA 标量同步回 host；正式 DARTEL 的
    # solver_initial 是带一基坐标的变换图，显式 graph-safe 路径可以跳过
    # 这个检查。默认仍保留原来的全零检查，避免改变 reference 语义。
    if initial is None:
        use_coarse_initialisation = True
    elif assume_initial_nonzero:
        use_coarse_initialisation = False
    else:
        use_coarse_initialisation = not bool(torch.any(u_levels[0]).item())

    if use_coarse_initialisation:
        solve_coarse()
        for level in range(len(specs) - 2, -1, -1):
            u_levels[level] = resize_field(
                u_levels[level + 1],
                specs[level],
                kernel=kernel,
                device=target_device,
                dtype=dtype,
            )
            if level > 0:
                b_levels[level] = original_b_levels[level].clone()
            for _ in range(cycles):
                cycle(level)
    else:
        for _ in range(cycles):
            cycle(0)

    return u_levels[0].contiguous()


def dartel_step(
    source: torch.Tensor,
    target: torch.Tensor,
    velocity: torch.Tensor,
    *,
    k: int,
    rtype: int = 1,
    params: Sequence[float] = (1.0, 1.0, 0.125, 0.0, 0.0),
    lmreg: float = 1e-3,
    cycles: int = 3,
    nit: int = 3,
    code: int = 1,
    distortion: torch.Tensor | None = None,
    kernel: str = "auto",
    squaring_kernel: str | None = None,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
    assume_initial_nonzero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行官方 DARTEL `dartel()` 的一个膜正则更新步骤。

    当前实现覆盖 CAT 默认的膜正则 ``rtype=1`` 以及平方和目标
    ``code=0/1``；返回更新后的速度场和 ``[ssl, ssp/2, normb]`` 诊断量。
    ``source``、``target`` 是二维标量图，``velocity`` 是 ``[2, ny, nx]``。
    ``optimized=False`` 保留逐通道插值的 reference 计算路径。
    """

    if k < 0:
        raise ValueError(f"DARTEL k 不能为负数，得到 {k}")
    if cycles < 0 or nit < 0:
        raise ValueError(f"DARTEL cycles/nit 不能为负数，得到 {(cycles, nit)}")
    if squaring_kernel not in {None, "auto", "torch", "triton"}:
        raise ValueError(
            "squaring_kernel 必须为 None、auto、torch 或 triton"
        )
    if code not in {0, 1}:
        raise NotImplementedError("当前 Python DARTEL 步骤只支持 code=0 或 code=1")
    if rtype != 1:
        raise NotImplementedError("当前 Python DARTEL 步骤只支持膜正则 rtype=1")
    if len(params) != 5:
        raise ValueError(f"DARTEL 正则参数必须有5个元素，得到 {len(params)}")
    resolved_squaring_kernel = kernel if squaring_kernel is None else squaring_kernel

    target_device = resolve_device(device)
    source_value = torch.as_tensor(source, dtype=dtype, device=target_device)
    target_value = torch.as_tensor(target, dtype=dtype, device=target_device)
    if source_value.ndim != 2 or target_value.ndim != 2:
        raise ValueError("DARTEL source/target 必须是二维标量图")
    if tuple(source_value.shape) != tuple(target_value.shape):
        raise ValueError("DARTEL source/target 形状必须一致")
    spec = GridSpec(nx=int(source_value.shape[1]), ny=int(source_value.shape[0]))
    source_value = source_value.contiguous()
    target_value = target_value.contiguous()
    velocity_value = _as_grid(
        velocity,
        spec,
        target_device,
        dtype,
        "velocity",
    )
    if distortion is None:
        distortion_value = None
    else:
        distortion_value = torch.as_tensor(
            distortion,
            dtype=dtype,
            device=target_device,
        )
        if distortion_value.numel() != spec.points:
            raise ValueError(f"distortion 元素数必须是 {spec.points}")
        distortion_value = distortion_value.reshape(spec.ny, spec.nx).contiguous()

    scale = 1.0 / float(2**k)
    transformation, jacobian = expdef(
        velocity_value,
        k=k,
        return_jacobian=True,
        device=target_device,
        dtype=dtype,
        optimized=optimized,
    )
    jacobian = jac_div_smalldef(
        jacobian,
        velocity_value,
        scale,
        device=target_device,
        dtype=dtype,
    )
    objective, right_hand, hessian = initialise_objfun(
        source_value,
        target_value,
        transformation,
        jacobian,
        distortion_value,
        device=target_device,
        dtype=dtype,
    )
    transformation, jacobian = smalldef_jac(
        velocity_value,
        -scale,
        device=target_device,
        dtype=dtype,
    )
    (
        right_hand,
        hessian,
        transformation,
        jacobian,
        solver_initial,
    ) = squaring_update(
        right_hand,
        hessian,
        transformation,
        jacobian,
        k=k,
        save_transformation=code == 1,
        return_scratch=True,
        device=target_device,
        dtype=dtype,
        optimized=optimized,
        kernel=resolved_squaring_kernel,
    )

    if code == 1:
        reverse_jacobian = jac_div_smalldef(
            jacobian,
            velocity_value,
            -scale,
            device=target_device,
            dtype=dtype,
        )
        reverse_objective, reverse_right_hand, reverse_hessian = initialise_objfun(
            target_value,
            source_value,
            transformation,
            reverse_jacobian,
            distortion_value,
            device=target_device,
            dtype=dtype,
        )
        objective = objective + reverse_objective
        reverse_transformation, reverse_small_jacobian = smalldef_jac(
            velocity_value,
            scale,
            device=target_device,
            dtype=dtype,
        )
        reverse_right_hand, reverse_hessian, _, _, solver_initial = squaring_update(
            reverse_right_hand,
            reverse_hessian,
            reverse_transformation,
            reverse_small_jacobian,
            k=k,
            save_transformation=False,
            return_scratch=True,
            device=target_device,
            dtype=dtype,
            optimized=optimized,
            kernel=resolved_squaring_kernel,
        )
        right_hand = right_hand - reverse_right_hand
        hessian = hessian + reverse_hessian

    base_params = tuple(float(item) for item in params)
    regularized = regularization_operator(
        velocity_value,
        base_params,
        rtype=rtype,
        device=target_device,
        dtype=dtype,
    )
    right_hand = right_hand * scale + regularized
    regularization_objective = torch.sum(regularized * velocity_value)
    norm_right_hand = torch.sqrt(torch.sum(right_hand * right_hand))
    hessian = hessian * scale
    solver_params = list(base_params)
    solver_params[4] += float(lmreg)
    solution = fmg2_membrane(
        hessian,
        right_hand,
        solver_params,
        cycles=cycles,
        nit=nit,
        initial=solver_initial,
        kernel=kernel,
        device=target_device,
        dtype=dtype,
        assume_initial_nonzero=assume_initial_nonzero,
    )
    if distortion_value is None:
        update_weight = 1.0
    else:
        # CAT 的 dartel() 使用每一行第一个 dj 值作为该行的球面权重。
        update_weight = distortion_value[:, :1]
    updated_velocity = velocity_value - solution * update_weight
    metrics = torch.stack(
        (objective, regularization_objective * 0.5, norm_right_hand)
    )
    return updated_velocity.contiguous(), metrics.contiguous()


def expdef(
    displacement: torch.Tensor,
    k: int,
    *,
    return_jacobian: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """计算 CAT DARTEL 的 scaling-and-squaring 指数形变。

    ``optimized=False`` 保留逐分量插值的 reference 组合路径。
    """

    if k < 0:
        raise ValueError(f"指数形变步数 k 不能为负数，得到 {k}")
    target_device = resolve_device(device)
    spec = GridSpec.from_shape(displacement.shape)
    velocity = _as_grid(displacement, spec, target_device, dtype, "displacement")
    scale = 1.0 / float(2**k)
    current = make_identity_map(spec, device=target_device, dtype=dtype)
    current = current + velocity * scale

    if return_jacobian:
        current_jacobian = jacobian_of_displacement(
            velocity,
            scale=scale,
            device=target_device,
            dtype=dtype,
        )
        for _ in range(k):
            current, current_jacobian = composition_jacobian(
                current,
                current_jacobian,
                current,
                current_jacobian,
                device=target_device,
                dtype=dtype,
                optimized=optimized,
            )
        return current.contiguous(), current_jacobian.contiguous()

    for _ in range(k):
        current = composition(
            current,
            current,
            device=target_device,
            dtype=dtype,
            optimized=optimized,
        )
    return current.contiguous()
