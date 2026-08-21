"""CAT-Surface 曲面阶段与 DARTEL 的设备常驻组合后端。"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import torch

from .dartel_grid import dartel_step, expdef, resolve_device, sample_field
from .surface_stencil import SurfaceStencil, SurfaceStencilDevice


@dataclass(frozen=True)
class DartelSolveResult:
    """保存设备上的最终变换、曲率图和诊断量。"""

    flow: torch.Tensor
    source_maps: torch.Tensor
    target_maps: torch.Tensor
    metrics: torch.Tensor


@dataclass
class _DartelCudaGraphEntry:
    """保存固定 shape/参数的 DARTEL CUDA Graph 及其静态张量。"""

    graph: torch.cuda.CUDAGraph
    source_maps: torch.Tensor
    target_maps: torch.Tensor
    flow: torch.Tensor
    metrics: torch.Tensor
    capture_seconds: float


_DARTEL_CUDA_GRAPH_CACHE: dict[tuple[object, ...], _DartelCudaGraphEntry] = {}


def _dartel_cuda_graph_key(
    source_maps: torch.Tensor,
    target_maps: torch.Tensor,
    *,
    loop: int,
    lmreg: float,
    cycles: int,
    nit: int,
    its: int,
    code: int,
    kernel: str,
    squaring_kernel: str | None,
    optimized: bool,
) -> tuple[object, ...]:
    """构造不会混用不同 DARTEL 配置的 Graph cache key。"""

    return (
        source_maps.device.type,
        source_maps.device.index,
        source_maps.dtype,
        tuple(source_maps.shape),
        tuple(target_maps.shape),
        int(loop),
        float(lmreg),
        int(cycles),
        int(nit),
        int(its),
        int(code),
        str(kernel),
        "auto" if squaring_kernel is None else str(squaring_kernel),
        bool(optimized),
    )


def default_dartel_parameters(
    loop: int = 6,
    *,
    mu: float = 0.125,
    lambda_: float = 0.0,
    lmreg: float = 1e-3,
    muchange: int = 4,
    murate: float = 1.25,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """构造 CAT_SurfWarp 默认的逐 loop 正则参数。"""

    if loop <= 0:
        raise ValueError(f"loop 必须为正数，得到 {loop}")
    if muchange <= 0 or murate <= 0:
        raise ValueError("muchange 和 murate 必须为正数")
    values: list[tuple[float, float, float, float, float]] = []
    current_mu = float(mu)
    current_lambda = float(lambda_)
    for index in range(loop):
        values.append(
            (
                1.0,
                1.0,
                current_mu,
                current_lambda,
                current_lambda / 2.0,
            )
        )
        if (index + 1) % muchange == 0:
            current_mu /= murate
        current_lambda /= 5.0
    return tuple(values)


def _capture_dartel_cuda_graph(
    source_maps: torch.Tensor,
    target_maps: torch.Tensor,
    *,
    loop: int,
    lmreg: float,
    cycles: int,
    nit: int,
    its: int,
    code: int,
    kernel: str,
    squaring_kernel: str | None,
    device: torch.device,
    dtype: torch.dtype,
    optimized: bool,
) -> _DartelCudaGraphEntry:
    """捕获一个固定 shape/参数的 DARTEL Graph。"""

    if device.type != "cuda":
        raise ValueError("DARTEL CUDA Graph 只支持 CUDA 设备")
    static_source = source_maps.detach().clone().contiguous()
    static_target = target_maps.detach().clone().contiguous()
    graph = torch.cuda.CUDAGraph()
    # 捕获前等待曲率 stream 和上一轮 eager warm-up，避免把外部依赖带入图。
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            captured = solve_dartel_maps(
                static_source,
                static_target,
                loop=loop,
                lmreg=lmreg,
                cycles=cycles,
                nit=nit,
                its=its,
                code=code,
                kernel=kernel,
                squaring_kernel=squaring_kernel,
                device=device,
                dtype=dtype,
                optimized=optimized,
                assume_initial_nonzero=True,
                cuda_graph=False,
            )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    return _DartelCudaGraphEntry(
        graph=graph,
        source_maps=static_source,
        target_maps=static_target,
        flow=captured.flow,
        metrics=captured.metrics,
        capture_seconds=time.perf_counter() - started,
    )


def prepare_dartel_cuda_graph(
    source_maps: torch.Tensor,
    target_maps: torch.Tensor,
    *,
    loop: int = 6,
    lmreg: float = 1e-3,
    cycles: int = 3,
    nit: int = 3,
    its: int = 3,
    code: int = 1,
    kernel: str = "auto",
    squaring_kernel: str | None = None,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
) -> float:
    """预热后捕获固定 DARTEL 配置，返回一次性捕获耗时。"""

    target_device = resolve_device(device)
    source = torch.as_tensor(source_maps, dtype=dtype, device=target_device).contiguous()
    target = torch.as_tensor(target_maps, dtype=dtype, device=target_device).contiguous()
    if target_device.type != "cuda":
        raise ValueError("DARTEL CUDA Graph 只支持 CUDA 设备")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("source_maps 和 target_maps 形状必须一致")
    key = _dartel_cuda_graph_key(
        source,
        target,
        loop=loop,
        lmreg=lmreg,
        cycles=cycles,
        nit=nit,
        its=its,
        code=code,
        kernel=kernel,
        squaring_kernel=squaring_kernel,
        optimized=optimized,
    )
    cached = _DARTEL_CUDA_GRAPH_CACHE.get(key)
    if cached is not None:
        return 0.0
    entry = _capture_dartel_cuda_graph(
        source,
        target,
        loop=loop,
        lmreg=lmreg,
        cycles=cycles,
        nit=nit,
        its=its,
        code=code,
        kernel=kernel,
        squaring_kernel=squaring_kernel,
        device=target_device,
        dtype=dtype,
        optimized=optimized,
    )
    _DARTEL_CUDA_GRAPH_CACHE[key] = entry
    return entry.capture_seconds


def clear_dartel_cuda_graph_cache() -> None:
    """释放当前进程保存的 DARTEL CUDA Graph。"""

    _DARTEL_CUDA_GRAPH_CACHE.clear()


def solve_dartel_maps(
    source_maps: torch.Tensor,
    target_maps: torch.Tensor,
    *,
    loop: int = 6,
    lmreg: float = 1e-3,
    cycles: int = 3,
    nit: int = 3,
    its: int = 3,
    code: int = 1,
    kernel: str = "auto",
    squaring_kernel: str | None = None,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    optimized: bool = True,
    assume_initial_nonzero: bool = False,
    cuda_graph: bool = False,
) -> DartelSolveResult:
    """在同一设备上执行 CAT 默认多 step DARTEL solve。

    ``optimized=False`` 透传到逐通道插值的 reference DARTEL 路径。
    ``cuda_graph=True`` 使用同一 shape/参数的 CUDA Graph；调用方应先完成
    一次 eager warm-up 或显式调用 ``prepare_dartel_cuda_graph``。
    """

    if cycles < 0 or nit < 0 or its < 0:
        raise ValueError("cycles、nit 和 its 不能为负数")
    if squaring_kernel not in {None, "auto", "torch", "triton"}:
        raise ValueError(
            "squaring_kernel 必须为 None、auto、torch 或 triton"
        )
    target_device = resolve_device(device)
    source = torch.as_tensor(source_maps, dtype=dtype, device=target_device)
    target = torch.as_tensor(target_maps, dtype=dtype, device=target_device)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source_maps 和 target_maps 必须是 [steps, ny, nx]")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("source_maps 和 target_maps 形状必须一致")
    if source.shape[0] < 1:
        raise ValueError("至少需要一个 DARTEL step")
    source = source.contiguous()
    target = target.contiguous()
    params = default_dartel_parameters(loop=loop)

    if cuda_graph:
        if target_device.type != "cuda":
            raise ValueError("cuda_graph=True 要求 CUDA 设备")
        key = _dartel_cuda_graph_key(
            source,
            target,
            loop=loop,
            lmreg=lmreg,
            cycles=cycles,
            nit=nit,
            its=its,
            code=code,
            kernel=kernel,
            squaring_kernel=squaring_kernel,
            optimized=optimized,
        )
        entry = _DARTEL_CUDA_GRAPH_CACHE.get(key)
        if entry is None:
            # 显式 graph 请求不能静默退回 eager；直接捕获当前输入。
            entry = _capture_dartel_cuda_graph(
                source,
                target,
                loop=loop,
                lmreg=lmreg,
                cycles=cycles,
                nit=nit,
                its=its,
                code=code,
                kernel=kernel,
                squaring_kernel=squaring_kernel,
                device=target_device,
                dtype=dtype,
                optimized=optimized,
            )
            _DARTEL_CUDA_GRAPH_CACHE[key] = entry
        else:
            entry.source_maps.copy_(source)
            entry.target_maps.copy_(target)
            entry.graph.replay()
        return DartelSolveResult(
            # Graph replay 会复用同一组静态输出；返回独立快照，避免
            # 后续 run/avg replay 覆盖调用方仍在使用的上一轮 flow。
            flow=entry.flow.clone().contiguous(),
            source_maps=entry.source_maps.clone().contiguous(),
            target_maps=entry.target_maps.clone().contiguous(),
            metrics=entry.metrics.clone().contiguous(),
        )

    velocities: list[torch.Tensor] = []
    diagnostics: list[torch.Tensor] = []

    for step in range(int(source.shape[0])):
        velocity = torch.zeros(
            (2, int(source.shape[1]), int(source.shape[2])),
            dtype=dtype,
            device=target_device,
        )
        step_metrics: list[torch.Tensor] = []
        for index, loop_params in enumerate(params):
            for _ in range(its):
                velocity, metrics = dartel_step(
                    source[step],
                    target[step],
                    velocity,
                    k=index,
                    params=loop_params,
                    lmreg=lmreg,
                    cycles=cycles,
                    nit=nit,
                    code=code,
                    kernel=kernel,
                    squaring_kernel=squaring_kernel,
                    device=target_device,
                    dtype=dtype,
                    optimized=optimized,
                    assume_initial_nonzero=assume_initial_nonzero,
                )
                step_metrics.append(metrics)
        velocities.append(velocity)
        diagnostics.append(torch.stack(step_metrics, dim=0))

    final_velocity = velocities[-1]
    flow = expdef(
        final_velocity,
        k=10,
        device=target_device,
        dtype=dtype,
        optimized=optimized,
    ).contiguous()
    return DartelSolveResult(
        flow=flow,
        source_maps=source,
        target_maps=target,
        metrics=torch.stack(diagnostics, dim=0).contiguous(),
    )


def solve_dartel_from_surfaces(
    source_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    source_stencil: SurfaceStencil | SurfaceStencilDevice,
    target_stencil: SurfaceStencil | SurfaceStencilDevice,
    *,
    fwhm: Sequence[float] = (5.0, 5.0 / 3.0, 5.0 / 9.0),
    curvtypes: Sequence[int] = (5, 5, 2),
    loop: int = 6,
    n_steps: int = 2,
    lmreg: float = 1e-3,
    cycles: int = 3,
    nit: int = 3,
    its: int = 3,
    code: int = 1,
    kernel: str = "auto",
    squaring_kernel: str | None = None,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    parallel_sides: bool = True,
    optimized: bool = True,
    assume_initial_nonzero: bool = False,
    cuda_graph: bool = False,
) -> DartelSolveResult:
    """曲率、sheet 映射和 DARTEL 在同一设备上连续执行。

    source 与 target 的曲率图彼此独立；CUDA 下默认用两个 stream 重叠
    两侧的重采样、曲率计算和 sheet 映射。``parallel_sides=False`` 保留
    原来的顺序路径，便于 reference A/B。``optimized=False`` 保留逐通道
    DARTEL 插值路径；``cuda_graph=True`` 只在 CUDA 上启用固定 shape 的
    DARTEL Graph。
    """

    if n_steps < 1 or n_steps > len(fwhm) or n_steps > len(curvtypes):
        raise ValueError("n_steps 必须落在 fwhm 和 curvtypes 的有效范围内")
    if squaring_kernel not in {None, "auto", "torch", "triton"}:
        raise ValueError(
            "squaring_kernel 必须为 None、auto、torch 或 triton"
        )
    target_device = resolve_device(device)
    if isinstance(source_stencil, SurfaceStencil):
        source_device = source_stencil.to(target_device, geometry_dtype=torch.float32)
    else:
        source_device = source_stencil
    if isinstance(target_stencil, SurfaceStencil):
        target_device_stencil = target_stencil.to(
            target_device, geometry_dtype=torch.float32
        )
    else:
        target_device_stencil = target_stencil
    if source_device.sphere_points.device != target_device:
        raise ValueError("source_stencil 必须已经位于请求设备")
    if target_device_stencil.sphere_points.device != target_device:
        raise ValueError("target_stencil 必须已经位于请求设备")

    source = torch.as_tensor(source_vertices, device=target_device)
    target = torch.as_tensor(target_vertices, device=target_device)
    source = source.to(torch.float32).contiguous()
    target = target.to(torch.float32).contiguous()

    for curvtype in curvtypes[:n_steps]:
        if int(curvtype) not in (2, 5):
            raise NotImplementedError(
                "当前设备常驻曲面后端只覆盖官方 curvtype=2 和 curvtype=5"
            )

    def build_maps(
        stencil: SurfaceStencilDevice,
        mapped_surface: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """按官方 steps 顺序生成一侧的曲率 sheet 图。"""

        requested_types = tuple(int(item) for item in curvtypes[:n_steps])
        if n_steps > 1 and all(item == 5 for item in requested_types):
            # type5 的 50 mm 几何平滑和法向只依赖当前 mapped_surface；同一
            # steps 内的不同 fwhm 只改变标量 heat-kernel，可安全复用前两项。
            return stencil.curvature_type5_to_sheet_many(
                mapped_surface,
                tuple(float(item) for item in fwhm[:n_steps]),
            )
        maps: list[torch.Tensor] = []
        for index in range(n_steps):
            maps.append(
                stencil.curvature_to_sheet(
                    mapped_surface,
                    float(fwhm[index]),
                    curvtype=int(curvtypes[index]),
                )
            )
        return tuple(maps)

    if parallel_sides and target_device.type == "cuda":
        current_stream = torch.cuda.current_stream(device=target_device)
        source_stream = torch.cuda.Stream(device=target_device)
        target_stream = torch.cuda.Stream(device=target_device)
        source_stream.wait_stream(current_stream)
        target_stream.wait_stream(current_stream)
        with torch.cuda.stream(source_stream):
            source_surface = source_device.resample_vertices(source)
            source_maps = build_maps(source_device, source_surface)
        with torch.cuda.stream(target_stream):
            target_surface = target_device_stencil.resample_vertices(target)
            target_maps = build_maps(target_device_stencil, target_surface)
        current_stream.wait_stream(source_stream)
        current_stream.wait_stream(target_stream)
    else:
        source_surface = source_device.resample_vertices(source)
        target_surface = target_device_stencil.resample_vertices(target)
        source_maps = build_maps(source_device, source_surface)
        target_maps = build_maps(target_device_stencil, target_surface)

    stacked_source = torch.stack(source_maps, dim=0).to(dtype=dtype).contiguous()
    stacked_target = torch.stack(target_maps, dim=0).to(dtype=dtype).contiguous()
    return solve_dartel_maps(
        stacked_source,
        stacked_target,
        loop=loop,
        lmreg=lmreg,
        cycles=cycles,
        nit=nit,
        its=its,
        code=code,
        kernel=kernel,
        squaring_kernel=squaring_kernel,
        device=target_device,
        dtype=dtype,
        optimized=optimized,
        assume_initial_nonzero=assume_initial_nonzero,
        cuda_graph=cuda_graph,
    )


def _wrap_displacement(value: torch.Tensor) -> torch.Tensor:
    """复现 CAT apply_warp 对周期形变分量的逐条件折返。"""

    value = torch.where(value >= 1.0, value - torch.floor(value), value)
    value = torch.where(value <= -1.0, value + torch.floor(-value), value)
    value = torch.where(value >= 0.5, value - 1.0, value)
    return torch.where(value <= -0.5, value + 1.0, value)


def apply_flow_to_sphere(
    sphere_vertices: torch.Tensor,
    flow: torch.Tensor,
    *,
    unit_sphere_vertices: torch.Tensor | None = None,
    inverse: bool = True,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """在 GPU 上复现 CAT ``apply_warp`` 的二维 flow 到球面变换。"""

    target_device = resolve_device(device)
    sphere = torch.as_tensor(sphere_vertices, device=target_device)
    deformation = torch.as_tensor(flow, dtype=dtype, device=target_device)
    if sphere.ndim != 2 or sphere.shape[1] != 3:
        raise ValueError("sphere_vertices 形状必须是 [points, 3]")
    if deformation.ndim != 3 or deformation.shape[0] != 2:
        raise ValueError("flow 形状必须是 [2, ny, nx]")
    if unit_sphere_vertices is None:
        sphere = sphere.to(torch.float32)
        sphere = sphere / torch.linalg.vector_norm(
            sphere, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).eps)
    else:
        sphere = torch.as_tensor(
            unit_sphere_vertices, device=target_device, dtype=torch.float32
        )
        if sphere.shape != sphere_vertices.shape:
            raise ValueError("unit_sphere_vertices 必须与 sphere_vertices 形状一致")
    points = sphere.to(dtype)
    x_coord = points[:, 0]
    y_coord = points[:, 1]
    z_coord = points[:, 2].clamp(-1.0, 1.0)
    theta = torch.acos(z_coord)
    clockwise = -torch.atan2(x_coord, y_coord)
    clockwise = torch.where(clockwise < 0.0, clockwise + 2.0 * torch.pi, clockwise)
    u = (clockwise + torch.pi / 2.0) / (2.0 * torch.pi)
    u = torch.where(u > 1.0, u - 1.0, u)
    v = theta / torch.pi

    ny = int(deformation.shape[1])
    nx = int(deformation.shape[2])
    row = torch.arange(ny, device=target_device, dtype=dtype).view(ny, 1)
    column = torch.arange(nx, device=target_device, dtype=dtype).view(1, nx)
    row = row.expand(ny, nx)
    column = column.expand(ny, nx)
    weight = torch.sin(((row + 0.5) / float(ny)) * torch.pi)
    u_deformation = _wrap_displacement(
        (deformation[0] - column - 1.0) / float(nx)
    ) * weight
    # 官方 C 实现对两个分量都先做周期折返，再乘投影面积权重。
    v_deformation = _wrap_displacement(
        (deformation[1] - row - 1.0) / float(ny)
    ) * weight
    x = u * float(nx) - 0.5
    y = v * float(ny) - 0.5
    sampled_u = sample_field(
        u_deformation, x, y, device=target_device, dtype=dtype
    )
    sampled_v = sample_field(
        v_deformation, x, y, device=target_device, dtype=dtype
    )
    if inverse:
        sampled_u = -sampled_u
        sampled_v = -sampled_v
    u = u + sampled_u
    v = v + sampled_v
    u = torch.where(v < 0.0, u + 0.5, u)
    v = torch.where(v < 0.0, -v, v)
    u = torch.where(v > 1.0, u + 0.5, u)
    v = torch.where(v > 1.0, 2.0 - v, v)
    u = u - torch.floor(u)
    v = v.clamp(0.0, 1.0)
    phi = u * (2.0 * torch.pi)
    theta = v * torch.pi
    sin_theta = torch.sin(theta)
    warped = torch.stack(
        (
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=-1,
    )
    return (
        warped
        / torch.linalg.vector_norm(warped, dim=-1, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )
    ).contiguous()


def apply_flow_to_stenciled_sphere(
    sphere_vertices: torch.Tensor,
    flow: torch.Tensor,
    stencil: SurfaceStencil | SurfaceStencilDevice,
    *,
    inverse: bool = True,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """使用 stencil 中缓存的精确单位球面点完成最终 GPU warp。

    单位球面点由官方 C 的 ``map_point_to_unit_sphere`` 在 CPU 端一次生
    成，随后与 flow 一起留在目标设备，避免最终阶段再次做不规则三角形
    定位。
    """

    target_device = resolve_device(device)
    device_stencil = (
        stencil.to(target_device, geometry_dtype=torch.float32)
        if isinstance(stencil, SurfaceStencil)
        else stencil
    )
    if device_stencil.sphere_points.device != target_device:
        raise ValueError("stencil 必须已经位于请求设备")
    if device_stencil.unit_sphere_points is None:
        raise ValueError("stencil 缺少 apply_warp 所需的 unit_sphere_points")
    return apply_flow_to_sphere(
        sphere_vertices,
        flow,
        unit_sphere_vertices=device_stencil.unit_sphere_points,
        inverse=inverse,
        device=target_device,
        dtype=dtype,
    )
