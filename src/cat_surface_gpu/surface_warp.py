"""Device-resident surface feature, DARTEL, and final-warp operations."""

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
    """Store a solved flow, warped feature maps, and DARTEL diagnostics."""

    flow: torch.Tensor
    source_maps: torch.Tensor
    target_maps: torch.Tensor
    metrics: torch.Tensor


@dataclass
class _DartelCudaGraphEntry:
    """Store one captured DARTEL CUDA Graph and its static tensors."""

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
    """Dartel cuda graph key."""

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
    """Return upstream-compatible regularization parameters for each loop."""

    if loop <= 0:
        raise ValueError(f"loop must be positive, got {loop}")
    if muchange <= 0 or murate <= 0:
        raise ValueError("muchange and murate must be positive")
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
    """Capture dartel cuda graph."""

    if device.type != "cuda":
        raise ValueError("DARTEL CUDA Graph capture requires a CUDA device")
    static_source = source_maps.detach().clone().contiguous()
    static_target = target_maps.detach().clone().contiguous()
    graph = torch.cuda.CUDAGraph()
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
    """Prepare dartel cuda graph."""

    target_device = resolve_device(device)
    source = torch.as_tensor(
        source_maps, dtype=dtype, device=target_device
    ).contiguous()
    target = torch.as_tensor(
        target_maps, dtype=dtype, device=target_device
    ).contiguous()
    if target_device.type != "cuda":
        raise ValueError("DARTEL CUDA Graph capture requires a CUDA device")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("source_maps and target_maps must have the same shape")
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
    """Clear dartel cuda graph cache."""

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
    """Solve DARTEL feature maps while preserving upstream loop semantics."""

    if cycles < 0 or nit < 0 or its < 0:
        raise ValueError("cycles, nit, and its must be non-negative")
    if squaring_kernel not in {None, "auto", "torch", "triton"}:
        raise ValueError("squaring_kernel must be None, 'auto', 'torch', or 'triton'")
    target_device = resolve_device(device)
    source = torch.as_tensor(source_maps, dtype=dtype, device=target_device)
    target = torch.as_tensor(target_maps, dtype=dtype, device=target_device)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source_maps and target_maps must have shape [steps, ny, nx]")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("Source_maps target_maps shape must")
    if source.shape[0] < 1:
        raise ValueError("At least one DARTEL step is required")
    source = source.contiguous()
    target = target.contiguous()
    params = default_dartel_parameters(loop=loop)

    if cuda_graph:
        if target_device.type != "cuda":
            raise ValueError("cuda_graph=True requires a CUDA device")
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
    """Build curvature sheets from surfaces and solve their DARTEL flow."""

    if n_steps < 1 or n_steps > len(fwhm) or n_steps > len(curvtypes):
        raise ValueError("n_steps must not exceed the lengths of fwhm and curvtypes")
    if squaring_kernel not in {None, "auto", "torch", "triton"}:
        raise ValueError("squaring_kernel must be None, 'auto', 'torch', or 'triton'")
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
        raise ValueError("source_stencil must be on the requested device")
    if target_device_stencil.sphere_points.device != target_device:
        raise ValueError("target_stencil must be on the requested device")

    source = torch.as_tensor(source_vertices, device=target_device)
    target = torch.as_tensor(target_vertices, device=target_device)
    source = source.to(torch.float32).contiguous()
    target = target.to(torch.float32).contiguous()

    for curvtype in curvtypes[:n_steps]:
        if int(curvtype) not in (2, 5):
            raise NotImplementedError(
                "The device surface backend supports only upstream curvtype 2 and 5"
            )

    def build_maps(
        stencil: SurfaceStencilDevice,
        mapped_surface: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Build maps."""

        requested_types = tuple(int(item) for item in curvtypes[:n_steps])
        if n_steps > 1 and all(item == 5 for item in requested_types):
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
    """Wrap displacement."""

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
    """Apply flow to sphere."""

    target_device = resolve_device(device)
    sphere = torch.as_tensor(sphere_vertices, device=target_device)
    deformation = torch.as_tensor(flow, dtype=dtype, device=target_device)
    if sphere.ndim != 2 or sphere.shape[1] != 3:
        raise ValueError("sphere_vertices must have shape [points, 3]")
    if deformation.ndim != 3 or deformation.shape[0] != 2:
        raise ValueError("flow must have shape [2, ny, nx]")
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
            raise ValueError("unit_sphere_vertices must match sphere_vertices shape")
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
    u_deformation = (
        _wrap_displacement((deformation[0] - column - 1.0) / float(nx)) * weight
    )
    v_deformation = (
        _wrap_displacement((deformation[1] - row - 1.0) / float(ny)) * weight
    )
    x = u * float(nx) - 0.5
    y = v * float(ny) - 0.5
    sampled_u = sample_field(u_deformation, x, y, device=target_device, dtype=dtype)
    sampled_v = sample_field(v_deformation, x, y, device=target_device, dtype=dtype)
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
    """Apply flow to stenciled sphere."""

    target_device = resolve_device(device)
    device_stencil = (
        stencil.to(target_device, geometry_dtype=torch.float32)
        if isinstance(stencil, SurfaceStencil)
        else stencil
    )
    if device_stencil.sphere_points.device != target_device:
        raise ValueError("stencil must be on the requested device")
    if device_stencil.unit_sphere_points is None:
        raise ValueError(
            "stencil does not contain unit-sphere points required by apply_warp"
        )
    return apply_flow_to_sphere(
        sphere_vertices,
        flow,
        unit_sphere_vertices=device_stencil.unit_sphere_points,
        inverse=inverse,
        device=target_device,
        dtype=dtype,
    )
