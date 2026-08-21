"""End-to-end GIFTI pipeline for CAT_SurfWarp-compatible GPU registration."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Sequence

import numpy as np
import torch

from .dartel_grid import resolve_device
from .rotation_pipeline import RotationPipeline, read_rotation_values
from .rotation_feature import compute_rotation_features
from .surface_stencil import SurfaceStencil
from .surface_warp import (
    apply_flow_to_stenciled_sphere,
    prepare_dartel_cuda_graph,
    solve_dartel_from_surfaces,
    solve_dartel_maps,
)


@dataclass(frozen=True)
class GiftiMesh:
    """Store one GIFTI surface's vertices, faces, and source path."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class CatSurfaceGpuResult:
    """Store final mesh data, rotation diagnostics, timings, and output path."""

    vertices: np.ndarray
    faces: np.ndarray
    rotation_angle: np.ndarray
    rotation_cost: float
    timings: dict[str, float]
    output_path: Path


def read_gifti_mesh(path: str | Path) -> GiftiMesh:
    """Read the first point-set and triangle arrays from a GIFTI surface."""

    import nibabel as nib

    image = nib.load(str(path))
    if len(image.darrays) < 2:
        raise ValueError(f"GIFTI file must contain vertex and face arrays: {path}")
    vertices = np.asarray(image.darrays[0].data)
    faces = np.asarray(image.darrays[1].data)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Vertex array must have shape [n, 3]: {path} -> {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Face array must have shape [m, 3]: {path} -> {faces.shape}")
    return GiftiMesh(
        vertices=np.ascontiguousarray(vertices, dtype=np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.int32),
    )


def write_gifti_mesh(
    path: str | Path,
    mesh: GiftiMesh,
    *,
    reference_path: str | Path | None = None,
) -> None:
    """Write vertices and faces while preserving reference GIFTI intents."""

    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    reference = None if reference_path is None else nib.load(str(reference_path))
    point_intent = (
        int(reference.darrays[0].intent)
        if reference is not None and len(reference.darrays) >= 1
        else 1008
    )
    triangle_intent = (
        int(reference.darrays[1].intent)
        if reference is not None and len(reference.darrays) >= 2
        else 1009
    )
    image = GiftiImage(
        darrays=[
            GiftiDataArray(
                data=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                intent=point_intent,
            ),
            GiftiDataArray(
                data=np.ascontiguousarray(mesh.faces, dtype=np.int32),
                intent=triangle_intent,
            ),
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output))


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA at an explicit stage boundary."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_dartel_dtype(value: str | torch.dtype) -> torch.dtype:
    """Resolve explicit FP32/FP64 DARTEL precision without autocast."""

    if isinstance(value, torch.dtype):
        resolved = value
    else:
        normalized = str(value).lower().replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        try:
            resolved = mapping[normalized]
        except KeyError as error:
            raise ValueError(
                "dartel_dtype must be float32/fp32 or float64/fp64"
            ) from error
    if resolved not in {torch.float32, torch.float64}:
        raise ValueError("dartel_dtype must be torch.float32 or torch.float64")
    return resolved


def _run_checked(
    command: Sequence[str],
    label: str,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    """Run an external helper and preserve diagnostic output on failure."""

    completed = subprocess.run(
        [str(item) for item in command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-2000:]
        stdout = completed.stdout.strip()[-1000:]
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


def _prepare_rotation_values(
    probe: str | Path,
    source_surface: str | Path,
    source_sphere: str | Path,
    target_surface: str | Path,
    target_sphere: str | Path,
    output_path: Path,
    *,
    raw_depth: bool = False,
) -> Path:
    """Prepare rotation values."""

    environment = None
    if raw_depth:
        environment = os.environ.copy()
        environment["CAT_SURFACE_GPU_ROTATION_RAW_DEPTH"] = "1"
        environment["CAT_SURFACE_GPU_ROTATION_RAW_DEPTH_GEOMETRY"] = "1"

    _run_checked(
        [
            str(probe),
            str(source_surface),
            str(source_sphere),
            str(target_surface),
            str(target_sphere),
            str(output_path),
        ],
        "Rotation feature",
        environment=environment,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Rotation feature helper did not produce a valid file: {output_path}"
        )
    return output_path


def _prepare_rotation_geometry(
    probe: str | Path,
    surface: str | Path,
    sphere: str | Path,
    heat_fwhm: float,
    output_path: Path,
) -> Path:
    """Prepare rotation geometry."""

    _run_checked(
        [
            str(probe),
            str(surface),
            str(sphere),
            str(float(heat_fwhm)),
            str(output_path),
        ],
        "Upstream coarse feature geometry",
    )
    if not output_path.is_file() or output_path.stat().st_size < 4:
        raise RuntimeError(
            f"Coarse feature geometry helper did not produce a valid file: {output_path}"
        )
    return output_path


def _read_rotation_geometry(path: str | Path) -> np.ndarray:
    """Read rotation geometry."""

    raw = Path(path).read_bytes()
    if len(raw) < 4:
        raise ValueError(f"Coarse feature geometry file is too short: {path}")
    count = int(np.frombuffer(raw, dtype=np.int32, count=1)[0])
    expected = 4 + count * 3 * np.dtype(np.float64).itemsize
    if count <= 0 or len(raw) < expected:
        raise ValueError(f"Coarse feature geometry file has an invalid length: {path}")
    points = np.frombuffer(raw, dtype=np.float64, count=count * 3, offset=4).reshape(
        count, 3
    )
    return np.ascontiguousarray(points, dtype=np.float32)


def _average_xz_surfaces(
    x_surface: torch.Tensor,
    z_surface: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Average xz surfaces."""

    x_value = torch.as_tensor(x_surface, dtype=torch.float64, device=device)
    z_value = torch.as_tensor(z_surface, dtype=torch.float64, device=device)
    if x_value.shape != z_value.shape or x_value.ndim != 2 or x_value.shape[1] != 3:
        raise ValueError("Both sphere arrays must have shape [points, 3]")
    phi_x = torch.acos(x_value[:, 0].clamp(-1.0, 1.0)) / torch.pi
    phi_z = torch.acos(z_value[:, 2].clamp(-1.0, 1.0)) / torch.pi
    weight_x = torch.exp(-((2.0 * phi_x - 1.0) ** 2) / 0.1).clamp_min(1e-19)
    weight_z = torch.exp(-((2.0 * phi_z - 1.0) ** 2) / 0.1).clamp_min(1e-19)
    total = weight_x + weight_z
    averaged = (weight_x[:, None] * x_value + weight_z[:, None] * z_value) / total[
        :, None
    ]
    averaged = averaged / torch.linalg.vector_norm(
        averaged, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float64).eps)
    return averaged.to(torch.float32).contiguous()


def _normalise_sphere(vertices: np.ndarray) -> np.ndarray:
    """Normalise sphere."""

    value = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"Sphere vertices must have shape [n, 3], got {value.shape}")
    centre = np.zeros(3, dtype=np.float64)
    for point in value:
        centre[0] += float(point[0])
        centre[1] += float(point[1])
        centre[2] += float(point[2])

    centered = np.empty_like(value)
    point_count = value.shape[0]
    for index, point in enumerate(value):
        centered[index, 0] = np.float32(float(point[0]) - centre[0] / point_count)
        centered[index, 1] = np.float32(float(point[1]) - centre[1] / point_count)
        centered[index, 2] = np.float32(float(point[2]) - centre[2] / point_count)

    normalised = np.empty_like(centered)
    for index, point in enumerate(centered):
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        length = math.sqrt(x * x + y * y + z * z)
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("Sphere contains a vertex that cannot be normalized")
        scale = 1.0 / length
        normalised[index, 0] = np.float32(x * scale)
        normalised[index, 1] = np.float32(y * scale)
        normalised[index, 2] = np.float32(z * scale)
    return np.ascontiguousarray(normalised)


def _rotation_matrix_from_angles(angles: np.ndarray) -> torch.Tensor:
    """Rotation matrix from angles."""

    value = torch.as_tensor(angles, dtype=torch.float64).reshape(-1)
    if value.numel() != 3:
        raise ValueError("Rotation must contain exactly three angles")
    alpha, beta, gamma = value.unbind()
    zero = torch.zeros((), dtype=torch.float64)
    one = torch.ones((), dtype=torch.float64)
    rx = torch.stack(
        (
            one,
            zero,
            zero,
            zero,
            torch.cos(alpha),
            torch.sin(alpha),
            zero,
            -torch.sin(alpha),
            torch.cos(alpha),
        )
    )
    ry = torch.stack(
        (
            torch.cos(beta),
            zero,
            torch.sin(beta),
            zero,
            one,
            zero,
            -torch.sin(beta),
            zero,
            torch.cos(beta),
        )
    )
    rz = torch.stack(
        (
            torch.cos(gamma),
            torch.sin(gamma),
            zero,
            -torch.sin(gamma),
            torch.cos(gamma),
            zero,
            zero,
            zero,
            one,
        )
    )
    intermediate = torch.empty(9, dtype=torch.float64)
    result = torch.empty(9, dtype=torch.float64)
    for row in range(3):
        for column in range(3):
            intermediate[row + 3 * column] = sum(
                ry[row + 3 * index] * rx[index + 3 * column] for index in range(3)
            )
    for row in range(3):
        for column in range(3):
            result[row + 3 * column] = sum(
                rz[row + 3 * index] * intermediate[index + 3 * column]
                for index in range(3)
            )
    return result.reshape(3, 3)


def _rotate_vertices(vertices: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate vertices."""

    sphere = torch.as_tensor(vertices, dtype=torch.float64)
    matrix = _rotation_matrix_from_angles(angles)
    rotated = sphere @ matrix.transpose(0, 1)
    return np.ascontiguousarray(rotated.numpy(), dtype=np.float32)


def _prepare_artifact(
    provided_path: str | Path | None,
    builder: str | Path | None,
    builder_input: str | Path,
    output_path: Path,
    label: str,
    builder_options: Sequence[str] = (),
) -> Path:
    """Prepare artifact."""

    if provided_path is not None:
        path = Path(provided_path)
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return path
    if builder is None:
        raise ValueError(f"Neither an existing {label} nor a builder was provided")
    _run_checked(
        [str(builder), str(builder_input), str(output_path), *builder_options],
        f"{label}",
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"{label} did not produce a valid file: {output_path}")
    return output_path


def _timed_prepare(function, *args, **kwargs):
    """Timed prepare."""

    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def _warmup_and_prepare_dartel_graph(
    maps: torch.Tensor,
    *,
    loop: int,
    cycles: int,
    nit: int,
    its: int,
    code: int,
    kernel: str,
    device: torch.device,
    dtype: torch.dtype,
    squaring_kernel: str | None,
    optimized: bool,
) -> tuple[float, float]:
    """Warmup and prepare dartel graph."""

    started = time.perf_counter()
    warmup = solve_dartel_maps(
        maps,
        maps,
        loop=loop,
        lmreg=1e-3,
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
    _synchronize(device)
    capture_seconds = prepare_dartel_cuda_graph(
        warmup.source_maps,
        warmup.target_maps,
        loop=loop,
        lmreg=1e-3,
        cycles=cycles,
        nit=nit,
        its=its,
        code=code,
        kernel=kernel,
        squaring_kernel=squaring_kernel,
        device=device,
        dtype=dtype,
        optimized=optimized,
    )
    return capture_seconds, time.perf_counter() - started


def run_cat_surface_gpu_pipeline(
    source_surface_path: str | Path,
    source_sphere_path: str | Path,
    target_surface_path: str | Path,
    target_sphere_path: str | Path,
    output_path: str | Path,
    *,
    source_stencil_path: str | Path | None = None,
    target_stencil_path: str | Path | None = None,
    rotation_values_path: str | Path | None = None,
    rotation_values_probe: str | Path | None = None,
    rotation_geometry_probe: str | Path | None = None,
    rotation_depth_probe: str | Path | None = None,
    stencil_builder: str | Path | None = None,
    rotated_stencil_builder: str | Path | None = None,
    stencil_threads: int = 8,
    device: str | torch.device = "cuda",
    kernel: str = "triton",
    dartel_dtype: str | torch.dtype = torch.float64,
    squaring_kernel: str | None = None,
    steps: int = 2,
    runs: int = 1,
    avg: bool = False,
    curvtypes: Sequence[int] = (5, 5, 2),
    fwhm: Sequence[float] = (5.0, 5.0 / 3.0, 5.0 / 9.0),
    loop: int = 6,
    cycles: int = 3,
    nit: int = 3,
    its: int = 3,
    code: int = 1,
    point_chunk: int = 4096,
    rotation_grid_size: int = 128,
    rotation_margin: int = 1,
    rotation_refine: bool = True,
    rotation_max_iter: int = 500,
    rotation_tol: float = 1.0e-4,
    rotation_simplex_step: float = 0.1,
    rotation_feature_backend: str = "auto",
    rotation_feature_solver: str = "colored-sor",
    parallel_sides: bool = True,
    optimized_dartel: bool = True,
    cuda_graph: bool = True,
) -> CatSurfaceGpuResult:
    """Run initial rotation, FP64 DARTEL, averaging, warp, and GIFTI output."""

    if steps < 1 or steps > 3:
        raise ValueError("steps must be between 1 and 3")
    if runs < 1:
        raise ValueError("runs must be positive")
    if stencil_threads < 1 or stencil_threads > 256:
        raise ValueError("stencil_threads must be between 1 and 256")
    if len(curvtypes) < steps or len(fwhm) < steps:
        raise ValueError("curvtypes and fwhm must contain at least 'steps' entries")
    if rotation_max_iter < 1 or rotation_tol < 0.0 or rotation_simplex_step <= 0.0:
        raise ValueError(
            "rotation_max_iter, rotation_tol, and rotation_simplex_step must be positive"
        )
    if rotation_feature_backend not in {
        "auto",
        "cuda",
        "cuda-official-geometry",
        "cuda-official-depth",
        "official-cpu",
    }:
        raise ValueError(
            "rotation_feature_backend must be 'auto', 'cuda', "
            "cuda-official-geometry、cuda-official-depth or official-cpu"
        )
    if rotation_feature_solver not in {"colored-sor", "pcg"}:
        raise ValueError("rotation_feature_solver must be 'colored-sor' or 'pcg'")
    solve_dtype = _resolve_dartel_dtype(dartel_dtype)
    if squaring_kernel not in {None, "auto", "torch", "triton"}:
        raise ValueError("squaring_kernel must be None, 'auto', 'torch', or 'triton'")
    if (
        rotation_feature_backend
        in {
            "cuda-official-geometry",
            "cuda-official-depth",
        }
        and rotation_values_path is not None
    ):
        raise ValueError("GPU official hybrid backend cannot rotation_values")
    if rotation_feature_backend == "auto":
        rotation_feature_backend = (
            "official-cpu" if rotation_values_probe is not None else "cuda"
        )
    if device == "cpu" or str(device) == "cpu":
        raise ValueError("The explicit GPU pipeline requires a CUDA device")
    target_device = resolve_device(device)
    if target_device.type != "cuda":
        raise RuntimeError(
            f"CUDA device is unavailable for the GPU pipeline: {target_device}"
        )
    graph_requested = bool(cuda_graph and (runs + int(avg) >= 3))

    total_start = time.perf_counter()
    output = Path(output_path)
    source_surface = read_gifti_mesh(source_surface_path)
    source_sphere = read_gifti_mesh(source_sphere_path)
    target_surface = read_gifti_mesh(target_surface_path)
    target_sphere = read_gifti_mesh(target_sphere_path)
    timings: dict[str, float] = {
        "input_read_seconds": time.perf_counter() - total_start
    }

    with tempfile.TemporaryDirectory(prefix="cat_surface_gpu_") as temp_name:
        temp_dir = Path(temp_name)

        prepare_specs: dict[
            str, tuple[object, tuple[object, ...], dict[str, object]]
        ] = {}
        prepared: dict[str, Path] = {}
        if source_stencil_path is None:
            prepare_specs["source_stencil"] = (
                _timed_prepare,
                (
                    _prepare_artifact,
                    None,
                    stencil_builder,
                    source_sphere_path,
                    temp_dir / "source.stencil",
                    "source stencil",
                ),
                {
                    "builder_options": (
                        "--threads",
                        str(stencil_threads),
                        "--surface",
                        str(source_surface_path),
                    )
                },
            )
        else:
            prepared["source_stencil"] = _prepare_artifact(
                source_stencil_path,
                None,
                source_sphere_path,
                temp_dir / "source.stencil",
                "source stencil",
            )
        if target_stencil_path is None:
            prepare_specs["target_stencil"] = (
                _timed_prepare,
                (
                    _prepare_artifact,
                    None,
                    stencil_builder,
                    target_sphere_path,
                    temp_dir / "target.stencil",
                    "target stencil",
                ),
                {
                    "builder_options": (
                        "--threads",
                        str(stencil_threads),
                        "--surface",
                        str(target_surface_path),
                    )
                },
            )
        else:
            prepared["target_stencil"] = _prepare_artifact(
                target_stencil_path,
                None,
                target_sphere_path,
                temp_dir / "target.stencil",
                "target stencil",
            )
        if rotation_values_path is None:
            if rotation_feature_backend == "official-cpu":
                if rotation_values_probe is None:
                    raise ValueError(
                        "official-cpu rotation features require rotation_values_probe"
                    )
                prepare_specs["rotation_values"] = (
                    _timed_prepare,
                    (
                        _prepare_rotation_values,
                        rotation_values_probe,
                        source_surface_path,
                        source_sphere_path,
                        target_surface_path,
                        target_sphere_path,
                        temp_dir / "rotation-values.bin",
                    ),
                    {},
                )
            elif rotation_feature_backend in {
                "cuda-official-geometry",
                "cuda-official-depth",
            }:
                if rotation_feature_backend == "cuda-official-geometry":
                    if rotation_geometry_probe is None:
                        raise ValueError(
                            "cuda-official-geometry requires rotation_geometry_probe"
                        )
                elif rotation_depth_probe is None:
                    raise ValueError(
                        "cuda-official-depth requires rotation_depth_probe"
                    )

                if (
                    rotation_feature_backend == "cuda-official-geometry"
                    or rotation_geometry_probe is not None
                ):
                    prepare_specs["rotation_geometry_source"] = (
                        _timed_prepare,
                        (
                            _prepare_rotation_geometry,
                            rotation_geometry_probe,
                            source_surface_path,
                            source_sphere_path,
                            15.0,
                            temp_dir / "rotation-geometry-source.bin",
                        ),
                        {},
                    )
                    prepare_specs["rotation_geometry_target"] = (
                        _timed_prepare,
                        (
                            _prepare_rotation_geometry,
                            rotation_geometry_probe,
                            target_surface_path,
                            target_sphere_path,
                            10.0,
                            temp_dir / "rotation-geometry-target.bin",
                        ),
                        {},
                    )
                if rotation_feature_backend == "cuda-official-depth":
                    prepare_specs["rotation_depth_values"] = (
                        _timed_prepare,
                        (
                            _prepare_rotation_values,
                            rotation_depth_probe,
                            source_surface_path,
                            source_sphere_path,
                            target_surface_path,
                            target_sphere_path,
                            temp_dir / "rotation-depth-values.bin",
                        ),
                        {"raw_depth": True},
                    )
            elif rotation_values_probe is not None:
                raise ValueError(
                    "rotation_values_probe is used only with rotation_feature_backend='official-cpu'"
                )
        else:
            rotation_values_path = Path(rotation_values_path)
            if not rotation_values_path.is_file():
                raise FileNotFoundError(
                    f"Rotation values file does not exist: {rotation_values_path}"
                )
            prepared["rotation_values"] = rotation_values_path

        prepare_executor = None
        prepare_futures: dict[str, object] = {}
        if prepare_specs:
            prepare_executor = ThreadPoolExecutor(max_workers=len(prepare_specs))
            prepare_futures = {
                key: prepare_executor.submit(function, *args, **kwargs)
                for key, (function, args, kwargs) in prepare_specs.items()
            }

        for key in ("source_stencil", "target_stencil"):
            future = prepare_futures.get(key)
            if future is not None:
                prepared[key], elapsed = future.result()
                timings[f"{key}_seconds"] = float(elapsed)

        if "source_stencil" not in prepared or "target_stencil" not in prepared:
            raise RuntimeError("Source/target stencil")
        source_stencil = prepared["source_stencil"]
        target_stencil = prepared["target_stencil"]
        source_stencil_data = SurfaceStencil.from_file(source_stencil)
        target_stencil_data = SurfaceStencil.from_file(target_stencil)

        index_ready = False
        if (
            rotation_feature_backend == "official-cpu"
            or rotation_values_path is not None
        ):
            index_start = time.perf_counter()
            rotation_pipeline = RotationPipeline.from_stencil(
                target_stencil_data,
                device=target_device,
                grid_size=rotation_grid_size,
                margin=rotation_margin,
            )
            _synchronize(target_device)
            timings["rotation_index_upload_seconds"] = time.perf_counter() - index_start
            index_ready = True

        for key, future in prepare_futures.items():
            if key in prepared:
                continue
            prepared[key], elapsed = future.result()
            timings[f"{key}_seconds"] = float(elapsed)
        if prepare_executor is not None:
            prepare_executor.shutdown(wait=True)
        if rotation_values_path is None and "rotation_values" in prepared:
            rotation_values_path = prepared["rotation_values"]
        if not prepare_specs:
            timings["source_stencil_seconds"] = 0.0
            timings["target_stencil_seconds"] = 0.0
            timings["rotation_values_seconds"] = 0.0
        timings["initial_prepare_seconds"] = max(
            timings.get("source_stencil_seconds", 0.0),
            timings.get("target_stencil_seconds", 0.0),
            timings.get("rotation_values_seconds", 0.0),
            timings.get("rotation_geometry_source_seconds", 0.0),
            timings.get("rotation_geometry_target_seconds", 0.0),
            timings.get("rotation_depth_values_seconds", 0.0),
        )
        timings["initial_stencil_seconds"] = max(
            timings.get("source_stencil_seconds", 0.0),
            timings.get("target_stencil_seconds", 0.0),
        )
        timings["rotation_feature_seconds"] = max(
            timings.get("rotation_values_seconds", 0.0),
            timings.get("rotation_depth_values_seconds", 0.0),
        )
        rotation_geometry = None
        if rotation_feature_backend in {
            "cuda-official-geometry",
            "cuda-official-depth",
        }:
            if rotation_feature_backend == "cuda-official-depth" and (
                "rotation_geometry_source" not in prepared
            ):
                raw_depth_path = prepared["rotation_depth_values"]
                prepared["rotation_geometry_source"] = Path(
                    f"{raw_depth_path}.source.geometry"
                )
                prepared["rotation_geometry_target"] = Path(
                    f"{raw_depth_path}.target.geometry"
                )
                for geometry_path in (
                    prepared["rotation_geometry_source"],
                    prepared["rotation_geometry_target"],
                ):
                    if not geometry_path.is_file() or geometry_path.stat().st_size < 4:
                        raise RuntimeError(
                            f"Raw depth helper geometry sidecar:{geometry_path}"
                        )
            rotation_geometry = (
                _read_rotation_geometry(prepared["rotation_geometry_source"]),
                _read_rotation_geometry(prepared["rotation_geometry_target"]),
            )
            timings["rotation_feature_seconds"] = max(
                timings.get("rotation_geometry_source_seconds", 0.0),
                timings.get("rotation_geometry_target_seconds", 0.0),
                timings.get("rotation_depth_values_seconds", 0.0),
            )
        target_feature_stencil = None
        rotation_depth_values = None
        if rotation_feature_backend == "cuda-official-depth":
            rotation_depth_values = read_rotation_values(
                prepared["rotation_depth_values"]
            )
        if rotation_values_path is not None:
            source_values, target_values = read_rotation_values(rotation_values_path)
        else:
            feature_start = time.perf_counter()
            source_feature_stencil = source_stencil_data.to(
                target_device, geometry_dtype=torch.float32
            )
            target_feature_stencil = target_stencil_data.to(
                target_device, geometry_dtype=torch.float32
            )
            source_surface_for_feature = torch.as_tensor(
                source_surface.vertices,
                dtype=torch.float32,
                device=target_device,
            ).contiguous()
            target_surface_for_feature = torch.as_tensor(
                target_surface.vertices,
                dtype=torch.float32,
                device=target_device,
            ).contiguous()
            feature_results = compute_rotation_features(
                (source_feature_stencil, target_feature_stencil),
                (source_surface_for_feature, target_surface_for_feature),
                heat_fwhms=(15.0, 10.0),
                smoothed_surfaces=(
                    None
                    if rotation_geometry is None
                    else tuple(
                        torch.as_tensor(
                            item, dtype=torch.float32, device=target_device
                        ).contiguous()
                        for item in rotation_geometry
                    )
                ),
                depth_values=(
                    None
                    if rotation_depth_values is None
                    else tuple(
                        torch.as_tensor(
                            item, dtype=torch.float64, device=target_device
                        ).contiguous()
                        for item in rotation_depth_values
                    )
                ),
                solver=rotation_feature_solver,
            )
            _synchronize(target_device)
            feature_compute_seconds = time.perf_counter() - feature_start
            timings["rotation_feature_compute_seconds"] = feature_compute_seconds
            if rotation_feature_backend not in {
                "cuda-official-geometry",
                "cuda-official-depth",
            }:
                timings["rotation_feature_seconds"] = feature_compute_seconds
            timings["rotation_feature_source_iterations"] = float(
                feature_results[0].iterations
            )
            timings["rotation_feature_target_iterations"] = float(
                feature_results[1].iterations
            )
            timings["rotation_feature_source_residual"] = float(
                feature_results[0].relative_residual
            )
            timings["rotation_feature_target_residual"] = float(
                feature_results[1].relative_residual
            )
            source_values = feature_results[0].values
            target_values = feature_results[1].values

        if not index_ready:
            index_start = time.perf_counter()
            rotation_pipeline = RotationPipeline.from_stencil(
                target_stencil_data,
                device=target_device,
                grid_size=rotation_grid_size,
                margin=rotation_margin,
            )
            _synchronize(target_device)
            timings["rotation_index_upload_seconds"] = time.perf_counter() - index_start

        search_start = time.perf_counter()
        rotation_result = rotation_pipeline.search(
            source_stencil_data.sphere_points,
            source_values,
            target_values,
            point_chunk=point_chunk,
            refine=rotation_refine,
            max_iter=rotation_max_iter,
            tol=rotation_tol,
            simplex_step=rotation_simplex_step,
        )
        _synchronize(target_device)
        timings["rotation_search_seconds"] = time.perf_counter() - search_start

        angle = rotation_result.angle.detach().cpu().numpy()
        current_sphere = _rotate_vertices(
            _normalise_sphere(source_sphere.vertices), angle
        )
        current_stencil_path = temp_dir / "rotated-000.stencil"
        rotated_sphere_path = temp_dir / "rotated-000.gii"
        write_gifti_mesh(
            rotated_sphere_path,
            GiftiMesh(current_sphere, source_sphere.faces),
            reference_path=source_sphere_path,
        )

        pole_angles = np.asarray((0.0, np.pi / 2.0, 0.0), dtype=np.float64)
        avg_target_surface = _rotate_vertices(target_surface.vertices, pole_angles)
        avg_target_sphere = _rotate_vertices(
            _normalise_sphere(target_sphere.vertices), pole_angles
        )
        avg_target_sphere_path = temp_dir / "avg-target.gii"
        write_gifti_mesh(
            avg_target_sphere_path,
            GiftiMesh(avg_target_sphere, target_sphere.faces),
            reference_path=target_sphere_path,
        )

        graph_executor = None
        graph_future = None
        if graph_requested:
            dummy_maps = torch.zeros(
                (steps, target_stencil_data.ny, target_stencil_data.nx),
                dtype=solve_dtype,
                device=target_device,
            ).contiguous()
            graph_executor = ThreadPoolExecutor(max_workers=1)
            graph_future = graph_executor.submit(
                _warmup_and_prepare_dartel_graph,
                dummy_maps,
                loop=loop,
                cycles=cycles,
                nit=nit,
                its=its,
                code=code,
                kernel=kernel,
                device=target_device,
                dtype=solve_dtype,
                squaring_kernel=squaring_kernel,
                optimized=optimized_dartel,
            )

        if rotated_stencil_builder is None:
            raise ValueError(
                "The rotated source stencil requires an explicit rotated_stencil_builder"
            )
        rotated_stencil_started = time.perf_counter()
        rotated_specs: dict[
            str, tuple[object, tuple[object, ...], dict[str, object]]
        ] = {
            "rotated_source_stencil": (
                _timed_prepare,
                (
                    _prepare_artifact,
                    None,
                    rotated_stencil_builder,
                    rotated_sphere_path,
                    current_stencil_path,
                    "Rotation source stencil",
                ),
                {
                    "builder_options": (
                        "--no-normalize",
                        "--threads",
                        str(stencil_threads),
                        "--surface",
                        str(source_surface_path),
                    )
                },
            )
        }
        if avg:
            rotated_specs["avg_target_stencil"] = (
                _timed_prepare,
                (
                    _prepare_artifact,
                    None,
                    rotated_stencil_builder,
                    avg_target_sphere_path,
                    temp_dir / "avg-target.stencil",
                    "avg rotated target stencil",
                ),
                {
                    "builder_options": (
                        "--no-normalize",
                        "--threads",
                        str(stencil_threads),
                        "--surface",
                        str(target_surface_path),
                    )
                },
            )
        with ThreadPoolExecutor(max_workers=len(rotated_specs)) as executor:
            futures = {
                key: executor.submit(function, *args, **kwargs)
                for key, (function, args, kwargs) in rotated_specs.items()
            }
            rotated_results: dict[str, Path] = {}
            for key, future in futures.items():
                rotated_results[key], elapsed = future.result()
                timings[f"{key}_seconds"] = float(elapsed)
        timings["rotated_stencil_seconds"] = (
            time.perf_counter() - rotated_stencil_started
        )
        graph_enabled = False
        if graph_future is not None:
            try:
                capture_seconds, prepare_seconds = graph_future.result()
            finally:
                if graph_executor is not None:
                    graph_executor.shutdown(wait=True)
            timings["cuda_graph_capture_seconds"] = capture_seconds
            timings["cuda_graph_prepare_seconds"] = prepare_seconds
            graph_enabled = True
        else:
            timings["cuda_graph_capture_seconds"] = 0.0
            timings["cuda_graph_prepare_seconds"] = 0.0
        current_stencil_path = rotated_results["rotated_source_stencil"]

        if target_feature_stencil is not None:
            target_stencil_device = target_feature_stencil
        else:
            target_stencil_device = target_stencil_data.to(
                target_device, geometry_dtype=torch.float32
            )
        current_stencil_data = SurfaceStencil.from_file(current_stencil_path)
        current_stencil_device = current_stencil_data.to(
            target_device, geometry_dtype=torch.float32
        )
        source_surface_device = torch.as_tensor(
            source_surface.vertices, dtype=torch.float32, device=target_device
        ).contiguous()
        target_surface_device = torch.as_tensor(
            target_surface.vertices, dtype=torch.float32, device=target_device
        ).contiguous()

        avg_target_stencil_device = None
        avg_target_surface_device = None
        if avg:
            avg_target_data = SurfaceStencil.from_file(
                rotated_results["avg_target_stencil"]
            )
            avg_target_stencil_device = avg_target_data.to(
                target_device, geometry_dtype=torch.float32
            )
            avg_target_surface_device = torch.as_tensor(
                avg_target_surface, dtype=torch.float32, device=target_device
            ).contiguous()

        def solve_one(
            source_vertices: torch.Tensor,
            target_vertices: torch.Tensor,
            source_stencil_device,
            target_stencil_device_local,
            solve_fwhm: Sequence[float],
            solve_loop: int,
            use_cuda_graph: bool = False,
        ):
            """Solve one."""

            return solve_dartel_from_surfaces(
                source_vertices,
                target_vertices,
                source_stencil_device,
                target_stencil_device_local,
                fwhm=solve_fwhm,
                curvtypes=tuple(curvtypes),
                n_steps=steps,
                loop=solve_loop,
                cycles=cycles,
                nit=nit,
                its=its,
                code=code,
                kernel=kernel,
                device=target_device,
                dtype=solve_dtype,
                squaring_kernel=squaring_kernel,
                parallel_sides=parallel_sides,
                optimized=optimized_dartel,
                cuda_graph=use_cuda_graph,
            )

        warmup_start = time.perf_counter()
        solve_one(
            source_surface_device,
            target_surface_device,
            current_stencil_device,
            target_stencil_device,
            tuple(fwhm),
            1,
            use_cuda_graph=False,
        )
        _synchronize(target_device)
        timings["rotation_warmup_seconds"] = time.perf_counter() - warmup_start

        graph_ready = graph_enabled

        dartel_start = time.perf_counter()
        last_result = None
        final_prewarp_sphere: np.ndarray | None = None
        avg_source_stencil_executor = None
        avg_source_stencil_future = None
        avg_source_surface = None
        avg_source_sphere = None
        avg_source_sphere_path = None
        for run_index in range(runs):
            run_start = time.perf_counter()
            base_fwhm = float(fwhm[0])
            run_fwhm = tuple(
                base_fwhm / (3.0 ** (steps * (run_index + 1) + index))
                for index in range(steps)
            )
            if avg and run_index == runs - 1:
                final_prewarp_sphere = current_sphere.copy()
                avg_source_surface = _rotate_vertices(
                    source_surface.vertices, pole_angles
                )
                avg_source_sphere = _rotate_vertices(final_prewarp_sphere, pole_angles)
                avg_source_sphere_path = temp_dir / "avg-source.gii"
                write_gifti_mesh(
                    avg_source_sphere_path,
                    GiftiMesh(avg_source_sphere, source_sphere.faces),
                    reference_path=source_sphere_path,
                )
                avg_source_stencil_executor = ThreadPoolExecutor(max_workers=1)
                avg_source_stencil_future = avg_source_stencil_executor.submit(
                    _timed_prepare,
                    _prepare_artifact,
                    None,
                    rotated_stencil_builder,
                    avg_source_sphere_path,
                    temp_dir / "avg-source.stencil",
                    "avg rotated source stencil",
                    builder_options=(
                        "--no-normalize",
                        "--threads",
                        str(stencil_threads),
                        "--surface",
                        str(source_surface_path),
                    ),
                )
            solve = solve_one(
                source_surface_device,
                target_surface_device,
                current_stencil_device,
                target_stencil_device,
                run_fwhm,
                loop,
                use_cuda_graph=graph_enabled and graph_ready,
            )
            current_sphere_tensor = torch.as_tensor(
                current_sphere, dtype=torch.float32, device=target_device
            )
            if not (avg and run_index == runs - 1):
                current_sphere = (
                    apply_flow_to_stenciled_sphere(
                        current_sphere_tensor,
                        solve.flow,
                        current_stencil_device,
                        inverse=True,
                        device=target_device,
                        dtype=solve_dtype,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            _synchronize(target_device)
            timings[f"dartel_run_{run_index}_seconds"] = time.perf_counter() - run_start
            last_result = solve
            if run_index + 1 < runs:
                next_sphere_path = temp_dir / f"rotated-{run_index + 1:03d}.gii"
                write_gifti_mesh(
                    next_sphere_path,
                    GiftiMesh(current_sphere, source_sphere.faces),
                    reference_path=source_sphere_path,
                )
                next_stencil_path = temp_dir / f"rotated-{run_index + 1:03d}.stencil"
                _prepare_artifact(
                    None,
                    rotated_stencil_builder,
                    next_sphere_path,
                    next_stencil_path,
                    f"{run_index + 1} run source stencil",
                    builder_options=(
                        "--no-normalize",
                        "--threads",
                        str(stencil_threads),
                        "--surface",
                        str(source_surface_path),
                    ),
                )
                current_stencil_device = SurfaceStencil.from_file(next_stencil_path).to(
                    target_device, geometry_dtype=torch.float32
                )
        timings["source_dartel_total_seconds"] = time.perf_counter() - dartel_start

        if last_result is None:
            raise RuntimeError("GPU DARTEL")

        if avg:
            if avg_target_stencil_device is None or avg_target_surface_device is None:
                raise RuntimeError("Avg target stencil")
            if final_prewarp_sphere is None:
                raise RuntimeError("Average mode is missing the warped source sphere")
            if avg_source_stencil_future is None:
                raise RuntimeError("Avg source stencil start")
            try:
                avg_source_stencil_path, avg_stencil_elapsed = (
                    avg_source_stencil_future.result()
                )
            finally:
                if avg_source_stencil_executor is not None:
                    avg_source_stencil_executor.shutdown(wait=True)
            timings["avg_source_stencil_seconds"] = float(avg_stencil_elapsed)
            if avg_source_surface is None or avg_source_sphere is None:
                raise RuntimeError("Avg source geometry")
            avg_source_stencil_device = SurfaceStencil.from_file(
                avg_source_stencil_path
            ).to(target_device, geometry_dtype=torch.float32)
            avg_source_surface_device = torch.as_tensor(
                avg_source_surface, dtype=torch.float32, device=target_device
            ).contiguous()
            avg_source_sphere_device = torch.as_tensor(
                avg_source_sphere, dtype=torch.float32, device=target_device
            ).contiguous()
            avg_fwhm = tuple(
                float(fwhm[0]) / (3.0 ** (steps * (runs + 1) + index))
                for index in range(steps)
            )
            avg_solve_start = time.perf_counter()
            avg_result = solve_one(
                avg_source_surface_device,
                avg_target_surface_device,
                avg_source_stencil_device,
                avg_target_stencil_device,
                avg_fwhm,
                loop,
                use_cuda_graph=graph_enabled and graph_ready,
            )
            _synchronize(target_device)
            timings["avg_dartel_seconds"] = time.perf_counter() - avg_solve_start

            apply_start = time.perf_counter()
            main_warped = apply_flow_to_stenciled_sphere(
                torch.as_tensor(
                    final_prewarp_sphere, dtype=torch.float32, device=target_device
                ),
                last_result.flow,
                current_stencil_device,
                inverse=True,
                device=target_device,
                dtype=solve_dtype,
            )
            avg_warped = apply_flow_to_stenciled_sphere(
                avg_source_sphere_device,
                avg_result.flow,
                avg_source_stencil_device,
                inverse=True,
                device=target_device,
                dtype=solve_dtype,
            )
            _synchronize(target_device)
            timings["avg_apply_seconds"] = time.perf_counter() - apply_start

            combine_start = time.perf_counter()
            reverse_matrix = _rotation_matrix_from_angles(-pole_angles).to(
                device=target_device
            )
            unrotated_avg = avg_warped.to(torch.float64) @ reverse_matrix.transpose(
                0, 1
            )
            combined = _average_xz_surfaces(
                unrotated_avg,
                main_warped,
                device=target_device,
            )
            _synchronize(target_device)
            current_sphere = (
                combined.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            timings["average_combine_seconds"] = time.perf_counter() - combine_start

        timings["dartel_total_seconds"] = time.perf_counter() - dartel_start
        write_start = time.perf_counter()
        final_mesh = GiftiMesh(current_sphere, source_sphere.faces)
        write_gifti_mesh(output, final_mesh, reference_path=source_sphere_path)
        timings["output_write_seconds"] = time.perf_counter() - write_start

    timings["total_seconds"] = time.perf_counter() - total_start
    return CatSurfaceGpuResult(
        vertices=final_mesh.vertices,
        faces=final_mesh.faces,
        rotation_angle=np.ascontiguousarray(angle, dtype=np.float64),
        rotation_cost=float(rotation_result.cost.detach().cpu()),
        timings=timings,
        output_path=output,
    )
