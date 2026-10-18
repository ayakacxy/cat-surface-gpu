"""按设备调用球面计算，保留原 CUDA 数组接口。"""

from .io import GiftiSurface
from .parameters import Surf2SphereParameters


def surf_to_sphere(
    vertices, faces, parameters=None, topology=None, *, device="cuda", progress=None
):
    """验证输入和输出，再调用显式选择的计算后端。"""
    from .runtime import resolve_device

    device = resolve_device(device)
    GiftiSurface(vertices, faces)
    parameters = parameters or Surf2SphereParameters()
    if topology is None:
        from .templates import fixed_surface_topology

        topology = fixed_surface_topology(faces, len(vertices))
    if device == "cpu":
        from .backends.cpu.surf2sphere import surf_to_sphere_cpu as compute
    elif device == "metal":
        from .backends.metal.surf2sphere import surf_to_sphere_metal as compute
    elif device == "cuda":
        from .backends.cuda.surf2sphere import surf_to_sphere_cupy as compute
    else:
        raise ValueError(
            "device must be auto, cpu, metal or cuda; select a CUDA device using its context"
        )
    if progress:
        progress(f"Surf2Sphere: {device} stages 1–{parameters.stop_at}")
        if parameters.stop_at >= 6:
            progress(
                f"Surf2Sphere: area smoothing ({parameters.areal_iterations - 1} sweeps)"
            )
    output, timings = compute(vertices, faces, parameters, topology)
    GiftiSurface(output, faces)
    return output, timings


def surf_to_sphere_cupy(
    vertices, faces, parameters=None, topology=None, *, progress=None
):
    """保留 v0.2 CUDA 调用方式，使用当前 CUDA 上下文。"""
    return surf_to_sphere(
        vertices, faces, parameters, topology, device="cuda", progress=progress
    )
