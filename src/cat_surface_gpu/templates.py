"按内容寻址缓存固定 CAT 目标的二进制准备结果。"

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

from functools import lru_cache
import hashlib
from importlib.metadata import version
import logging
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from cat_surface_gpu.runtime import cache_directory

_INCIDENCE_CACHE = {}


@lru_cache(maxsize=4)
def _implementation_digest(backend="cpu") -> bytes:
    "实现或数值依赖变化时使目标准备缓存失效。"
    root = Path(__file__).parent
    digest = hashlib.sha256(b"cat-target-platform-v3" + backend.encode())
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for package in ("numpy", "scipy", "nibabel") + (
        ("mlx",) if backend == "metal" else ()
    ):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not-installed"
        digest.update(f"{package}={installed}".encode())
    if backend == "cupy":
        import cupy as cp

        digest.update(
            f"cupy={cp.__version__};cuda={cp.cuda.runtime.runtimeGetVersion()}".encode()
        )
        digest.update(str(cp.cuda.Device().compute_capability).encode())
    elif backend not in ("cpu", "metal"):
        raise ValueError("Unsupported CAT cache backend")
    return digest.digest()


def _cached_arrays(kind, inputs, build, validate, *, backend="cpu"):
    "读取并验证缓存或原子重建；不缓存被试配准结果。"
    digest = hashlib.sha256(_implementation_digest(backend) + kind.encode())
    for value in inputs:
        array = np.ascontiguousarray(value)
        digest.update(repr((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    destination = cache_directory("cat-targets") / f"{kind}-{digest.hexdigest()}.npz"
    try:
        with np.load(destination, allow_pickle=False) as saved:
            arrays = tuple(saved[f"arr_{i}"] for i in range(len(saved.files)))
        validate(arrays)
        return arrays
    except FileNotFoundError:
        pass
    except (ValueError, KeyError, EOFError, BadZipFile) as error:
        logging.getLogger(__name__).warning(
            "Rebuilding invalid CAT target cache %s: %s", destination, error
        )
    arrays = tuple(np.ascontiguousarray(value) for value in build())
    validate(arrays)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            # ZIP_STORED 保存原始 NPY 数组，复用时无需文本解析或解压。
            np.savez(stream, *arrays)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return arrays


def load_template_geometry(path):
    "仅解码一次固定模板几何，保留原始压缩 GIFTI。"
    from .io import GiftiSurface, load_gifti_surface

    path = Path(path)
    content_digest = np.frombuffer(
        hashlib.sha256(path.read_bytes()).digest(), dtype=np.uint8
    )

    def validate(arrays):
        if (
            len(arrays) != 2
            or arrays[0].dtype != np.float32
            or arrays[1].dtype != np.int32
        ):
            raise ValueError("Invalid CAT template array types")
        GiftiSurface(*arrays)

    def build():
        surface = load_gifti_surface(path)
        return surface.vertices, surface.faces

    return GiftiSurface(*_cached_arrays("geometry", (content_digest,), build, validate))


def target_features(kind, vertices, faces, fwhms, build, *, backend="cupy"):
    "按几何、尺度和数值后端标识复用固定目标特征。"
    if kind not in ("depth", "type5"):
        raise ValueError("Unsupported CAT target feature kind")
    scales = np.asarray(fwhms, dtype=np.float64)

    def validate(arrays):
        if len(arrays) != len(scales):
            raise ValueError("Invalid CAT target feature count")
        for array in arrays:
            if (
                array.shape != (len(vertices),)
                or array.dtype != np.float64
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid CAT target feature array")

    return _cached_arrays(
        kind, (vertices, faces, scales), build, validate, backend=backend
    )


def fixed_surface_topology(faces, n_vertices):
    "复用固定粗模板已验证的整数邻环与颜色组。"
    from .topology import SurfaceTopology, build_surface_topology

    def validate(arrays):
        if len(arrays) < 5:
            raise ValueError("Incomplete CAT topology cache")
        neighbours, mask, incident, incident_mask = arrays[:4]
        for table, active, limit in (
            (neighbours, mask, n_vertices),
            (incident, incident_mask, len(faces)),
        ):
            if (
                table.ndim != 2
                or len(table) != n_vertices
                or table.dtype != np.int32
                or active.shape != table.shape
                or active.dtype != np.bool_
                or np.any(table < 0)
                or np.any(table >= limit)
                or not active.any(axis=1).all()
            ):
                raise ValueError("Invalid CAT topology tables")
        groups = arrays[4:]
        if any(group.ndim != 1 or group.dtype != np.int32 for group in groups):
            raise ValueError("Invalid CAT topology color groups")
        ids = np.concatenate(groups)
        if (
            len(ids) != n_vertices
            or ids.min() < 0
            or ids.max() >= n_vertices
            or not np.all(np.bincount(ids, minlength=n_vertices) == 1)
        ):
            raise ValueError("CAT topology colors must partition the vertices")

    def build():
        topology = build_surface_topology(faces, n_vertices)
        return (
            topology.neighbours,
            topology.neighbour_mask,
            topology.incident_faces,
            topology.incident_mask,
            *topology.color_groups,
        )

    arrays = _cached_arrays(
        "coarse-topology", (faces, np.asarray([n_vertices], np.int64)), build, validate
    )
    for array in arrays:
        array.setflags(write=False)
    return SurfaceTopology(*arrays[:4], tuple(arrays[4:]))


def fixed_resampling_topology(faces, n_vertices):
    "在连接关系相同的被试间复用整数关联表。"
    from .resampling import ResamplingTopology, build_resampling_topology

    faces = np.ascontiguousarray(faces, dtype=np.int32)
    key = (
        hashlib.sha256(memoryview(faces).cast("B")).digest(),
        faces.shape,
        int(n_vertices),
        _implementation_digest(),
    )
    if key in _INCIDENCE_CACHE:
        return _INCIDENCE_CACHE[key]

    def build():
        topology = build_resampling_topology(faces, n_vertices)
        return topology.incident_faces, topology.incident_counts

    def validate(arrays):
        if len(arrays) != 2:
            raise ValueError("Incomplete CAT resampling topology cache")
        table, counts = arrays
        if (
            table.ndim != 2
            or len(table) != n_vertices
            or table.dtype != np.int32
            or counts.shape != (n_vertices,)
            or counts.dtype != np.int32
            or np.any(counts < 0)
            or np.any(counts > table.shape[1])
        ):
            raise ValueError("Invalid CAT incidence table dimensions")
        active = np.arange(table.shape[1])[None, :] < counts[:, None]
        if np.any((table < 0) & active) or np.any((table >= len(faces)) & active):
            raise ValueError("Invalid CAT incident face indices")

    arrays = _cached_arrays(
        "incidence", (faces, np.asarray([n_vertices], np.int64)), build, validate
    )
    for array in arrays:
        array.setflags(write=False)
    result = ResamplingTopology(*arrays)
    if len(_INCIDENCE_CACHE) >= 4:
        del _INCIDENCE_CACHE[next(iter(_INCIDENCE_CACHE))]
    _INCIDENCE_CACHE[key] = result
    return result


def fixed_metal_tables(faces, topology):
    "复用整数依赖分块和邻环，不保存被试坐标。"
    from .backends.metal.surf2sphere import _make_tables, _pack_rings

    def build():
        arrays, _ = _make_tables(faces, topology)
        rings = _pack_rings(arrays[0], arrays[-1], topology.neighbours, topology.degree)
        return (*arrays, rings)

    def validate(arrays):
        if len(arrays) != 6:
            raise ValueError("Incomplete CAT Metal dependency tables")
        ids, triangles, counts, owned, offsets, rings = arrays
        if ids.ndim != 2 or not 0 < ids.shape[1] <= 1024:
            raise ValueError("Invalid CAT Metal tile width")
        patches, width = ids.shape
        degree = topology.neighbours.shape[1]
        expected = (
            (ids, np.int32, (patches, width)),
            (triangles, np.uint32, (patches, width, topology.incident_faces.shape[1])),
            (counts, np.uint16, (patches, width)),
            (offsets, np.uint16, (patches, len(topology.color_groups) + 1)),
            (rings, np.uint16, (patches, width, degree)),
        )
        if any(
            array.dtype != dtype or array.shape != shape
            for array, dtype, shape in expected
        ):
            raise ValueError("Invalid CAT Metal table dimensions or types")
        if owned.ndim != 2 or len(owned) != patches or owned.dtype != np.int16:
            raise ValueError("Invalid CAT Metal ownership table")
        if (
            np.any(ids < -1)
            or np.any(ids >= len(topology.neighbours))
            or np.any(owned < -1)
            or np.any(owned >= width)
            or np.any(rings >= width)
            or np.any(counts > triangles.shape[2])
            or np.any(offsets > width)
            or np.any(np.diff(offsets.astype(np.int32), axis=1) < 0)
            or np.any(offsets[:, 0] != 0)
        ):
            raise ValueError("Invalid CAT Metal table indices")
        for shift in (0, 10, 20):
            if np.any(((triangles >> shift) & 1023) >= width):
                raise ValueError("Invalid CAT Metal packed triangle index")
        owners = ids[np.arange(patches)[:, None], np.maximum(owned, 0)][owned >= 0]
        if (
            len(owners) != len(topology.neighbours)
            or np.any(owners < 0)
            or not np.all(np.bincount(owners, minlength=len(topology.neighbours)) == 1)
        ):
            raise ValueError("CAT Metal tiles must own every vertex exactly once")

    inputs = (
        faces,
        topology.neighbours,
        topology.neighbour_mask,
        topology.incident_faces,
        topology.incident_mask,
        *topology.color_groups,
    )
    return _cached_arrays("metal-tables", inputs, build, validate)


def fixed_dependency_tables(faces, topology):
    "缓存闭合依赖表，验证索引边界和输出顶点的唯一归属。"
    from cat_surface_gpu.backends.cuda.dependency_tiles import _make_tables, _pack_rings

    def build():
        arrays, _ = _make_tables(faces, topology)
        rings = _pack_rings(arrays[0], arrays[-1], topology.neighbours, topology.degree)
        return (*arrays, rings)

    def validate(arrays):
        if len(arrays) != 6:
            raise ValueError("Incomplete CAT CUDA dependency tables")
        ids, triangles, counts, owned, offsets, rings = arrays
        if ids.ndim != 2 or not 0 < ids.shape[1] <= 1024:
            raise ValueError("Invalid CAT CUDA tile width")
        patches, width = ids.shape
        degree = topology.neighbours.shape[1]
        expected = (
            (ids, np.int32, (patches, width)),
            (triangles, np.uint32, (patches, width, topology.incident_faces.shape[1])),
            (counts, np.uint16, (patches, width)),
            (offsets, np.uint16, (patches, len(topology.color_groups) + 1)),
            (rings, np.uint16, (patches, width, degree)),
        )
        if any(
            array.dtype != dtype or array.shape != shape
            for array, dtype, shape in expected
        ):
            raise ValueError("Invalid CAT CUDA table dimensions or types")
        if owned.ndim != 2 or len(owned) != patches or owned.dtype != np.int16:
            raise ValueError("Invalid CAT CUDA ownership table")
        if (
            np.any(ids < -1)
            or np.any(ids >= len(topology.neighbours))
            or np.any(owned < -1)
            or np.any(owned >= width)
            or np.any(rings >= width)
            or np.any(counts > triangles.shape[2])
            or np.any(offsets > width)
            or np.any(np.diff(offsets.astype(np.int32), axis=1) < 0)
            or np.any(offsets[:, 0] != 0)
        ):
            raise ValueError("Invalid CAT CUDA table indices")
        for shift in (0, 10, 20):
            if np.any(((triangles >> shift) & 1023) >= width):
                raise ValueError("Invalid CAT CUDA packed triangle index")
        owners = ids[np.arange(patches)[:, None], np.maximum(owned, 0)][owned >= 0]
        if (
            len(owners) != len(topology.neighbours)
            or np.any(owners < 0)
            or not np.all(np.bincount(owners, minlength=len(topology.neighbours)) == 1)
        ):
            raise ValueError("CAT CUDA tiles must own every vertex exactly once")

    inputs = (
        faces,
        topology.neighbours,
        topology.neighbour_mask,
        topology.incident_faces,
        topology.incident_mask,
        *topology.color_groups,
    )
    return _cached_arrays("dependency-tables", inputs, build, validate)
