"在线程组内按顺序更新的 FP32 分块 CAT 面积平滑。每块拥有互不重叠的输出顶点，包含颜色顺序更新所需的传递依赖；边界依赖在块内重算。每块读取前一完整迭代，因此无需跨线程组屏障或原子自旋等待。几何和选择核将整球保留在 GPU。"

from __future__ import annotations

from functools import lru_cache
import hashlib
import sys
import time

import numpy as np
from numba import njit
from numba.typed import List


@njit(cache=True)
def _dependency_patches(neighbours, degree, colors, size):
    "分割顶点，并补齐每个分块对较早颜色的依赖。"
    count = len(neighbours)
    used = np.zeros(count, np.bool_)
    queued = np.zeros(count, np.int32)
    needed = np.full(count, -1, np.int32)
    queue = np.empty(count * 2, np.int32)
    patches = List()
    stamp = 0
    for seed in range(count):
        if used[seed]:
            continue
        stamp += 1
        targets = np.empty(size, np.int32)
        queue[0] = seed
        queued[seed] = stamp
        head, tail, owned = 0, 1, 0
        while head < tail and owned < size:
            vertex = queue[head]
            head += 1
            if used[vertex]:
                continue
            used[vertex] = True
            targets[owned] = vertex
            owned += 1
            for slot in range(degree[vertex]):
                other = neighbours[vertex, slot]
                if not used[other] and queued[other] != stamp:
                    queued[other] = stamp
                    queue[tail] = other
                    tail += 1
        needed[:] = -1
        nodes = np.empty(count, np.int32)
        node_count = owned
        head, tail = 0, owned
        for i in range(owned):
            vertex = targets[i]
            needed[vertex] = 1
            nodes[i] = vertex
            queue[i] = vertex
        while head < tail:
            vertex = queue[head]
            head += 1
            for slot in range(degree[vertex]):
                other = neighbours[vertex, slot]
                required = int(colors[other] < colors[vertex])
                if needed[other] < 0:
                    nodes[node_count] = other
                    node_count += 1
                if required > needed[other]:
                    needed[other] = required
                    if required:
                        queue[tail] = other
                        tail += 1
        vertices = nodes[:node_count].copy()
        patches.append((vertices, needed[vertices].copy(), targets[:owned].copy()))
    return patches


@njit(cache=True)
def _pack_tables(
    patches, colors, faces, incident_faces, incident_mask, width, size, ncolors
):
    degree = incident_faces.shape[1]
    ids = np.full((len(patches), width), -1, np.int32)
    triangles = np.zeros((len(patches), width, degree), np.uint32)
    counts = np.zeros((len(patches), width), np.uint16)
    owned = np.full((len(patches), size), -1, np.int16)
    offsets = np.zeros((len(patches), ncolors + 1), np.uint16)
    local = np.empty(len(colors), np.int32)
    for patch, (nodes, active, targets) in enumerate(patches):
        row = 0
        for color in range(ncolors):
            for i in range(len(nodes)):
                vertex = nodes[i]
                if active[i] and colors[vertex] == color:
                    ids[patch, row] = vertex
                    local[vertex] = row
                    row += 1
            offsets[patch, color + 1] = row
        active_count = row
        for i in range(len(nodes)):
            if not active[i]:
                vertex = nodes[i]
                ids[patch, row] = vertex
                local[vertex] = row
                row += 1
        for i in range(len(targets)):
            owned[patch, i] = local[targets[i]]
        for i in range(active_count):
            vertex = ids[patch, i]
            slot = 0
            for j in range(degree):
                if incident_mask[vertex, j]:
                    face = faces[incident_faces[vertex, j]]
                    a, b, c = local[face[0]], local[face[1]], local[face[2]]
                    triangles[patch, i, slot] = a | (b << 10) | (c << 20)
                    slot += 1
            counts[patch, i] = slot
    return ids, triangles, counts, owned, offsets


def _make_tables(faces, topology, size=256):
    count = len(topology.neighbours)
    colors = np.full(count, -1, np.int32)
    for color, group in enumerate(topology.color_groups):
        if np.any(colors[group] >= 0):
            raise ValueError("CAT color groups contain duplicate vertices")
        colors[group] = color
    if np.any(colors < 0):
        raise ValueError("CAT color groups must cover every vertex")
    if np.any(
        (colors[topology.neighbours] == colors[:, None]) & topology.neighbour_mask
    ):
        raise ValueError("CAT color groups must contain independent vertices")
    patches = _dependency_patches(topology.neighbours, topology.degree, colors, size)
    width = max(len(patch[0]) for patch in patches)
    # 三个十位局部索引装入一个 uint；float3 占十六字节共享内存。
    if width > 1024:
        if size > 1:
            return _make_tables(faces, topology, max(1, size // 2))
        raise ValueError(
            "CAT dependency tile exceeds the Metal threadgroup memory budget"
        )
    ncolors = len(topology.color_groups)
    tables = _pack_tables(
        patches,
        colors,
        faces,
        topology.incident_faces,
        topology.incident_mask,
        width,
        size,
        ncolors,
    )
    return tables, width


_SOURCE = r"""
uint patch=threadgroup_position_in_grid.x, lane=thread_position_in_threadgroup.x;
uint W=uint(ids_shape[1]), D=uint(triangles_shape[2]), O=uint(owned_shape[1]);
uint C=uint(offsets_shape[1])-1;
threadgroup float3 positions[WIDTH];
for(uint i=lane;i<W;i+=128){
    int id=ids[patch*W+i];
    positions[i]=id>=0?float3(points[id*3],points[id*3+1],points[id*3+2]):float3(0.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint color=0;color<C;++color){
    uint begin=uint(offsets[patch*(C+1)+color]);
    uint end=uint(offsets[patch*(C+1)+color+1]);
    for(uint i=begin+lane;i<end;i+=128){
        uint row=patch*W+i;
        float total=0.0f; float3 weighted=0.0f;
        for(uint k=0;k<uint(counts[row]);++k){
            uint packed=triangles[row*D+k];
            float3 a=positions[packed&1023u];
            float3 b=positions[(packed>>10)&1023u];
            float3 c=positions[packed>>20];
            float area=0.5f*metal::length(metal::cross(b-a,c-a));
            area=metal::isfinite(area)?metal::clamp(area,0.0f,1.0f):0.0f;
            total+=area;
            weighted+=area*(a+b+c)/3.0f;
        }
        if(total>0.0f)positions[i]=weighted/total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint o=lane;o<O;o+=128){
    int i=owned[patch*O+o];
    if(i<0)continue;
    uint v=uint(ids[patch*W+uint(i)]);
    float3 p=positions[i];
    output[v*3]=p.x;output[v*3+1]=p.y;output[v*3+2]=p.z;
}
"""


_DISTANCE_SOURCE = r"""
uint patch=threadgroup_position_in_grid.x, lane=thread_position_in_threadgroup.x;
uint W=uint(ids_shape[1]), D=uint(rings_shape[2]), O=uint(owned_shape[1]);
uint C=uint(offsets_shape[1])-1;
threadgroup float3 positions[WIDTH];
for(uint i=lane;i<W;i+=128){
    int id=ids[patch*W+i];
    positions[i]=id>=0?float3(points[id*3],points[id*3+1],points[id*3+2]):float3(0.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint color=0;color<C;++color){
    uint begin=uint(offsets[patch*(C+1)+color]),end=uint(offsets[patch*(C+1)+color+1]);
    for(uint i=begin+lane;i<end;i+=128){
        uint row=patch*W+i;
        if(!selected[ids[row]])continue;
        float3 old=positions[i],weighted=0.0f;float total=0.0f;
        for(uint k=0;k<uint(counts[row]);++k){
            float3 other=positions[rings[row*D+k]],delta=metal::abs(other-old);
            float distance=(delta.x+delta.y)+delta.z;
            total+=distance;weighted+=distance*other;
        }
        if(total>0.0f)positions[i]=old*(1.0f-strength[0])+(weighted/total)*strength[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint o=lane;o<O;o+=128){
    int i=owned[patch*O+o];if(i<0)continue;
    uint v=uint(ids[patch*W+uint(i)]);float3 p=positions[i];
    output[v*3]=p.x;output[v*3+1]=p.y;output[v*3+2]=p.z;
}
"""


@njit(cache=True)
def _pack_rings(ids, offsets, neighbours, degree):
    rings = np.zeros((ids.shape[0], ids.shape[1], neighbours.shape[1]), np.uint16)
    local = np.empty(len(neighbours), np.int32)
    for patch in range(len(ids)):
        for i in range(ids.shape[1]):
            if ids[patch, i] >= 0:
                local[ids[patch, i]] = i
        for i in range(offsets[patch, -1]):
            vertex = ids[patch, i]
            for slot in range(degree[vertex]):
                rings[patch, i, slot] = local[neighbours[vertex, slot]]
    return rings


@lru_cache(maxsize=1)
def _mlx():
    if sys.platform != "darwin":
        raise RuntimeError("CAT Metal area smoothing requires macOS with a Metal GPU")
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("CAT Metal area smoothing requires an available Metal GPU")
    return mx


class _Plan:
    def __init__(self, faces, topology):
        mx = _mlx()
        from ...templates import fixed_metal_tables

        *arrays, self.host_rings = fixed_metal_tables(faces, topology)
        width = arrays[0].shape[1]
        self.ids, self.offsets = arrays[0], arrays[-1]
        self.width = width
        self.rings = None
        with mx.stream(mx.gpu):
            self.tables = [mx.array(array) for array in arrays]
        self.kernel = mx.fast.metal_kernel(
            name=f"cat_areal_ordered_tiles_{width}",
            input_names=["points", "ids", "triangles", "counts", "owned", "offsets"],
            output_names=["output"],
            source=_SOURCE.replace("WIDTH", str(width)),
            compile_options={"math_mode": "safe"},
        )
        self.batch = mx.compile(self._batch)
        self.short_batch = mx.compile(self._short_batch)
        self.table_bytes = sum(array.nbytes for array in arrays)
        mx.eval(self.tables)

    def step(self, points):
        mx = _mlx()
        return self.kernel(
            inputs=[points, *self.tables],
            output_shapes=[points.shape],
            output_dtypes=[mx.float32],
            grid=(self.tables[0].shape[0] * 128, 1, 1),
            threadgroup=(128, 1, 1),
            stream=mx.gpu,
        )[0]

    def _batch(self, points):
        for _ in range(250):
            points = self.step(points)
        return points

    def _short_batch(self, points):
        for _ in range(50):
            points = self.step(points)
        return points

    def prepare_distance(self, topology):
        if self.rings is not None:
            return
        mx = _mlx()
        self.rings = mx.array(self.host_rings)
        self.distance_kernel = mx.fast.metal_kernel(
            name=f"cat_distance_ordered_tiles_{self.width}",
            input_names=[
                "points",
                "ids",
                "rings",
                "counts",
                "owned",
                "offsets",
                "selected",
                "strength",
            ],
            output_names=["output"],
            source=_DISTANCE_SOURCE.replace("WIDTH", str(self.width)),
            compile_options={"math_mode": "safe"},
        )
        self.distance_batch = mx.compile(self._distance_batch)
        mx.eval(self.rings)

    def distance_step(self, points, selected, strength):
        mx = _mlx()
        ids, _, counts, owned, offsets = self.tables
        return self.distance_kernel(
            inputs=[
                points,
                ids,
                self.rings,
                counts,
                owned,
                offsets,
                selected,
                strength,
            ],
            output_shapes=[points.shape],
            output_dtypes=[mx.float32],
            grid=(ids.shape[0] * 128, 1, 1),
            threadgroup=(128, 1, 1),
            stream=mx.gpu,
        )[0]

    def _distance_batch(self, points, selected, strength):
        for _ in range(10):
            points = self.distance_step(points, selected, strength)
        return points

    def distance(self, points, selected, strength, iterations):
        mx = _mlx()
        strength = mx.array([strength], mx.float32)
        remaining = max(0, iterations - 1)
        while remaining >= 10:
            points = self.distance_batch(points, selected, strength)
            mx.eval(points)
            remaining -= 10
        for _ in range(remaining):
            points = self.distance_step(points, selected, strength)
        return points

    def areal(self, state, iterations):
        mx = _mlx()
        radius = mx.max(mx.sqrt(mx.sum(state * state, axis=1)))
        mx.eval(state, radius)
        done = 0
        while done < iterations - 1:
            remaining = min(1000 - done % 1000, iterations - 1 - done)
            while remaining >= 250:
                state = self.batch(state)
                mx.eval(state)
                done += 250
                remaining -= 250
            while remaining >= 50:
                state = self.short_batch(state)
                mx.eval(state)
                done += 50
                remaining -= 50
            for _ in range(remaining):
                state = self.step(state)
                done += 1
            if done % 1000 == 0:
                length = mx.sqrt(mx.sum(state * state, axis=1))
                state = (
                    state
                    * mx.where(length > 0, radius / mx.maximum(length, 1e-30), 1)[
                        :, None
                    ]
                )
        return state


_PLAN_CACHE = None


def _get_plan(faces, topology):
    global _PLAN_CACHE
    digest = hashlib.sha256()
    for array in (
        faces,
        topology.neighbours,
        topology.neighbour_mask,
        topology.incident_faces,
        topology.incident_mask,
        *topology.color_groups,
    ):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    key = digest.digest()
    if _PLAN_CACHE is None or _PLAN_CACHE[0] != key:
        _PLAN_CACHE = (key, _Plan(faces, topology))
    return _PLAN_CACHE[1]


def smooth_areal_metal(points, faces, topology, iterations):
    "原地更新 FP32 坐标，保留 CPU 颜色顺序和投影频率。单项有界缓存只保存拓扑；每次调用和每个半球均重建坐标、半径及迭代状态。"
    mx = _mlx()
    if points.dtype != np.float32 or points.shape != (len(topology.neighbours), 3):
        raise ValueError(
            "CAT Metal points must be an (N, 3) FP32 array matching the topology"
        )
    if not np.isfinite(points).all():
        raise ValueError("CAT Metal points must be finite")
    if iterations < 1:
        return {"metal_setup": 0.0, "metal_compute": 0.0, "metal_dispatches": 0}
    started = time.perf_counter()
    plan = _get_plan(faces, topology)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    with mx.stream(mx.gpu):
        state = mx.array(points)
        state = plan.areal(state, iterations)
        mx.eval(state)
        result = np.array(state)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "CAT Metal area smoothing produced non-finite coordinates"
        )
    points[:] = result
    return {
        "metal_setup": setup,
        "metal_compute": time.perf_counter() - started,
        "metal_dispatches": iterations - 1,
        "metal_table_bytes": plan.table_bytes,
    }


@lru_cache(maxsize=1)
def _geometry_kernels():
    mx = _mlx()
    specifications = {
        "areas": (
            ["points", "faces"],
            r"""
            uint f=thread_position_in_grid.x;if(f>=uint(faces_shape[0]))return;
            uint a=uint(faces[f*3]),b=uint(faces[f*3+1]),c=uint(faces[f*3+2]);
            float3 pa=float3(points[a*3],points[a*3+1],points[a*3+2]);
            float3 pb=float3(points[b*3],points[b*3+1],points[b*3+2]);
            float3 pc=float3(points[c*3],points[c*3+1],points[c*3+2]);
            float area=0.5f*metal::length(metal::cross(pb-pa,pc-pa));
            output[f]=metal::isfinite(area)?area:0.0f;
        """,
        ),
        "vertex_areas": (
            ["areas", "incident", "degree"],
            r"""
            uint v=thread_position_in_grid.x;if(v>=uint(incident_shape[0]))return;
            uint D=uint(incident_shape[1]);float total=0.0f;
            for(uint k=0;k<uint(degree[v]);++k)total+=areas[incident[v*D+k]]/3.0f;
            output[v]=total;
        """,
        ),
        "compression": (
            [
                "points",
                "reference",
                "areas",
                "reference_areas",
                "neighbours",
                "degree",
                "area_ratio",
            ],
            r"""
            uint v=thread_position_in_grid.x;if(v>=uint(points_shape[0]))return;
            uint D=uint(neighbours_shape[1]);float maximum=0.0f;
            float3 p=float3(points[v*3],points[v*3+1],points[v*3+2]);
            float3 r=float3(reference[v*3],reference[v*3+1],reference[v*3+2]);
            for(uint k=0;k<uint(degree[v]);++k){
                uint u=uint(neighbours[v*D+k]);
                float3 p2=float3(points[u*3],points[u*3+1],points[u*3+2]);
                float3 r2=float3(reference[u*3],reference[u*3+1],reference[u*3+2]);
                float original=metal::length(r2-r);
                if(original>0.0f)maximum=metal::max(maximum,metal::length(p2-p)/original);
            }
            float ratio=areas[v]>0.0f?reference_areas[v]/areas[v]:(reference_areas[v]!=0.0f?10000.0f:1.0f);
            output[v]=(maximum*metal::max(ratio,1e-8f))*area_ratio[0];
        """,
        ),
        "select": (
            ["compression", "neighbours", "degree", "threshold"],
            r"""
            uint v=thread_position_in_grid.x;if(v>=uint(neighbours_shape[0]))return;
            uint D=uint(neighbours_shape[1]);float total=0.0f;
            for(uint k=0;k<uint(degree[v]);++k)total+=compression[neighbours[v*D+k]];
            output[v]=(compression[v]+total)/(float(degree[v])+1.0f)>threshold[0];
        """,
        ),
    }
    return {
        name: mx.fast.metal_kernel(
            name=f"cat_geometry_{name}",
            input_names=inputs,
            output_names=["output"],
            source=source,
            compile_options={"math_mode": "safe"},
        )
        for name, (inputs, source) in specifications.items()
    }


def surf_to_sphere_metal(vertices, faces, parameters=None, topology=None):
    "通过保序 FP32 Metal 更新和 CPU 拓扑准备生成球面。"
    from cat_surface_gpu.parameters import Surf2SphereParameters
    from cat_surface_gpu.topology import build_surface_topology

    mx = _mlx()
    options = parameters or Surf2SphereParameters()
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("CAT Metal vertices must be finite with shape (N, 3)")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not faces.size
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("CAT Metal faces must be valid nonempty triangle indices")
    topology = topology or build_surface_topology(faces, len(vertices))
    if len(topology.neighbours) != len(vertices):
        raise ValueError("CAT Metal topology does not match the vertex count")
    total_started = time.perf_counter()
    with mx.stream(mx.gpu):
        plan = _get_plan(faces, topology)
        plan.prepare_distance(topology)
        kernels = _geometry_kernels()
        points, device_faces = mx.array(vertices), mx.array(faces)
        neighbours, degree = mx.array(topology.neighbours), mx.array(topology.degree)
        incident = mx.array(topology.incident_faces)
        incident_degree = mx.array(topology.incident_mask.sum(axis=1, dtype=np.int32))
        all_selected = mx.ones((len(vertices),), mx.bool_)

        def dispatch(name, inputs, count, dtype=mx.float32):
            return kernels[name](
                inputs=inputs,
                output_shapes=[(count,)],
                output_dtypes=[dtype],
                grid=(count, 1, 1),
                threadgroup=(128, 1, 1),
                stream=mx.gpu,
            )[0]

        def areas(state):
            face = dispatch("areas", [state, device_faces], len(faces))
            vertex = dispatch(
                "vertex_areas", [face, incident, incident_degree], len(vertices)
            )
            return mx.sum(face), vertex

        def project(state, area):
            axes = mx.maximum(
                (mx.abs(mx.min(state, axis=0)) + mx.abs(mx.max(state, axis=0))) * 0.5,
                1e-20,
            )
            factors = mx.sqrt(mx.sum((state / axes) ** 2, axis=1))
            factors = mx.where(factors != 0, factors, 1)
            radius = mx.sqrt(area / mx.array(4 * np.pi, mx.float32))
            return (state / factors[:, None]) * (radius / axes)

        desired_area, _ = areas(points)
        mx.eval(
            points,
            desired_area,
            neighbours,
            degree,
            incident,
            incident_degree,
            all_selected,
        )
        if float(desired_area.item()) <= 0:
            raise ValueError("CAT Metal surface must have positive area")
        timings = {"metal_setup": time.perf_counter() - total_started}
        factor = len(faces) / 350000.0 if len(faces) > 350000 else 1.0
        stages = (
            (1, "low_smooth", 1, 0.2, round(factor * 50), 1.0, 3.0, 0),
            (2, "inflate", 2, 1.0, round(factor * 30), 1.4, 3.0, 30),
            (3, "very_inflate", 4, 1.0, round(factor * 30), 1.1, 3.0, 0),
            (4, "high_smooth", 6, 1.0, round(factor * 60), 1.6, 3.0, 60),
            (5, "ellipsoid", 6, 1.0, round(factor * 50), 1.4, 4.0, 60),
        )
        for (
            index,
            name,
            cycles,
            strength,
            iterations,
            inflation,
            threshold,
            finger_iterations,
        ) in stages:
            if index > options.stop_at:
                break
            started = time.perf_counter()
            reference = points
            spans = mx.maximum(
                mx.max(reference, axis=0) - mx.min(reference, axis=0), 1e-20
            )
            points = points - mx.mean(points, axis=0)
            reference_area, reference_areas = areas(reference)
            for cycle in range(cycles + 1):
                if cycle < cycles:
                    points = plan.distance(points, all_selected, strength, iterations)
                    radius = mx.sqrt(mx.sum((points / spans) ** 2, axis=1))
                    points = points * (1 + (inflation - 1) * (1 - radius))[:, None]
                # 参考实现不使用最后一轮的选择结果。
                if cycle < cycles and finger_iterations > 0:
                    area, vertex_areas = areas(points)
                    compression = dispatch(
                        "compression",
                        [
                            points,
                            reference,
                            vertex_areas,
                            reference_areas,
                            neighbours,
                            degree,
                            mx.reshape(area / reference_area, (1,)),
                        ],
                        len(vertices),
                    )
                    selected = dispatch(
                        "select",
                        [
                            compression,
                            neighbours,
                            degree,
                            mx.array([threshold], mx.float32),
                        ],
                        len(vertices),
                        mx.bool_,
                    )
                    points = plan.distance(points, selected, 1.0, finger_iterations)
            mx.eval(points)
            timings[name] = time.perf_counter() - started
        if options.stop_at > 4:
            started = time.perf_counter()
            points = project(points, desired_area)
            mx.eval(points)
            timings["ellipsoid_projection"] = time.perf_counter() - started
        if options.areal_iterations:
            started = time.perf_counter()
            points = plan.areal(points, options.areal_iterations)
            mx.eval(points)
            timings["areal_smoothing"] = time.perf_counter() - started
            started = time.perf_counter()
            points = project(points, desired_area)
            mx.eval(points)
            timings["final_projection"] = time.perf_counter() - started
        result = np.array(points)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "CAT Metal sphere generation produced non-finite coordinates"
        )
    timings["total"] = time.perf_counter() - total_started
    return result, timings
