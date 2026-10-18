"构建包含颜色闭合依赖的整数分块拓扑。"

from __future__ import annotations


import numpy as np


from numba import njit


from numba.typed import List


@njit(cache=True)
def _dependency_patches(neighbours, degree, colors, size):
    "收集每个输出分块按颜色排序的完整依赖。"
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
    # 共享内存和十位局部索引将分块宽度限制为一千零二十四。
    if width > 1024:
        if size > 1:
            return _make_tables(faces, topology, max(1, size // 2))
        raise ValueError("CAT dependency tile exceeds the CUDA shared memory budget")
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
