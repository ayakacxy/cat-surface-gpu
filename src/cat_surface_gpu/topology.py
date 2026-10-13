"""从三角形构建 CAT 球面算法复用的拓扑表。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class SurfaceTopology:
    """保存邻点环、入射面和无相邻顶点的颜色分组。"""

    neighbours: np.ndarray
    neighbour_mask: np.ndarray
    incident_faces: np.ndarray
    incident_mask: np.ndarray
    color_groups: tuple[np.ndarray, ...]

    @property
    def degree(self) -> np.ndarray:
        """返回每个顶点的邻点数。"""

        return self.neighbour_mask.sum(axis=1, dtype=np.int32)


def _cyclic_neighbours(
    faces: np.ndarray, incident: list[list[int]], vertex: int
) -> list[int]:
    """利用入射三角形在局部邻点图中恢复闭合环。"""

    graph: dict[int, set[int]] = {}
    for face_index in incident[vertex]:
        face = faces[face_index]
        pair = [int(item) for item in face if int(item) != vertex]
        if len(pair) != 2:
            raise ValueError(f"顶点 {vertex} 所在三角形退化")
        graph.setdefault(pair[0], set()).add(pair[1])
        graph.setdefault(pair[1], set()).add(pair[0])
    if not graph:
        return []
    if any(len(adjacent) != 2 for adjacent in graph.values()):
        raise ValueError(f"顶点 {vertex} 的邻域不是闭合二维流形")
    first = min(graph)
    second = min(graph[first])
    ring = [first, second]
    previous, current = first, second
    while len(ring) < len(graph):
        candidates = graph[current] - {previous}
        if len(candidates) != 1:
            raise ValueError(f"顶点 {vertex} 的邻点环存在歧义")
        following = next(iter(candidates))
        if following == first:
            break
        if following in ring:
            raise ValueError(f"顶点 {vertex} 的邻点环提前闭合")
        ring.append(following)
        previous, current = current, following
    if len(ring) != len(graph) or first not in graph[ring[-1]]:
        raise ValueError(f"顶点 {vertex} 的邻点环不闭合")
    return ring


def _color_groups(neighbours: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """贪心构建顶点独立集，供单次 CUDA 原地更新安全并行。"""

    colors = _greedy_colors_numba(neighbours, mask)
    return tuple(
        np.flatnonzero(colors == color).astype(np.int32, copy=False)
        for color in range(int(colors.max()) + 1)
    )


@njit(cache=True, nogil=True)
def _greedy_colors_numba(neighbours: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """以稳定的顶点顺序执行独立集着色。"""

    count, width = neighbours.shape
    colors = np.full(count, -1, dtype=np.int32)
    used = np.zeros(width + 1, dtype=np.uint8)
    for vertex in range(count):
        used[:] = 0
        for slot in range(width):
            if not mask[vertex, slot]:
                continue
            neighbour = neighbours[vertex, slot]
            if neighbour < 0:
                continue
            colour = colors[neighbour]
            if colour >= 0:
                used[colour] = 1
        colour = 0
        while colour < used.size and used[colour] != 0:
            colour += 1
        colors[vertex] = colour
    return colors


@njit(cache=True, nogil=True)
def _build_cyclic_neighbours_numba(
    faces: np.ndarray,
    incident_faces: np.ndarray,
    incident_counts: np.ndarray,
    output_width: int,
    adjacency_width: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """用定宽表重建闭合三角网格的稳定邻点环。"""

    count = incident_faces.shape[0]
    neighbours = np.zeros((count, output_width), dtype=np.int32)
    mask = np.zeros((count, output_width), dtype=np.bool_)
    for vertex in range(count):
        local_nodes = np.full(adjacency_width, -1, dtype=np.int32)
        local_adjacency = np.full(
            (adjacency_width, adjacency_width), -1, dtype=np.int32
        )
        node_count = 0
        for incident_index in range(incident_counts[vertex]):
            face = faces[incident_faces[vertex, incident_index]]
            if face[0] == vertex:
                first = face[1]
                second = face[2]
            elif face[1] == vertex:
                first = face[0]
                second = face[2]
            elif face[2] == vertex:
                first = face[0]
                second = face[1]
            else:
                return neighbours, mask, False

            first_slot = -1
            second_slot = -1
            for slot in range(node_count):
                if local_nodes[slot] == first:
                    first_slot = slot
                if local_nodes[slot] == second:
                    second_slot = slot
            if first_slot < 0:
                if node_count >= adjacency_width:
                    return neighbours, mask, False
                first_slot = node_count
                local_nodes[node_count] = first
                node_count += 1
            if second_slot < 0:
                if node_count >= adjacency_width:
                    return neighbours, mask, False
                second_slot = node_count
                local_nodes[node_count] = second
                node_count += 1

            exists = False
            for slot in range(adjacency_width):
                if local_adjacency[first_slot, slot] == second:
                    exists = True
                    break
            if not exists:
                first_degree = 0
                while (
                    first_degree < adjacency_width
                    and local_adjacency[first_slot, first_degree] >= 0
                ):
                    first_degree += 1
                second_degree = 0
                while (
                    second_degree < adjacency_width
                    and local_adjacency[second_slot, second_degree] >= 0
                ):
                    second_degree += 1
                if first_degree >= adjacency_width or second_degree >= adjacency_width:
                    return neighbours, mask, False
                local_adjacency[first_slot, first_degree] = second
                local_adjacency[second_slot, second_degree] = first

        if node_count != output_width or node_count < 3:
            return neighbours, mask, False
        for slot in range(node_count):
            degree = 0
            while degree < adjacency_width and local_adjacency[slot, degree] >= 0:
                degree += 1
            if degree != 2:
                return neighbours, mask, False

        first = local_nodes[0]
        for slot in range(1, node_count):
            if local_nodes[slot] < first:
                first = local_nodes[slot]
        first_slot = 0
        while local_nodes[first_slot] != first:
            first_slot += 1
        second = local_adjacency[first_slot, 0]
        if local_adjacency[first_slot, 1] < second:
            second = local_adjacency[first_slot, 1]

        ring_length = 0
        neighbours[vertex, ring_length] = first
        mask[vertex, ring_length] = True
        ring_length += 1
        neighbours[vertex, ring_length] = second
        mask[vertex, ring_length] = True
        ring_length += 1
        previous = first
        current = second
        while ring_length < node_count:
            current_slot = 0
            while local_nodes[current_slot] != current:
                current_slot += 1
            following = local_adjacency[current_slot, 0]
            if following == previous:
                following = local_adjacency[current_slot, 1]
            if following == first:
                break
            for slot in range(ring_length):
                if neighbours[vertex, slot] == following:
                    return neighbours, mask, False
            neighbours[vertex, ring_length] = following
            mask[vertex, ring_length] = True
            ring_length += 1
            previous = current
            current = following
        if ring_length != node_count:
            return neighbours, mask, False
        last_slot = 0
        while local_nodes[last_slot] != current:
            last_slot += 1
        closes = (
            local_adjacency[last_slot, 0] == first
            or local_adjacency[last_slot, 1] == first
        )
        if not closes:
            return neighbours, mask, False
    return neighbours, mask, True


def build_surface_topology(
    faces: np.ndarray,
    n_vertices: int | None = None,
) -> SurfaceTopology:
    """验证闭合三角网格并生成稳定的稠密拓扑表。"""

    triangles = np.ascontiguousarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("三角形数组必须是 (M, 3)")
    if triangles.size == 0:
        raise ValueError("三角形数组不能为空")
    inferred_vertices = int(triangles.max()) + 1
    count = inferred_vertices if n_vertices is None else int(n_vertices)
    if count < inferred_vertices or triangles.min() < 0:
        raise ValueError("三角形索引超出顶点范围")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    ):
        raise ValueError("网格包含退化三角形索引")

    # 先用稳定的 CSR 风格入射表和 Numba 环遍历；非法/非闭合网格回退到
    # 原始 Python 校验路径，以保留详细错误语义。
    flat_vertices = np.ascontiguousarray(triangles.ravel(), dtype=np.int32)
    flat_faces = np.repeat(np.arange(triangles.shape[0], dtype=np.int32), 3)
    order = np.argsort(flat_vertices, kind="stable")
    sorted_vertices = flat_vertices[order]
    sorted_faces = flat_faces[order]
    counts = np.bincount(sorted_vertices, minlength=count).astype(np.int32)
    max_incident = int(counts.max())
    starts = np.cumsum(counts, dtype=np.int64) - counts
    ranks = np.arange(sorted_vertices.size, dtype=np.int64) - np.repeat(starts, counts)
    incident_table = np.zeros((count, max_incident), dtype=np.int32)
    incident_table[sorted_vertices, ranks] = sorted_faces
    fast_neighbours, fast_mask, valid = _build_cyclic_neighbours_numba(
        triangles.astype(np.int32),
        incident_table,
        counts,
        max_incident,
        max_incident * 2,
    )
    if valid:
        incident_mask = np.arange(max_incident)[None, :] < counts[:, None]
        return SurfaceTopology(
            neighbours=fast_neighbours,
            neighbour_mask=fast_mask,
            incident_faces=incident_table,
            incident_mask=np.ascontiguousarray(incident_mask),
            color_groups=_color_groups(fast_neighbours, fast_mask),
        )

    incident: list[list[int]] = [[] for _ in range(count)]
    for face_index, face in enumerate(triangles):
        for vertex in face:
            incident[int(vertex)].append(face_index)
    if any(not items for items in incident):
        raise ValueError("网格包含孤立顶点")

    rings = [_cyclic_neighbours(triangles, incident, vertex) for vertex in range(count)]
    max_degree = max(map(len, rings))
    max_incident = max(map(len, incident))
    neighbours = np.zeros((count, max_degree), dtype=np.int32)
    neighbour_mask = np.zeros((count, max_degree), dtype=bool)
    incident_faces = np.zeros((count, max_incident), dtype=np.int32)
    incident_mask = np.zeros((count, max_incident), dtype=bool)
    for vertex, ring in enumerate(rings):
        neighbours[vertex, : len(ring)] = ring
        neighbour_mask[vertex, : len(ring)] = True
        items = incident[vertex]
        incident_faces[vertex, : len(items)] = items
        incident_mask[vertex, : len(items)] = True
    return SurfaceTopology(
        neighbours=neighbours,
        neighbour_mask=neighbour_mask,
        incident_faces=incident_faces,
        incident_mask=incident_mask,
        color_groups=_color_groups(neighbours, neighbour_mask),
    )
