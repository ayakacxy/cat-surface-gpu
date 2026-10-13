"""官方 CAT 初始刚体旋转的独立 CuPy 实现。

这里保留当前 CAT-Surface 的三个关键约定：

* 使用 ``rotation_to_matrix`` 的 X/Y/Z 欧拉角和矩阵布局；
* 用目标球面三角形上的最近点重心插值计算代价；
* 先执行 46 个固定 seed，再按 C 版本顺序执行 Nelder--Mead。

目标球面的空间索引只构建一次，之后所有角度候选都在 GPU 上批量定位，
避免为每个 Nelder--Mead 点重新建立 KD-tree 或三角形索引。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit


def official_rotation_seed_angles() -> np.ndarray:
    """返回官方初始旋转搜索的 46 个 seed 角度。"""

    seed0 = (-0.6, -0.3, 0.0, 0.3, 0.6)
    seed12 = (-0.3, 0.0, 0.3)
    values = [(0.0, 0.0, 0.0)]
    values.extend(
        (alpha, beta, gamma) for alpha in seed0 for beta in seed12 for gamma in seed12
    )
    return np.asarray(values, dtype=np.float64)


def rotation_matrix_from_angles(angles: np.ndarray) -> np.ndarray:
    """按官方 ``rotation_to_matrix`` 生成一个行优先三乘三矩阵。"""

    alpha, beta, gamma = np.asarray(angles, dtype=np.float64).reshape(3)
    rx = np.asarray(
        (
            1.0,
            0.0,
            0.0,
            0.0,
            math.cos(alpha),
            math.sin(alpha),
            0.0,
            -math.sin(alpha),
            math.cos(alpha),
        ),
        dtype=np.float64,
    )
    ry = np.asarray(
        (
            math.cos(beta),
            0.0,
            math.sin(beta),
            0.0,
            1.0,
            0.0,
            -math.sin(beta),
            0.0,
            math.cos(beta),
        ),
        dtype=np.float64,
    )
    rz = np.asarray(
        (
            math.cos(gamma),
            math.sin(gamma),
            0.0,
            -math.sin(gamma),
            math.cos(gamma),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        dtype=np.float64,
    )
    # C 源码虽把常量写成行优先数组，却按 i+3*j 访问；这里逐项保留
    # 该索引约定，随后 rotate_polygons 再按连续三元素读取。
    first = np.empty((3, 3), dtype=np.float64)
    result = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            first.flat[i + 3 * j] = sum(ry[i + 3 * k] * rx[k + 3 * j] for k in range(3))
    for i in range(3):
        for j in range(3):
            result.flat[i + 3 * j] = sum(
                rz[i + 3 * k] * first.flat[k + 3 * j] for k in range(3)
            )
    return np.ascontiguousarray(result)


def rotate_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """按 CAT 的行公式旋转顶点。"""

    points = np.asarray(vertices, dtype=np.float64)
    result = points @ np.asarray(matrix, dtype=np.float64).T
    return np.ascontiguousarray(result, dtype=np.float32)


@njit(cache=True, nogil=True)
def _pack_grid_cells(
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按原始面片/坐标循环顺序把面片压缩到空间网格。"""

    cell_count = int(grid_size) ** 3
    counts = np.zeros(cell_count, dtype=np.int32)
    for face_index in range(lower.shape[0]):
        for z in range(int(lower[face_index, 2]), int(upper[face_index, 2]) + 1):
            base = int(grid_size) * (z * int(grid_size))
            for y in range(int(lower[face_index, 1]), int(upper[face_index, 1]) + 1):
                row = base + int(grid_size) * y
                for x in range(
                    int(lower[face_index, 0]), int(upper[face_index, 0]) + 1
                ):
                    counts[row + x] += 1
    offsets = np.empty(cell_count + 1, dtype=np.int32)
    offsets[0] = 0
    for cell in range(cell_count):
        offsets[cell + 1] = offsets[cell] + counts[cell]
    packed = np.empty(int(offsets[cell_count]), dtype=np.int32)
    cursor = offsets[:-1].copy()
    for face_index in range(lower.shape[0]):
        for z in range(int(lower[face_index, 2]), int(upper[face_index, 2]) + 1):
            base = int(grid_size) * (z * int(grid_size))
            for y in range(int(lower[face_index, 1]), int(upper[face_index, 1]) + 1):
                row = base + int(grid_size) * y
                for x in range(
                    int(lower[face_index, 0]), int(upper[face_index, 0]) + 1
                ):
                    cell = row + x
                    packed[cursor[cell]] = face_index
                    cursor[cell] += 1
    return offsets, packed


@dataclass
class RotationGridIndex:
    """目标球面三角形的固定空间索引。"""

    points: np.ndarray
    faces: np.ndarray
    offsets: np.ndarray
    candidate_faces: np.ndarray
    grid_size: int
    margin: int
    scale: float

    @classmethod
    def from_geometry(
        cls,
        points: np.ndarray,
        faces: np.ndarray,
        *,
        grid_size: int = 128,
        margin: int = 1,
    ) -> "RotationGridIndex":
        """按目标三角形包围盒建立压缩网格索引。"""

        if grid_size < 8 or margin < 0:
            raise ValueError("grid_size 必须不小于 8，margin 不能为负数")
        target_points = np.ascontiguousarray(points, dtype=np.float32)
        target_faces = np.ascontiguousarray(faces, dtype=np.int32)
        if target_points.ndim != 2 or target_points.shape[1] != 3:
            raise ValueError("目标球面顶点必须是 [N,3]")
        if target_faces.ndim != 2 or target_faces.shape[1] != 3:
            raise ValueError("目标球面面片必须是 [M,3]")
        scale = float(np.max(np.abs(target_points.astype(np.float64))))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("目标球面尺度非法")

        triangles = target_points[target_faces].astype(np.float64)
        cell_size = 2.0 / float(grid_size)
        lower = np.floor((triangles.min(axis=1) / scale + 1.0) / cell_size)
        upper = np.floor((triangles.max(axis=1) / scale + 1.0) / cell_size)
        lower = np.clip(lower.astype(np.int64) - margin, 0, grid_size - 1)
        upper = np.clip(upper.astype(np.int64) + margin, 0, grid_size - 1)

        offsets, packed = _pack_grid_cells(lower, upper, grid_size)
        if packed.size == 0:
            raise ValueError("目标球面没有覆盖空间网格")
        return cls(
            points=target_points,
            faces=target_faces,
            offsets=np.ascontiguousarray(offsets),
            candidate_faces=np.ascontiguousarray(
                packed,
            ),
            grid_size=int(grid_size),
            margin=int(margin),
            scale=scale,
        )

    @property
    def max_candidates(self) -> int:
        """返回单个空间单元的最大候选面片数。"""

        return int(np.diff(self.offsets).max())

    def search(
        self,
        source_points: np.ndarray,
        source_values: np.ndarray,
        target_values: np.ndarray,
        *,
        point_chunk: int = 2048,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        refine: bool = True,
    ) -> "RotationSearchResult":
        """在 GPU 上执行官方 seed 和 Nelder--Mead 搜索。"""

        import cupy as cp

        source = cp.asarray(np.asarray(source_points, dtype=np.float32))
        source_map = cp.asarray(np.asarray(source_values, dtype=np.float64))
        target_map = cp.asarray(np.asarray(target_values, dtype=np.float64))
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source_points 必须是 [N,3]")
        if source_map.size != source.shape[0]:
            raise ValueError("source_values 长度与 source_points 不一致")
        if target_map.size != self.points.shape[0]:
            raise ValueError("target_values 长度与目标球面不一致")

        target_points = cp.asarray(self.points, dtype=cp.float32)
        target_faces = cp.asarray(self.faces, dtype=cp.int32)
        offsets = cp.asarray(self.offsets, dtype=cp.int32)
        candidate_faces = cp.asarray(self.candidate_faces, dtype=cp.int32)
        # 目标面片几何与被试无关。原实现会在每次 Nelder--Mead 代价
        # 评估中重复 gather、边向量和法向计算，这里在一次 search 内
        # 预计算并复用；目标曲率值仍按被试输入动态 gather。
        face_triangles = target_points[target_faces]
        face_first = face_triangles[:, 0, :]
        face_second = face_triangles[:, 1, :]
        face_third = face_triangles[:, 2, :]
        face_edge_a = face_second - face_first
        face_edge_b = face_third - face_first
        face_normal = cp.cross(face_edge_a, face_edge_b)
        face_normal_sq = cp.sum(face_normal * face_normal, axis=-1)
        face_xx = cp.sum(face_edge_a * face_edge_a, axis=-1)
        face_xy = cp.sum(face_edge_a * face_edge_b, axis=-1)
        face_yy = cp.sum(face_edge_b * face_edge_b, axis=-1)
        target_face_values = target_map[target_faces.astype(cp.int64)]
        max_candidates = self.max_candidates
        scale = cp.asarray(self.scale, dtype=cp.float32)

        def matrices(angles: cp.ndarray) -> cp.ndarray:
            """批量生成与官方数组布局一致的旋转矩阵。"""

            alpha, beta, gamma = angles.T
            zero = cp.zeros_like(alpha)
            one = cp.ones_like(alpha)
            rx = cp.stack(
                (
                    one,
                    zero,
                    zero,
                    zero,
                    cp.cos(alpha),
                    cp.sin(alpha),
                    zero,
                    -cp.sin(alpha),
                    cp.cos(alpha),
                ),
                axis=1,
            )
            ry = cp.stack(
                (
                    cp.cos(beta),
                    zero,
                    cp.sin(beta),
                    zero,
                    one,
                    zero,
                    -cp.sin(beta),
                    zero,
                    cp.cos(beta),
                ),
                axis=1,
            )
            rz = cp.stack(
                (
                    cp.cos(gamma),
                    cp.sin(gamma),
                    zero,
                    -cp.sin(gamma),
                    cp.cos(gamma),
                    zero,
                    zero,
                    zero,
                    one,
                ),
                axis=1,
            )
            # C 源码按 i+3*j 访问这些数组；逐项乘法才能保持相同方向。
            first = cp.empty_like(rx)
            result = cp.empty_like(rx)
            for row in range(3):
                for column in range(3):
                    first[:, row + 3 * column] = sum(
                        ry[:, row + 3 * index] * rx[:, index + 3 * column]
                        for index in range(3)
                    )
            for row in range(3):
                for column in range(3):
                    result[:, row + 3 * column] = sum(
                        rz[:, row + 3 * index] * first[:, index + 3 * column]
                        for index in range(3)
                    )
            return result.reshape(-1, 3, 3)

        def locate(query: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
            """定位 [候选数, 点数, 3] 查询并返回面片及重心权重。"""

            scaled = query / scale
            cell = cp.floor((scaled + 1.0) * (self.grid_size / 2.0)).astype(cp.int32)
            cell = cp.clip(cell, 0, self.grid_size - 1)
            cell_id = cell[..., 0] + self.grid_size * (
                cell[..., 1] + self.grid_size * cell[..., 2]
            )
            starts = offsets[cell_id]
            stops = offsets[cell_id + 1]
            local = cp.arange(max_candidates, dtype=cp.int32)
            positions = starts[..., None] + local
            valid = positions < stops[..., None]
            safe = cp.minimum(positions, candidate_faces.size - 1)
            candidates = candidate_faces[safe]
            triangles = face_triangles[candidates.astype(cp.int64)]
            first = triangles[..., 0, :]
            edge_a = face_edge_a[candidates.astype(cp.int64)]
            edge_b = face_edge_b[candidates.astype(cp.int64)]
            normal = face_normal[candidates.astype(cp.int64)]
            normal_sq = face_normal_sq[candidates.astype(cp.int64)]
            query_expanded = query[..., None, :]
            plane_numerator = cp.sum((first - query_expanded) * normal, axis=-1)
            plane_safe = cp.where(normal_sq > 0.0, normal_sq, 1.0)
            plane_t = cp.where(normal_sq > 0.0, plane_numerator / plane_safe, 0.0)
            projected = query_expanded + plane_t[..., None] * normal
            offset_point = projected - first
            xx = face_xx[candidates.astype(cp.int64)]
            xy = face_xy[candidates.astype(cp.int64)]
            xv = cp.sum(edge_a * offset_point, axis=-1)
            yy = face_yy[candidates.astype(cp.int64)]
            yv = cp.sum(edge_b * offset_point, axis=-1)
            denominator = xx * yy - xy * xy
            denominator_safe = cp.where(denominator != 0.0, denominator, 1.0)
            xpos = cp.where(
                denominator != 0.0,
                (xv * yy - yv * xy) / denominator_safe,
                0.0,
            )
            ypos = cp.where(
                denominator != 0.0,
                (yv * xx - xv * xy) / denominator_safe,
                0.0,
            )
            inside = (
                (denominator != 0.0)
                & (xpos >= -1.0e-3)
                & (xpos <= 1.0 + 1.0e-3)
                & (ypos >= -1.0e-3)
                & (ypos <= 1.0 + 1.0e-3)
                & (xpos + ypos >= -1.0e-3)
                & (xpos + ypos <= 1.0 + 1.0e-3)
            )
            plane_distance = plane_t * plane_t * normal_sq
            vertex_distance = cp.sum(
                (query_expanded[..., None, :] - triangles) ** 2, axis=-1
            )
            closest_vertex = cp.argmin(vertex_distance, axis=-1)
            vertex_index = closest_vertex[..., None, None]
            closest = cp.take_along_axis(
                triangles, vertex_index.repeat(3, axis=-1), axis=-2
            )[..., 0, :]
            previous_index = ((closest_vertex - 1) % 3)[..., None, None]
            next_index = ((closest_vertex + 1) % 3)[..., None, None]
            previous = cp.take_along_axis(
                triangles, previous_index.repeat(3, axis=-1), axis=-2
            )[..., 0, :]
            following = cp.take_along_axis(
                triangles, next_index.repeat(3, axis=-1), axis=-2
            )[..., 0, :]

            def segment_distance(
                point: cp.ndarray, first_point: cp.ndarray, second_point: cp.ndarray
            ) -> tuple[cp.ndarray, cp.ndarray]:
                direction = second_point - first_point
                offset_point = point - first_point
                denominator_seg = cp.sum(direction * direction, axis=-1)
                denominator_seg_safe = cp.where(
                    denominator_seg != 0.0, denominator_seg, 1.0
                )
                fraction = cp.where(
                    denominator_seg != 0.0,
                    cp.sum(offset_point * direction, axis=-1) / denominator_seg_safe,
                    0.0,
                )
                fraction = cp.clip(fraction, 0.0, 1.0)
                closest_point = first_point + fraction[..., None] * direction
                distance = cp.sum((point - closest_point) ** 2, axis=-1)
                return distance, closest_point

            edge_one_distance, edge_one_point = segment_distance(
                query_expanded, previous, closest
            )
            edge_two_distance, edge_two_point = segment_distance(
                query_expanded, closest, following
            )
            use_first = edge_one_distance < edge_two_distance
            outside_distance = cp.where(use_first, edge_one_distance, edge_two_distance)
            outside_point = cp.where(
                use_first[..., None], edge_one_point, edge_two_point
            )
            distance = cp.where(inside, plane_distance, outside_distance)
            closest_point = cp.where(inside[..., None], projected, outside_point)
            distance = cp.where(valid, distance, cp.inf)
            best = cp.argmin(distance, axis=-1)
            best_face = cp.take_along_axis(candidates, best[..., None], axis=-1)[..., 0]
            best_point = cp.take_along_axis(
                closest_point,
                best[..., None, None].repeat(3, axis=-1),
                axis=-2,
            )[..., 0, :]
            best_triangles = cp.take_along_axis(
                triangles,
                best[..., None, None, None].repeat(3, axis=-2).repeat(3, axis=-1),
                axis=-3,
            )[..., 0, :, :]
            a = best_triangles[..., 0, :]
            b = best_triangles[..., 1, :]
            c = best_triangles[..., 2, :]
            v0 = b - a
            v1 = c - a
            v2 = best_point - a
            d00 = cp.sum(v0 * v0, axis=-1)
            d01 = cp.sum(v0 * v1, axis=-1)
            d11 = cp.sum(v1 * v1, axis=-1)
            d20 = cp.sum(v2 * v0, axis=-1)
            d21 = cp.sum(v2 * v1, axis=-1)
            denom = d00 * d11 - d01 * d01
            denom_safe = cp.where(denom != 0.0, denom, 1.0)
            weight1 = cp.where(denom != 0.0, (d11 * d20 - d01 * d21) / denom_safe, 0.0)
            weight2 = cp.where(denom != 0.0, (d00 * d21 - d01 * d20) / denom_safe, 0.0)
            weights = cp.stack((1.0 - weight1 - weight2, weight1, weight2), axis=-1)
            return best_face, weights

        def costs(angles: cp.ndarray) -> cp.ndarray:
            """批量计算若干角度的曲率平方误差。"""

            angle_value = cp.asarray(angles, dtype=cp.float64).reshape(-1, 3)
            matrices_value = matrices(angle_value)
            rotated = cp.matmul(source[None, :, :], matrices_value.transpose(0, 2, 1))
            result = cp.zeros(angle_value.shape[0], dtype=cp.float64)
            for start in range(0, source.shape[0], int(point_chunk)):
                stop = min(start + int(point_chunk), int(source.shape[0]))
                selected_faces, weights = locate(rotated[:, start:stop, :])
                sampled = (
                    target_face_values[selected_faces.astype(cp.int64)]
                    * weights.astype(cp.float64)
                ).sum(axis=-1)
                delta = source_map[start:stop][None, :] - sampled
                result += cp.sum(delta * delta, axis=1)
            return result

        def one_cost(angle: cp.ndarray) -> float:
            return float(costs(angle.reshape(1, 3))[0].item())

        seeds = cp.asarray(official_rotation_seed_angles(), dtype=cp.float64)
        seed_costs = costs(seeds)
        seed_index = int(cp.argmin(seed_costs).item())
        seed = seeds[seed_index].copy()
        seed_cost = float(seed_costs[seed_index].item())
        if not refine:
            angle = seed
            cost = seed_cost
            iterations = 0
        else:
            simplex = cp.stack(
                (
                    seed,
                    seed + cp.asarray((0.1, 0.0, 0.0)),
                    seed + cp.asarray((0.0, 0.1, 0.0)),
                    seed + cp.asarray((0.0, 0.0, 0.1)),
                )
            )
            f_values = costs(simplex)
            iterations = 0
            for iterations in range(1, int(max_iter) + 1):
                highest = 0
                lowest = 0
                second_highest = 1
                for index in range(4):
                    if float(f_values[index].item()) > float(f_values[highest].item()):
                        second_highest = highest
                        highest = index
                    elif (
                        float(f_values[index].item())
                        > float(f_values[second_highest].item())
                        and index != highest
                    ):
                        second_highest = index
                    if float(f_values[index].item()) < float(f_values[lowest].item()):
                        lowest = index
                centroid = cp.sum(cp.delete(simplex, highest, axis=0), axis=0) / 3.0
                reflected = centroid + (centroid - simplex[highest])
                reflected_cost = one_cost(reflected)
                if reflected_cost < float(f_values[lowest].item()):
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded_cost = one_cost(expanded)
                    simplex[highest] = (
                        expanded if expanded_cost < reflected_cost else reflected
                    )
                    f_values[highest] = min(expanded_cost, reflected_cost)
                elif reflected_cost < float(f_values[second_highest].item()):
                    simplex[highest] = reflected
                    f_values[highest] = reflected_cost
                else:
                    contracted = centroid + 0.5 * (simplex[highest] - centroid)
                    contracted_cost = one_cost(contracted)
                    if contracted_cost < float(f_values[highest].item()):
                        simplex[highest] = contracted
                        f_values[highest] = contracted_cost
                    else:
                        for index in range(4):
                            if index != lowest:
                                simplex[index] = simplex[lowest] + 0.5 * (
                                    simplex[index] - simplex[lowest]
                                )
                                f_values[index] = one_cost(simplex[index])
                current_lowest = int(cp.argmin(f_values).item())
                max_diff = float(
                    cp.max(cp.abs(f_values - f_values[current_lowest])).item()
                )
                if max_diff < float(tol):
                    break
            lowest = int(cp.argmin(f_values).item())
            angle = simplex[lowest].copy()
            cost = float(f_values[lowest].item())

        angle_cpu = cp.asnumpy(angle).astype(np.float64, copy=False)
        return RotationSearchResult(
            seed_angle=cp.asnumpy(seed).astype(np.float64, copy=False),
            seed_cost=seed_cost,
            seed_costs=cp.asnumpy(seed_costs).astype(np.float64, copy=False),
            angle=angle_cpu,
            cost=float(cost),
            rotation_matrix=rotation_matrix_from_angles(angle_cpu),
            iterations=int(iterations),
        )


@dataclass(frozen=True)
class RotationSearchResult:
    """保存初始旋转搜索结果及诊断量。"""

    seed_angle: np.ndarray
    seed_cost: float
    seed_costs: np.ndarray
    angle: np.ndarray
    cost: float
    rotation_matrix: np.ndarray
    iterations: int


def average_xz_sphere(
    x_surface: np.ndarray,
    z_surface: np.ndarray,
) -> np.ndarray:
    """复现官方 ``average_xz_surf`` 的逐点加权单位化。"""

    x_value = np.asarray(x_surface, dtype=np.float64)
    z_value = np.asarray(z_surface, dtype=np.float64)
    if x_value.shape != z_value.shape or x_value.ndim != 2 or x_value.shape[1] != 3:
        raise ValueError("两个球面必须同为 [N,3]")
    phi_x = np.arccos(np.clip(x_value[:, 0], -1.0, 1.0)) / np.pi
    phi_z = np.arccos(np.clip(z_value[:, 2], -1.0, 1.0)) / np.pi
    weight_x = np.maximum(np.exp(-((2.0 * phi_x - 1.0) ** 2) / 0.1), 1.0e-19)
    weight_z = np.maximum(np.exp(-((2.0 * phi_z - 1.0) ** 2) / 0.1), 1.0e-19)
    averaged = (weight_x[:, None] * x_value + weight_z[:, None] * z_value) / (
        weight_x + weight_z
    )[:, None]
    length = np.linalg.norm(averaged, axis=1)
    averaged = averaged / np.maximum(length[:, None], np.finfo(np.float64).eps)
    return np.ascontiguousarray(averaged, dtype=np.float32)


__all__ = [
    "RotationGridIndex",
    "RotationSearchResult",
    "average_xz_sphere",
    "official_rotation_seed_angles",
    "rotate_vertices",
    "rotation_matrix_from_angles",
]
