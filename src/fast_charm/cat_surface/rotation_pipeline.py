"""CAT 初始旋转的输入读取、索引构建和 seed/精化编排。"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dartel_grid import resolve_device
from .surface_rotation import (
    RotationCostInputs,
    RotationGridIndex,
    RotationGridIndexDevice,
    RotationSearchResult,
)
from .surface_stencil import SurfaceStencil


def read_rotation_values(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取官方 probe 导出的 source/target 双精度旋转特征。"""

    path = Path(path)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if header.size != 2:
            raise ValueError(f"旋转特征文件头不完整：{path}")
        source = np.fromfile(stream, dtype="<f8", count=int(header[0]))
        target = np.fromfile(stream, dtype="<f8", count=int(header[1]))
        trailing = stream.read(1)
    if source.size != int(header[0]) or target.size != int(header[1]):
        raise ValueError(f"旋转特征文件长度不匹配：{path}")
    if trailing:
        raise ValueError(f"旋转特征文件包含未消费尾部：{path}")
    return np.ascontiguousarray(source), np.ascontiguousarray(target)


def read_rotation_points(path: str | Path) -> np.ndarray:
    """读取官方 map probe 导出的粗网格 source 点。

    官方 probe 的点类型会随编译配置成为单精度或双精度；两种格式的
    文件头相同，均为 ``int32`` 点数后紧跟 ``x/y/z`` 交错坐标。读取时
    只在文件长度明确匹配时选择对应精度，避免把双精度 payload 错读成
    单精度并把尾部误报成损坏。
    """

    path = Path(path)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=1)
        if header.size != 1:
            raise ValueError(f"旋转点文件头不完整：{path}")
        n_points = int(header[0])
        if n_points < 0:
            raise ValueError(f"旋转点文件点数非法：{path}")
        payload = stream.read()

    n_values = 3 * n_points
    expected_float32_bytes = n_values * np.dtype("<f4").itemsize
    expected_float64_bytes = n_values * np.dtype("<f8").itemsize
    if len(payload) == expected_float32_bytes:
        points = np.frombuffer(payload, dtype="<f4", count=n_values)
    elif len(payload) == expected_float64_bytes:
        points = np.frombuffer(payload, dtype="<f8", count=n_values)
    else:
        raise ValueError(
            f"旋转点文件长度不匹配：{path}，实际 {len(payload)} 字节，"
            f"期望 {expected_float32_bytes} 或 {expected_float64_bytes} 字节"
        )
    return np.ascontiguousarray(points.reshape(n_points, 3))


@dataclass
class RotationPipeline:
    """保存一次目标球面索引和设备端旋转搜索入口。"""

    index_cpu: RotationGridIndex
    index: RotationGridIndexDevice

    @classmethod
    def from_stencil(
        cls,
        target_stencil: SurfaceStencil,
        *,
        device: str | torch.device = "auto",
        grid_size: int = 128,
        margin: int = 1,
        candidate_table_dtype: torch.dtype = torch.int32,
        geometry_dtype: torch.dtype = torch.float32,
        compressed_candidate_table: bool = True,
    ) -> "RotationPipeline":
        """从目标粗球面 stencil 建立一次可复用的旋转 pipeline。"""

        target_device = resolve_device(device)
        index_cpu = RotationGridIndex.from_stencil(
            target_stencil,
            grid_size=grid_size,
            margin=margin,
            include_dense_candidate_table=not compressed_candidate_table,
        )
        index = index_cpu.to(
            target_device,
            geometry_dtype=geometry_dtype,
            candidate_table_dtype=candidate_table_dtype,
            compressed_candidate_table=compressed_candidate_table,
        )
        return cls(index_cpu=index_cpu, index=index)

    @classmethod
    def from_index_cpu(
        cls,
        index_cpu: RotationGridIndex,
        *,
        device: str | torch.device = "auto",
        candidate_table_dtype: torch.dtype = torch.int32,
        geometry_dtype: torch.dtype = torch.float32,
    ) -> "RotationPipeline":
        """从已经在 CPU 构建好的索引完成设备上传。"""

        target_device = resolve_device(device)
        index = index_cpu.to(
            target_device,
            geometry_dtype=geometry_dtype,
            candidate_table_dtype=candidate_table_dtype,
        )
        return cls(index_cpu=index_cpu, index=index)

    @property
    def device(self) -> torch.device:
        """返回旋转 pipeline 的设备。"""

        return self.index.device

    def prepare_inputs(
        self,
        source_points: np.ndarray | torch.Tensor,
        source_values: np.ndarray | torch.Tensor,
        target_values: np.ndarray | torch.Tensor,
    ) -> RotationCostInputs:
        """把一次旋转搜索的三组输入固定在 pipeline 设备。"""

        return self.index.prepare_cost_inputs(
            source_points, source_values, target_values
        )

    def search(
        self,
        source_points: np.ndarray | torch.Tensor,
        source_values: np.ndarray | torch.Tensor,
        target_values: np.ndarray | torch.Tensor,
        *,
        point_chunk: int = 1024,
        refine: bool = True,
        max_iter: int = 500,
        tol: float = 1.0e-4,
        simplex_step: float = 0.1,
    ) -> RotationSearchResult:
        """执行官方 seed 和可选 C 顺序 Nelder–Mead。"""

        inputs = self.prepare_inputs(
            source_points, source_values, target_values
        )
        return self.index.search_rotation_prepared(
            inputs,
            point_chunk=point_chunk,
            refine=refine,
            max_iter=max_iter,
            tol=tol,
            simplex_step=simplex_step,
        )


def run_rotation_pipeline(
    target_stencil: SurfaceStencil,
    source_points: np.ndarray | torch.Tensor,
    source_values: np.ndarray | torch.Tensor,
    target_values: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "auto",
    grid_size: int = 128,
    margin: int = 1,
    candidate_table_dtype: torch.dtype = torch.int32,
    geometry_dtype: torch.dtype = torch.float32,
    point_chunk: int = 1024,
    refine: bool = True,
    max_iter: int = 500,
    tol: float = 1.0e-4,
    simplex_step: float = 0.1,
) -> RotationSearchResult:
    """从固定 C probe 输入完成一次完整的初始旋转搜索。"""

    pipeline = RotationPipeline.from_stencil(
        target_stencil,
        device=device,
        grid_size=grid_size,
        margin=margin,
        candidate_table_dtype=candidate_table_dtype,
        geometry_dtype=geometry_dtype,
    )
    return pipeline.search(
        source_points,
        source_values,
        target_values,
        point_chunk=point_chunk,
        refine=refine,
        max_iter=max_iter,
        tol=tol,
        simplex_step=simplex_step,
    )


def run_rotation_pipeline_from_files(
    source_points_path: str | Path,
    target_stencil_path: str | Path,
    rotation_values_path: str | Path,
    *,
    device: str | torch.device = "auto",
    grid_size: int = 128,
    margin: int = 1,
    candidate_table_dtype: torch.dtype = torch.int32,
    geometry_dtype: torch.dtype = torch.float32,
    point_chunk: int = 1024,
    refine: bool = True,
    max_iter: int = 500,
    tol: float = 1.0e-4,
    simplex_step: float = 0.1,
) -> RotationSearchResult:
    """读取官方 probe 文件并完成一次初始旋转搜索。"""

    source_points = read_rotation_points(source_points_path)
    source_values, target_values = read_rotation_values(rotation_values_path)
    target_stencil = SurfaceStencil.from_file(target_stencil_path)
    return run_rotation_pipeline(
        target_stencil,
        source_points,
        source_values,
        target_values,
        device=device,
        grid_size=grid_size,
        margin=margin,
        candidate_table_dtype=candidate_table_dtype,
        geometry_dtype=geometry_dtype,
        point_chunk=point_chunk,
        refine=refine,
        max_iter=max_iter,
        tol=tol,
        simplex_step=simplex_step,
    )
