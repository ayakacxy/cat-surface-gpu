"""CAT-Surface 球面生成与配准参数。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral


def _positive_integer(name: str, value: int, *, minimum: int = 1) -> None:
    """拒绝小数、布尔值和范围外的计数参数。"""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} 必须是大于或等于 {minimum} 的整数")


@dataclass(frozen=True)
class Surf2SphereParameters:
    """对应官方 ``CAT_Surf2Sphere`` 命令行参数。"""

    stop_at: int = 10
    verbose: bool = False

    def __post_init__(self) -> None:
        _positive_integer("stop_at", self.stop_at)

    @property
    def areal_iterations(self) -> int:
        """返回官方 ``1000 * (stop_at - 5)`` 面积平滑次数。"""

        return 1000 * max(0, self.stop_at - 5)


@dataclass(frozen=True)
class SurfWarpParameters:
    """对应锁定提交中 ``CAT_SurfWarp.c`` 的默认参数。"""

    rotate: bool = True
    average: bool = True
    code: int = 1
    loop: int = 6
    regularization_type: int = 1
    curvature_types: tuple[int, int, int] = (5, 5, 2)
    mu_change: int = 4
    multires_levels: int = 0
    map_size: tuple[int, int] = (512, 256)
    resampled_triangles: int = 81920
    steps: int = 2
    runs: int = 2
    mu_rate: float = 1.25
    lambda_value: float = 0.0
    mu: float = 0.125
    lm_regularization: float = 1.0e-3
    curvature_fwhm: float = 5.0
    surface_fwhm: float = 0.0
    cycles: int = 3
    iterations: int = 3

    def __post_init__(self) -> None:
        for name in ("loop", "steps", "runs", "mu_change", "cycles", "iterations"):
            _positive_integer(name, getattr(self, name))
        _positive_integer("multires_levels", self.multires_levels, minimum=0)
        if len(self.map_size) != 2:
            raise ValueError("map_size 必须包含宽度和高度两个整数")
        for dimension in self.map_size:
            _positive_integer("map_size", dimension, minimum=2)
        if len(self.curvature_types) != 3:
            raise ValueError("curvature_types 必须包含三个曲率类型")
        for name in ("mu", "mu_rate", "lm_regularization", "curvature_fwhm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是正有限数")
        if self.code not in (0, 1, 2):
            raise ValueError("code 必须是 0、1 或 2")
        if self.regularization_type not in (0, 1, 2):
            raise ValueError("regularization_type 必须是 0、1 或 2")
        if not 1 <= self.loop <= 6:
            raise ValueError("loop 必须位于 1 到 6")
        if not 1 <= self.steps <= 3:
            raise ValueError("steps 必须位于 1 到 3")
        if self.runs < 1:
            raise ValueError("runs 必须大于或等于 1")
        if not 0 <= self.multires_levels <= 3:
            raise ValueError("multires_levels 必须位于 0 到 3")
        if min(self.map_size) <= 0:
            raise ValueError("map_size 必须为正整数")
        if self.cycles < 1 or self.iterations < 1:
            raise ValueError("cycles 和 iterations 必须为正整数")
        fixed = {
            "code": 1,
            "regularization_type": 1,
            "multires_levels": 0,
            "resampled_triangles": 81920,
            "lambda_value": 0.0,
            "surface_fwhm": 0.0,
            "cycles": 3,
        }
        for name, supported in fixed.items():
            if getattr(self, name) != supported:
                raise ValueError(f"当前实现仅支持 {name}={supported}")
        if self.steps > 2 or any(
            kind != 5 for kind in self.curvature_types[: self.steps]
        ):
            raise ValueError("当前实现支持一层或两层 type-5 曲率特征")

    def dartel_schedule(self) -> tuple[dict[str, float | int], ...]:
        """按官方循环规则生成 DARTEL 参数表。"""

        mu = float(self.mu)
        lambda_value = float(self.lambda_value)
        schedule: list[dict[str, float | int]] = []
        for index in range(self.loop):
            schedule.append(
                {
                    "rtype": self.regularization_type,
                    "rparam0": 1.0,
                    "rparam1": 1.0,
                    "mu": mu,
                    "lambda": lambda_value,
                    "lambda_identity": lambda_value / 2.0,
                    "lmreg": self.lm_regularization,
                    "cycles": self.cycles,
                    "iterations": self.iterations,
                    "time_step": index,
                    "code": self.code,
                }
            )
            if (index + 1) % self.mu_change == 0:
                mu /= self.mu_rate
            lambda_value /= 5.0
        return tuple(schedule)
