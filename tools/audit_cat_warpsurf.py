#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""审计 CAT_WarpSurf 的 CPU 输入、输出和运行合同。

这个工具只调用当前 reference 二进制，不替换 CAT 算法，也不读取大型文件内容做哈希。
它为未来的 GPU 实现保存命令行、版本、GIFTI 几何结构、输出尺寸和墙钟基线。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import nibabel as nib


@dataclass(frozen=True)
class GeometrySummary:
    """GIFTI 几何数组的结构摘要。"""

    path: str
    size_bytes: int
    arrays: tuple[dict[str, Any], ...]


def summarize_geometry(path: Path) -> GeometrySummary:
    """读取 GIFTI 头和数组形状，不计算内容哈希。"""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GIFTI 文件不存在: {path}")
    image = nib.load(str(path))
    arrays = tuple(
        {
            "shape": list(array.data.shape),
            "dtype": str(array.data.dtype),
            "intent": int(array.intent),
        }
        for array in image.darrays
    )
    return GeometrySummary(
        path=str(path),
        size_bytes=path.stat().st_size,
        arrays=arrays,
    )


def _children_usage() -> tuple[float, float, int]:
    """读取当前脚本已等待子进程的累计 CPU/RSS。"""

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sys.platform == "darwin":
        peak_rss = int(usage.ru_maxrss)
    else:
        peak_rss = int(usage.ru_maxrss * 1024)
    return usage.ru_utime + usage.ru_stime, peak_rss, int(usage.ru_maxrss)


def _read_version(binary: Path) -> str:
    """读取 reference 二进制自带版本输出。"""

    completed = subprocess.run(
        [str(binary), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        # 部分 CAT-Surface 新版程序的通用 -version 参数本身会触发参数类型错误；
        # 不能因为元数据读取失败而阻断真正的数值和性能审计。
        return f"unavailable (exit {completed.returncode}): {detail}"
    return (completed.stdout + completed.stderr).strip()


def run_one(
    binary: Path,
    surface_root: Path,
    template_root: Path,
    output_root: Path,
    hemisphere: str,
    steps: int,
) -> dict[str, Any]:
    """执行一个半球的 reference CAT_WarpSurf 并记录结构基线。"""

    surface_root = Path(surface_root).expanduser().resolve()
    template_root = Path(template_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{hemisphere}.sphere.reg.gii"
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有 CAT 输出: {output}")

    white = surface_root / f"{hemisphere}.white.gii"
    sphere = surface_root / f"{hemisphere}.sphere.gii"
    template_white = template_root / f"{hemisphere}.white.gii"
    template_sphere = template_root / f"{hemisphere}.sphere.gii"
    command = [
        str(binary),
        "-steps",
        str(steps),
        "-avg",
        "-i",
        str(white),
        "-is",
        str(sphere),
        "-t",
        str(template_white),
        "-ts",
        str(template_sphere),
        "-ws",
        str(output),
    ]
    inputs = {
        "white": summarize_geometry(white),
        "sphere": summarize_geometry(sphere),
        "template_white": summarize_geometry(template_white),
        "template_sphere": summarize_geometry(template_sphere),
    }
    child_cpu_before, _, _ = _children_usage()
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    child_cpu_after, child_peak_rss, _ = _children_usage()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"CAT_WarpSurf {hemisphere} 返回 {completed.returncode}: {detail[-4000:]}"
        )
    if not output.is_file():
        raise RuntimeError(f"CAT_WarpSurf 没有生成输出: {output}")
    return {
        "hemisphere": hemisphere,
        "steps": steps,
        "runs": 2,
        "average_solutions": True,
        "command": command,
        "inputs": {key: asdict(value) for key, value in inputs.items()},
        "output": asdict(summarize_geometry(output)),
        "timing_seconds": {
            "wall": wall_seconds,
            "child_cpu_delta": child_cpu_after - child_cpu_before,
            "child_peak_rss_cumulative": child_peak_rss,
        },
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CAT_WarpSurf 审计参数。"""

    parser = argparse.ArgumentParser(description="审计 CAT_WarpSurf CPU 合同")
    parser.add_argument("--surface-root", required=True, type=Path)
    parser.add_argument(
        "--template-root",
        type=Path,
        required=True,
        help="SimNIBS/CAT 模板曲面目录，不使用本机固定路径",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        required=True,
        help="要审计的 CAT_WarpSurf reference 二进制",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hemisphere", choices=("lh", "rh", "both"), default="both")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行审计并打印 JSON 结果。"""

    args = parse_args(argv)
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"CAT_WarpSurf 不存在: {binary}")
    hemispheres = ("lh", "rh") if args.hemisphere == "both" else (args.hemisphere,)
    result = {
        "binary": str(binary),
        "version": _read_version(binary),
        "hemispheres": [
            run_one(
                binary,
                args.surface_root,
                args.template_root,
                args.output_root,
                hemisphere,
                args.steps,
            )
            for hemisphere in hemispheres
        ],
    }
    output_json = args.output_root.expanduser().resolve() / "cat_warpsurf_audit.json"
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
