#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""运行最新版 CAT_Surf2Sphere 的 GPU 面积平滑路径。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cat_surface_gpu import run_cat_surf2sphere_gpu


def main() -> int:
    """执行一次显式 GPU CAT_Surf2Sphere，并输出阶段计时。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cli", type=Path, required=True)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--output-surface", type=Path, required=True)
    parser.add_argument("--stop-at", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--kernel", choices=("torch", "triton"), default="triton")
    parser.add_argument(
        "--preprocess-kernel",
        choices=("cpu", "triton"),
        default="cpu",
        help="前五个膨胀/诊断阶段使用的显式后端",
    )
    parser.add_argument(
        "--preprocess-block-size",
        type=int,
        default=131072,
        help="前处理按顶点编号分块的大小",
    )
    parser.add_argument(
        "--areal-block-size",
        type=int,
        help="面积平滑按顶点编号分块的大小；未设置时使用整图着色",
    )
    parser.add_argument(
        "--areal-schedule",
        choices=("ordered", "color"),
        default="ordered",
        help="面积平滑调度；ordered 保持官方顶点依赖顺序",
    )
    parser.add_argument(
        "--areal-arithmetic",
        choices=("cat", "fp32"),
        default="cat",
        help="面积平滑算术；cat 使用官方 float 存储/double 累加边界",
    )
    args = parser.parse_args()

    result = run_cat_surf2sphere_gpu(
        args.input_surface,
        args.output_surface,
        reference_cli=args.reference_cli,
        stop_at=args.stop_at,
        device=args.device,
        dtype=args.dtype,
        kernel=args.kernel,
        preprocess_kernel=args.preprocess_kernel,
        preprocess_block_size=args.preprocess_block_size,
        areal_schedule=args.areal_schedule,
        areal_arithmetic=args.areal_arithmetic,
        areal_block_size=args.areal_block_size,
    )
    payload = {
        "backend": (
            f"surf2sphere_pre_{args.preprocess_kernel}_"
            f"{args.areal_schedule}_gauss_seidel_{args.kernel}_cuda"
        ),
        "device": args.device,
        "dtype": args.dtype,
        "areal_arithmetic": args.areal_arithmetic,
        "stop_at": args.stop_at,
        "output_surface": str(result.output_path),
        "timings": result.timings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
