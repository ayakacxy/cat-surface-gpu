"""交错测量两个源码版本的完整 CAT 命令，并保存输入身份与输出差异。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


def compare(first: Path, second: Path) -> dict:
    """核对拓扑、有限值、退化面和球面角差，不把几何检查当作解剖精度验收。"""
    a, b = nib.load(first), nib.load(second)
    x, y = [image.agg_data("pointset").astype(np.float64) for image in (a, b)]
    faces = a.agg_data("triangle")
    if not np.array_equal(faces, b.agg_data("triangle")):
        raise ValueError("Triangle indices differ")
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid output coordinates")
    normals = []
    for points in (x, y):
        triangles = points[faces]
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        if np.any(np.linalg.norm(normal, axis=1) == 0):
            raise ValueError("Zero-area output triangle")
        normals.append(normal)
    first_unit, second_unit = [p / np.linalg.norm(p, axis=1)[:, None] for p in (x, y)]
    angle = np.degrees(
        np.arctan2(
            np.linalg.norm(np.cross(first_unit, second_unit), axis=1),
            np.sum(first_unit * second_unit, axis=1),
        )
    )
    distances = np.linalg.norm(x - y, axis=1)
    return {
        "faces_exact": True,
        "finite": True,
        "zero_area_triangles": 0,
        "opposed_corresponding_normals": int(
            np.count_nonzero(np.sum(normals[0] * normals[1], axis=1) < 0)
        ),
        "coordinate_max_abs": float(np.max(np.abs(x - y))),
        "vertex_distance_max": float(distances.max()),
        "angle_degrees_max": float(angle.max()),
        "angle_degrees_p99": float(np.percentile(angle, 99)),
    }


def main():
    """记录空应用缓存首调用，再交错运行已有缓存的新进程命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        required=True,
        help="Directory containing the baseline cat_surface_gpu package",
    )
    parser.add_argument(
        "--candidate-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src",
    )
    for name in ("source-surface", "source-sphere", "target-surface", "target-sphere"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    inputs = {
        k: getattr(args, k).resolve()
        for k in ("source_surface", "source_sphere", "target_surface", "target_sphere")
    }
    report = {
        "date": "2026-10-18",
        "platform": platform.platform(),
        "python": sys.version,
        "device": args.device,
        "threads": args.threads,
        "solver": "PCG, 40 iterations, check interval 0",
        "timing_scope": "Fresh process launch through exit, including imports, input validation, reading, cache loading or compilation, computation and compressed GIFTI writing; excludes validation below.",
        "first_call_scope": "Empty application caches. The first warp call can reuse kernels compiled by the first sphere call. OS page caches and driver state are not reset.",
        "inputs": {
            k: {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in inputs.items()
        },
        "source_hashes": {},
        "records": [],
        "comparisons": {},
    }
    sources = {
        "baseline": args.baseline_source.resolve(),
        "candidate": args.candidate_source.resolve(),
    }
    for variant, source in sources.items():
        report["source_hashes"][variant] = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "cat_surface_gpu").rglob("*.py"))
        }
    for repeat in range(args.repeats + 1):
        for variant in (
            ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        ):
            env = os.environ.copy()
            for key, leaf in {
                "TMPDIR": "tmp",
                "TMP": "tmp",
                "TEMP": "tmp",
                "XDG_CACHE_HOME": "cache",
                "NUMBA_CACHE_DIR": "numba",
                "CUPY_CACHE_DIR": "cupy",
                "CUDA_CACHE_PATH": "cuda",
                "CAT_SURFACE_CACHE_ROOT": "targets",
            }.items():
                path = work / variant / leaf
                path.mkdir(parents=True, exist_ok=True)
                env[key] = str(path)
            env.update(
                PYTHONPATH=str(sources[variant]),
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS=str(args.threads),
                OPENBLAS_NUM_THREADS=str(args.threads),
                MKL_NUM_THREADS=str(args.threads),
                NUMBA_NUM_THREADS=str(args.threads),
            )
            for component in ("sphere", "warp"):
                name = f"{variant}-{component}-{repeat}"
                output = work / f"{name}.gii"
                function = (
                    "surf2sphere_main" if component == "sphere" else "surfwarp_main"
                )
                command = [
                    sys.executable,
                    "-c",
                    f"from cat_surface_gpu.cli import {function}; raise SystemExit({function}())",
                    "--device",
                    args.device,
                    "--threads",
                    str(args.threads),
                ]
                if component == "sphere":
                    command += [
                        "--input-surface",
                        str(inputs["source_surface"]),
                        "--output-surface",
                        str(output),
                        "--stop-at",
                        "10",
                    ]
                else:
                    for key, path in inputs.items():
                        command += ["--" + key.replace("_", "-"), str(path)]
                    command += [
                        "--output",
                        str(output),
                        "--solver-iterations",
                        "40",
                        "--rotation-point-chunk",
                        "12288",
                        "--pcg-check-interval",
                        "0",
                    ]
                print(f"[{repeat + 1}/{args.repeats + 1}] {name}", flush=True)
                with (
                    (work / f"{name}.stdout.log").open("w") as out,
                    (work / f"{name}.stderr.log").open("w") as err,
                ):
                    start = time.perf_counter()
                    subprocess.run(command, env=env, stdout=out, stderr=err, check=True)
                    seconds = time.perf_counter() - start
                report["records"].append(
                    {
                        "variant": variant,
                        "component": component,
                        "repeat": repeat,
                        "wall_seconds": seconds,
                        "command": command,
                    }
                )
                (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["summary"] = {}
    for component in ("sphere", "warp"):
        medians = {
            v: statistics.median(
                r["wall_seconds"]
                for r in report["records"]
                if r["variant"] == v and r["component"] == component and r["repeat"] > 0
            )
            for v in sources
        }
        report["summary"][component] = {
            **medians,
            "speedup": medians["baseline"] / medians["candidate"],
        }
        report["comparisons"][component] = compare(
            work / f"baseline-{component}-1.gii", work / f"candidate-{component}-1.gii"
        )
        for variant in sources:
            for repeat in range(2, args.repeats + 1):
                difference = compare(
                    work / f"{variant}-{component}-1.gii",
                    work / f"{variant}-{component}-{repeat}.gii",
                )
                if difference["coordinate_max_abs"] != 0:
                    raise ValueError(
                        f"{variant} {component} repeated output differs: {difference}"
                    )
    report["repeated_outputs_exact"] = True
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
