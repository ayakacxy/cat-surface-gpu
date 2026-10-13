#!/usr/bin/env python3
"""用相同 GIFTI 输入比较固定 CAT-Surface CPU 参考与已安装的 GPU 命令。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from cat_surface_gpu import load_gifti_surface


def sha256(path: Path) -> str:
    """计算输入、参考程序和分发文件的摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference: Path, candidate: Path, input_surface: Path) -> dict:
    """分别记录逐坐标绝对差和逐顶点欧氏距离，避免混用误差单位。"""
    cpu = load_gifti_surface(reference)
    gpu = load_gifti_surface(candidate)
    source = load_gifti_surface(input_surface)
    if cpu.vertices.shape != gpu.vertices.shape:
        raise RuntimeError("CPU 和 GPU 输出顶点数量不同")
    if not np.array_equal(cpu.faces, gpu.faces) or not np.array_equal(
        source.faces, gpu.faces
    ):
        raise RuntimeError("CPU、GPU 与输入的三角形数组不一致")
    if not np.isfinite(cpu.vertices).all() or not np.isfinite(gpu.vertices).all():
        raise RuntimeError("输出含非有限坐标")
    difference = gpu.vertices.astype(np.float64) - cpu.vertices.astype(np.float64)
    absolute = np.abs(difference)
    distances = np.linalg.norm(difference, axis=1)
    result = {
        "faces_exact": True,
        "finite": True,
        "vertices": len(gpu.vertices),
        "triangles": len(gpu.faces),
        "coordinate_absolute_difference": {
            "max": float(absolute.max()),
            "mean": float(absolute.mean()),
            "p99": float(np.percentile(absolute, 99)),
        },
        "vertex_euclidean_distance": {
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "p99": float(np.percentile(distances, 99)),
        },
    }
    for name, surface in [("cpu", cpu), ("gpu", gpu)]:
        edges1 = (
            surface.vertices[surface.faces[:, 1]].astype(np.float64)
            - surface.vertices[surface.faces[:, 0]]
        )
        edges2 = (
            surface.vertices[surface.faces[:, 2]].astype(np.float64)
            - surface.vertices[surface.faces[:, 0]]
        )
        result[f"{name}_zero_area_triangles"] = int(
            np.count_nonzero(np.linalg.norm(np.cross(edges1, edges2), axis=1) == 0)
        )
    return result


def measure(command: list[str], directory: Path, name: str) -> dict:
    """计时覆盖新进程启动到退出，日志和输出文件均留在指定任务目录。"""
    stdout_path = directory / f"{name}.stdout.log"
    stderr_path = directory / f"{name}.stderr.log"
    print(f"[benchmark] Running {name}", flush=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        started = time.perf_counter()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        _, status, usage = os.wait4(process.pid, 0)
        wall = time.perf_counter() - started
        process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode:
        raise RuntimeError(
            f"{name} failed with exit {process.returncode}; see {stderr_path}"
        )
    record = {
        "name": name,
        "command": command,
        "wall_seconds": wall,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
    }
    if name.startswith("gpu"):
        record["cli_summary"] = json.loads(stdout_path.read_text())
    print(f"[benchmark] {name}: {wall:.4f} s", flush=True)
    return record


def main() -> int:
    """先记录首次 GPU 调用，再交错执行重复的 CPU/GPU 完整命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-sphere", type=Path, required=True)
    parser.add_argument("--reference-warp", type=Path, required=True)
    parser.add_argument("--source-surface", type=Path, required=True)
    parser.add_argument("--source-sphere", type=Path, required=True)
    parser.add_argument("--target-surface", type=Path, required=True)
    parser.add_argument("--target-sphere", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--include-fmg", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    logs = work / "logs"
    outputs = work / "outputs"
    logs.mkdir()
    outputs.mkdir()
    inputs = {
        key: getattr(args, key).resolve()
        for key in [
            "source_surface",
            "source_sphere",
            "target_surface",
            "target_sphere",
        ]
    }
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
    paths = {key: str(path) for key, path in inputs.items()}
    binaries = {
        "sphere": args.reference_sphere.resolve(),
        "warp": args.reference_warp.resolve(),
    }
    gpu_variants = ["pcg", "fmg"] if args.include_fmg else ["pcg"]
    scripts = Path(sys.executable).parent
    sphere_command = scripts / "cat-surf2sphere-gpu"
    warp_command = scripts / "cat-surfwarp-gpu"
    if not sphere_command.is_file() or not warp_command.is_file():
        parser.error("Install cat-surface-gpu into the benchmark Python environment")

    import cupy as cp

    gpu = cp.cuda.runtime.getDeviceProperties(0)
    report = {
        "version": importlib.metadata.version("cat-surface-gpu"),
        "date": time.strftime("%Y-%m-%d"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "numba", "nibabel"]
            },
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "gpu": gpu["name"].decode()
            if isinstance(gpu["name"], bytes)
            else gpu["name"],
            "threads_requested": args.threads,
        },
        "inputs": {
            key: {"name": path.name, "sha256": sha256(path)}
            for key, path in inputs.items()
        },
        "reference_programs": {
            key: {"name": path.name, "sha256": sha256(path)}
            for key, path in binaries.items()
        },
        "timing_scope": "Fresh process launch through exit: interpreter/imports, input loading, CPU preparation, CUDA setup, compilation/cache loading, computation, transfers, GIFTI writing. Validation is outside the timed command. Reference compilation and wheel installation are excluded.",
        "first_gpu_calls": {},
        "measurements": {"sphere": [], "warp": []},
        "comparisons": [],
    }

    def command(component: str, backend: str, output: Path) -> list[str]:
        if component == "sphere":
            if backend == "cpu":
                return [
                    str(binaries[component]),
                    paths["source_surface"],
                    str(output),
                    "10",
                ]
            return [
                str(sphere_command),
                "--input-surface",
                paths["source_surface"],
                "--output-surface",
                str(output),
                "--stop-at",
                "10",
                "--threads",
                str(args.threads),
            ]
        if backend == "cpu":
            return [
                str(binaries[component]),
                "-i",
                paths["source_surface"],
                "-is",
                paths["source_sphere"],
                "-t",
                paths["target_surface"],
                "-ts",
                paths["target_sphere"],
                "-ws",
                str(output),
                "-steps",
                "2",
                "-runs",
                "2",
                "-avg",
                "-size",
                "512",
                "256",
            ]
        return [
            str(warp_command),
            "--source-surface",
            paths["source_surface"],
            "--source-sphere",
            paths["source_sphere"],
            "--target-surface",
            paths["target_surface"],
            "--target-sphere",
            paths["target_sphere"],
            "--output",
            str(output),
            "--steps",
            "2",
            "--runs",
            "2",
            "--average",
            "--solver-backend",
            backend,
            "--solver-iterations",
            "40",
            "--threads",
            str(args.threads),
        ]

    def save() -> None:
        """每条命令后写入检查点，便于长任务恢复和阶段审计。"""
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    for component in ["sphere", "warp"]:
        variants = ["cupy"] if component == "sphere" else gpu_variants
        for backend in variants:
            name = f"gpu-{component}-{backend}-first"
            result = measure(
                command(component, backend, outputs / f"{name}.gii"), logs, name
            )
            report["first_gpu_calls"][f"{component}-{backend}"] = result
            save()

    for repeat in range(1, args.repeats + 1):
        for component in ["sphere", "warp"]:
            variants = ["cupy"] if component == "sphere" else gpu_variants
            order = ["cpu", *variants] if repeat % 2 else [*variants, "cpu"]
            for backend in order:
                name = f"{'cpu' if backend == 'cpu' else 'gpu'}-{component}-{backend}-r{repeat}"
                result = measure(
                    command(component, backend, outputs / f"{name}.gii"), logs, name
                )
                result.update({"repeat": repeat, "backend": backend})
                report["measurements"][component].append(result)
                save()
            for backend in variants:
                cpu_output = outputs / f"cpu-{component}-cpu-r{repeat}.gii"
                gpu_output = outputs / f"gpu-{component}-{backend}-r{repeat}.gii"
                comparison = compare(cpu_output, gpu_output, inputs["source_surface"])
                comparison.update(
                    {"component": component, "backend": backend, "repeat": repeat}
                )
                report["comparisons"].append(comparison)
                save()

    summaries = {}
    for component, records in report["measurements"].items():
        by_backend = {}
        for backend in sorted({record["backend"] for record in records}):
            values = [
                record["wall_seconds"]
                for record in records
                if record["backend"] == backend
            ]
            by_backend[backend] = {
                "repeats": len(values),
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        for backend, summary in by_backend.items():
            if backend != "cpu":
                summary["cpu_to_gpu_median_time_ratio"] = (
                    by_backend["cpu"]["median_seconds"] / summary["median_seconds"]
                )
        summaries[component] = by_backend
    report["summary"] = summaries
    report["inputs_unchanged"] = all(
        sha256(path) == report["inputs"][key]["sha256"] for key, path in inputs.items()
    )
    report["status"] = "completed"
    report["numerical_note"] = (
        "CPU uses upstream FMG. Default GPU PCG and the Python FMG implementation can follow different numerical paths. Elapsed-time ratios alone do not establish equivalent registration accuracy. Coordinate units are those stored in the GIFTI files."
    )
    save()
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
