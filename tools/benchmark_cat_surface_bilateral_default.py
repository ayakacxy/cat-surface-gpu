#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""按 SimNIBS 默认粗化参数测量双侧 CAT 球面注册 GPU 链路。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from fast_charm.cat_surface import read_gifti_mesh


def _run_parallel(
    commands: list[list[str]],
    logs: list[Path],
    *,
    root: Path,
    pythonpath: str,
) -> tuple[float, list[int]]:
    """并行运行两个半球进程，并保留各自 stdout/stderr。"""

    handles = []
    processes = []
    start = time.perf_counter()
    try:
        for command, log in zip(commands, logs):
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            handles.append(handle)
            environment = dict(**__import__("os").environ)
            environment["PYTHONPATH"] = (
                str(root / "src")
                + ":"
                + environment.get("PYTHONPATH", "")
            )
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        return_codes = [process.wait() for process in processes]
        return time.perf_counter() - start, return_codes
    finally:
        for handle in handles:
            handle.close()


def _upsample(
    simnibs_python: Path,
    pairs: list[tuple[Path, Path]],
) -> float:
    """复用 SimNIBS 的拓扑上采样逻辑。"""

    encoded = json.dumps(
        [(str(source), str(destination)) for source, destination in pairs]
    )
    script = (
        "import brainnet, cortech, torch\n"
        "from pathlib import Path\n"
        "topology = brainnet.DeepSurferTopology.recursive_subdivision(5)[-1]\n"
        f"pairs = {encoded}\n"
        "for source, destination in pairs:\n"
        "    sphere = cortech.Sphere.from_file(source, normalize=False)\n"
        "    vertices = topology.subdivide_vertices(torch.tensor(sphere.vertices).T).T.numpy()\n"
        "    faces = topology.subdivide_faces().numpy()\n"
        "    cortech.Sphere(vertices, faces, normalize=False).save(Path(destination))\n"
    )
    start = time.perf_counter()
    completed = subprocess.run(
        [str(simnibs_python), "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SimNIBS sphere upsample 失败:\n"
            + completed.stderr[-4000:]
        )
    return time.perf_counter() - start


def _compare_mesh(reference: Path, candidate: Path) -> dict[str, object]:
    """比较最终 GIFTI 的拓扑和逐元素坐标误差。"""

    expected = read_gifti_mesh(reference)
    actual = read_gifti_mesh(candidate)
    if expected.vertices.shape != actual.vertices.shape:
        raise ValueError(
            f"顶点形状不一致: {expected.vertices.shape} vs {actual.vertices.shape}"
        )
    if expected.faces.shape != actual.faces.shape:
        raise ValueError(
            f"面片形状不一致: {expected.faces.shape} vs {actual.faces.shape}"
        )
    if not np.array_equal(expected.faces, actual.faces):
        raise ValueError("最终 GIFTI 面片数组不一致")
    difference = np.abs(
        expected.vertices.astype(np.float64)
        - actual.vertices.astype(np.float64)
    )
    return {
        "faces_exact": True,
        "vertices_max_abs": float(difference.max()),
        "vertices_mean_abs": float(difference.mean()),
        "vertices_p99_abs": float(np.quantile(difference, 0.99)),
    }


def main() -> int:
    """执行默认双侧 CAT_Surf2Sphere + CAT_WarpSurf GPU 闭环。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--simnibs-python", type=Path, required=True)
    parser.add_argument("--reference-cli", type=Path, required=True)
    parser.add_argument("--lh-coarse", type=Path, required=True)
    parser.add_argument("--rh-coarse", type=Path, required=True)
    parser.add_argument("--lh-source", type=Path, required=True)
    parser.add_argument("--rh-source", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--rotation-depth-probe", type=Path, required=True)
    parser.add_argument("--stencil-builder", type=Path, required=True)
    parser.add_argument("--rotated-stencil-builder", type=Path, required=True)
    parser.add_argument(
        "--stencil-threads",
        type=int,
        default=8,
        help="每个 stencil helper 的 CPU worker 数；默认 8",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--areal-block-size", type=int, default=8192)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lh-cpu-reference", type=Path)
    parser.add_argument("--rh-cpu-reference", type=Path)
    parser.add_argument("--max-vertex-error", type=float, default=1.0e-3)
    parser.add_argument("--max-mean-error", type=float, default=1.0e-3)
    parser.add_argument("--max-p99-error", type=float, default=1.0e-3)

    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        (
            "lh",
            args.lh_coarse,
            args.lh_source,
            args.template_dir / "lh.white.gii",
            args.template_dir / "lh.sphere.gii",
        ),
        (
            "rh",
            args.rh_coarse,
            args.rh_source,
            args.template_dir / "rh.white.gii",
            args.template_dir / "rh.sphere.gii",
        ),
    )

    coarse_outputs = {
        hemi: output_dir / f"{hemi}.sphere.coarse.gii"
        for hemi, *_ in specs
    }
    full_outputs = {
        hemi: output_dir / f"{hemi}.sphere.gii"
        for hemi, *_ in specs
    }
    sphere_commands = []
    sphere_logs = []
    for hemi, coarse, _source, _target, _target_sphere in specs:
        sphere_commands.append(
            [
                str(args.python),
                "tools/run_cat_surf2sphere_gpu.py",
                "--reference-cli",
                str(args.reference_cli),
                "--input-surface",
                str(coarse),
                "--output-surface",
                str(coarse_outputs[hemi]),
                "--stop-at",
                "10",
                "--device",
                args.device,
                "--dtype",
                "float32",
                "--kernel",
                "triton",
                "--preprocess-kernel",
                "cpu",
                "--areal-schedule",
                args.areal_schedule,
                "--areal-arithmetic",
                args.areal_arithmetic,
                "--areal-block-size",
                str(args.areal_block_size),
            ]
        )
        sphere_logs.append(output_dir / f"{hemi}.surf2sphere.log")

    sphere_wall, sphere_codes = _run_parallel(
        sphere_commands,
        sphere_logs,
        root=root,
        pythonpath=str(root / "src"),
    )
    if any(sphere_codes):
        raise RuntimeError(f"CAT_Surf2Sphere 双侧失败: {sphere_codes}")

    upsample_wall = _upsample(
        args.simnibs_python,
        [(coarse_outputs[hemi], full_outputs[hemi]) for hemi, *_ in specs],
    )

    warp_commands = []
    warp_logs = []
    warp_outputs = {}
    for hemi, _coarse, source, target, target_sphere in specs:
        output = output_dir / f"{hemi}.sphere.reg.gii"
        warp_outputs[hemi] = output
        warp_commands.append(
            [
                str(args.python),
                "tools/run_cat_surface_gpu_pipeline.py",
                "--source-surface",
                str(source),
                "--source-sphere",
                str(full_outputs[hemi]),
                "--target-surface",
                str(target),
                "--target-sphere",
                str(target_sphere),
                "--output",
                str(output),
                "--rotation-depth-probe",
                str(args.rotation_depth_probe),
                "--rotation-feature-backend",
                "cuda-official-depth",
                "--stencil-builder",
                str(args.stencil_builder),
                "--rotated-stencil-builder",
                str(args.rotated_stencil_builder),
                "--stencil-threads",
                str(args.stencil_threads),
                "--device",
                args.device,
                "--kernel",
                "triton",
                "--dartel-dtype",
                "float64",
                "--squaring-kernel",
                "triton",
                "--steps",
                "2",
                "--runs",
                "2",
                "--avg",
                "--cuda-graph",
            ]
        )
        warp_logs.append(output_dir / f"{hemi}.warpsurf.log")

    warp_wall, warp_codes = _run_parallel(
        warp_commands,
        warp_logs,
        root=root,
        pythonpath=str(root / "src"),
    )
    if any(warp_codes):
        raise RuntimeError(f"CAT_WarpSurf 双侧失败: {warp_codes}")

    comparisons = None
    passed = None
    references = {
        "lh": args.lh_cpu_reference,
        "rh": args.rh_cpu_reference,
    }
    if all(reference is not None for reference in references.values()):
        comparisons = {
            hemi: _compare_mesh(reference, warp_outputs[hemi])
            for hemi, reference in references.items()
        }
        passed = all(
            comparison["vertices_max_abs"] <= args.max_vertex_error
            and comparison["vertices_mean_abs"] <= args.max_mean_error
            and comparison["vertices_p99_abs"] <= args.max_p99_error
            for comparison in comparisons.values()
        )

    payload = {
        "sphere_wall_seconds": sphere_wall,
        "upsample_wall_seconds": upsample_wall,
        "warp_wall_seconds": warp_wall,
        "total_wall_seconds": sphere_wall + upsample_wall + warp_wall,
        "sphere_returncodes": sphere_codes,
        "warp_returncodes": warp_codes,
        "areal_schedule": args.areal_schedule,
        "areal_arithmetic": args.areal_arithmetic,
        "outputs": {key: str(value) for key, value in warp_outputs.items()},
        "comparisons": comparisons,
        "tolerance": {
            "max_vertex_error": args.max_vertex_error,
            "max_mean_error": args.max_mean_error,
            "max_p99_error": args.max_p99_error,
        },
        "passed": passed,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
