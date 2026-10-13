"""文件读写、命令行检查及独立运行依赖的回归测试。"""

import ast
from contextlib import nullcontext
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import cat_surface_gpu
from cat_surface_gpu import cli
from cat_surface_gpu.io import GiftiSurface, load_gifti_surface, save_gifti_surface
from cat_surface_gpu.parameters import Surf2SphereParameters, SurfWarpParameters
from cat_surface_gpu.resampling import create_icosphere
from cat_surface_gpu.surfwarp import SurfWarpResult, surfwarp_cupy


def _surface_files(tmp_path):
    vertices, faces = create_icosphere(1)
    paths = {}
    for name in ("source.surface", "source.sphere", "target.surface", "target.sphere"):
        path = tmp_path / f"{name}.gii"
        save_gifti_surface(GiftiSurface(vertices, faces), path)
        paths[name] = path
    return vertices, faces, paths


def test_gifti_round_trip_preserves_geometry_and_metadata(tmp_path):
    vertices, faces = create_icosphere(1)
    image = nib.gifti.GiftiImage()
    point = nib.gifti.GiftiDataArray(vertices, intent="NIFTI_INTENT_POINTSET")
    point.coordsys = nib.gifti.GiftiCoordSystem(
        dataspace=1, xformspace=3, xform=np.diag([2.0, 3.0, 4.0, 1.0])
    )
    point.meta["AnatomicalStructurePrimary"] = "CortexLeft"
    image.add_gifti_data_array(point)
    image.add_gifti_data_array(
        nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")
    )
    image.meta["SubjectID"] = "synthetic"
    path = tmp_path / "surface.gii"
    save_gifti_surface(GiftiSurface(vertices, faces, image), path)
    result = load_gifti_surface(path)
    assert np.array_equal(result.vertices, vertices)
    assert np.array_equal(result.faces, faces)
    assert result.reference.meta["SubjectID"] == "synthetic"
    output_points = result.reference.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0]
    assert output_points.meta["AnatomicalStructurePrimary"] == "CortexLeft"
    assert np.array_equal(output_points.coordsys.xform, point.coordsys.xform)
    assert output_points.coordsys.xformspace == 3


@pytest.mark.parametrize(
    "options",
    [
        {"mu_change": 0},
        {"mu_rate": 0.0},
        {"mu": float("nan")},
        {"steps": 3},
        {"map_size": (512,)},
        {"map_size": (0, 256)},
        {"map_size": (32.5, 16)},
        {"runs": 1.5},
        {"regularization_type": 2},
        {"code": 0},
        {"resampled_triangles": 320},
        {"lambda_value": 0.1},
        {"surface_fwhm": 1},
        {"cycles": 4},
        {"multires_levels": 1},
    ],
)
def test_unsupported_or_invalid_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        SurfWarpParameters(**options)


@pytest.mark.parametrize("stop_at", [0, -1, 1.5, True])
def test_invalid_sphere_stage_is_rejected(stop_at):
    with pytest.raises(ValueError):
        Surf2SphereParameters(stop_at=stop_at)


def test_fractional_faces_are_rejected():
    vertices, faces = create_icosphere(0)
    with pytest.raises(ValueError, match="整数"):
        GiftiSurface(vertices, faces.astype(float) + 0.5)


def test_zero_area_faces_are_rejected_before_cuda():
    """不同顶点编号但坐标重合也必须作为退化网格拒绝。"""

    from cat_surface_gpu.surf2sphere import surf_to_sphere_cupy

    vertices, faces = create_icosphere(0)
    vertices[:] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="20 个面积为零"):
        surf_to_sphere_cupy(vertices, faces)


def test_warp_api_checks_paired_vertex_count_before_cuda():
    vertices, faces = create_icosphere(0)
    with pytest.raises(ValueError, match="数量"):
        surfwarp_cupy(
            vertices,
            np.concatenate((vertices, vertices[:1])),
            faces,
            vertices,
            vertices,
            faces,
        )


def test_warp_cli_rejects_mismatched_faces_before_cuda(tmp_path, monkeypatch, capsys):
    vertices, faces, paths = _surface_files(tmp_path)
    save_gifti_surface(GiftiSurface(vertices, faces[::-1]), paths["source.sphere"])

    def unexpected_device(name):
        raise AssertionError("不应在网格校验前访问 CUDA")

    monkeypatch.setattr(cli, "_cuda_device", unexpected_device)
    with pytest.raises(SystemExit) as error:
        cli.surfwarp_main(
            [
                "--source-surface",
                str(paths["source.surface"]),
                "--source-sphere",
                str(paths["source.sphere"]),
                "--target-surface",
                str(paths["target.surface"]),
                "--target-sphere",
                str(paths["target.sphere"]),
                "--output",
                str(tmp_path / "registered.gii"),
            ]
        )
    assert error.value.code == 2
    assert "三角形编号或顺序不一致" in capsys.readouterr().err


def test_sphere_cli_preserves_faces_and_reports_progress(tmp_path, monkeypatch, capsys):
    vertices, faces, paths = _surface_files(tmp_path)
    output = tmp_path / "output.gii"
    seen = {}
    monkeypatch.setattr(cli, "_cuda_device", lambda name: nullcontext())

    def candidate(points, triangles, parameters, topology, *, progress):
        seen["stop_at"] = parameters.stop_at
        progress("test stage")
        return points * 2, {"total": 0.25}

    monkeypatch.setattr(cli, "surf_to_sphere_cupy", candidate)
    assert (
        cli.surf2sphere_main(
            [
                "--input-surface",
                str(paths["source.surface"]),
                "--output-surface",
                str(output),
                "--stop-at",
                "5",
            ]
        )
        == 0
    )
    result = load_gifti_surface(output)
    assert np.array_equal(result.faces, faces)
    assert np.array_equal(result.vertices, vertices * 2)
    assert seen["stop_at"] == 5
    captured = capsys.readouterr()
    assert "test stage" in captured.err
    assert json.loads(captured.out)["backend"] == "python-cupy"


def test_sphere_cli_rejects_degenerate_output_without_writing(
    tmp_path, monkeypatch, capsys
):
    """有限坐标和相同半径不足以构成可用球面，不得写出退化结果。"""

    vertices, _, paths = _surface_files(tmp_path)
    output = tmp_path / "invalid.gii"
    monkeypatch.setattr(cli, "_cuda_device", lambda name: nullcontext())
    monkeypatch.setattr(
        cli,
        "surf_to_sphere_cupy",
        lambda *args, **kwargs: (np.repeat(vertices[:1], len(vertices), axis=0), {}),
    )
    with pytest.raises(SystemExit) as error:
        cli.surf2sphere_main(
            [
                "--input-surface",
                str(paths["source.surface"]),
                "--output-surface",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert "面积为零" in capsys.readouterr().err
    assert not output.exists()


def test_warp_cli_passes_parameters_and_preserves_source_faces(
    tmp_path, monkeypatch, capsys
):
    vertices, faces, paths = _surface_files(tmp_path)
    output = tmp_path / "registered.gii"
    monkeypatch.setattr(cli, "_cuda_device", lambda name: nullcontext())

    def candidate(
        source,
        sphere,
        source_faces,
        target,
        target_sphere,
        target_faces,
        parameters,
        **kwargs,
    ):
        assert parameters.average is False
        assert parameters.map_size == (32, 16)
        assert kwargs["solver_iterations"] == 7
        return SurfWarpResult(
            sphere, np.zeros((2, 16, 32)), np.zeros((2, 16, 32)), {"total": 0.5}, ()
        )

    monkeypatch.setattr(cli, "surfwarp_cupy", candidate)
    assert (
        cli.surfwarp_main(
            [
                "--source-surface",
                str(paths["source.surface"]),
                "--source-sphere",
                str(paths["source.sphere"]),
                "--target-surface",
                str(paths["target.surface"]),
                "--target-sphere",
                str(paths["target.sphere"]),
                "--output",
                str(output),
                "--no-avg",
                "--map-size",
                "32",
                "16",
                "--solver-iterations",
                "7",
            ]
        )
        == 0
    )
    assert np.array_equal(load_gifti_surface(output).faces, faces)
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def test_cli_rejects_cpu_device():
    with pytest.raises(ValueError, match="CUDA"):
        cli._cuda_device("cpu")


def test_runtime_has_no_cat_executables_or_old_backends():
    package = Path(cat_surface_gpu.__file__).parent
    blocked = {"subprocess", "ctypes", "torch", "triton", "simnibs"}
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not blocked.intersection(
                    alias.name.split(".")[0] for alias in node.names
                )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in blocked
    assert not any(
        path.suffix in {".so", ".dll", ".exe"} for path in package.rglob("*")
    )
