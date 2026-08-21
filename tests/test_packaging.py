"""Packaging, resource, and synthetic-fixture contracts."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np

from cat_surface_gpu.resources import bundled_binary, bundled_binary_names


def _load_fixture_module():
    path = Path(__file__).parents[1] / "tools" / "generate_synthetic_fixture.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_binaries_are_verified_and_executable():
    """Every declared helper must pass its committed SHA-256 contract."""

    assert bundled_binary_names() == (
        "CAT_Surf2Sphere",
        "CAT_SurfWarp",
        "cat_surface_rotation_depth",
        "cat_surface_stencil_builder",
    )
    for name in bundled_binary_names():
        path = bundled_binary(name)
        assert path.is_file()
        assert path.stat().st_mode & 0o111


def test_synthetic_fixture_is_deterministic(tmp_path):
    """The public smoke fixture must contain matching, deterministic topology."""

    module = _load_fixture_module()
    first = module.generate_fixture(tmp_path / "first", subdivisions=2)
    second = module.generate_fixture(tmp_path / "second", subdivisions=2)
    assert set(first) == set(second)
    for name in first:
        first_image = nib.load(first[name])
        second_image = nib.load(second[name])
        assert len(first_image.darrays) == 2
        assert np.array_equal(first_image.darrays[0].data, second_image.darrays[0].data)
        assert np.array_equal(first_image.darrays[1].data, second_image.darrays[1].data)
