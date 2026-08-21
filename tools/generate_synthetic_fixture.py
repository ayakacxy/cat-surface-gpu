#!/usr/bin/env python3
"""Generate deterministic, non-anatomical GIFTI meshes for smoke testing."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return a unit icosahedron with consistently oriented triangular faces."""

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices, faces


def _subdivide(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide every triangle once and project new vertices to the unit sphere."""

    points = [point.copy() for point in vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(first: int, second: int) -> int:
        edge = tuple(sorted((first, second)))
        if edge not in midpoint_cache:
            point = (points[first] + points[second]) * 0.5
            point /= np.linalg.norm(point)
            midpoint_cache[edge] = len(points)
            points.append(point)
        return midpoint_cache[edge]

    output_faces: list[tuple[int, int, int]] = []
    for first, second, third in faces:
        first_second = midpoint(int(first), int(second))
        second_third = midpoint(int(second), int(third))
        third_first = midpoint(int(third), int(first))
        output_faces.extend(
            [
                (int(first), first_second, third_first),
                (int(second), second_third, first_second),
                (int(third), third_first, second_third),
                (first_second, second_third, third_first),
            ]
        )
    return np.asarray(points, dtype=np.float64), np.asarray(
        output_faces, dtype=np.int32
    )


def create_icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic unit icosphere."""

    if subdivisions < 0 or subdivisions > 6:
        raise ValueError("subdivisions must be between 0 and 6")
    vertices, faces = _icosahedron()
    for _ in range(subdivisions):
        vertices, faces = _subdivide(vertices, faces)
    return vertices, faces


def _write_gifti(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write one two-array GIFTI surface."""

    point_array = GiftiDataArray(
        np.ascontiguousarray(vertices, dtype=np.float32),
        intent="NIFTI_INTENT_POINTSET",
    )
    face_array = GiftiDataArray(
        np.ascontiguousarray(faces, dtype=np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
    )
    face_array.coordsys = None
    image = GiftiImage(darrays=[point_array, face_array])
    nib.save(image, str(path))


def generate_fixture(output: Path, subdivisions: int = 3) -> dict[str, Path]:
    """Write source and target white/sphere meshes without anatomical data."""

    output.mkdir(parents=True, exist_ok=True)
    sphere, faces = create_icosphere(subdivisions)
    x_value, y_value, z_value = sphere.T
    source_radius = 80.0 + 4.0 * x_value * y_value + 2.0 * z_value**2
    target_radius = 82.0 + 3.0 * y_value * z_value - 1.5 * x_value**2
    meshes = {
        "source.white.gii": sphere * source_radius[:, None],
        "source.sphere.gii": sphere,
        "target.white.gii": sphere * target_radius[:, None],
        "target.sphere.gii": sphere,
    }
    paths = {}
    for name, vertices in meshes.items():
        path = output / name
        _write_gifti(path, vertices, faces)
        paths[name] = path
    return paths


def main() -> int:
    """Generate a fixture from command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subdivisions", type=int, default=3)
    args = parser.parse_args()
    paths = generate_fixture(args.output, args.subdivisions)
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
