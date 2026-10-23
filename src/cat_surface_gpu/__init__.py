"""基于官方 CAT-Surface 算法独立实现的 Python/CuPy 后端。"""

__version__ = "0.3.1"

from .parameters import Surf2SphereParameters, SurfWarpParameters
from .dartel import (
    compose,
    dartel_symmetric_step,
    exponentiate_velocity,
    identity_mapping,
    membrane_operator,
    objective_terms,
    small_deformation,
)
from .geometry import ellipsoid_to_sphere, surface_area, vertex_areas
from .features import (
    build_smoothing_topology,
    depth_potential_feature_cupy,
    type5_depth_feature_cupy,
)
from .io import GiftiSurface, load_gifti_surface, save_gifti_surface
from .mapping import (
    SphereSheetMapping,
    apply_sphere_sheet_mapping,
    build_sphere_sheet_mapping,
    map_sphere_values_to_sheet,
    sheet_unit_points,
)
from .resampling import (
    ResamplingStencil,
    ResamplingTopology,
    SphericalResampler,
    apply_resampling_stencil,
    build_resampling_stencil,
    build_resampling_topology,
    cat_resampling_sphere,
    create_icosphere,
    nearest_vertices_cpu,
    nearest_vertices_cupy,
    prepare_spherical_resampler,
)
from .surf2sphere import surf_to_sphere, surf_to_sphere_cupy
from .rotation import (
    RotationGridIndex,
    RotationSearchResult,
    average_xz_sphere,
    official_rotation_seed_angles,
    rotate_vertices,
    rotation_matrix_from_angles,
)
from .surfwarp import (
    SurfWarpResult,
    apply_deformation_to_sphere,
    surfwarp_cupy,
    surfwarp_single_run_cupy,
)
from .topology import SurfaceTopology, build_surface_topology

__all__ = [
    "GiftiSurface",
    "ResamplingStencil",
    "ResamplingTopology",
    "SphericalResampler",
    "Surf2SphereParameters",
    "SurfWarpParameters",
    "SurfWarpResult",
    "SurfaceTopology",
    "SphereSheetMapping",
    "apply_sphere_sheet_mapping",
    "apply_deformation_to_sphere",
    "build_surface_topology",
    "build_smoothing_topology",
    "depth_potential_feature_cupy",
    "build_sphere_sheet_mapping",
    "build_resampling_stencil",
    "build_resampling_topology",
    "cat_resampling_sphere",
    "compose",
    "dartel_symmetric_step",
    "ellipsoid_to_sphere",
    "exponentiate_velocity",
    "identity_mapping",
    "create_icosphere",
    "membrane_operator",
    "objective_terms",
    "load_gifti_surface",
    "map_sphere_values_to_sheet",
    "save_gifti_surface",
    "apply_resampling_stencil",
    "surface_area",
    "sheet_unit_points",
    "small_deformation",
    "nearest_vertices_cupy",
    "nearest_vertices_cpu",
    "prepare_spherical_resampler",
    "surf_to_sphere_cupy",
    "surf_to_sphere",
    "surfwarp_single_run_cupy",
    "surfwarp_cupy",
    "type5_depth_feature_cupy",
    "RotationGridIndex",
    "RotationSearchResult",
    "average_xz_sphere",
    "official_rotation_seed_angles",
    "rotate_vertices",
    "rotation_matrix_from_angles",
    "vertex_areas",
]
