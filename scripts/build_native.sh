#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${repo_root}/UPSTREAM.lock"

jobs=${BUILD_JOBS:-8}
build_root=${BUILD_ROOT:-"${repo_root}/build/native"}
output_dir=${OUTPUT_DIR:-"${repo_root}/build/native-output"}
source_dir="${build_root}/CAT-Surface"

if [[ ${jobs} -lt 1 ]]; then
    echo "BUILD_JOBS must be positive" >&2
    exit 2
fi

mkdir -p "${build_root}" "${output_dir}"
if [[ ! -d "${source_dir}/.git" ]]; then
    git clone --filter=blob:none "${CAT_SURFACE_REPOSITORY}" "${source_dir}"
fi

git -C "${source_dir}" fetch --depth 1 origin "${CAT_SURFACE_COMMIT}"
git -C "${source_dir}" checkout --detach "${CAT_SURFACE_COMMIT}"
git -C "${source_dir}" clean -ffdX

pushd "${source_dir}" >/dev/null
./autogen.sh
CFLAGS="-O2 -fPIC -ffile-prefix-map=${source_dir}=CAT-Surface" \
    ./configure --disable-shared
make -j"${jobs}"
popd >/dev/null

include_flags=(
    -DHAVE_ZLIB
    -I"${source_dir}/Include"
    -I"${source_dir}/3rdparty/dartel"
    -I"${source_dir}/3rdparty/genus0"
    -I"${source_dir}/3rdparty/nifti"
    -I"${source_dir}/3rdparty/s2kit10"
    -I"${source_dir}/3rdparty/gifticlib"
    -I"${source_dir}/3rdparty/expat/lib"
    -I"${source_dir}/3rdparty/zlib"
    -I"${source_dir}/3rdparty/volume_io/Include"
    -I"${source_dir}/3rdparty/bicpl-surface/Include"
    -I"${source_dir}/3rdparty/bicpl-surface/Include/bicpl"
    -I"${source_dir}/3rdparty/nii2mesh"
)
common_flags=(
    -O2
    -fPIC
    -ffile-prefix-map="${source_dir}=CAT-Surface"
    -ffile-prefix-map="${repo_root}=cat-surface-gpu"
    -Wl,--build-id=none
)
libcat="${source_dir}/.libs/libCAT.a"

if [[ ! -f "${libcat}" ]]; then
    echo "Static CAT-Surface library was not produced: ${libcat}" >&2
    exit 1
fi

gcc "${common_flags[@]}" "${include_flags[@]}" \
    "${source_dir}/Progs/CAT_Surf2Sphere.c" "${libcat}" -lm -lpthread \
    -o "${output_dir}/CAT_Surf2Sphere"
gcc "${common_flags[@]}" "${include_flags[@]}" \
    "${source_dir}/Progs/CAT_SurfWarp.c" "${libcat}" -lm -lpthread \
    -o "${output_dir}/CAT_SurfWarp"
gcc "${common_flags[@]}" "${include_flags[@]}" \
    "${repo_root}/native/cat_surface_rotation_depth.c" "${libcat}" -lm -lpthread \
    -o "${output_dir}/cat_surface_rotation_depth"
gcc "${common_flags[@]}" "${include_flags[@]}" \
    "${repo_root}/native/cat_surface_stencil_builder.c" "${libcat}" -lm -lpthread \
    -o "${output_dir}/cat_surface_stencil_builder"

strip --strip-unneeded "${output_dir}/CAT_Surf2Sphere" \
    "${output_dir}/CAT_SurfWarp" \
    "${output_dir}/cat_surface_rotation_depth" \
    "${output_dir}/cat_surface_stencil_builder"

(
    cd "${output_dir}"
    sha256sum CAT_Surf2Sphere CAT_SurfWarp \
        cat_surface_rotation_depth cat_surface_stencil_builder > SHA256SUMS
)

echo "Native binaries written to ${output_dir}"
