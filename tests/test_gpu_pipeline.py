"""CAT-Surface GPU implementation."""

# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from cat_surface_gpu import run_cat_surface_gpu_pipeline


@pytest.mark.parametrize("thread_count", (0, 257))
def test_stencil_thread_count_rejects_invalid_values(thread_count):
    """Test stencil thread count rejects invalid values."""

    with pytest.raises(ValueError, match="stencil_threads"):
        run_cat_surface_gpu_pipeline(
            "source.white.gii",
            "source.sphere.gii",
            "target.white.gii",
            "target.sphere.gii",
            "output.gii",
            stencil_threads=thread_count,
        )
