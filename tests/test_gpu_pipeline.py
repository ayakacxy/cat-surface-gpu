"""CAT 曲面 GPU 总入口的参数合同测试。"""

# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from cat_surface_gpu import run_cat_surface_gpu_pipeline


@pytest.mark.parametrize("thread_count", (0, 257))
def test_stencil_thread_count_rejects_invalid_values(thread_count):
    """stencil worker 数超出公开范围时应在读取输入前失败。"""

    with pytest.raises(ValueError, match="stencil_threads"):
        run_cat_surface_gpu_pipeline(
            "source.white.gii",
            "source.sphere.gii",
            "target.white.gii",
            "target.sphere.gii",
            "output.gii",
            stencil_threads=thread_count,
        )
