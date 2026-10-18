"""独立包的任务缓存与设备选择，不依赖 FlashCHARM。"""

from contextlib import nullcontext
import os
from pathlib import Path
import sys


def cache_directory(name: str) -> Path:
    """将目标缓存写入调用者指定位置，默认归入当前目录的任务子目录。"""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Cache name must be a single directory component")
    root = Path(
        os.environ.get("CAT_SURFACE_CACHE_ROOT", Path.cwd() / ".work/cat-surface/cache")
    )
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_device(device: str) -> str:
    """显式设备不会静默回退；自动模式优先选择本机可用 GPU。"""
    if device != "auto":
        return device
    if sys.platform == "darwin":
        try:
            from .backends.metal.surf2sphere import _mlx

            _mlx()
            return "metal"
        except (ImportError, RuntimeError):
            return "cpu"
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cuda"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def host_device_context(device: str):
    """检查 CPU 或 Metal 可用性；CUDA 上下文由原 CLI 管理。"""
    if device == "metal":
        from .backends.metal.surf2sphere import _mlx

        _mlx()
    elif device != "cpu":
        raise ValueError("Device must be auto, cpu, metal, cuda or cuda:N")
    return nullcontext()
