"""Build a platform wheel containing verified Linux helpers and notices."""

# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildPyWithReleaseAssets(build_py):
    """Copy repository-level binaries and notices into the installed package."""

    def run(self) -> None:
        super().run()
        package = Path(self.build_lib) / "cat_surface_gpu"
        shutil.copytree(
            ROOT / "bin" / "linux-x86_64",
            package / "bin" / "linux-x86_64",
            dirs_exist_ok=True,
        )
        licenses = package / "licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        for source in (
            ROOT / "LICENSE",
            ROOT / "THIRD_PARTY_NOTICES.md",
            ROOT / "upstream" / "CAT-Surface-LICENSE.txt",
        ):
            shutil.copy2(source, licenses / source.name)


class LinuxPlatformWheel(bdist_wheel):
    """Mark the wheel as platform-specific because it contains Linux ELF files."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(
    cmdclass={
        "build_py": BuildPyWithReleaseAssets,
        "bdist_wheel": LinuxPlatformWheel,
    }
)
