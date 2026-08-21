#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compatibility wrapper for the installed CAT-Surf2Sphere GPU CLI."""

from cat_surface_gpu.cli import surf2sphere_main


def main() -> int:
    """Forward to the stable package CLI."""

    return surf2sphere_main()


if __name__ == "__main__":
    raise SystemExit(main())
