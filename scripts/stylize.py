#!/usr/bin/env python3
"""Stylize a fitted image with CLIP (image or text style) while its geometry stays fixed.

    python scripts/stylize.py --config configs/style_image.toml --checkpoint ... --style_image ... --output_path ...
"""
import sys

from facet.cli import run

if __name__ == "__main__":
    sys.exit(run("facet.stylization.stylize", "stylize"))
