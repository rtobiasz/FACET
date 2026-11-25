#!/usr/bin/env python3
"""Render a fitted image (any resolution) and report its PSNR against the source image.

    python scripts/render.py --checkpoint checkpoints/kodim23.checkpoint --source_path data/kodak/kodim23.png \
        --output renders/kodim23.png [--scale 4]
"""
import sys

from facet.cli import run

if __name__ == "__main__":
    sys.exit(run("facet.evaluation.render", "render"))
