#!/usr/bin/env python3
"""Fit FACET to one image. Defaults are the paper's Kodak setting; see configs/ for every result.

    python scripts/train.py --source_path data/kodak/kodim01.png --output_path runs/kodim01 [--config configs/kodak.toml]
"""
import sys

from facet.cli import run

if __name__ == "__main__":
    sys.exit(run("facet.training.fit", "train"))
