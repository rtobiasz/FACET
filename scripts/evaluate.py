#!/usr/bin/env python3
"""Evaluate a fitted image: PSNR, SSIM, MS-SSIM, LPIPS and render time against its source image.

    python scripts/evaluate.py --checkpoint runs/kodim01/best.checkpoint --output runs/kodim01/evaluation
"""
import sys

from facet.cli import run

if __name__ == "__main__":
    sys.exit(run("facet.evaluation.evaluate_checkpoint", "evaluate"))
