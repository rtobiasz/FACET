#!/usr/bin/env python3
"""Super-resolve a fitted low-resolution image and score it against the high-resolution source.

Fit first with `scripts/train.py --config configs/sr_x8.toml` (degradation-consistent objective).

    python scripts/super_resolve.py --config configs/sr_x8.toml --checkpoint ... --source_path LR.png \
        --target_path HR.png --output_path ...
"""
import sys

from facet.cli import run

if __name__ == "__main__":
    sys.exit(run("facet.superres.evaluate", "super_resolve"))
