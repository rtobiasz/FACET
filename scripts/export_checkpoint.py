#!/usr/bin/env python3
"""Strip a training checkpoint to its inference state (no optimizer; the same renders).

    python scripts/export_checkpoint.py --input runs/kodim01/best.checkpoint --output kodim01.checkpoint
"""
import sys

from facet.cli import run

if __name__ == "__main__":
    sys.exit(run("facet.io.export_inference", "export"))
