#!/usr/bin/env python3
"""Lifting into depth and ray tracing. Commands (run one with --help for its options):

  render         lift a fitted image with a depth map and render target views or ray-traced scenes
  evaluate       score lifted renders against held-out views (PSNR, SSIM, LPIPS, holes)
  depth          estimate a depth map with Depth Pro
  mesh-baseline  the textured depth-mesh baseline of the lifting table
  fill-depth     fill invalid ground-truth depth inside the object mask from the nearest valid pixel
  prepare-nerf-synthetic, prepare-dtu   build a lifting scene (source view, depth, target cameras and views)

    python scripts/lift.py render --config configs/lift_hotdog.toml --checkpoint ... --depth ... --scene-dir ...
"""
import sys

from facet.cli import dispatch

COMMANDS = {
    "render": ("facet.lifting.lift", "lift", "lift and render"),
    "evaluate": ("facet.lifting.evaluate", "lift_evaluate", "evaluate lifted views"),
    "depth": ("facet.lifting.depth_pro", "depth", "Depth Pro depth map"),
    "mesh-baseline": ("facet.lifting.mesh_baseline", "mesh_baseline", "textured depth-mesh baseline"),
    "fill-depth": ("facet.lifting.fill_depth", "fill_depth", "fill GT depth holes inside the mask"),
    "prepare-nerf-synthetic": ("facet.lifting.prepare_nerf_synthetic", "prepare", "NeRF-Synthetic lifting scene"),
    "prepare-dtu": ("facet.lifting.prepare_dtu", "prepare", "DTU lifting scene"),
}

if __name__ == "__main__":
    sys.exit(dispatch(COMMANDS, __doc__))
