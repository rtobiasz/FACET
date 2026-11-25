#!/usr/bin/env python3
"""Physics on a fitted image. Commands (run one with --help for its options):

  mpm     material point method (elastic or sand) on the components of an editable state; render the
          resulting states with `scripts/edit.py render --transport_deformation`
  cloth   cloth simulation of a triangle cage (NVIDIA Warp) that `scripts/lift.py --cage` deforms
"""
import sys

from facet.cli import dispatch

COMMANDS = {
    "mpm": ("facet.physics.simulate_mpm", "simulate", "MPM simulation of an editable state"),
    "cloth": ("facet.physics.simulate_cloth", "cloth", "cloth cage simulation"),
}

if __name__ == "__main__":
    sys.exit(dispatch(COMMANDS, __doc__))
