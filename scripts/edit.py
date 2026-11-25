#!/usr/bin/env python3
"""Geometric editing with frame transport. Commands (run one with --help for its options):

  mask          foreground mask of an RGBA image (selects the components an edit moves)
  export-state  write a checkpoint's components as an editable state (npz)
  make-states   apply an edit spec (configs/edits/*.json) to a state, one state per frame
  render        render a sequence of states (edits or simulations), optionally as a video

    python scripts/edit.py render --config configs/edit_lighthouse_bend.toml --checkpoint ... --edit_state_dir ...
"""
import sys

from facet.cli import dispatch

COMMANDS = {
    "mask": ("facet.editing.foreground_mask", "mask", "foreground mask of an RGBA image"),
    "export-state": ("facet.editing.export_state", "export_state", "checkpoint -> editable state"),
    "make-states": ("facet.editing.make_states", "states", "edit spec -> per-frame states"),
    "render": ("facet.editing.render_sequence", "render_edit", "render a state sequence"),
}

if __name__ == "__main__":
    sys.exit(dispatch(COMMANDS, __doc__))
