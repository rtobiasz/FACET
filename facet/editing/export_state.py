"""Export a FACET 2D checkpoint to the stable planar-edit NPZ sidecar."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import torch

from facet.io.checkpoint import load_training_checkpoint
from facet.editing.state_io import file_sha256, save_planar_edit_state, state_from_mean_xy


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    state = load_training_checkpoint(checkpoint, torch.device("cpu"), require_rgb=True)
    if state.config is None or state.config.get("mode") != "image_2d":
        raise ValueError("checkpoint is not an image_2d checkpoint")
    edit_state = state_from_mean_xy(
        state.model["mean_xy"],
        metadata={
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha256(checkpoint),
            "source_iteration": state.iteration,
        },
    )
    save_planar_edit_state(args.output, edit_state)
    print(
        f"Exported {edit_state.primitive_count} primitives to "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
