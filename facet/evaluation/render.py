"""Render a fitted image at any resolution and, optionally, score it against its source image.

    python scripts/render.py --checkpoint checkpoints/kodim23.checkpoint \
        --source_path data/kodak/kodim23.png --output renders/kodim23.png [--scale 4]

The source image fixes the canvas (width, height, background); rendering at --scale > 1 samples
the same continuous image plane on a finer grid. Writes the PNG and, next to it, a JSON with the
float PSNR of the native-resolution render against the source (the metric of the paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from facet.data.image_dataset import Image2DDataset
from facet.evaluation.metrics import psnr
from facet.evaluation.reconstruction import _save_rgb
from facet.io.checkpoint import load_training_checkpoint
from facet.model.components import PlanarGaussianModel
from facet.renderer.facade import FACETRenderer
from facet.renderer.planar import render_planar, render_planar_resolution
from facet.training.config import Image2DConfig


def load_model(checkpoint: Path, source_path: Path | None, device: torch.device):
    """Model, dataset and renderer of an image checkpoint."""
    state = load_training_checkpoint(checkpoint, device, require_rgb=True)
    if state.config is None or state.config.get("mode") != "image_2d":
        raise ValueError(f"{checkpoint} is not an image checkpoint")
    image = dict(state.config["image"])
    if source_path is not None:
        image["source_path"] = str(source_path.resolve())
    dataset = Image2DDataset(Image2DConfig(**image), device=device)
    model = PlanarGaussianModel(
        state.model,
        plane_z=dataset.config.plane_z,
        requires_grad=False,
        active_conditioning_dimensions=state.config.get("ablation", {}).get("active_conditioning_dimensions", 96),
    )
    return state, model, dataset, FACETRenderer(8, 11.3449, dataset.ray_count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source_path", type=Path, help="the fitted image (default: the path stored in the checkpoint)")
    parser.add_argument("--output", type=Path, required=True, help="PNG to write")
    parser.add_argument("--scale", type=int, default=1, help="render at this multiple of the source resolution")
    parser.add_argument("--background_rgb", type=float, nargs=3, help="default: the training background")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    state, model, dataset, renderer = load_model(args.checkpoint, args.source_path, device)
    background = tuple(args.background_rgb) if args.background_rgb else None
    with torch.no_grad():
        native = render_planar(renderer, model, dataset, background_rgb=background)
        target = dataset.target.reshape(dataset.height, dataset.width, 3)
        # as at training time (the PSNR stored with the checkpoint)
        score = float(psnr(native.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0).contiguous(),
                           target.permute(2, 0, 1).unsqueeze(0).contiguous()).mean().item())
        image = native if args.scale == 1 else render_planar_resolution(
            renderer, model, dataset, dataset.width * args.scale, dataset.height * args.scale,
            background_rgb=background)
    _save_rgb(image, args.output)
    report = {"checkpoint": str(args.checkpoint), "iteration": state.iteration, "width": dataset.width,
              "height": dataset.height, "scale": args.scale, "psnr": score}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
