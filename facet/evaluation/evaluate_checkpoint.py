"""Evaluate a fitted image (PSNR, SSIM, MS-SSIM, LPIPS, render time) against its source image."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from facet.training.config import Image2DConfig
from facet.io.checkpoint import load_training_checkpoint
from facet.evaluation.reconstruction import evaluate_planar_reconstruction
from facet.renderer.facade import FACETRenderer
from facet.data.image_dataset import Image2DDataset
from facet.model.components import PlanarGaussianModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source_path', type=Path,
                        help='the fitted image (default: the path stored in the checkpoint)')
    args = parser.parse_args()
    state = load_training_checkpoint(args.checkpoint, 'cuda', require_rgb=True)
    if not state.config or state.config.get('mode') != 'image_2d':
        raise ValueError('Expected image_2d checkpoint')
    image = dict(state.config['image'])
    if args.source_path is not None:
        image['source_path'] = str(args.source_path.resolve())
    dataset = Image2DDataset(Image2DConfig(**image), device='cuda')
    active = state.config.get('ablation', {}).get(
        'active_conditioning_dimensions', 96
    )
    model = PlanarGaussianModel(
        state.model, plane_z=dataset.config.plane_z, requires_grad=False,
        active_conditioning_dimensions=active,
    )
    renderer = FACETRenderer(8, 11.3449, dataset.ray_count)
    with torch.no_grad():
        result = evaluate_planar_reconstruction(renderer, model, dataset, args.output)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'iteration': state.iteration, 'evaluation': asdict(result)}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
