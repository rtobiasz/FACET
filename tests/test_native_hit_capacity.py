from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
RENDERER_DIR = Path(
    os.environ.get("FACET_RENDERER_DIRECTORY", SOURCE_DIR / "renderer" / "output")
)
RENDERER_SO = RENDERER_DIR / "PYOPTIXFACETRENDERER.so"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from facet.renderer.planar import summarize_hit_batching
from facet.renderer.facade import FACETRenderer
from facet.model.rays import generate_indexed_orthographic_rays


@unittest.skipUnless(
    torch.cuda.is_available() and RENDERER_SO.is_file(),
    "native hit diagnostics require CUDA and a built FACET renderer",
)
class NativeHitCapacityTest(unittest.TestCase):
    def test_equal_depth_order_is_stable_and_batch_load_is_reported(self) -> None:
        device = torch.device("cuda:0")
        primitive_count = 20
        renderer = FACETRenderer(8, 11.3449, 1)
        means = torch.zeros((primitive_count, 3), device=device)
        scales = torch.full((primitive_count, 2), math.log(0.5), device=device)
        quaternions = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device
        ).expand(primitive_count, -1).contiguous()
        opacity = torch.full((primitive_count, 1), 4.0, device=device)
        kappa = torch.zeros((primitive_count, 1), device=device)
        renderer.sync_geometry(
            SimpleNamespace(
                renderer_geometry=lambda: (
                    means,
                    torch.exp(scales),
                    quaternions,
                    torch.sigmoid(opacity),
                    1.0 + torch.nn.functional.softplus(kappa),
                )
            )
        )
        rays = generate_indexed_orthographic_rays(
            torch.tensor([0], device=device), 1, 1, camera_distance=1.0
        )
        t_min = torch.zeros((1,), device=device)

        first = renderer.trace_hits(rays.origins, rays.directions, t_min, 1)
        second = renderer.trace_hits(rays.origins, rays.directions, t_min, 1)

        for first_tensor, second_tensor in zip(first, second):
            self.assertTrue(torch.equal(first_tensor, second_tensor))
        torch.testing.assert_close(
            first[1][:16, 0], torch.arange(16, dtype=torch.int32, device=device)
        )
        report = summarize_hit_batching(first[2], capacity=16)
        self.assertEqual(report.maximum, primitive_count)
        self.assertEqual(report.rays_exceeding_batch_capacity, 1)
        self.assertEqual(report.fraction_exceeding_batch_capacity, 1.0)

    def test_equal_depth_hits_continue_beyond_one_batch(self) -> None:
        device = torch.device("cuda:0")
        primitive_count = 20
        renderer = FACETRenderer(8, 11.3449, 1)
        means = torch.zeros((primitive_count, 3), device=device)
        log_scales = torch.full(
            (primitive_count, 2), math.log(0.5), device=device
        )
        quaternions = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device
        ).expand(primitive_count, -1).contiguous()
        opacity_logits = torch.full((primitive_count, 1), -2.0, device=device)
        raw_kappa = torch.zeros((primitive_count, 1), device=device)
        renderer.sync_geometry(
            SimpleNamespace(
                renderer_geometry=lambda: (
                    means,
                    torch.exp(log_scales),
                    quaternions,
                    torch.sigmoid(opacity_logits),
                    1.0 + torch.nn.functional.softplus(raw_kappa),
                )
            )
        )
        rays = generate_indexed_orthographic_rays(
            torch.tensor([0], device=device), 1, 1, camera_distance=1.0
        )
        rgb = torch.zeros((primitive_count, 3), device=device)
        rgb[16:, 0] = 1.0
        rgba = torch.cat((rgb, opacity_logits), dim=1)
        image = torch.zeros((1, 3), device=device)

        renderer.forward_inference_base(
            rays.origins, rays.directions, image,
            0.0, 0.0, 0.0,
            means, log_scales, quaternions, rgba, raw_kappa,
            1.0e-4,
        )
        torch.cuda.synchronize()

        self.assertGreater(float(image[0, 0]), 0.01)


if __name__ == "__main__":
    unittest.main()
