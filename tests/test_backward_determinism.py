"""The renderer backward must return the same gradients for the same inputs.

Without it, three identical training runs
of kodim18 diverged at iteration 4 by one float32 ULP and were 16.20 dB apart by
the end. The cause was float `atomicAdd` accumulation whose summation order
followed whatever the hardware scheduled, so this test reruns one backward pass
on unchanged inputs and demands bit-identical output.

The scene is deliberately larger than the one in test_native_renderer.py: the
divergence needs many rays landing on the same primitive from different warps,
which a handful of rays will not produce.
"""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import sys
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

from facet.renderer.backend import NativeFACETBackend
from facet.renderer.facade import FACETRenderer


class _Geometry:
    def __init__(self, test) -> None:
        self.test = test

    def renderer_geometry(self):
        test = self.test
        return (
            test.means,
            torch.exp(test.log_scales),
            test.quaternions,
            torch.sigmoid(test.opacity_logits),
            1.0 + torch.nn.functional.softplus(test.raw_kappa),
        )


@unittest.skipUnless(
    torch.cuda.is_available() and RENDERER_SO.is_file(),
    "backward determinism requires CUDA and a built FACET renderer",
)
class BackwardDeterminismTest(unittest.TestCase):
    """Two identical backward passes must agree bit for bit."""

    PRIMITIVES = 96
    SIDE = 96

    @classmethod
    def setUpClass(cls) -> None:
        if str(RENDERER_DIR) not in sys.path:
            sys.path.insert(0, str(RENDERER_DIR))
        cls.backend = NativeFACETBackend(__import__("PYOPTIXFACETRENDERER"))
        cls.device = torch.device("cuda:0")
        cls.ray_count = cls.SIDE * cls.SIDE
        cls.background = (0.01, 0.02, 0.03)
        cls.threshold = 1.0e-4
        cls.reg_a, cls.reg_b = -0.01, 1.01

        directions = cls.backend.generate_rays(
            torch.tensor([1.0, 0.0, 0.0], device=cls.device),
            torch.tensor([0.0, 1.0, 0.0], device=cls.device),
            torch.tensor([0.0, 0.0, 1.0], device=cls.device),
            cls.SIDE, cls.SIDE, 1.0, 1.0,
        ).reshape(cls.ray_count, 3)
        cls.directions = torch.nn.functional.normalize(directions, dim=1).contiguous()
        cls.origins = torch.zeros_like(cls.directions)

        generator = torch.Generator(device="cpu").manual_seed(178)

        def sample(shape, scale, dtype=torch.float32):
            return (
                torch.randn(shape, generator=generator)
                .mul(scale).to(device=cls.device, dtype=dtype).contiguous()
            )

        # Primitives are stacked in front of the camera and are wide enough to
        # overlap, so most rays traverse most of them and every primitive is
        # written by rays from many different warps.
        count = cls.PRIMITIVES
        centres = sample((count, 3), 0.25)
        centres[:, 2] = torch.linspace(2.5, 4.0, count, device=cls.device)
        cls.means = centres.contiguous()
        cls.log_scales = torch.full((count, 2), math.log(0.6), device=cls.device)
        quaternions = sample((count, 4), 1.0)
        quaternions[:, 0] += 3.0
        cls.quaternions = torch.nn.functional.normalize(quaternions, dim=1).contiguous()
        cls.rgb = sample((count, 3), 0.3).abs().clamp(0.0, 1.0).contiguous()
        cls.opacity_logits = sample((count, 1), 0.5).contiguous()
        cls.rgba = torch.cat((cls.rgb, cls.opacity_logits), dim=1).contiguous()
        cls.raw_kappa = sample((count, 1), 0.2).contiguous()
        cls.target = torch.linspace(
            0.05, 0.95, cls.ray_count * 3, device=cls.device
        ).reshape(cls.ray_count, 3)

        cls.renderer = FACETRenderer(8, 11.3449, cls.ray_count, backend=cls.backend)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.renderer
        gc.collect()
        torch.cuda.empty_cache()

    def setUp(self) -> None:
        self.renderer.sync_geometry(_Geometry(self))

    def _auxiliary(self) -> dict[str, torch.Tensor]:
        return {
            "depth": torch.zeros((self.ray_count, 4), device=self.device),
            "depth_and_index": torch.zeros((self.ray_count, 2), device=self.device),
            "surface_normal": torch.zeros((self.ray_count, 3), device=self.device),
            "normal": torch.zeros((self.ray_count, 4), device=self.device),
        }

    def _backward_once(self) -> dict[str, torch.Tensor]:
        image = torch.zeros((self.ray_count, 3), device=self.device)
        auxiliary = self._auxiliary()
        self.renderer.forward_training_base(
            self.origins, self.directions, image, *self.background,
            self.means, self.log_scales, self.quaternions, self.rgba,
            self.raw_kappa, self.threshold, auxiliary["depth"],
            self.reg_a, self.reg_b, auxiliary["depth_and_index"],
            auxiliary["surface_normal"], auxiliary["normal"],
        )
        image_gradient = (2.0 * (image - self.target) / image.numel()).contiguous()
        gradients = {
            "RGB": torch.zeros_like(self.rgb),
            "A": torch.zeros_like(self.opacity_logits),
            "k": torch.zeros_like(self.raw_kappa),
            "m": torch.zeros_like(self.means),
            "s": torch.zeros_like(self.log_scales),
            "q": torch.zeros_like(self.quaternions),
        }
        self.renderer.backward_base(
            self.origins, self.directions, *self.background,
            self.means, self.log_scales, self.quaternions, self.rgba,
            self.raw_kappa, image.clone(), image_gradient,
            gradients["RGB"], gradients["A"], gradients["k"],
            gradients["m"], gradients["s"], gradients["q"],
            self.threshold, auxiliary["depth"],
            torch.zeros((self.ray_count, 4), device=self.device),
            0.0, self.reg_a, self.reg_b, auxiliary["depth_and_index"],
            auxiliary["surface_normal"], auxiliary["normal"], 0.0,
        )
        return gradients

    def test_repeated_backward_is_bitwise_identical(self) -> None:
        reference = self._backward_once()
        # A scene that never contends would pass this test for the wrong
        # reason, so require that the pass actually produced gradients.
        self.assertTrue(
            any(tensor.abs().sum().item() > 0.0 for tensor in reference.values()),
            "the fixture produced no gradients, so the test proves nothing",
        )
        for attempt in range(3):
            self.setUp()
            repeat = self._backward_once()
            for name, expected in reference.items():
                self.assertTrue(
                    torch.equal(expected, repeat[name]),
                    f"{name} differs on repeat {attempt}: greatest absolute "
                    f"difference {(expected - repeat[name]).abs().max().item():.3e}",
                )


if __name__ == "__main__":
    unittest.main()
