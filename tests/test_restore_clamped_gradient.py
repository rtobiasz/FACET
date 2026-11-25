"""The optional inward gradient through the [0, 1] render clamp."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from facet.training.step import _loss_and_image_gradient


def step(target: torch.Tensor, restore: bool) -> SimpleNamespace:
    count = target.shape[0]
    return SimpleNamespace(
        rays=SimpleNamespace(flat_indices=torch.arange(count)),
        foreground=target, alpha=torch.ones(count, 1), background=torch.zeros(count, 3),
        loss_weights=None, restore_clamped_gradient=restore)


class RestoreClampedGradientTest(unittest.TestCase):
    # Rows: over-bright pixel above a target below 1; over-bright pixel on a
    # white target; negative render above a positive target; in range.
    rendered = torch.tensor([[1.2, 1.2, 0.5], [1.3, 1.3, 1.3], [-0.2, 0.4, 0.4], [0.3, 0.6, 0.9]])
    target = torch.tensor([[0.9, 1.0, 0.5], [1.0, 1.0, 1.0], [0.1, 0.4, 0.4], [0.2, 0.6, 0.8]])

    def gradients(self, restore: bool):
        return _loss_and_image_gradient(step(self.target, restore), self.rendered)

    def test_default_drops_the_gradient_outside_the_range(self):
        _, image, gradient = self.gradients(False)
        outside = (self.rendered > 1.0) | (self.rendered < 0.0)
        self.assertTrue(torch.all(gradient[outside] == 0.0))
        self.assertTrue(torch.equal(gradient[~outside], image.grad[~outside]))

    def test_restore_keeps_only_the_inward_gradient(self):
        loss_default, _, default = self.gradients(False)
        loss_restore, image, restored = self.gradients(True)
        self.assertEqual(loss_default.item(), loss_restore.item())
        # 1.2 over 0.9 and -0.2 over 0.1 now pull back into range.
        self.assertGreater(restored[0, 0].item(), 0.0)
        self.assertLess(restored[2, 0].item(), 0.0)
        # Over a white target the clamped render is already exact.
        self.assertTrue(torch.all(restored[1] == 0.0))
        self.assertEqual(restored[0, 1].item(), 0.0)
        inside = (self.rendered >= 0.0) & (self.rendered <= 1.0)
        self.assertTrue(torch.equal(restored[inside], default[inside]))


if __name__ == "__main__":
    unittest.main()
