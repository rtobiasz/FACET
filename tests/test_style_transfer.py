import unittest

import torch

from facet.stylization.losses import (
    LatentOptimizationConfig,
    restrict_conditioning_dimensions,
    style_optimizer_param_groups,
)


class ConditioningRestrictionTests(unittest.TestCase):
    def test_zero_dimensions_zero_values_and_gradients(self):
        source = torch.randn(3, 96, requires_grad=True)
        restricted = restrict_conditioning_dimensions(source, 0)
        self.assertTrue(torch.equal(restricted, torch.zeros_like(restricted)))
        restricted.sum().backward()
        self.assertTrue(torch.equal(source.grad, torch.zeros_like(source)))

    def test_prefix_is_preserved_and_suffix_is_inactive(self):
        source = torch.randn(2, 96, requires_grad=True)
        restricted = restrict_conditioning_dimensions(source, 8)
        self.assertTrue(torch.equal(restricted[:, :8], source[:, :8]))
        self.assertTrue(torch.equal(restricted[:, 8:], torch.zeros_like(source[:, 8:])))
        restricted.sum().backward()
        self.assertTrue(torch.equal(source.grad[:, :8], torch.ones_like(source[:, :8])))
        self.assertTrue(torch.equal(source.grad[:, 8:], torch.zeros_like(source[:, 8:])))

    def test_invalid_dimension_is_rejected(self):
        source = torch.zeros(1, 96)
        for dimensions in (-1, 97):
            with self.assertRaises(ValueError):
                restrict_conditioning_dimensions(source, dimensions)

    def test_rgb_is_an_appearance_optimizer_parameter(self):
        rgb = torch.nn.Parameter(torch.zeros(2, 3))
        groups = style_optimizer_param_groups(
            {"RGB": rgb}, LatentOptimizationConfig(model_lr=0.0123)
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["lr"], 0.0123)
        self.assertIs(groups[0]["params"][0], rgb)


if __name__ == "__main__":
    unittest.main()
