from __future__ import annotations

import unittest

import torch

from facet.evaluation.reconstruction import (
    metric_image,
    multi_scale_ssim,
    optional_multi_scale_ssim,
)


class PlanarEvaluationTest(unittest.TestCase):
    def test_metric_image_matches_training_and_saved_rgb_domain(self) -> None:
        image = torch.tensor([-2.0, 0.25, 1.0, 10.0])
        torch.testing.assert_close(
            metric_image(image), torch.tensor([0.0, 0.25, 1.0, 1.0])
        )

    def test_multi_scale_ssim_uses_float_tensor_values(self) -> None:
        target = torch.full((1, 3, 192, 192), 64.0 / 255.0)
        sub_png_difference = 0.25 / 255.0
        rendered = target + sub_png_difference

        self.assertTrue(
            torch.equal((rendered * 255.0).round(), (target * 255.0).round())
        )

        identical = multi_scale_ssim(target, target)
        perturbed = multi_scale_ssim(rendered, target)

        torch.testing.assert_close(identical, torch.ones_like(identical))
        self.assertLess(float(perturbed.item()), float(identical.item()))

    def test_multi_scale_ssim_is_omitted_when_default_window_does_not_fit(self) -> None:
        image = torch.zeros((1, 3, 160, 255))
        self.assertIsNone(optional_multi_scale_ssim(image, image))

    def test_multi_scale_ssim_size_error_is_omitted(self) -> None:
        image = torch.zeros((1, 3, 169, 255))

        def rejects_size(*_args):
            raise ValueError("image height must be larger than 160")

        self.assertIsNone(optional_multi_scale_ssim(image, image, rejects_size))

    def test_multi_scale_ssim_width_error_is_omitted(self) -> None:
        image = torch.zeros((1, 3, 169, 255))

        def rejects_size(*_args):
            raise ValueError("image width must be larger than 160")

        self.assertIsNone(optional_multi_scale_ssim(image, image, rejects_size))

    def test_multi_scale_ssim_unrelated_value_error_is_not_hidden(self) -> None:
        image = torch.zeros((1, 3, 169, 255))

        def rejects_input(*_args):
            raise ValueError("unrelated metric failure")

        with self.assertRaisesRegex(ValueError, "unrelated metric failure"):
            optional_multi_scale_ssim(image, image, rejects_input)


if __name__ == "__main__":
    unittest.main()
