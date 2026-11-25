from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from facet.training.config import Image2DConfig
from facet.data.image_dataset import Image2DDataset, resolve_single_image_path
from facet.training.trainer import Image2DFullBatchSampler


class Image2DDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "corners.png"
        pixels = np.array(
            [
                [[255, 0, 0, 255], [0, 255, 0, 128]],
                [[0, 0, 255, 0], [255, 255, 255, 64]],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(pixels, mode="RGBA").save(self.image_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_rgba_is_flattened_row_major_and_composited_explicitly(self):
        config = Image2DConfig(
            source_path=str(self.image_path),
            bg_color_R=0.25,
            bg_color_G=0.5,
            bg_color_B=0.75,
        )
        dataset = Image2DDataset(config, dtype=torch.float64)

        self.assertEqual((dataset.width, dataset.height), (2, 2))
        self.assertEqual(dataset.ray_count, 4)
        self.assertEqual(dataset.camera_count, 1)
        expected_foreground = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        expected_alpha = torch.tensor(
            [[1.0], [128.0 / 255.0], [0.0], [64.0 / 255.0]],
            dtype=torch.float64,
        )
        background = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float64)
        expected_target = (
            expected_foreground * expected_alpha
            + background * (1.0 - expected_alpha)
        )
        torch.testing.assert_close(dataset.foreground, expected_foreground)
        torch.testing.assert_close(dataset.alpha, expected_alpha)
        torch.testing.assert_close(dataset.target, expected_target)
        torch.testing.assert_close(dataset.loss_weights, torch.ones((4, 1), dtype=torch.float64))
        torch.testing.assert_close(dataset.flat_indices, torch.arange(4))

    def test_mask_mode_preserves_rgb_and_uses_alpha_as_loss_weight(self):
        dataset = Image2DDataset(
            Image2DConfig(source_path=str(self.image_path), alpha_mode="mask")
        )
        torch.testing.assert_close(dataset.target, dataset.foreground)
        torch.testing.assert_close(dataset.loss_weights, dataset.alpha)

    def test_composite_sampler_preserves_alpha_for_dynamic_backgrounds(self):
        config = Image2DConfig(
            source_path=str(self.image_path),
            bg_color_R=0.25,
            bg_color_G=0.5,
            bg_color_B=0.75,
        )
        dataset = Image2DDataset(config)
        sampler = Image2DFullBatchSampler(dataset)
        batch = sampler.next()
        torch.testing.assert_close(batch.foreground, dataset.foreground * dataset.alpha)
        torch.testing.assert_close(batch.alpha, dataset.alpha)
        background = torch.tensor([[0.25, 0.5, 0.75]])
        torch.testing.assert_close(
            batch.foreground + background * (1.0 - batch.alpha), dataset.target
        )

    def test_mask_sampler_keeps_straight_rgb_target(self):
        dataset = Image2DDataset(
            Image2DConfig(source_path=str(self.image_path), alpha_mode="mask")
        )
        batch = Image2DFullBatchSampler(dataset).next()
        torch.testing.assert_close(batch.foreground, dataset.foreground)
        torch.testing.assert_close(batch.alpha, torch.ones_like(dataset.alpha))

    def test_rays_and_pixels_share_the_documented_corner_order(self):
        dataset = Image2DDataset(Image2DConfig(source_path=str(self.image_path)))
        rays = dataset.rays(dataset.flat_indices)
        torch.testing.assert_close(
            rays.origins[:, :2],
            torch.tensor(
                [
                    [-0.5, -0.5],
                    [0.5, -0.5],
                    [-0.5, 0.5],
                    [0.5, 0.5],
                ]
            ),
        )
        torch.testing.assert_close(
            dataset.foreground,
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
            ),
        )

    def test_directory_resolution_is_unambiguous_and_extension_checked(self):
        self.assertEqual(resolve_single_image_path(self.root), self.image_path.resolve())
        second = self.root / "second.jpg"
        Image.new("RGB", (1, 1)).save(second)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_single_image_path(self.root)
        unsupported = self.root / "image.bmp"
        Image.new("RGB", (1, 1)).save(unsupported)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_single_image_path(unsupported)


if __name__ == "__main__":
    unittest.main()
