from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from facet.editing.foreground_mask import alpha_mask, colour_key_mask, dilate_mask


class ForegroundMaskTest(unittest.TestCase):
    def test_uses_border_median_as_colour_key(self):
        image = np.full((3, 3, 3), 200, dtype=np.uint8)
        image[1, 1] = [10, 20, 30]
        mask, background = colour_key_mask(image, 20.0)
        np.testing.assert_array_equal(background, [200, 200, 200])
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(mask[1, 1])

    def test_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            colour_key_mask(np.zeros((2, 2, 3), dtype=np.uint8), -1.0)

    def test_dilation_expands_mask_and_preserves_zero_radius(self):
        mask = np.zeros((5, 5), dtype=np.bool_)
        mask[2, 2] = True
        np.testing.assert_array_equal(dilate_mask(mask, 0), mask)
        expected = np.zeros((5, 5), dtype=np.bool_)
        expected[1:4, 1:4] = True
        np.testing.assert_array_equal(dilate_mask(mask, 1), expected)

    def test_dilation_rejects_negative_radius(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dilate_mask(np.zeros((2, 2), dtype=np.bool_), -1)

    def test_alpha_mask_uses_requested_threshold(self):
        image = np.zeros((1, 3, 4), dtype=np.uint8)
        image[0, :, 3] = [0, 1, 255]
        np.testing.assert_array_equal(alpha_mask(image, 1), [[False, True, True]])

    def test_alpha_mask_requires_rgba(self):
        with self.assertRaisesRegex(ValueError, "four channels"):
            alpha_mask(np.zeros((2, 2, 3), dtype=np.uint8), 1)


if __name__ == "__main__":
    unittest.main()
