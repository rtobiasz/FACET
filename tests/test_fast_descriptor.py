import types
import unittest

import torch

from facet.training.step import _compact_descriptor_width, _zero_descriptor_fast_path


def _model(dimensions):
    return types.SimpleNamespace(conditioning_variable=torch.zeros((3, dimensions)))


def _step(**flags):
    return types.SimpleNamespace(**flags)


class CompactDescriptorWidthTest(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(_compact_descriptor_width(_step(), _model(4)), 0)
        self.assertEqual(_compact_descriptor_width(_step(fast_zero_conditioning=True), _model(4)), 0)

    def test_rounds_up_to_kernel_widths(self):
        step = _step(fast_descriptor=True)
        for dimensions, width in ((1, 4), (4, 4), (5, 8), (8, 8), (9, 16), (16, 16)):
            self.assertEqual(_compact_descriptor_width(step, _model(dimensions)), width)

    def test_zero_descriptor_uses_stride_zero_path(self):
        step = _step(fast_descriptor=True)
        self.assertEqual(_compact_descriptor_width(step, _model(0)), 0)
        self.assertTrue(_zero_descriptor_fast_path(step, _model(0)))
        self.assertFalse(_zero_descriptor_fast_path(step, _model(4)))
        self.assertFalse(_zero_descriptor_fast_path(_step(), _model(0)))

    def test_rejects_wide_descriptor(self):
        with self.assertRaises(ValueError):
            _compact_descriptor_width(_step(fast_descriptor=True), _model(17))


if __name__ == "__main__":
    unittest.main()
