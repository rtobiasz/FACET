import unittest
import numpy as np
from facet.superres.protocol import bicubic_resize, modcrop


class SRProtocolTest(unittest.TestCase):
    def test_identity_and_constant(self):
        x = np.random.default_rng(0).random((17,23,3))
        np.testing.assert_allclose(bicubic_resize(x,(17,23)),x,atol=1e-14)
        np.testing.assert_allclose(bicubic_resize(np.ones((17,23,3)),(3,4)),1,atol=1e-14)

    def test_reflection_and_axis_symmetry(self):
        x = np.random.default_rng(1).random((16,24,3))
        y = bicubic_resize(x,(2,3))
        np.testing.assert_allclose(y[::-1],bicubic_resize(x[::-1],(2,3)),atol=1e-14)
        np.testing.assert_allclose(y.transpose(1,0,2),bicubic_resize(x.transpose(1,0,2),(3,2)),atol=1e-14)

    def test_modcrop_retains_top_left_pixels(self):
        x = np.arange(20*28*3).reshape(20,28,3)
        np.testing.assert_array_equal(modcrop(x),x[:16,:24])
