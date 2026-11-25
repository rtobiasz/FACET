"""--disable_component holds inherited FACET components fixed."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from facet.training.fit import _OPACITY_OUTPUTS, _impose_disabled_components


def model():
    return SimpleNamespace(
        k=torch.randn(5, 1), features=torch.randn(10), w1_uv=torch.randn(64, 8),
        w1_v=torch.randn(64, 24), w3=torch.randn(16, 64), b3=torch.randn(16),
    )


class DisableComponentTest(unittest.TestCase):
    def test_each_component_is_fixed_and_others_untouched(self):
        m = model()
        before = {name: value.clone() for name, value in vars(m).items()}
        _impose_disabled_components(m, ("kappa", "opacity_modulation"), -4.0)
        self.assertTrue(torch.all(m.k == -4.0))
        rows = list(_OPACITY_OUTPUTS)
        self.assertTrue(torch.all(m.w3[rows] == 0.0) and torch.all(m.b3[rows] == 6.0))
        others = [r for r in range(16) if r not in rows]
        self.assertTrue(torch.equal(m.w3[others], before["w3"][others]))
        for name in ("features", "w1_uv", "w1_v"):
            self.assertTrue(torch.equal(getattr(m, name), before[name]))

    def test_zeroed_descriptor_dimensions(self):
        m = model()
        m.conditioning_variable = torch.randn(5, 8)
        m.w1_conditioning = torch.randn(64, 8)
        kept = m.conditioning_variable[:, :4].clone()
        _impose_disabled_components(m, (), 0.0, zero_conditioning=4)
        self.assertTrue(torch.all(m.conditioning_variable[:, 4:] == 0) and torch.all(m.w1_conditioning[:, 4:] == 0))
        self.assertTrue(torch.equal(m.conditioning_variable[:, :4], kept))

    def test_uv_and_view(self):
        m = model()
        _impose_disabled_components(m, ("uv", "view"), 0.0)
        self.assertEqual(float(m.features.abs().sum() + m.w1_uv.abs().sum() + m.w1_v.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
