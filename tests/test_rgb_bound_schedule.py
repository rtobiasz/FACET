"""The base-colour bound and its optional step schedule."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from facet.training.trainer import PlanarTopologyController


class RgbBoundScheduleTest(unittest.TestCase):
    def controller(self, **kwargs) -> PlanarTopologyController:
        return PlanarTopologyController(2.0, 1.333, **kwargs)

    def test_unset_keeps_lower_only_bound(self) -> None:
        self.assertIsNone(self.controller()._rgb_bound())

    def test_fixed_bound_holds_throughout(self) -> None:
        c = self.controller(maximum_rgb=1.0)
        for iteration in (1, 15000, 30000):
            c.set_iteration(iteration)
            self.assertEqual(c._rgb_bound(), 1.0)

    def test_schedule_relaxes_at_the_iteration(self) -> None:
        c = self.controller(maximum_rgb=1.0, maximum_rgb_final=1.25,
                            maximum_rgb_relax_iteration=15000)
        c.set_iteration(14999)
        self.assertEqual(c._rgb_bound(), 1.0)
        c.set_iteration(15000)
        self.assertEqual(c._rgb_bound(), 1.25)

    def test_schedule_needs_both_parts_and_a_start(self) -> None:
        with self.assertRaises(ValueError):
            self.controller(maximum_rgb=1.0, maximum_rgb_final=1.25)
        with self.assertRaises(ValueError):
            self.controller(maximum_rgb_final=1.25, maximum_rgb_relax_iteration=10)


if __name__ == "__main__":
    unittest.main()
