from __future__ import annotations

import unittest

from facet.superres.evaluate import (
    require_compatible_dimensions,
    scaled_dimensions,
)


class SuperresolutionEvaluationTest(unittest.TestCase):
    def test_scaled_dimensions(self) -> None:
        self.assertEqual(scaled_dimensions(255, 169, 8), (2040, 1352))

    def test_scaled_dimensions_reject_nonpositive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            scaled_dimensions(255, 169, 0)

    def test_compatible_dimensions_allow_div2k_rounding(self) -> None:
        require_compatible_dimensions((2040, 1356), (255, 169), 8)

    def test_incompatible_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            require_compatible_dimensions((2040, 1370), (255, 169), 8)


if __name__ == "__main__":
    unittest.main()
