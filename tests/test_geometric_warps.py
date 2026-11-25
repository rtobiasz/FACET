from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from facet.editing.warps import unwarp_points, warp_points, warp_with_jacobian  # noqa: E402


SMUDGE = {"type": "smudge", "strokes": [{"path": [[20, 50], [80, 40], [120, 60]], "radius": 12, "strength": 0.85}]}


class SmudgeTest(unittest.TestCase):
    def setUp(self) -> None:
        xs, ys = torch.meshgrid(torch.arange(0.0, 160.0, 2.0), torch.arange(0.0, 100.0, 2.0), indexing="xy")
        self.points = torch.stack((xs, ys), -1).reshape(-1, 2)

    def test_identity_at_zero(self) -> None:
        self.assertTrue(torch.equal(warp_points(self.points, SMUDGE, 0.0), self.points))

    def test_one_to_one(self) -> None:
        _, jacobian = warp_with_jacobian(self.points, SMUDGE, 1.0)
        self.assertGreater(float(torch.linalg.det(jacobian).min()), 0.0)

    def test_drags_along_and_leaves_far_points(self) -> None:
        start, far = torch.tensor([[20.0, 50.0]]), torch.tensor([[150.0, 5.0]])
        self.assertGreater(float(warp_points(start, SMUDGE, 1.0)[0, 0]), 40.0)
        self.assertTrue(torch.equal(warp_points(far, SMUDGE, 1.0), far))
        half = warp_points(start, SMUDGE, 0.5)[0, 0]
        self.assertLessEqual(float(half), float(warp_points(start, SMUDGE, 1.0)[0, 0]))

    def test_inverse(self) -> None:
        for t in (0.37, 1.0):
            back = unwarp_points(warp_points(self.points, SMUDGE, t), SMUDGE, t)
            self.assertLess(float((back - self.points).abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
