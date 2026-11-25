"""Ellipse transport through a deformation gradient (physics with orientation)."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import numpy as np

from facet.editing.transport import (
    initial_neighbours,
    neighbourhood_deformation_gradient,
    transport_by_deformation_gradient,
)
from facet.physics.mpm import MPMTransform, plane_deformation_gradient


def rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def wrapped(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()


class DeformationTransportTest(unittest.TestCase):
    theta = torch.tensor([0.0, 0.3, -1.2, 2.5], dtype=torch.float32)
    log_scales = torch.log(torch.tensor([[1.0, 0.5], [0.2, 0.9], [0.4, 0.4], [3.0, 1.0]], dtype=torch.float32))

    def transport(self, deformation: torch.Tensor):
        batch = deformation.expand(4, 2, 2).clone()
        return transport_by_deformation_gradient(self.theta, self.log_scales, batch)

    def test_identity_keeps_every_ellipse(self):
        theta, scales = self.transport(torch.eye(2, dtype=torch.float64))
        self.assertTrue(torch.all(wrapped(theta, self.theta) < 1e-6))
        self.assertTrue(torch.allclose(scales, self.log_scales, atol=1e-6))

    def test_rotation_turns_the_frame_without_swapping_axes(self):
        theta, scales = self.transport(rotation(0.7))
        # The circle (third row) has no preferred axis; the others must turn by 0.7.
        for row in (0, 1, 3):
            self.assertLess(float(wrapped(theta[row], self.theta[row] + 0.7)), 1e-5)
        self.assertTrue(torch.allclose(scales[[0, 1, 3]], self.log_scales[[0, 1, 3]], atol=1e-6))

    def test_stretch_along_an_axis_scales_that_axis(self):
        theta, scales = self.transport(torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64)))
        self.assertLess(float(wrapped(theta[0], self.theta[0])), 1e-6)
        self.assertTrue(torch.allclose(scales[0], torch.log(torch.tensor([2.0, 0.5])), atol=1e-6))

    def test_minor_axis_first_is_not_reordered(self):
        # Row 1 has its longer axis second; an eigen-ordered transport
        # would turn its frame by 90 degrees.
        theta, scales = self.transport(torch.diag(torch.tensor([1.1, 0.95], dtype=torch.float64)))
        self.assertLess(float(wrapped(theta[1], self.theta[1])), 0.1)
        self.assertLess(float(scales[1, 0]), float(scales[1, 1]))

    def test_rejects_inverted_deformation(self):
        with self.assertRaises(ValueError):
            self.transport(torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)))


class PlaneDeformationGradientTest(unittest.TestCase):
    def test_matches_a_finite_difference_of_the_plane_map(self):
        transform = MPMTransform(np.array([-1.0, -0.6]), np.array([1.0, 0.6]), 0.5,
                                 np.array([0.25, 0.4]), True)
        extent = transform.maximum - transform.minimum

        def to_mpm(point):
            normalized = (point - transform.minimum) / extent
            normalized[1] = 1.0 - normalized[1]
            return transform.scale * normalized + transform.offset

        def to_plane(position):
            normalized = (position - transform.offset) / transform.scale
            normalized[1] = 1.0 - normalized[1]
            return normalized * extent + transform.minimum

        mpm_gradient = np.array([[1.2, 0.3], [-0.1, 0.9]])
        point, step = np.array([0.1, 0.2]), 1e-6
        base = to_mpm(point.copy())
        expected = np.column_stack([
            (to_plane(base + mpm_gradient @ ((to_mpm(point + step * axis) - base))) - to_plane(base)) / step
            for axis in np.eye(2)
        ])
        plane = plane_deformation_gradient(mpm_gradient[None], transform)[0]
        self.assertTrue(np.allclose(plane, expected, atol=1e-6))


class NeighbourhoodDeformationTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.initial = torch.rand(400, 2, generator=generator, dtype=torch.float64)
        self.neighbours = initial_neighbours(self.initial, 8)

    def test_affine_motion_is_recovered_exactly(self):
        gradient = torch.tensor([[1.3, 0.2], [-0.4, 0.8]], dtype=torch.float64)
        current = self.initial @ gradient.T + torch.tensor([0.5, -2.0], dtype=torch.float64)
        fitted = neighbourhood_deformation_gradient(self.initial, current, self.neighbours)
        self.assertTrue(torch.allclose(fitted, gradient.expand_as(fitted), atol=1e-6))

    def test_rigid_rotation_gives_a_rotation(self):
        current = self.initial @ rotation(0.7).T
        fitted = neighbourhood_deformation_gradient(self.initial, current, self.neighbours)
        self.assertTrue(torch.allclose(fitted, rotation(0.7).expand_as(fitted), atol=1e-6))

    def test_stretch_is_clamped_and_folds_stay_positive(self):
        stretched = self.initial * torch.tensor([10.0, 1.0], dtype=torch.float64)
        fitted = neighbourhood_deformation_gradient(self.initial, stretched, self.neighbours, maximum_stretch=4.0)
        self.assertTrue(torch.allclose(torch.linalg.svdvals(fitted)[:, 0], torch.full((400,), 4.0, dtype=torch.float64)))
        mirrored = self.initial * torch.tensor([-1.0, 1.0], dtype=torch.float64)
        fitted = neighbourhood_deformation_gradient(self.initial, mirrored, self.neighbours)
        self.assertTrue(bool((torch.linalg.det(fitted) > 0.0).all()))

    def test_cutoff_keeps_torn_off_points_unstretched(self):
        current = self.initial.clone()
        current[0] += 100.0
        fitted = neighbourhood_deformation_gradient(
            self.initial, current, self.neighbours, neighbour_cutoff=4.0
        )
        self.assertTrue(torch.allclose(fitted[0], torch.eye(2, dtype=torch.float64)))
        untouched = (self.neighbours != 0).all(dim=1)
        untouched[0] = False
        self.assertTrue(torch.allclose(fitted[untouched], torch.eye(2, dtype=torch.float64).expand(int(untouched.sum()), 2, 2), atol=1e-6))

    def test_neighbours_exclude_the_point_itself(self):
        rows = torch.arange(400)[:, None]
        self.assertFalse(bool((self.neighbours == rows).any()))


if __name__ == "__main__":
    unittest.main()
