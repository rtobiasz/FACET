#!/usr/bin/env python3
"""CPU tests for facet/lifting/lift.py geometry (no CUDA, no checkpoint)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from facet.lifting.lift import lift, rotmat_to_quaternion, target_cameras, view_encoding_training_direction  # noqa: E402

FOV = 0.6911112070083618


def quat_to_rot(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def primitives(n=500, seed=0, h=64, w=96):
    rng = np.random.default_rng(seed)
    xy = np.column_stack([rng.uniform(-0.9, 0.9, n), rng.uniform(-0.9 * h / w, 0.9 * h / w, n)])
    return xy, rng.uniform(-math.pi, math.pi, n), np.log(rng.uniform(0.002, 0.01, (n, 2)))


class LiftTest(unittest.TestCase):
    def test_constant_plane_is_similarity(self):
        xy, theta, log_s = primitives()
        depth = np.full((64, 96), 2.0)
        for transport in ("legacy", "jacobian"):
            m, q, s, active, _ = lift(xy, theta, log_s, depth, FOV, "bilinear", transport, "deactivate")
            t = math.tan(FOV / 2)
            np.testing.assert_allclose(m, np.column_stack([xy * t * 2.0, np.full(len(xy), 2.0)]), atol=1e-12)
            np.testing.assert_allclose(s, log_s + math.log(2.0 * t), atol=1e-9)
            r = quat_to_rot(q)
            np.testing.assert_allclose(r[:, :, 2], np.tile([0, 0, 1.0], (len(xy), 1)), atol=1e-9)
            np.testing.assert_allclose(np.arctan2(r[:, 1, 0], r[:, 0, 0]), theta, atol=1e-9)
            self.assertTrue(active.all())

    def test_tilted_plane_12_degrees(self):
        h, w = 64, 96
        fx = 0.5 * w / math.tan(FOV / 2)
        angle = math.radians(12.0)
        normal = np.array([math.sin(angle), 0.0, -math.cos(angle)])  # plane n.P = -2 (Z=2 at centre)
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([(uu - w / 2) / fx, (vv - h / 2) / fx, np.ones_like(uu)], -1)
        depth = -2.0 / (rays @ normal)
        xy, theta, log_s = primitives(h=h, w=w)
        m, q, s, _, _ = lift(xy, theta, log_s, depth, FOV, "bilinear", "jacobian", "deactivate")
        np.testing.assert_allclose(m @ normal, -2.0, atol=1e-5)  # bilinear Z is not exact: 1/Z is linear in u
        n_lift = quat_to_rot(q)[:, :, 2]
        self.assertGreater(np.abs(n_lift @ normal).min(), 1 - 1e-6)  # surfels tangent to it
        m_leg, q_leg, *_ = lift(xy, theta, log_s, depth, FOV, "bilinear", "legacy", "deactivate")
        self.assertLess(np.abs(quat_to_rot(q_leg)[:, :, 2] @ normal).max(), math.cos(angle) + 1e-6)

    def test_edge_guard(self):
        h, w = 64, 96
        depth = np.where(np.arange(w)[None, :] < w // 2, 1.0, 1.5) * np.ones((h, 1))
        xy = np.array([[(w // 2 - 0.5) / (w / 2) - 1.0, 0.0], [-0.5, 0.0]])  # at the step, far from it
        theta, log_s = np.zeros(2), np.log(np.full((2, 2), 0.005))
        for sampling in ("floor", "bilinear"):
            _, _, s_raw, _, _ = lift(xy, theta, log_s, depth, FOV, sampling, "jacobian", "deactivate")
            _, _, s_cut, _, _ = lift(xy, theta, log_s, depth, FOV, sampling, "jacobian", "deactivate",
                                     edge_max_slant_deg=85.0)
            self.assertGreater(s_raw[0, 0] - s_cut[0, 0], 1.0)  # the cut keeps the step from stretching it
            np.testing.assert_allclose(s_cut[1], s_raw[1])      # flat region untouched

    def test_edge_guard_keeps_slopes(self):
        h, w = 64, 96
        fx = 0.5 * w / math.tan(FOV / 2)
        normal = np.array([math.sin(math.radians(60)), 0.0, -math.cos(math.radians(60))])
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([(uu - w / 2) / fx, (vv - h / 2) / fx, np.ones_like(uu)], -1)
        depth = -2.0 / (rays @ normal)
        xy, theta, log_s = primitives(h=h, w=w)
        a = lift(xy, theta, log_s, depth, FOV, "bilinear", "jacobian", "deactivate")
        b = lift(xy, theta, log_s, depth, FOV, "bilinear", "jacobian", "deactivate", edge_max_slant_deg=85.0)
        np.testing.assert_allclose(a[1], b[1])

    def test_invalid_policies_keep_rows(self):
        xy, theta, log_s = primitives(n=50)
        depth = np.zeros((64, 96))
        _, _, _, active, stats = lift(xy, theta, log_s, depth, FOV, "round", "legacy", "deactivate")
        self.assertEqual(active.shape, (50,))
        self.assertFalse(active.any())
        self.assertEqual(stats["invalid_depth"], 50)
        m, _, _, active, _ = lift(xy, theta, log_s, depth, FOV, "round", "legacy", "plane")
        self.assertTrue(active.all())
        np.testing.assert_allclose(m[:, 2], 1.0)

    def test_quaternion_roundtrip(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=(200, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q *= np.where(q[:, :1] < 0, -1, 1)
        np.testing.assert_allclose(rotmat_to_quaternion(quat_to_rot(q)), q, atol=1e-9)

    def test_reference_camera_is_identity(self):
        c2w = np.eye(4)
        c2w[2, 3] = 1.0
        path = {"render_width": 96, "render_height": 64,
                "camera_path": [{"camera_to_world": c2w.flatten().tolist(), "fov": 30.0, "file_path": "a"}]}
        _, _, cams = target_cameras(path)
        np.testing.assert_allclose(cams[0]["c2w_cv"], np.eye(4), atol=1e-12)

    def test_training_direction_encoding(self):
        e = view_encoding_training_direction()
        self.assertEqual(e.shape, (24,))
        np.testing.assert_allclose(e[1::2], [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
