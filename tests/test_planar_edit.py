from pathlib import Path
import sys
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from facet.editing.transport import edit_mean_xy, edit_planar_state, transport_planar_ellipses


class PlanarEditTest(unittest.TestCase):
    def test_identity_is_exact_and_does_not_alias(self):
        source = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
        result = edit_mean_xy(source, "identity", 0.0)
        torch.testing.assert_close(result, source, rtol=0, atol=0)
        self.assertNotEqual(result.data_ptr(), source.data_ptr())

    def test_translate_and_stretch_preserve_order(self):
        source = torch.tensor([[-1.0, 2.0], [3.0, 4.0]])
        translated = edit_mean_xy(source, "translate", 0.5, axis="x")
        stretched = edit_mean_xy(source, "stretch", 2.0, axis="y")
        torch.testing.assert_close(translated, torch.tensor([[-0.5, 2.0], [3.5, 4.0]]))
        torch.testing.assert_close(stretched, torch.tensor([[-1.0, 4.0], [3.0, 8.0]]))

    def test_bend_changes_only_secondary_coordinate(self):
        source = torch.tensor([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0]])
        result = edit_mean_xy(source, "bend", 0.25, axis="x")
        torch.testing.assert_close(result[:, 0], source[:, 0])
        self.assertGreater(float(result[:, 1].abs().max()), 0.0)

    def test_selection_leaves_other_primitives_exact(self):
        source = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        selection = torch.tensor([False, True, False])
        result = edit_mean_xy(source, "translate", 0.5, selection=selection)
        torch.testing.assert_close(result, torch.tensor([[-1.0, 0.0], [0.5, 0.0], [1.0, 0.0]]))

    def test_cut_uses_activity_without_deleting_or_moving(self):
        source = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        selection = torch.tensor([False, True, False])
        means, active = edit_planar_state(
            source, "cut", 0.0, selection=selection
        )
        torch.testing.assert_close(means, source, rtol=0, atol=0)
        torch.testing.assert_close(active, torch.tensor([True, False, True]))

    def test_displacement_is_local_translation(self):
        source = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        selection = torch.tensor([False, True, False])
        means, active = edit_planar_state(
            source, "displace", 0.25, axis="y", selection=selection
        )
        torch.testing.assert_close(
            means, torch.tensor([[-1.0, 0.0], [0.0, 0.25], [1.0, 0.0]])
        )
        self.assertTrue(bool(active.all()))

    def test_ellipse_transport_is_exact_for_identity_and_translation(self):
        means = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        theta = torch.tensor([0.2, -0.4])
        log_scales = torch.log(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        for mode in ("identity", "translate"):
            with self.subTest(mode=mode):
                out_theta, out_scales = transport_planar_ellipses(
                    means, theta, log_scales, mode, 0.25
                )
                torch.testing.assert_close(out_theta, theta, rtol=0, atol=0)
                torch.testing.assert_close(out_scales, log_scales, rtol=0, atol=0)

    def test_stretch_transports_axis_aligned_scales(self):
        means = torch.tensor([[0.0, 0.0]])
        theta = torch.zeros(1)
        log_scales = torch.log(torch.tensor([[2.0, 1.0]]))
        _, transported = transport_planar_ellipses(
            means, theta, log_scales, "stretch", 1.5
        )
        torch.testing.assert_close(torch.exp(transported), torch.tensor([[3.0, 1.0]]))

    def test_bend_changes_orientation_and_preserves_positive_scales(self):
        means = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        theta = torch.zeros(3)
        log_scales = torch.log(torch.full((3, 2), 0.1))
        transported_theta, transported_scales = transport_planar_ellipses(
            means, theta, log_scales, "bend", 0.2
        )
        self.assertGreater(float(transported_theta.abs().max()), 0.0)
        self.assertTrue(bool(torch.isfinite(transported_scales).all()))
        self.assertTrue(bool((torch.exp(transported_scales) > 0).all()))

    def test_cut_and_displacement_require_selection(self):
        source = torch.zeros((2, 2))
        for mode in ("cut", "displace"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "requires an explicit selection"
            ):
                edit_planar_state(source, mode, 0.1)


if __name__ == "__main__":
    unittest.main()
