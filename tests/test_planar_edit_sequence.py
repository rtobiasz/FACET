from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

import torch

from facet.editing.sequence import (
    animation_strength,
    collect_edit_state_paths,
    natural_sort_key,
    progressive_cut_selection,
)


class PlanarEditSequenceTest(unittest.TestCase):
    def test_natural_sort_orders_numeric_frame_names(self):
        paths = [Path("frame10.npz"), Path("frame2.npz"), Path("frame1.npz")]
        self.assertEqual(
            [path.name for path in sorted(paths, key=natural_sort_key)],
            ["frame1.npz", "frame2.npz", "frame10.npz"],
        )

    def test_natural_sort_accepts_mixed_numeric_and_text_prefixes(self):
        paths = [Path("frame2.npz"), Path("10.npz"), Path("2.npz")]
        self.assertEqual(
            [path.name for path in sorted(paths, key=natural_sort_key)],
            ["2.npz", "10.npz", "frame2.npz"],
        )

    def test_collect_filters_and_sorts_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("edit10.npz", "edit2.npz", "ignored.txt"):
                (root / name).touch()
            self.assertEqual(
                [path.name for path in collect_edit_state_paths(root, "*.npz")],
                ["edit2.npz", "edit10.npz"],
            )

    def test_collect_rejects_empty_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No planar edit sidecars"):
                collect_edit_state_paths(Path(directory), "*.npz")

    def test_animation_strength_has_smooth_endpoints(self):
        self.assertEqual(animation_strength(0, 5, cycle=False), 0.0)
        self.assertEqual(animation_strength(4, 5, cycle=False), 1.0)
        self.assertEqual(animation_strength(0, 5, cycle=True), 0.0)
        self.assertAlmostEqual(animation_strength(2, 5, cycle=True), 1.0)
        self.assertEqual(animation_strength(4, 5, cycle=True), 0.0)

    def test_progressive_cut_preserves_selection_and_endpoints(self):
        means = torch.tensor([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
        selection = torch.tensor([False, True, True, False])
        torch.testing.assert_close(
            progressive_cut_selection(means, selection, 0.0, axis="x"),
            torch.zeros(4, dtype=torch.bool),
        )
        torch.testing.assert_close(
            progressive_cut_selection(means, selection, 0.5, axis="x"),
            torch.tensor([False, True, False, False]),
        )
        torch.testing.assert_close(
            progressive_cut_selection(means, selection, 1.0, axis="x"),
            selection,
        )


if __name__ == "__main__":
    unittest.main()
