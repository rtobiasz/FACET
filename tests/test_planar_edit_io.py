from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from facet.editing.state_io import (
    PlanarEditState,
    load_planar_edit_state,
    ordered_edit_tensors,
    save_planar_edit_state,
    state_from_mean_xy,
)


class PlanarEditIOTest(unittest.TestCase):
    def test_identity_round_trip_is_exact_and_pickle_free(self):
        mean_xy = torch.tensor([[-1.0, 0.25], [0.5, 2.0]], dtype=torch.float32)
        state = state_from_mean_xy(mean_xy, metadata={"source": "fixture", "step": 7})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "edit.npz"
            save_planar_edit_state(path, state)
            loaded = load_planar_edit_state(path)
            with np.load(path, allow_pickle=False) as payload:
                self.assertNotEqual(payload["metadata_json"].dtype, np.dtype("O"))
        torch.testing.assert_close(loaded.mean_xy, mean_xy, rtol=0, atol=0)
        torch.testing.assert_close(loaded.primitive_ids, torch.arange(2), rtol=0, atol=0)
        self.assertTrue(bool(loaded.active.all()))
        self.assertEqual(loaded.metadata, {"source": "fixture", "step": 7})

    def test_ids_restore_checkpoint_order_after_external_reordering(self):
        state = PlanarEditState(
            primitive_ids=torch.tensor([2, 0, 1], dtype=torch.int64),
            mean_xy=torch.tensor([[20.0, 21.0], [0.0, 1.0], [10.0, 11.0]]),
            active=torch.tensor([False, True, True]),
            metadata={},
        )
        mean_xy, active = ordered_edit_tensors(state, 3)
        torch.testing.assert_close(
            mean_xy, torch.tensor([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])
        )
        torch.testing.assert_close(active, torch.tensor([True, True, False]))

    def test_invalid_duplicate_ids_are_rejected(self):
        state = PlanarEditState(
            primitive_ids=torch.tensor([0, 0], dtype=torch.int64),
            mean_xy=torch.zeros((2, 2)),
            active=torch.ones(2, dtype=torch.bool),
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, "permutation"):
            ordered_edit_tensors(state, 2)

    def test_checkpoint_count_mismatch_is_rejected(self):
        state = state_from_mean_xy(torch.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            ordered_edit_tensors(state, 3)

    def test_save_requires_explicit_npz_extension(self):
        state = state_from_mean_xy(torch.zeros((1, 2)))
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "npz"):
                save_planar_edit_state(Path(directory) / "edit", state)


if __name__ == "__main__":
    unittest.main()
