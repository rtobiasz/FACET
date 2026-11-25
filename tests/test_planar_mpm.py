from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from facet.editing.state_format import PlanarEditArrays
from facet.physics.mpm import (
    isolate_planar_selection,
    prepare_mpm_particles,
    restore_planar_state,
    selection_from_image_mask,
)
from facet.physics.simulate_mpm import summarize_simulation


def sample_state() -> PlanarEditArrays:
    return PlanarEditArrays(
        primitive_ids=np.array([2, 0, 3, 1], dtype=np.int64),
        mean_xy=np.array(
            [[-1.0, -2.0], [0.0, 4.0], [2.0, 8.0], [4.0, 10.0]],
            dtype=np.float32,
        ),
        active=np.array([True, False, True, True], dtype=np.bool_),
        metadata={"source": "test"},
    )


class PlanarMPMTest(unittest.TestCase):
    def test_isolate_selection_changes_only_activity(self):
        source = sample_state()
        isolated = isolate_planar_selection(
            source, np.array([True, True, False, True], dtype=np.bool_)
        )
        np.testing.assert_array_equal(isolated.active, [True, False, False, True])
        np.testing.assert_array_equal(isolated.primitive_ids, source.primitive_ids)
        np.testing.assert_array_equal(isolated.mean_xy, source.mean_xy)
        self.assertTrue(isolated.metadata["isolated_for_mpm"])

    def test_isolate_selection_rejects_empty_result(self):
        with self.assertRaisesRegex(ValueError, "active primitive"):
            isolate_planar_selection(
                sample_state(), np.zeros(4, dtype=np.bool_)
            )

    def test_image_mask_selection_uses_row_major_planar_coordinates(self):
        mask = np.array([[True, False], [False, True]], dtype=np.bool_)
        points = np.array(
            [[-1.0, -0.5], [0.75, -0.25], [-0.75, 0.25], [1.0, 0.5], [2.0, 0.0]],
            dtype=np.float32,
        )
        selection = selection_from_image_mask(points, mask, (-1.0, 1.0, -0.5, 0.5))
        np.testing.assert_array_equal(selection, [True, False, False, True, False])

    def test_image_mask_selection_rejects_invalid_inputs(self):
        points = np.zeros((1, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "boolean"):
            selection_from_image_mask(points, np.ones((2, 2)), (-1, 1, -1, 1))
        with self.assertRaisesRegex(ValueError, "ordered"):
            selection_from_image_mask(
                points, np.ones((2, 2), dtype=np.bool_), (1, -1, -1, 1)
            )

    def test_prepare_uses_only_active_rows_and_keeps_ids(self):
        particles = prepare_mpm_particles(sample_state())
        np.testing.assert_array_equal(particles.source_rows, [0, 2, 3])
        np.testing.assert_array_equal(particles.primitive_ids, [2, 3, 1])
        np.testing.assert_allclose(particles.positions.min(axis=0), [0.25, 0.4])
        np.testing.assert_allclose(particles.positions.max(axis=0), [0.75, 0.9])
        self.assertAlmostEqual(float(particles.positions[0, 1]), 0.9)
        self.assertAlmostEqual(float(particles.positions[-1, 1]), 0.4)

    def test_restore_keeps_order_activity_and_inactive_positions(self):
        source = sample_state()
        particles = prepare_mpm_particles(source)
        moved = particles.positions.copy()
        moved[:, 0] += 0.1
        restored = restore_planar_state(source, particles, moved)
        np.testing.assert_array_equal(restored.primitive_ids, source.primitive_ids)
        np.testing.assert_array_equal(restored.active, source.active)
        np.testing.assert_array_equal(restored.mean_xy[1], source.mean_xy[1])
        self.assertTrue(
            np.all(
                restored.mean_xy[particles.source_rows, 0]
                > source.mean_xy[particles.source_rows, 0]
            )
        )

    def test_mpm_downward_motion_maps_to_image_downward_motion(self):
        source = sample_state()
        particles = prepare_mpm_particles(source)
        moved = particles.positions.copy()
        moved[:, 1] -= 0.1
        restored = restore_planar_state(source, particles, moved)
        self.assertTrue(
            np.all(
                restored.mean_xy[particles.source_rows, 1]
                > source.mean_xy[particles.source_rows, 1]
            )
        )

    def test_prepare_rejects_empty_active_set(self):
        state = sample_state()
        empty = PlanarEditArrays(
            state.primitive_ids,
            state.mean_xy,
            np.zeros(4, dtype=np.bool_),
            state.metadata,
        )
        with self.assertRaisesRegex(ValueError, "at least one active"):
            prepare_mpm_particles(empty)

    def test_prepare_rejects_domain_overflow(self):
        with self.assertRaisesRegex(ValueError, r"inside \[0, 1\]"):
            prepare_mpm_particles(sample_state(), scale=0.8, offset=(0.25, 0.4))

    def test_prepare_limits_particles_deterministically(self):
        particles = prepare_mpm_particles(sample_state(), maximum_particles=2)
        np.testing.assert_array_equal(particles.source_rows, [0, 3])
        np.testing.assert_array_equal(particles.primitive_ids, [2, 1])

    def test_prepare_rejects_invalid_particle_limit(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            prepare_mpm_particles(sample_state(), maximum_particles=0)

    def test_prepare_intersects_selection_with_activity(self):
        selection = np.array([False, True, True, False], dtype=np.bool_)
        particles = prepare_mpm_particles(sample_state(), selection=selection)
        np.testing.assert_array_equal(particles.source_rows, [2])
        np.testing.assert_array_equal(particles.primitive_ids, [3])

    def test_prepare_rejects_invalid_selection(self):
        with self.assertRaisesRegex(ValueError, "boolean vector"):
            prepare_mpm_particles(sample_state(), selection=np.ones(4))
        with self.assertRaisesRegex(ValueError, "at least one active"):
            prepare_mpm_particles(
                sample_state(), selection=np.zeros(4, dtype=np.bool_)
            )

    def test_restore_rejects_primitive_id_mismatch(self):
        source = sample_state()
        particles = prepare_mpm_particles(source)
        invalid = replace(
            particles, primitive_ids=particles.primitive_ids[::-1].copy()
        )
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            restore_planar_state(source, invalid, particles.positions)

    def test_simulation_summary_checks_identity_and_selection(self):
        source = sample_state()
        final = PlanarEditArrays(
            source.primitive_ids.copy(),
            source.mean_xy.copy(),
            source.active.copy(),
            source.metadata,
        )
        final.mean_xy[2, 1] += 0.25
        summary = summarize_simulation(source, final, np.array([2]))
        self.assertEqual(summary["changed_primitive_count"], 1)
        self.assertEqual(summary["simulated_primitive_count"], 1)
        self.assertAlmostEqual(summary["maximum_selected_displacement"], 0.25)
        np.testing.assert_allclose(summary["mean_selected_delta_xy"], [0.0, 0.25])

        invalid = PlanarEditArrays(
            final.primitive_ids.copy(),
            final.mean_xy.copy(),
            final.active.copy(),
            final.metadata,
        )
        invalid.mean_xy[0, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "unselected"):
            summarize_simulation(source, invalid, np.array([2]))


if __name__ == "__main__":
    unittest.main()
