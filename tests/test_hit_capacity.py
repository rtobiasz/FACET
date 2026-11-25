from __future__ import annotations

import unittest

import torch

from facet.renderer.planar import order_hits_lexicographically, summarize_hit_batching


class HitCapacityReferenceTest(unittest.TestCase):
    def test_equal_depth_hits_are_ordered_by_primitive_id(self) -> None:
        t = torch.tensor([2.0, 1.0, 1.0, 1.5, 1.0])
        primitive_ids = torch.tensor([8, 9, 2, 7, 4], dtype=torch.int32)

        ordered_t, ordered_ids = order_hits_lexicographically(
            t, primitive_ids, capacity=4
        )

        torch.testing.assert_close(ordered_t, torch.tensor([1.0, 1.0, 1.0, 1.5]))
        torch.testing.assert_close(
            ordered_ids, torch.tensor([2, 4, 9, 7], dtype=torch.int32)
        )

    def test_more_than_sixteen_candidates_exposes_multi_pass_load(self) -> None:
        report = summarize_hit_batching(
            torch.tensor([0, 16, 17, 20], dtype=torch.int32), capacity=16
        )

        self.assertEqual(report.maximum, 20)
        self.assertEqual(report.rays_exceeding_batch_capacity, 2)
        self.assertEqual(report.fraction_exceeding_batch_capacity, 0.5)
        self.assertEqual(report.median, 16.5)
        self.assertGreater(report.p99_9, 19.0)

    def test_invalid_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty vector"):
            summarize_hit_batching(torch.empty((0,), dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_hit_batching(torch.ones(2), capacity=0)
        with self.assertRaisesRegex(ValueError, "equal-length vectors"):
            order_hits_lexicographically(torch.ones(2, 1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
