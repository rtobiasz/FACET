import unittest

import torch

from facet.model.rays import generate_indexed_orthographic_rays


class IndexedOrthographicRayTest(unittest.TestCase):
    def test_landscape_corners_are_pixel_centred_and_aspect_preserving(self):
        rays = generate_indexed_orthographic_rays(
            torch.tensor([0, 3, 4, 7]),
            4,
            2,
            plane_extent_x=2.0,
            plane_z=1.0,
            camera_distance=3.0,
        )

        expected_origins = torch.tensor(
            [
                [-0.75, -0.25, -2.0],
                [0.75, -0.25, -2.0],
                [-0.75, 0.25, -2.0],
                [0.75, 0.25, -2.0],
            ]
        )
        torch.testing.assert_close(rays.origins, expected_origins)
        torch.testing.assert_close(
            rays.directions,
            torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1),
        )
        intersections = rays.origins + 3.0 * rays.directions
        torch.testing.assert_close(intersections[:, 2], torch.ones(4))
        torch.testing.assert_close(rays.camera_indices, torch.zeros(4, dtype=torch.int64))
        torch.testing.assert_close(rays.pixel_indices, torch.tensor([0, 3, 4, 7]))

    def test_portrait_extent_and_odd_centre_are_exact(self):
        portrait = generate_indexed_orthographic_rays(
            torch.tensor([0, 1, 6, 7]),
            2,
            4,
            plane_extent_x=2.0,
        )
        torch.testing.assert_close(
            portrait.origins[:, :2],
            torch.tensor(
                [
                    [-0.5, -1.5],
                    [0.5, -1.5],
                    [-0.5, 1.5],
                    [0.5, 1.5],
                ]
            ),
        )

        centre = generate_indexed_orthographic_rays(torch.tensor([4]), 3, 3)
        torch.testing.assert_close(centre.origins[0], torch.tensor([0.0, 0.0, -1.0]))

    def test_preserves_device_and_requested_float_dtype(self):
        indices = torch.tensor([1, 4], dtype=torch.int32)
        rays = generate_indexed_orthographic_rays(
            indices,
            3,
            2,
            dtype=torch.float64,
        )
        self.assertEqual(rays.origins.dtype, torch.float64)
        self.assertEqual(rays.directions.dtype, torch.float64)
        self.assertEqual(rays.flat_indices.dtype, torch.int64)
        self.assertEqual(rays.flat_indices.device, indices.device)

    def test_rejects_invalid_domain_and_indices(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            generate_indexed_orthographic_rays(torch.tensor([0]), 0, 1)
        with self.assertRaisesRegex(ValueError, "plane_extent_x"):
            generate_indexed_orthographic_rays(
                torch.tensor([0]), 1, 1, plane_extent_x=0.0
            )
        with self.assertRaisesRegex(ValueError, "camera_distance"):
            generate_indexed_orthographic_rays(
                torch.tensor([0]), 1, 1, camera_distance=0.0
            )
        with self.assertRaisesRegex(IndexError, "address the image"):
            generate_indexed_orthographic_rays(torch.tensor([-1]), 1, 1)
        with self.assertRaisesRegex(IndexError, "address the image"):
            generate_indexed_orthographic_rays(torch.tensor([1]), 1, 1)
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            generate_indexed_orthographic_rays(torch.tensor([0.0]), 1, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            generate_indexed_orthographic_rays(torch.tensor([0]), 1, 1, plane_z=float("nan"))


if __name__ == "__main__":
    unittest.main()
