from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "facet" / "editing" / "blender_export.py"
SPEC = importlib.util.spec_from_file_location("blender_planar_edit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
blender_planar_edit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(blender_planar_edit)


class BlenderPlanarEditTest(unittest.TestCase):
    def test_transform_coordinates_applies_world_matrix(self):
        coordinates = np.asarray([[1.0, 2.0, 0.0], [-1.0, 0.0, 0.0]])
        matrix = np.asarray(
            [
                [0.0, -2.0, 0.0, 3.0],
                [2.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = blender_planar_edit.transform_coordinates(coordinates, matrix)
        np.testing.assert_allclose(
            result, np.asarray([[-1.0, 1.0, 0.5], [3.0, -3.0, 0.5]])
        )

    def test_cli_contract(self):
        imported = blender_planar_edit.parse_arguments(
            ["import", "--input", "edit.npz"]
        )
        self.assertEqual(imported.command, "import")
        self.assertEqual(imported.object, "FACETPlanarEdit")
        exported = blender_planar_edit.parse_arguments(
            ["export", "--output", "edited.npz", "--z_tolerance", "0.01"]
        )
        self.assertEqual(exported.command, "export")
        self.assertEqual(exported.z_tolerance, 0.01)

    def test_bpy_is_only_required_for_blender_actions(self):
        with self.assertRaisesRegex(RuntimeError, "Blender"):
            blender_planar_edit._bpy()


if __name__ == "__main__":
    unittest.main()
