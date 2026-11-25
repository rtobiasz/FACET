"""Import/export FACET 2D edit sidecars inside Blender.

Usage:
  blender scene.blend --python scripts/export_blender.py -- import \
    --input edit.npz --object FACETPlanarEdit
  blender scene.blend --python scripts/export_blender.py -- export \
    --output edited.npz --object FACETPlanarEdit
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np


# Blender runs this file with its own Python: make the facet package importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facet.editing.state_format import (  # noqa: E402
    PlanarEditArrays,
    load_planar_edit_arrays,
    save_planar_edit_arrays,
)


ID_ATTRIBUTE = "facet_primitive_id"
ACTIVE_ATTRIBUTE = "facet_active"
METADATA_PROPERTY = "facet_metadata_json"


def transform_coordinates(coordinates: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a Blender-style 4x4 world transform to N local XYZ points."""
    coordinates = np.asarray(coordinates)
    matrix = np.asarray(matrix)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if matrix.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    homogeneous = np.concatenate(
        (coordinates, np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype)),
        axis=1,
    )
    transformed = homogeneous @ matrix.astype(coordinates.dtype, copy=False).T
    return transformed[:, :3]


def _bpy():
    try:
        import bpy
    except ImportError as error:
        raise RuntimeError("run this command with Blender's Python") from error
    return bpy


def import_sidecar(path: Path, object_name: str) -> None:
    bpy = _bpy()
    if object_name in bpy.data.objects:
        raise ValueError(f"Blender object {object_name!r} already exists")
    state = load_planar_edit_arrays(path)
    coordinates = np.zeros((state.primitive_count, 3), dtype=np.float32)
    coordinates[:, :2] = state.mean_xy.astype(np.float32, copy=False)

    mesh = bpy.data.meshes.new(f"{object_name}Mesh")
    mesh.vertices.add(state.primitive_count)
    mesh.vertices.foreach_set("co", coordinates.reshape(-1))
    id_attribute = mesh.attributes.new(ID_ATTRIBUTE, type="INT", domain="POINT")
    id_attribute.data.foreach_set(
        "value", state.primitive_ids.astype(np.int32, copy=False)
    )
    active_attribute = mesh.attributes.new(
        ACTIVE_ATTRIBUTE, type="INT", domain="POINT"
    )
    active_attribute.data.foreach_set("value", state.active.astype(np.int32))
    mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    obj[METADATA_PROPERTY] = json.dumps(
        state.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    obj["facet_source_sidecar"] = str(path.resolve())
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    print(f"Imported {state.primitive_count} FACET primitives as {object_name!r}")


def _attribute_values(mesh, name: str, count: int) -> np.ndarray:
    attribute = mesh.attributes.get(name)
    if attribute is None or attribute.domain != "POINT":
        raise ValueError(f"mesh is missing POINT attribute {name!r}")
    values = np.empty(count, dtype=np.int32)
    attribute.data.foreach_get("value", values)
    return values


def export_sidecar(
    path: Path, object_name: str, *, z_tolerance: float = 1.0e-5
) -> None:
    bpy = _bpy()
    obj = bpy.data.objects.get(object_name)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Blender mesh object {object_name!r} was not found")
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    mesh = obj.data
    count = len(mesh.vertices)
    coordinates = np.empty((count, 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", coordinates.reshape(-1))
    world = transform_coordinates(coordinates, np.asarray(obj.matrix_world))
    if count:
        plane_z = float(np.median(world[:, 2]))
        deviation = float(np.max(np.abs(world[:, 2] - plane_z)))
        if deviation > z_tolerance:
            raise ValueError(
                "edited vertices are not planar in world space: "
                f"maximum Z deviation {deviation:.6g} exceeds {z_tolerance:.6g}"
            )
    else:
        plane_z = 0.0

    metadata_raw = obj.get(METADATA_PROPERTY, "{}")
    try:
        metadata = json.loads(metadata_raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Blender object has invalid FACET metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("Blender object FACET metadata must be a JSON object")
    metadata = dict(metadata)
    metadata["blender_object"] = object_name
    metadata["blender_plane_z"] = plane_z

    state = PlanarEditArrays(
        primitive_ids=_attribute_values(mesh, ID_ATTRIBUTE, count).astype(np.int64),
        mean_xy=world[:, :2].astype(np.float32),
        active=_attribute_values(mesh, ACTIVE_ATTRIBUTE, count).astype(np.bool_),
        metadata=metadata,
    )
    save_planar_edit_arrays(path, state)
    print(f"Exported {state.primitive_count} FACET primitives to {path.resolve()}")


def parse_arguments(argv: list[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import")
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--object", default="FACETPlanarEdit")
    exporter = commands.add_parser("export")
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--object", default="FACETPlanarEdit")
    exporter.add_argument("--z_tolerance", type=float, default=1.0e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = (
            sys.argv[sys.argv.index("--") + 1 :]
            if "--" in sys.argv
            else sys.argv[1:]
        )
    args = parse_arguments(argv)
    if args.command == "import":
        import_sidecar(args.input, args.object)
    else:
        if args.z_tolerance < 0.0:
            raise ValueError("--z_tolerance must be non-negative")
        export_sidecar(args.output, args.object, z_tolerance=args.z_tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
