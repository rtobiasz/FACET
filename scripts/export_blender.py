#!/usr/bin/env python3
"""Edit a fitted image in Blender: import an editable state as a point object, move it, export it back.

Runs inside Blender (tested with Blender 4.x):

    blender --background --python scripts/export_blender.py -- import --input state.npz
    blender --background scene.blend --python scripts/export_blender.py -- export --output edited.npz

Write the state with `scripts/edit.py export-state`, render the edited one with `scripts/edit.py render`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facet.editing.blender_export import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
