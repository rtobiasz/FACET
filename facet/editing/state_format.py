"""Pure-NumPy schema for portable FACET 2D edit sidecars."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_NAME = "FACETPlanarEdit"
FORMAT_VERSION = 1
FORMAT_FIELDS = {
    "format", "version", "primitive_ids", "mean_xy", "active", "metadata_json"
}


@dataclass(frozen=True)
class PlanarEditArrays:
    primitive_ids: np.ndarray
    mean_xy: np.ndarray
    active: np.ndarray
    metadata: dict[str, Any]

    @property
    def primitive_count(self) -> int:
        return int(self.primitive_ids.size)


def validate_planar_edit_arrays(state: PlanarEditArrays) -> None:
    count = state.primitive_count
    if state.primitive_ids.dtype != np.int64 or state.primitive_ids.shape != (count,):
        raise ValueError("primitive_ids must be an int64 vector")
    if state.mean_xy.shape != (count, 2) or state.mean_xy.dtype.kind != "f":
        raise ValueError("mean_xy must be a floating-point array with shape (N, 2)")
    if state.active.dtype != np.bool_ or state.active.shape != (count,):
        raise ValueError("active must be a boolean vector with shape (N,)")
    if not np.isfinite(state.mean_xy).all():
        raise ValueError("mean_xy must contain only finite values")
    if not np.array_equal(np.sort(state.primitive_ids), np.arange(count, dtype=np.int64)):
        raise ValueError("primitive_ids must be a permutation of [0, N)")
    try:
        json.dumps(state.metadata, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite and JSON-serializable") from error


def save_planar_edit_arrays(path: Path | str, state: PlanarEditArrays) -> None:
    validate_planar_edit_arrays(state)
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("planar edit sidecar must use the .npz extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        state.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    np.savez_compressed(
        destination,
        format=np.asarray(FORMAT_NAME),
        version=np.asarray(FORMAT_VERSION, dtype=np.int64),
        primitive_ids=state.primitive_ids,
        mean_xy=state.mean_xy,
        active=state.active,
        metadata_json=np.asarray(metadata),
    )


def load_planar_edit_arrays(path: Path | str) -> PlanarEditArrays:
    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) != FORMAT_FIELDS:
            raise ValueError(f"invalid planar edit fields: {sorted(payload.files)}")
        if str(payload["format"].item()) != FORMAT_NAME:
            raise ValueError("unsupported planar edit format")
        if int(payload["version"].item()) != FORMAT_VERSION:
            raise ValueError("unsupported planar edit version")
        try:
            metadata = json.loads(str(payload["metadata_json"].item()))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("invalid planar edit metadata") from error
        if not isinstance(metadata, dict):
            raise ValueError("planar edit metadata must be a JSON object")
        state = PlanarEditArrays(
            primitive_ids=np.asarray(payload["primitive_ids"]).copy(),
            mean_xy=np.asarray(payload["mean_xy"]).copy(),
            active=np.asarray(payload["active"]).copy(),
            metadata=metadata,
        )
    validate_planar_edit_arrays(state)
    return state
