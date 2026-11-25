"""Versioned, pickle-free interchange for FACET 2D primitive edits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from facet.editing.state_format import (
    PlanarEditArrays,
    load_planar_edit_arrays,
    save_planar_edit_arrays,
)


@dataclass(frozen=True)
class PlanarEditState:
    primitive_ids: torch.Tensor
    mean_xy: torch.Tensor
    active: torch.Tensor
    metadata: dict[str, Any]

    @property
    def primitive_count(self) -> int:
        return int(self.primitive_ids.numel())


def _validate_state(state: PlanarEditState) -> None:
    count = state.primitive_count
    if state.primitive_ids.dtype != torch.int64 or state.primitive_ids.shape != (count,):
        raise ValueError("primitive_ids must be an int64 vector")
    if state.mean_xy.shape != (count, 2) or not state.mean_xy.is_floating_point():
        raise ValueError("mean_xy must be a floating-point tensor with shape (N, 2)")
    if state.active.dtype != torch.bool or state.active.shape != (count,):
        raise ValueError("active must be a boolean vector with shape (N,)")
    if not bool(torch.isfinite(state.mean_xy).all()):
        raise ValueError("mean_xy must contain only finite values")
    sorted_ids = torch.sort(state.primitive_ids.detach().cpu()).values
    expected = torch.arange(count, dtype=torch.int64)
    if not torch.equal(sorted_ids, expected):
        raise ValueError("primitive_ids must be a permutation of [0, N)")


def state_from_mean_xy(
    mean_xy: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PlanarEditState:
    """Create an interchange state with stable IDs matching tensor rows."""
    count = int(mean_xy.shape[0]) if mean_xy.ndim >= 1 else 0
    if active is None:
        active = torch.ones(count, dtype=torch.bool, device=mean_xy.device)
    state = PlanarEditState(
        primitive_ids=torch.arange(count, dtype=torch.int64, device=mean_xy.device),
        mean_xy=mean_xy.detach().clone(),
        active=active.detach().clone(),
        metadata=dict(metadata or {}),
    )
    _validate_state(state)
    return state


def save_planar_edit_state(path: Path | str, state: PlanarEditState) -> None:
    """Save a portable NPZ without Python objects or executable pickle data."""
    _validate_state(state)
    save_planar_edit_arrays(
        path,
        PlanarEditArrays(
            primitive_ids=state.primitive_ids.detach().cpu().numpy(),
            mean_xy=state.mean_xy.detach().cpu().numpy(),
            active=state.active.detach().cpu().numpy(),
            metadata=state.metadata,
        ),
    )


def load_planar_edit_state(
    path: Path | str, *, device: torch.device | str = "cpu"
) -> PlanarEditState:
    """Load and strictly validate a FACET planar edit sidecar."""
    arrays = load_planar_edit_arrays(path)
    state = PlanarEditState(
        primitive_ids=torch.as_tensor(arrays.primitive_ids, device=device),
        mean_xy=torch.as_tensor(arrays.mean_xy, device=device),
        active=torch.as_tensor(arrays.active, device=device),
        metadata=arrays.metadata,
    )
    _validate_state(state)
    return state


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_edit_tensors(
    state: PlanarEditState, expected_primitive_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore checkpoint row order using persistent primitive IDs."""
    _validate_state(state)
    if state.primitive_count != expected_primitive_count:
        raise ValueError(
            "planar edit primitive count does not match checkpoint: "
            f"{state.primitive_count} != {expected_primitive_count}"
        )
    order = torch.argsort(state.primitive_ids)
    return state.mean_xy[order], state.active[order]
