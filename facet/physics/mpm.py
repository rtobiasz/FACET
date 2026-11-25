"""Stable-ID adapter between planar edit sidecars and 2D MPM particles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from facet.editing.state_format import PlanarEditArrays, validate_planar_edit_arrays


@dataclass(frozen=True)
class MPMTransform:
    minimum: np.ndarray
    maximum: np.ndarray
    scale: float
    offset: np.ndarray
    flip_y: bool = True

    def as_metadata(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "scale": self.scale,
            "offset": self.offset.tolist(),
            "flip_y": self.flip_y,
        }


@dataclass(frozen=True)
class MPMParticles:
    positions: np.ndarray
    primitive_ids: np.ndarray
    source_rows: np.ndarray
    transform: MPMTransform


def isolate_planar_selection(
    state: PlanarEditArrays, selection: np.ndarray
) -> PlanarEditArrays:
    """Keep selected active primitives visible without changing geometry or IDs."""
    validate_planar_edit_arrays(state)
    selected = np.asarray(selection)
    if selected.shape != state.active.shape or selected.dtype != np.bool_:
        raise ValueError(
            "MPM selection must be a boolean vector with one entry per primitive"
        )
    active = state.active & selected
    if not np.any(active):
        raise ValueError("isolated MPM selection must contain an active primitive")
    metadata = dict(state.metadata)
    metadata["isolated_for_mpm"] = True
    return PlanarEditArrays(
        primitive_ids=state.primitive_ids.copy(),
        mean_xy=state.mean_xy.copy(),
        active=active,
        metadata=metadata,
    )


def selection_from_image_mask(
    mean_xy: np.ndarray,
    mask: np.ndarray,
    domain: tuple[float, float, float, float],
) -> np.ndarray:
    """Sample a row-major image mask at planar primitive positions."""
    points = np.asarray(mean_xy)
    mask_array = np.asarray(mask)
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind != "f":
        raise ValueError("mean_xy must have floating-point shape (P, 2)")
    if mask_array.ndim != 2 or mask_array.dtype != np.bool_:
        raise ValueError("MPM image mask must be a two-dimensional boolean array")
    if mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        raise ValueError("MPM image mask must not be empty")
    xmin, xmax, ymin, ymax = (float(value) for value in domain)
    if not np.isfinite((xmin, xmax, ymin, ymax)).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("MPM mask domain must contain finite ordered bounds")

    normalized_x = (points[:, 0] - xmin) / (xmax - xmin)
    normalized_y = (points[:, 1] - ymin) / (ymax - ymin)
    inside = (
        (normalized_x >= 0.0)
        & (normalized_x <= 1.0)
        & (normalized_y >= 0.0)
        & (normalized_y <= 1.0)
    )
    columns = np.minimum(
        (normalized_x * mask_array.shape[1]).astype(np.int64),
        mask_array.shape[1] - 1,
    )
    rows = np.minimum(
        (normalized_y * mask_array.shape[0]).astype(np.int64),
        mask_array.shape[0] - 1,
    )
    result = np.zeros(points.shape[0], dtype=np.bool_)
    valid_rows = rows[inside]
    valid_columns = columns[inside]
    result[inside] = mask_array[valid_rows, valid_columns]
    return result


def prepare_mpm_particles(
    state: PlanarEditArrays,
    *,
    scale: float = 0.5,
    offset: tuple[float, float] = (0.25, 0.4),
    maximum_particles: int | None = None,
    selection: np.ndarray | None = None,
) -> MPMParticles:
    """Normalize active planar primitives into the MPM simulation domain."""
    validate_planar_edit_arrays(state)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("MPM scale must be finite and greater than zero")
    offset_array = np.asarray(offset, dtype=np.float64)
    if offset_array.shape != (2,) or not np.isfinite(offset_array).all():
        raise ValueError("MPM offset must contain two finite values")
    if np.any(offset_array < 0.0) or np.any(offset_array + scale > 1.0):
        raise ValueError("MPM scale and offset must place particles inside [0, 1]")
    selected = state.active.copy()
    if selection is not None:
        selection = np.asarray(selection)
        if selection.shape != state.active.shape or selection.dtype != np.bool_:
            raise ValueError(
                "MPM selection must be a boolean vector with one entry per primitive"
            )
        selected &= selection
    source_rows = np.flatnonzero(selected)
    if source_rows.size == 0:
        raise ValueError("MPM simulation requires at least one active primitive")
    if maximum_particles is not None:
        if maximum_particles <= 0:
            raise ValueError("maximum_particles must be greater than zero")
        if source_rows.size > maximum_particles:
            sample = np.linspace(
                0, source_rows.size - 1, maximum_particles, dtype=np.int64
            )
            source_rows = source_rows[sample]
    source = state.mean_xy[source_rows].astype(np.float64, copy=False)
    minimum = source.min(axis=0)
    maximum = source.max(axis=0)
    extent = maximum - minimum
    safe_extent = np.where(extent == 0.0, 1.0, extent)
    normalized = (source - minimum) / safe_extent
    # FACET image coordinates use +Y down, while Taichi MPM uses +Y up.
    normalized[:, 1] = 1.0 - normalized[:, 1]
    positions = scale * normalized + offset_array
    transform = MPMTransform(minimum, maximum, float(scale), offset_array)
    return MPMParticles(
        positions=positions.astype(np.float32),
        primitive_ids=state.primitive_ids[source_rows].copy(),
        source_rows=source_rows.astype(np.int64),
        transform=transform,
    )


def restore_planar_state(
    source: PlanarEditArrays,
    particles: MPMParticles,
    simulated_positions: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> PlanarEditArrays:
    """Map simulated particles back without reordering or dropping primitives."""
    validate_planar_edit_arrays(source)
    positions = np.asarray(simulated_positions)
    if (
        particles.source_rows.ndim != 1
        or particles.source_rows.shape != particles.primitive_ids.shape
        or particles.source_rows.shape[0] != particles.positions.shape[0]
        or np.any(particles.source_rows < 0)
        or np.any(particles.source_rows >= source.primitive_count)
        or np.unique(particles.source_rows).size != particles.source_rows.size
    ):
        raise ValueError("MPM source rows must be unique valid primitive rows")
    expected_ids = source.primitive_ids[particles.source_rows]
    if not np.array_equal(particles.primitive_ids, expected_ids):
        raise ValueError("MPM primitive IDs do not match their source rows")
    if not np.all(source.active[particles.source_rows]):
        raise ValueError("MPM particles may reference only active primitives")
    if positions.shape != particles.positions.shape or positions.dtype.kind != "f":
        raise ValueError("simulated positions must have floating-point shape (P, 2)")
    if not np.isfinite(positions).all():
        raise ValueError("simulated positions must contain only finite values")
    extent = particles.transform.maximum - particles.transform.minimum
    safe_extent = np.where(extent == 0.0, 1.0, extent)
    normalized = (
        positions.astype(np.float64) - particles.transform.offset
    ) / particles.transform.scale
    if particles.transform.flip_y:
        normalized[:, 1] = 1.0 - normalized[:, 1]
    restored = normalized * safe_extent + particles.transform.minimum
    means = source.mean_xy.copy()
    means[particles.source_rows] = restored.astype(means.dtype)
    output_metadata = dict(source.metadata)
    output_metadata.update(metadata or {})
    result = PlanarEditArrays(
        primitive_ids=source.primitive_ids.copy(),
        mean_xy=means,
        active=source.active.copy(),
        metadata=output_metadata,
    )
    validate_planar_edit_arrays(result)
    return result


def plane_deformation_gradient(
    deformation: np.ndarray, transform: MPMTransform
) -> np.ndarray:
    """Express MPM deformation gradients (P, 2, 2) in FACET plane coordinates.

    prepare_mpm_particles maps the plane to the MPM domain by the per-axis
    linear map M = diag(scale / extent_x, +-scale / extent_y), with the sign
    flipping +Y down to +Y up; a gradient F in MPM coordinates is M^-1 F M in
    the plane. The translation terms cancel.
    """
    deformation = np.asarray(deformation, dtype=np.float64)
    if deformation.ndim != 3 or deformation.shape[1:] != (2, 2):
        raise ValueError("deformation gradients must have shape (P, 2, 2)")
    extent = transform.maximum - transform.minimum
    safe_extent = np.where(extent == 0.0, 1.0, extent)
    diagonal = transform.scale / safe_extent
    if transform.flip_y:
        diagonal = diagonal * np.array([1.0, -1.0])
    return deformation * diagonal[None, None, :] / diagonal[None, :, None]
