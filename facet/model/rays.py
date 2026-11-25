from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class RayBundle:
    """Rays and their positions in a flattened rendering dataset."""

    origins: torch.Tensor
    directions: torch.Tensor
    flat_indices: torch.Tensor
    camera_indices: torch.Tensor
    pixel_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.origins.ndim != 2 or self.origins.shape[1] != 3:
            raise ValueError("ray origins must have shape (ray_count, 3)")
        if self.directions.shape != self.origins.shape:
            raise ValueError("ray directions must match ray origins")

        ray_count = self.origins.shape[0]
        for name, indices in (
            ("flat_indices", self.flat_indices),
            ("camera_indices", self.camera_indices),
            ("pixel_indices", self.pixel_indices),
        ):
            if indices.shape != (ray_count,):
                raise ValueError(f"{name} must have shape (ray_count,)")
            if indices.dtype != torch.int64:
                raise ValueError(f"{name} must use torch.int64")
            if indices.device != self.origins.device:
                raise ValueError(f"{name} must be on the ray device")

        if self.directions.device != self.origins.device:
            raise ValueError("ray origins and directions must share a device")
        if self.directions.dtype != self.origins.dtype:
            raise ValueError("ray origins and directions must share a dtype")


def generate_indexed_orthographic_rays(
    flat_indices: torch.Tensor,
    width: int,
    height: int,
    *,
    plane_extent_x: float = 2.0,
    plane_z: float = 0.0,
    camera_distance: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> RayBundle:
    """Generate pixel-centred rays perpendicular to an aspect-preserving plane."""

    if width <= 0 or height <= 0:
        raise ValueError("orthographic image dimensions must be positive")
    if not math.isfinite(plane_extent_x) or plane_extent_x <= 0.0:
        raise ValueError("plane_extent_x must be a positive finite value")
    if not math.isfinite(plane_z):
        raise ValueError("plane_z must be finite")
    if not math.isfinite(camera_distance) or camera_distance <= 0.0:
        raise ValueError("camera_distance must be a positive finite value")
    if not dtype.is_floating_point:
        raise TypeError("orthographic ray dtype must be floating point")
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )
    if flat_indices.dtype not in integer_dtypes:
        raise TypeError("orthographic flat indices must use an integer dtype")

    indices = flat_indices.to(dtype=torch.int64).reshape(-1)
    pixel_count = width * height
    if torch.any(indices < 0) or torch.any(indices >= pixel_count):
        raise IndexError("orthographic flat indices must address the image")

    pixel_indices = indices
    rows = torch.div(pixel_indices, width, rounding_mode="floor")
    columns = torch.remainder(pixel_indices, width)
    extent_y = plane_extent_x * height / width
    x = ((columns.to(dtype) + 0.5) / width - 0.5) * plane_extent_x
    y = ((rows.to(dtype) + 0.5) / height - 0.5) * extent_y
    z = torch.full_like(x, float(plane_z - camera_distance))
    ray_origins = torch.stack((x, y, z), dim=1)
    ray_directions = torch.zeros_like(ray_origins)
    ray_directions[:, 2] = 1.0
    return RayBundle(
        origins=ray_origins,
        directions=ray_directions,
        flat_indices=indices,
        camera_indices=torch.zeros_like(indices),
        pixel_indices=pixel_indices,
    )


def generate_indexed_perspective_rays(
    flat_indices: torch.Tensor,
    origins: torch.Tensor,
    rights: torch.Tensor,
    downs: torch.Tensor,
    forwards: torch.Tensor,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
) -> RayBundle:
    """Reproduce FACET's indexed, pixel-centred pinhole ray construction."""

    indices = flat_indices.to(device=origins.device, dtype=torch.int64).reshape(-1)
    pixels_per_camera = width * height
    camera_indices = torch.div(indices, pixels_per_camera, rounding_mode="floor")
    pixel_indices = torch.remainder(indices, pixels_per_camera)
    y = torch.div(pixel_indices, width, rounding_mode="floor")
    x = torch.remainder(pixel_indices, width)

    double_tan_half_fov_x = 2.0 * np.tan(0.5 * fov_x)
    double_tan_half_fov_y = 2.0 * np.tan(0.5 * fov_y)
    d_x = (-0.5 + ((x + 0.5) / width)) * double_tan_half_fov_x
    d_y = (-0.5 + ((y + 0.5) / height)) * double_tan_half_fov_y

    ray_origins = origins[camera_indices]
    directions = (
        rights[camera_indices] * d_x.unsqueeze(1)
        + downs[camera_indices] * d_y.unsqueeze(1)
        + forwards[camera_indices]
    )
    return RayBundle(
        origins=ray_origins,
        directions=directions,
        flat_indices=indices,
        camera_indices=camera_indices,
        pixel_indices=pixel_indices,
    )
