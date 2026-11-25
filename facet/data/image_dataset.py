"""Direct single-image dataset for the FACET 2D reference layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from facet.training.config import Image2DConfig
from facet.model.rays import RayBundle, generate_indexed_orthographic_rays


SUPPORTED_IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


def resolve_single_image_path(source_path: str | Path) -> Path:
    """Resolve one supported image without relying on directory iteration order."""

    path = Path(source_path).expanduser()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"unsupported 2D image extension: {path.suffix}")
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"2D image source does not exist: {path}")
    candidates = sorted(
        (
            candidate
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda candidate: candidate.name.casefold(),
    )
    if len(candidates) != 1:
        raise ValueError(
            f"2D image directory must contain exactly one supported image; "
            f"found {len(candidates)} in {path}"
        )
    return candidates[0].resolve()


class Image2DDataset:
    """Flattened row-major tensor view of one RGB or RGBA image."""

    def __init__(
        self,
        config: Image2DConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(config, Image2DConfig):
            raise TypeError("Image2DDataset requires an Image2DConfig")
        if not dtype.is_floating_point:
            raise TypeError("image dataset dtype must be floating point")

        self.config = config
        self.path = resolve_single_image_path(config.source_path)
        self.device = torch.device(device)
        self.dtype = dtype
        with Image.open(self.path) as image:
            rgba = np.array(image.convert("RGBA"), dtype=np.float32, copy=True)
        rgba_tensor = torch.from_numpy(rgba).to(
            device=self.device,
            dtype=self.dtype,
        )
        rgba_tensor.mul_(1.0 / 255.0)

        self.height = int(rgba_tensor.shape[0])
        self.width = int(rgba_tensor.shape[1])
        self.camera_count = 1
        self.rays_per_camera = self.width * self.height
        self.ray_count = self.rays_per_camera
        self.foreground = rgba_tensor[..., :3].reshape(-1, 3).contiguous()
        self.alpha = rgba_tensor[..., 3:4].reshape(-1, 1).contiguous()
        background = torch.tensor(
            config.background_rgb,
            dtype=self.dtype,
            device=self.device,
        ).reshape(1, 3)
        if config.alpha_mode == "composite":
            self.target = (
                self.foreground * self.alpha
                + background * (1.0 - self.alpha)
            ).contiguous()
            self.loss_weights = torch.ones_like(self.alpha)
        else:
            self.target = self.foreground.clone()
            self.loss_weights = self.alpha.clone()
        self.flat_indices = torch.arange(
            self.ray_count,
            device=self.device,
            dtype=torch.int64,
        )

    def rays(self, flat_indices: torch.Tensor) -> RayBundle:
        return generate_indexed_orthographic_rays(
            flat_indices.to(device=self.device),
            self.width,
            self.height,
            plane_extent_x=self.config.plane_extent_x,
            plane_z=self.config.plane_z,
            camera_distance=self.config.camera_distance,
            dtype=self.dtype,
        )
