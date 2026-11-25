"""Structurally planar FACET geometry for production image reconstruction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from facet.io.checkpoint import PLANAR_MODEL_TENSOR_NAMES
from facet.model.geometry import (
    angles_to_planar_quaternions,
    quaternion_gradient_to_angle,
)


_NATIVE_CONDITIONING_DIMENSIONS = 96


def materialize_renderer_conditioning(conditioning: torch.Tensor) -> torch.Tensor:
    """Materialize compact conditioning as a contiguous native 96D tensor."""
    if conditioning.ndim != 2:
        raise ValueError("conditioning must have shape (N, D)")
    dimensions = conditioning.shape[1]
    if not 0 <= dimensions <= _NATIVE_CONDITIONING_DIMENSIONS:
        raise ValueError("conditioning dimensions must be in [0, 96]")
    return torch.nn.functional.pad(
        conditioning, (0, _NATIVE_CONDITIONING_DIMENSIONS - dimensions)
    ).contiguous()


class PlanarGaussianModel(nn.Module):
    """Own only trainable XY means and scalar in-plane angles."""

    PARAMETER_NAMES = PLANAR_MODEL_TENSOR_NAMES
    RENDERER_CONDITIONING_DIMENSIONS = _NATIVE_CONDITIONING_DIMENSIONS

    def __init__(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        plane_z: float = 0.0,
        requires_grad: bool = False,
        active_conditioning_dimensions: int = 96,
    ) -> None:
        super().__init__()
        if not math.isfinite(plane_z):
            raise ValueError("plane_z must be finite")
        missing = set(self.PARAMETER_NAMES) - set(values)
        unknown = set(values) - set(self.PARAMETER_NAMES)
        if missing or unknown:
            raise ValueError(
                f"invalid planar tensors; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if not 0 <= active_conditioning_dimensions <= 96:
            raise ValueError("active_conditioning_dimensions must be in [0, 96]")
        self.plane_z = float(plane_z)
        self.active_conditioning_dimensions = int(active_conditioning_dimensions)
        values = dict(values)
        conditioning = values["conditioning_variable"]
        w1_conditioning = values["w1_conditioning"]
        active = self.active_conditioning_dimensions
        if conditioning.ndim == 2 and conditioning.shape[1] == 96 and active < 96:
            values["conditioning_variable"] = conditioning[:, :active].clone().contiguous()
        if w1_conditioning.ndim == 2 and w1_conditioning.shape[1] == 96 and active < 96:
            values["w1_conditioning"] = w1_conditioning[:, :active].clone().contiguous()
        for name in self.PARAMETER_NAMES:
            value = values[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            self.register_parameter(
                name, nn.Parameter(value, requires_grad=requires_grad)
            )
        if self.mean_xy.ndim != 2 or self.mean_xy.shape[1] != 2:
            raise ValueError("mean_xy must have shape (N, 2)")
        if self.theta.shape != (self.mean_xy.shape[0],):
            raise ValueError("theta must have shape (N,)")
        self.number_of_Gaussians = self.mean_xy.shape[0]
        if self.conditioning_variable.shape != (
            self.number_of_Gaussians,
            self.active_conditioning_dimensions,
        ):
            raise ValueError("conditioning_variable must have shape (N, active dimensions)")
        if self.w1_conditioning.shape != (
            64,
            self.active_conditioning_dimensions,
        ):
            raise ValueError("w1_conditioning must have shape (64, active dimensions)")

    def renderer_conditioning(self) -> torch.Tensor:
        """Materialize the fixed 96-channel descriptor view expected by the native renderer."""
        return materialize_renderer_conditioning(self.conditioning_variable)

    def native_conditioning(self) -> torch.Tensor:
        """Materialize the contiguous native ABI view without persistent storage."""
        return materialize_renderer_conditioning(self.conditioning_variable)

    def renderer_w1_conditioning(self) -> torch.Tensor:
        """Materialize the matching fixed-width first-layer weight block."""
        missing = self.RENDERER_CONDITIONING_DIMENSIONS - self.active_conditioning_dimensions
        return torch.nn.functional.pad(self.w1_conditioning, (0, missing))

    @property
    def m(self) -> torch.Tensor:
        z = torch.full_like(self.mean_xy[:, :1], self.plane_z)
        return torch.cat((self.mean_xy, z), 1)

    @property
    def q(self) -> torch.Tensor:
        return angles_to_planar_quaternions(self.theta)

    @property
    def means(self) -> torch.Tensor:
        return self.m

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.s)

    @property
    def quaternions(self) -> torch.Tensor:
        return self.q

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.A)

    @property
    def kappas(self) -> torch.Tensor:
        return 1.0 + torch.nn.functional.softplus(self.k)

    def render_geometry(self):
        """(means, log-scales, quaternions) as the renderer sees them."""
        return self.m, self.s, self.q

    def renderer_geometry(self):
        return self.means, self.scales, self.quaternions, self.opacities, self.kappas

    def model_tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.PARAMETER_NAMES}

    def replace_parameters(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        requires_grad: bool = True,
    ) -> None:
        """Replace owned tensors after a planar topology change."""
        unknown = set(values) - set(self.PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown planar parameters: {sorted(unknown)}")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            setattr(
                self, name, nn.Parameter(value, requires_grad=requires_grad)
            )
        self.number_of_Gaussians = int(self.mean_xy.shape[0])

    def export_geometry(self) -> dict[str, torch.Tensor]:
        """Materialize the renderer/interchange geometry view (XYZ means and quaternions)."""
        result = self.model_tensors()
        result.pop("mean_xy")
        result.pop("theta")
        result["m"] = self.m
        result["q"] = self.q
        result["conditioning_variable"] = self.conditioning_variable
        result["w1_conditioning"] = self.renderer_w1_conditioning()
        return result

    def assign_native_gradients(self, gradients) -> None:
        """Convert renderer XYZ/quaternion gradients to owned 2D parameters."""
        values = gradients.as_dict()
        for name, gradient in values.items():
            if name == "m":
                self.mean_xy.grad = gradient[:, :2].contiguous()
            elif name == "q":
                self.theta.grad = quaternion_gradient_to_angle(
                    self.theta.detach(), gradient
                )
            elif name == "conditioning_variable":
                self.conditioning_variable.grad = gradient[
                    :, :self.active_conditioning_dimensions
                ].contiguous()
            elif name == "w1_conditioning":
                self.w1_conditioning.grad = gradient[
                    :, :self.active_conditioning_dimensions
                ].contiguous()
            else:
                getattr(self, name).grad = gradient


@dataclass(frozen=True)
class PlanarInitializationConfig:
    """Reproducible image-domain primitive initialization."""

    primitive_count: int = 1024
    method: str = "random"
    seed: int = 0
    initial_conditioning_std: float = 0.01
    appearance_initialization: str = "legacy"
    active_conditioning_dimensions: int = 96
    initial_opacity: float = 0.01
    initial_kappa: float = 1.01

    def __post_init__(self) -> None:
        if self.primitive_count <= 0:
            raise ValueError("primitive_count must be positive")
        if self.method not in (
            "grid",
            "random",
            "jittered_grid",
            "hybrid",
            "hierarchical",
            "multiscale_random",
            "edge_weighted",
        ):
            raise ValueError(
                "initialization method must be grid, random, jittered_grid, "
                "hybrid, hierarchical, multiscale_random, or edge_weighted"
            )
        if self.initial_conditioning_std < 0.0:
            raise ValueError("initial_conditioning_std cannot be negative")
        if not 0 <= self.active_conditioning_dimensions <= 96:
            raise ValueError("active_conditioning_dimensions must be in [0, 96]")
        if self.appearance_initialization not in ("legacy", "neutral"):
            raise ValueError(
                "appearance_initialization must be legacy or neutral"
            )
        if not 0.0 < self.initial_opacity < 1.0:
            raise ValueError("initial_opacity must lie in (0, 1)")
        if self.initial_kappa <= 1.0:
            raise ValueError("initial_kappa must be greater than 1")


def _grid_shape(count: int, width: int, height: int) -> tuple[int, int]:
    target_aspect = width / height
    candidates = []
    for rows in range(1, count + 1):
        columns = int(math.ceil(count / rows))
        unused_fraction = (rows * columns - count) / count
        score = abs(columns / rows - target_aspect) + unused_fraction
        candidates.append((score, unused_fraction, rows, columns))
        if columns == 1:
            break
    _, _, rows, columns = min(candidates)
    return rows, columns


def _grid_points(
    count: int,
    width: int,
    height: int,
    extent_x: float,
    extent_y: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    jittered: bool,
) -> tuple[torch.Tensor, float, float]:
    rows, columns = _grid_shape(count, width, height)
    spacing_x = extent_x / columns
    spacing_y = extent_y / rows
    x = torch.arange(columns, device=device, dtype=dtype).add_(0.5)
    y = torch.arange(rows, device=device, dtype=dtype).add_(0.5)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)[:count]
    if jittered:
        points.add_(
            torch.rand(
                points.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            ).sub_(0.5)
        )
    points[:, 0].mul_(spacing_x).sub_(0.5 * extent_x)
    points[:, 1].mul_(spacing_y).sub_(0.5 * extent_y)
    return points, spacing_x, spacing_y


def _initial_shared_appearance(
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    appearance_initialization: str = "legacy",
) -> dict[str, torch.Tensor]:
    values = {
        "w1_uv": torch.empty((64, 8), dtype=dtype, device=device),
        "w1_v": torch.empty((64, 24), dtype=dtype, device=device),
        "w1_conditioning": torch.empty((64, 96), dtype=dtype, device=device),
        "b1": torch.zeros(64, dtype=dtype, device=device),
        "w2": torch.empty((64, 64), dtype=dtype, device=device),
        "b2": torch.zeros(64, dtype=dtype, device=device),
        "w3": torch.empty((16, 64), dtype=dtype, device=device),
        "b3": torch.zeros(16, dtype=dtype, device=device),
        "features": torch.empty(13154, dtype=dtype, device=device),
    }
    for name in ("w1_uv", "w1_v", "w1_conditioning", "w2", "w3"):
        nn.init.kaiming_normal_(
            values[name], mode="fan_in", nonlinearity="relu", generator=generator
        )
        values[name].mul_(0.1)
    values["features"].uniform_(-1.0e-4, 1.0e-4, generator=generator)
    if appearance_initialization == "neutral":
        # The decoder multiplicatively modulates base RGB and opacity. Start
        # close to the identity map: ReLU(RGB)=1 and sigmoid(alpha)~=0.9975.
        values["b3"].copy_(
            torch.tensor(
                [1.0, 1.0, 1.0, 6.0] * 4, device=device, dtype=dtype
            )
        )
    return values


def initialize_planar_model(
    dataset,
    config: PlanarInitializationConfig,
    *,
    requires_grad: bool = True,
) -> PlanarGaussianModel:
    """Create a structurally planar model with grid or uniform image support."""
    if not isinstance(config, PlanarInitializationConfig):
        raise TypeError("config must be a PlanarInitializationConfig")
    device = dataset.target.device
    dtype = dataset.target.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    count = config.primitive_count
    extent_x = float(dataset.config.plane_extent_x)
    extent_y = extent_x * dataset.height / dataset.width

    rows, columns = _grid_shape(count, dataset.width, dataset.height)
    spacing_x = extent_x / columns
    spacing_y = extent_y / rows
    scale_factors = torch.ones(count, device=device, dtype=dtype)
    if config.method in ("grid", "jittered_grid"):
        mean_xy, spacing_x, spacing_y = _grid_points(
            count,
            dataset.width,
            dataset.height,
            extent_x,
            extent_y,
            device=device,
            dtype=dtype,
            generator=generator,
            jittered=config.method == "jittered_grid",
        )
    elif config.method == "edge_weighted":
        target = dataset.target.reshape(dataset.height, dataset.width, 3)
        gradient = torch.zeros(
            (dataset.height, dataset.width), device=device, dtype=dtype
        )
        gradient[:, 1:] += torch.mean(
            torch.abs(target[:, 1:] - target[:, :-1]), dim=2
        )
        gradient[1:, :] += torch.mean(
            torch.abs(target[1:, :] - target[:-1, :]), dim=2
        )
        flat_gradient = gradient.reshape(-1)
        if float(flat_gradient.sum().item()) > 0.0:
            probabilities = 0.25 / flat_gradient.numel() + 0.75 * (
                flat_gradient / flat_gradient.sum()
            )
        else:
            probabilities = torch.ones_like(flat_gradient)
        pixel_indices = torch.multinomial(
            probabilities,
            count,
            replacement=count > flat_gradient.numel(),
            generator=generator,
        )
        pixel_y = torch.div(pixel_indices, dataset.width, rounding_mode="floor")
        pixel_x = pixel_indices.remainder(dataset.width)
        jitter = torch.rand(
            (count, 2), device=device, dtype=dtype, generator=generator
        ).sub_(0.5)
        mean_xy = torch.empty((count, 2), device=device, dtype=dtype)
        mean_xy[:, 0] = (
            (pixel_x.to(dtype) + 0.5 + jitter[:, 0]) / dataset.width - 0.5
        ) * extent_x
        mean_xy[:, 1] = (
            (pixel_y.to(dtype) + 0.5 + jitter[:, 1]) / dataset.height - 0.5
        ) * extent_y
    elif config.method in ("hybrid", "hierarchical"):
        coarse_count = max(1, count // 4)
        coarse, _, _ = _grid_points(
            coarse_count,
            dataset.width,
            dataset.height,
            extent_x,
            extent_y,
            device=device,
            dtype=dtype,
            generator=generator,
            jittered=True,
        )
        fine = torch.rand(
            (count - coarse_count, 2),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        fine[:, 0].mul_(extent_x).sub_(0.5 * extent_x)
        fine[:, 1].mul_(extent_y).sub_(0.5 * extent_y)
        mean_xy = torch.cat((coarse, fine), dim=0)
        if config.method == "hierarchical":
            scale_factors[:coarse_count] = 1.25
            scale_factors[coarse_count:] = 0.75
    else:
        mean_xy = torch.rand(
            (count, 2), device=device, dtype=dtype, generator=generator
        )
        mean_xy[:, 0].mul_(extent_x).sub_(0.5 * extent_x)
        mean_xy[:, 1].mul_(extent_y).sub_(0.5 * extent_y)
        if config.method == "multiscale_random":
            coarse_count = max(1, count // 4)
            scale_factors[:coarse_count] = 1.25
            scale_factors[coarse_count:] = 0.75

    pixel_x = ((mean_xy[:, 0] / extent_x + 0.5) * dataset.width).floor().long()
    pixel_y = ((mean_xy[:, 1] / extent_y + 0.5) * dataset.height).floor().long()
    pixel_x.clamp_(0, dataset.width - 1)
    pixel_y.clamp_(0, dataset.height - 1)
    initial_rgb = dataset.target[pixel_y * dataset.width + pixel_x].clone()
    raw_kappa = config.initial_kappa - 1.0
    raw_kappa = raw_kappa + math.log(-math.expm1(-raw_kappa))
    values = _initial_shared_appearance(
        device=device,
        dtype=dtype,
        generator=generator,
        appearance_initialization=config.appearance_initialization,
    )
    active_conditioning_dimensions = config.active_conditioning_dimensions
    if active_conditioning_dimensions < 96:
        values["w1_conditioning"].zero_()
        if active_conditioning_dimensions > 0:
            nn.init.kaiming_normal_(
                values["w1_conditioning"][:, :active_conditioning_dimensions],
                mode="fan_in",
                nonlinearity="relu",
                generator=generator,
            )
            values["w1_conditioning"][:, :active_conditioning_dimensions].mul_(0.1)
    conditioning = (
        torch.randn(
            (count, active_conditioning_dimensions),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * config.initial_conditioning_std
    )
    values["w1_conditioning"] = values["w1_conditioning"][
        :, :active_conditioning_dimensions
    ].clone().contiguous()
    values.update({
        "RGB": initial_rgb,
        "A": torch.full((count, 1), torch.logit(torch.tensor(config.initial_opacity)).item(), device=device, dtype=dtype),
        "k": torch.full((count, 1), raw_kappa, device=device, dtype=dtype),
        "conditioning_variable": conditioning,
        "mean_xy": mean_xy.contiguous(),
        "theta": torch.rand(count, device=device, dtype=dtype, generator=generator) * (2.0 * math.pi) - math.pi,
        "s": torch.log(
            torch.full(
                (count, 2),
                math.sqrt(spacing_x * spacing_y),
                device=device,
                dtype=dtype,
            )
            * scale_factors.unsqueeze(1)
        ),
    })
    return PlanarGaussianModel(
        values,
        plane_z=dataset.config.plane_z,
        requires_grad=requires_grad,
        active_conditioning_dimensions=config.active_conditioning_dimensions,
    )
