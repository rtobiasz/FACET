"""Degradation-consistent fitting for super-resolution (--sr_training_scale).

The model is rendered at K times the training image's resolution and the
render is mapped back to the training resolution with the antialiased
MATLAB-style bicubic that produced the DIV2K LR images (sr_protocol). The
loss compares that degraded render with the LR image, so the model has to
explain each LR pixel as a blurred patch of a sharper image instead of
interpolating the LR pixels directly. Off unless the flag is given.

The degradation is a fixed separable linear operator, applied as two dense
matrices (H x KH and W x KW; a few MB even at K = 8).
"""

from __future__ import annotations

import copy

import torch

import facet.training.step as training_step
from facet.renderer.planar import render_planar
from facet.superres.protocol import resize_weights
from facet.evaluation.metrics import psnr


def bicubic_operator(length: int, output_length: int, device) -> torch.Tensor:
    """Dense (output_length x length) antialiased bicubic resize matrix."""
    return torch.as_tensor(
        resize_weights(length, output_length).toarray(),
        dtype=torch.float32, device=device,
    )


class BicubicDegradation:
    """Maps a row-major (K*H*K*W, 3) render to a row-major (H*W, 3) image."""

    def __init__(self, width: int, height: int, scale: int, device) -> None:
        self.width, self.height, self.scale = width, height, scale
        self.render_width, self.render_height = width * scale, height * scale
        self.rows = bicubic_operator(self.render_height, height, device)
        self.columns = bicubic_operator(self.render_width, width, device)
        self.loss_calls = 0

    @property
    def render_pixel_count(self) -> int:
        return self.render_width * self.render_height

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image.reshape(self.render_height, self.render_width, 3)
        image = torch.einsum("hy,yxc->hxc", self.rows, image)
        image = torch.einsum("wx,hxc->hwc", self.columns, image)
        return image.reshape(-1, 3)

    def loss_and_image_gradient(self, step, image_unclamped):
        """Drop-in for training_step._loss_and_image_gradient on render rays.

        Same clamp and clamp-gradient rule as the original, but the squared
        error is taken after degradation, against the LR foreground.
        """
        image = torch.clamp(image_unclamped, min=0.0, max=1.0)
        image = image.detach().requires_grad_(True)
        target = step.foreground + step.background * (1.0 - step.alpha)
        loss = torch.mean((self(image) - target) ** 2)
        loss.backward()
        inside = image_unclamped == image
        if getattr(step, "restore_clamped_gradient", False):
            inside = (inside
                      | ((image_unclamped > 1.0) & (image.grad > 0.0))
                      | ((image_unclamped < 0.0) & (image.grad < 0.0)))
        self.loss_calls += 1
        return loss, image, image.grad * inside


def render_view(dataset, scale: int):
    """The dataset seen at K times its resolution, for rays only.

    Targets (foreground, alpha, target) stay at the training resolution;
    only width, height and the ray indices change.
    """
    view = copy.copy(dataset)
    view.width, view.height = dataset.width * scale, dataset.height * scale
    view.rays_per_camera = view.ray_count = view.width * view.height
    view.flat_indices = torch.arange(
        view.ray_count, device=dataset.device, dtype=torch.int64
    )
    return view


def install(degradation: BicubicDegradation) -> None:
    """Route the training loss of render-resolution batches through the
    degradation; any other batch keeps the original loss."""
    original = training_step._loss_and_image_gradient

    def loss_and_image_gradient(step, image_unclamped):
        if image_unclamped.shape[0] == degradation.render_pixel_count:
            return degradation.loss_and_image_gradient(step, image_unclamped)
        return original(step, image_unclamped)

    training_step._loss_and_image_gradient = loss_and_image_gradient


def degraded_psnr(renderer, model, view, dataset, degradation, *,
                  render_mode: str = "FACET") -> float:
    """PSNR of the degraded render-resolution image against the LR target."""
    with torch.no_grad():
        image = render_planar(renderer, model, view, mode=render_mode)
        degraded = degradation(image.clamp(0.0, 1.0))
        shape = (1, dataset.height, dataset.width, 3)
        return float(psnr(
            degraded.reshape(shape).permute(0, 3, 1, 2).contiguous(),
            dataset.target.reshape(shape).permute(0, 3, 1, 2).contiguous(),
        ).mean().item())
