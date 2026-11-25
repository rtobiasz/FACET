#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the 3DGS_LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from facet.renderer.facade import FACETRenderer
from facet.model.rays import RayBundle
from facet.training.surface import ray_parameters_to_depth_and_normal

if TYPE_CHECKING:
    from facet.model.components import PlanarGaussianModel as GaussianModel


@dataclass(frozen=True)
class FACETTrainingStepInput:
    """All data and coefficients consumed by one FACET optimizer step."""

    model: "GaussianModel"
    renderer: FACETRenderer
    optimizer: torch.optim.Optimizer
    rays: RayBundle
    foreground: torch.Tensor
    alpha: torch.Tensor
    background: torch.Tensor
    background_rgb: tuple[float, float, float]
    warmup_lambda: float
    ray_termination_threshold: float
    depth_lambda: float
    normal_lambda: float
    reg_depth_a: float
    reg_depth_b: float
    scale_regularization: float
    sm_count: int
    image_height: int
    image_width: int
    normal_depth_edge_threshold: float
    camera_forward: torch.Tensor | None = None
    loss_weights: torch.Tensor | None = None
    # The loss sees the render clamped to [0, 1], so a pixel rendered outside
    # that range gets no gradient and cannot come back (on one DIV2K image, 94% of
    # the error came from channels stuck at 1.0 over a target below 1).
    # When set, the gradient passes the clamp wherever it points back into
    # range; the loss value is unchanged.
    restore_clamped_gradient: bool = False
    # When the model has no active descriptor dimensions the descriptor is
    # zero and its gradient is sliced to nothing: pass it with stride 0 (the
    # renderer's zero-descriptor path) and an empty gradient buffer, which the
    # renderer skips (--fast_zero_conditioning).
    fast_zero_conditioning: bool = False
    # Pass a 1-16 dimensional descriptor as a compact contiguous N x 4/8/16
    # tensor (zero-padded to the next width) and request its gradient at
    # that width; a renderer built with COMPACT_CONDITIONING=1 then processes
    # only those inputs (--fast_descriptor). With 0D it acts like
    # fast_zero_conditioning.
    fast_descriptor: bool = False
    # Additional views rendered after the main view and added to the same Adam
    # step; empty for single-image fitting.
    auxiliary_views: tuple = ()
    # Keep the view-direction weights w1_v at their value.
    freeze_view_weights: bool = False


@dataclass(frozen=True)
class PhaseMetrics:
    loss: float
    psnr: float


@dataclass(frozen=True)
class NativeGradientBundle:
    """The exact sixteen gradients assigned to the FACET model parameters."""

    RGB: torch.Tensor
    A: torch.Tensor
    k: torch.Tensor
    w1_uv: torch.Tensor
    w1_v: torch.Tensor
    w1_conditioning: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    conditioning_variable: torch.Tensor
    features: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    q: torch.Tensor

    def assign_to(self, model: "GaussianModel") -> None:
        if hasattr(model, "assign_native_gradients"):
            model.assign_native_gradients(self)
            return
        for name, gradient in self.as_dict().items():
            getattr(model, name).grad = gradient

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "RGB": self.RGB,
            "A": self.A,
            "k": self.k,
            "w1_uv": self.w1_uv,
            "w1_v": self.w1_v,
            "w1_conditioning": self.w1_conditioning,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
            "conditioning_variable": self.conditioning_variable,
            "features": self.features,
            "m": self.m,
            "s": self.s,
            "q": self.q,
        }


@dataclass(frozen=True)
class FACETTrainingStepResult:
    base: PhaseMetrics | None
    facet: PhaseMetrics | None
    gradients: NativeGradientBundle


@dataclass
class _BlendedGradients:
    RGB: torch.Tensor
    A: torch.Tensor
    k: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    conditioning_variable: torch.Tensor
    features: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    q: torch.Tensor

    @classmethod
    def zeros(cls, model: "GaussianModel") -> "_BlendedGradients":
        return cls(
            RGB=torch.zeros_like(model.RGB),
            A=torch.zeros_like(model.A),
            k=torch.zeros_like(model.k),
            w3=torch.zeros(8, 64, dtype=torch.float32, device="cuda"),
            b3=torch.zeros(8, dtype=torch.float32, device="cuda"),
            w2=torch.zeros(64, 64, dtype=torch.float32, device="cuda"),
            b2=torch.zeros(64, dtype=torch.float32, device="cuda"),
            w1=torch.zeros(64, 128, dtype=torch.float32, device="cuda"),
            b1=torch.zeros(64, dtype=torch.float32, device="cuda"),
            conditioning_variable=torch.zeros_like(model.conditioning_variable),
            features=torch.zeros_like(
                model.features, dtype=torch.float32, device="cuda"
            ),
            m=torch.zeros_like(model.m),
            s=torch.zeros_like(model.s),
            q=torch.zeros_like(model.q),
        )

    def parameter_bundle(
        self,
        model: "GaussianModel",
        scale_regularization: float,
    ) -> NativeGradientBundle:
        scales_squared = torch.exp(model.s.detach()) ** 2
        scale_gradient = (
            scale_regularization
            / model.m.shape[0]
            * scales_squared
            / torch.sqrt(scales_squared.sum(1, keepdim=True))
        )
        return NativeGradientBundle(
            RGB=self.RGB,
            A=self.A,
            k=self.k,
            w1_uv=self.w1[:, 0:8],
            w1_v=self.w1[:, 8:32],
            w1_conditioning=self.w1[:, 32:128],
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=torch.nn.functional.pad(
                self.w3, (0, 0, 0, 8), "constant", 0.0
            ),
            b3=torch.nn.functional.pad(self.b3, (0, 8), "constant", 0.0),
            conditioning_variable=self.conditioning_variable,
            features=self.features,
            m=self.m,
            s=self.s + scale_gradient,
            q=self.q,
        )


@dataclass(frozen=True)
class _AuxiliaryBuffers:
    depth: torch.Tensor
    depth_and_index: torch.Tensor
    surface_normal: torch.Tensor
    normal: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int) -> "_AuxiliaryBuffers":
        return cls(
            depth=torch.zeros(
                (batch_size, 4), dtype=torch.float32, device="cuda"
            ),
            depth_and_index=torch.zeros(
                (batch_size, 2), dtype=torch.float32, device="cuda"
            ),
            surface_normal=torch.zeros(
                (batch_size, 3), dtype=torch.float32, device="cuda"
            ),
            normal=torch.zeros(
                (batch_size, 4), dtype=torch.float32, device="cuda"
            ),
        )


def _surface_normal(
    step: FACETTrainingStepInput,
    auxiliary: _AuxiliaryBuffers,
) -> torch.Tensor:
    if step.normal_lambda <= 0.0:
        return auxiliary.surface_normal
    if step.camera_forward is None:
        raise ValueError(
            "camera_forward is required when normal regularization is active"
        )
    expected_t = (
        auxiliary.depth[:, 3]
        / auxiliary.depth[:, 1].clamp_min(1.0e-8)
    )
    _, surface_normal, _ = ray_parameters_to_depth_and_normal(
        expected_t,
        auxiliary.depth[:, 1] > 0.01,
        step.rays.origins,
        step.rays.directions,
        step.camera_forward,
        step.image_height,
        step.image_width,
        step.normal_depth_edge_threshold,
    )
    return surface_normal.reshape(step.rays.origins.shape[0], 3).contiguous()


def _loss_and_image_gradient(
    step: FACETTrainingStepInput,
    image_unclamped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.clamp(image_unclamped, min=0.0, max=1.0)
    image = image.detach().requires_grad_(True)
    indices = step.rays.flat_indices
    foreground = torch.gather(
        step.foreground, 0, indices.unsqueeze(1).expand(-1, 3)
    )
    alpha = torch.gather(step.alpha, 0, indices.unsqueeze(1))
    target = foreground + step.background * (1.0 - alpha)
    squared_error = (image - target) ** 2
    if step.loss_weights is None:
        loss = torch.mean(squared_error)
    else:
        loss_weights = torch.gather(
            step.loss_weights, 0, indices.unsqueeze(1)
        )
        denominator = (3.0 * loss_weights.sum()).clamp_min(1.0e-12)
        loss = torch.sum(squared_error * loss_weights) / denominator
    loss.backward()
    inside = image_unclamped == image
    if getattr(step, "restore_clamped_gradient", False):
        inside = (inside
                  | ((image_unclamped > 1.0) & (image.grad > 0.0))
                  | ((image_unclamped < 0.0) & (image.grad < 0.0)))
    image_gradient = image.grad * inside
    return loss, image, image_gradient


def _metrics(loss: torch.Tensor) -> PhaseMetrics:
    psnr = (
        -10.0
        * (
            torch.log(loss)
            / torch.log(torch.tensor([10.0], device="cuda"))
        )
    ).item()
    return PhaseMetrics(loss=loss.item(), psnr=psnr)


def _render_geometry(model):
    """Means, log-scales, quaternions for the renderer (tilted with --relief_tilt)."""
    geometry = getattr(model, "render_geometry", None)
    if geometry is None:
        return model.m, model.s, model.q
    return geometry()


def _run_base_phase(
    step: FACETTrainingStepInput,
    blended: _BlendedGradients,
    rgba: torch.Tensor,
) -> PhaseMetrics:
    model = step.model
    render_m, render_s, render_q = _render_geometry(model)
    batch_size = step.rays.origins.shape[0]
    image_unclamped = torch.zeros(
        (batch_size, 3), dtype=torch.float32, device="cuda"
    )
    auxiliary = _AuxiliaryBuffers.zeros(batch_size)
    step.renderer.forward_training_base(
        step.rays.origins,
        step.rays.directions,
        image_unclamped,
        *step.background_rgb,
        render_m,
        render_s,
        render_q,
        rgba,
        model.k,
        step.ray_termination_threshold,
        auxiliary.depth,
        step.reg_depth_a,
        step.reg_depth_b,
        auxiliary.depth_and_index,
        auxiliary.surface_normal,
        auxiliary.normal,
    )
    surface_normal = _surface_normal(step, auxiliary)
    loss, image, image_gradient = _loss_and_image_gradient(
        step, image_unclamped
    )

    with torch.no_grad():
        dL_dRGB = torch.zeros_like(model.RGB)
        dL_dA = torch.zeros_like(model.A)
        dL_dk = torch.zeros_like(model.k)
        dL_dm = torch.zeros_like(model.m)
        dL_ds = torch.zeros_like(model.s)
        dL_dq = torch.zeros_like(model.q)
        prefix = torch.zeros(
            (batch_size, 4), dtype=torch.float32, device="cuda"
        )
        step.renderer.backward_base(
            step.rays.origins,
            step.rays.directions,
            *step.background_rgb,
            render_m,
            render_s,
            render_q,
            rgba,
            model.k,
            image,
            image_gradient,
            dL_dRGB,
            dL_dA,
            dL_dk,
            dL_dm,
            dL_ds,
            dL_dq,
            step.ray_termination_threshold,
            auxiliary.depth,
            prefix,
            step.depth_lambda / batch_size,
            step.reg_depth_a,
            step.reg_depth_b,
            auxiliary.depth_and_index,
            surface_normal,
            auxiliary.normal,
            step.normal_lambda / batch_size,
        )
        weight = 1.0 - step.warmup_lambda
        blended.RGB.add_(weight * dL_dRGB)
        blended.A.add_(weight * dL_dA)
        blended.k.add_(weight * dL_dk)
        blended.m.add_(weight * dL_dm)
        blended.s.add_(weight * dL_ds)
        blended.q.add_(weight * dL_dq)

    return _metrics(loss)


def _reduce_full_mlp_gradients(
    dL_dw3: torch.Tensor,
    dL_db3: torch.Tensor,
    dL_dw2: torch.Tensor,
    dL_db2: torch.Tensor,
    dL_dw1: torch.Tensor,
    dL_db1: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    dL_dw3 = dL_dw3.sum(0)
    dL_dw3 = dL_dw3.reshape((8, 2, 32))
    dL_dw3 = dL_dw3.transpose(1, 2)
    dL_dw3 = dL_dw3.reshape((8, 8, -1))
    dL_dw3 = dL_dw3.transpose(0, 1)
    dL_dw3 = dL_dw3.flatten(1, 2)
    dL_db3 = dL_db3.sum(0)
    dL_db3 = dL_db3.reshape((2, 32))
    dL_db3 = dL_db3.transpose(0, 1)
    dL_db3 = dL_db3.reshape((8, 8))
    dL_db3 = dL_db3.sum(1)
    dL_dw2 = dL_dw2.sum(0)
    dL_dw2 = dL_dw2.reshape((64, 2, 32))
    dL_dw2 = dL_dw2.transpose(1, 2)
    dL_dw2 = dL_dw2.reshape((8, 8, 8, -1))
    dL_dw2 = dL_dw2.transpose(1, 2)
    dL_dw2 = dL_dw2.flatten(0, 1)
    dL_dw2 = dL_dw2.flatten(1, 2)
    dL_db2 = dL_db2.sum(0)
    dL_db2 = dL_db2.reshape((8, 2, 32))
    dL_db2 = dL_db2.transpose(1, 2)
    dL_db2 = dL_db2.reshape((8, 8, -1))
    dL_db2 = dL_db2.sum(2)
    dL_db2 = dL_db2.flatten(0, 1)
    dL_dw1 = dL_dw1.sum(0)
    dL_dw1 = dL_dw1.reshape((128, 2, 32))
    dL_dw1 = dL_dw1.transpose(1, 2)
    dL_dw1 = dL_dw1.reshape((8, 16, 8, -1))
    dL_dw1 = dL_dw1.transpose(1, 2)
    dL_dw1 = dL_dw1.flatten(0, 1)
    dL_dw1 = dL_dw1.flatten(1, 2)
    dL_db1 = dL_db1.sum(0)
    dL_db1 = dL_db1.reshape((8, 2, 32))
    dL_db1 = dL_db1.transpose(1, 2)
    dL_db1 = dL_db1.reshape((8, 8, -1))
    dL_db1 = dL_db1.sum(2)
    dL_db1 = dL_db1.flatten(0, 1)
    return dL_dw3, dL_db3, dL_dw2, dL_db2, dL_dw1, dL_db1


COMPACT_DESCRIPTOR_WIDTHS = (4, 8, 16)


def _zero_descriptor_fast_path(step, model) -> bool:
    return model.conditioning_variable.shape[1] == 0 and (
        getattr(step, "fast_zero_conditioning", False)
        or getattr(step, "fast_descriptor", False)
    )


def _compact_descriptor_width(step, model) -> int:
    """Renderer descriptor width for --fast_descriptor, 0 for the 96D ABI."""
    dimensions = model.conditioning_variable.shape[1]
    if not getattr(step, "fast_descriptor", False) or dimensions == 0:
        return 0
    for width in COMPACT_DESCRIPTOR_WIDTHS:
        if dimensions <= width:
            return width
    raise ValueError("--fast_descriptor supports at most 16 descriptor dimensions")


def _run_full_phase(
    step: FACETTrainingStepInput,
    blended: _BlendedGradients,
    rgba: torch.Tensor,
) -> PhaseMetrics:
    model = step.model
    render_m, render_s, render_q = _render_geometry(model)
    batch_size = step.rays.origins.shape[0]
    renderer_w1_conditioning = (
        model.renderer_w1_conditioning()
        if hasattr(model, "renderer_w1_conditioning")
        else model.w1_conditioning
    )
    w1_fp16 = torch.cat(
        [
            model.w1_uv.detach(),
            model.w1_v.detach(),
            renderer_w1_conditioning.detach(),
        ],
        1,
    ).to(torch.float16)
    w2_fp16 = model.w2.detach().to(torch.float16)
    w3_fp16 = model.w3.detach().to(torch.float16)
    renderer_conditioning = (
        model.native_conditioning()
        if hasattr(model, "native_conditioning")
        else model.conditioning_variable
    )
    compact_width = _compact_descriptor_width(step, model)
    if compact_width:
        conditioning_fp16 = torch.nn.functional.pad(
            model.conditioning_variable.detach(),
            (0, compact_width - model.conditioning_variable.shape[1]),
        ).to(torch.float16).contiguous()
    elif _zero_descriptor_fast_path(step, model):
        conditioning_fp16 = torch.zeros(
            (1, renderer_conditioning.shape[1]), dtype=torch.float16, device="cuda"
        ).expand(renderer_conditioning.shape[0], -1)
    else:
        conditioning_fp16 = renderer_conditioning.detach().to(torch.float16)
    image_unclamped = torch.zeros(
        (batch_size, 3), dtype=torch.float32, device="cuda"
    )
    auxiliary = _AuxiliaryBuffers.zeros(batch_size)
    step.renderer.forward_training(
        conditioning_fp16,
        model.features,
        w1_fp16,
        model.b1,
        w2_fp16,
        model.b2,
        w3_fp16,
        model.b3,
        step.rays.origins,
        step.rays.directions,
        image_unclamped,
        *step.background_rgb,
        render_m,
        render_s,
        render_q,
        rgba,
        model.k,
        step.ray_termination_threshold,
        auxiliary.depth,
        step.reg_depth_a,
        step.reg_depth_b,
        auxiliary.depth_and_index,
        auxiliary.surface_normal,
        auxiliary.normal,
    )
    surface_normal = _surface_normal(step, auxiliary)
    loss, image, image_gradient = _loss_and_image_gradient(
        step, image_unclamped
    )

    with torch.no_grad():
        sm_buffers = 4 * step.sm_count
        dL_dRGB = torch.zeros_like(model.RGB)
        dL_dA = torch.zeros_like(model.A)
        dL_dk = torch.zeros_like(model.k)
        dL_dw3 = torch.zeros(
            (sm_buffers, 8 * 64), dtype=torch.float32, device="cuda"
        )
        dL_db3 = torch.zeros(
            (sm_buffers, 8 * 8), dtype=torch.float32, device="cuda"
        )
        dL_dw2 = torch.zeros(
            (sm_buffers, 64 * 64), dtype=torch.float32, device="cuda"
        )
        dL_db2 = torch.zeros(
            (sm_buffers, 64 * 8), dtype=torch.float32, device="cuda"
        )
        dL_dw1 = torch.zeros(
            (sm_buffers, 64 * 128), dtype=torch.float32, device="cuda"
        )
        dL_db1 = torch.zeros(
            (sm_buffers, 64 * 8), dtype=torch.float32, device="cuda"
        )
        native_conditioning_dimensions = getattr(
            model, "RENDERER_CONDITIONING_DIMENSIONS",
            model.conditioning_variable.shape[1],
        )
        if compact_width:
            native_conditioning_dimensions = compact_width
        elif _zero_descriptor_fast_path(step, model):
            native_conditioning_dimensions = 0
        dL_d_conditioning_native = torch.zeros(
            (model.m.shape[0], native_conditioning_dimensions),
            dtype=model.conditioning_variable.dtype,
            device=model.conditioning_variable.device,
        )
        dL_d_features = torch.zeros_like(
            model.features, dtype=torch.float32, device="cuda"
        )
        dL_dm = torch.zeros_like(model.m)
        dL_ds = torch.zeros_like(model.s)
        dL_dq = torch.zeros_like(model.q)
        prefix = torch.zeros(
            (batch_size, 4), dtype=torch.float32, device="cuda"
        )
        step.renderer.backward(
            conditioning_fp16,
            model.features,
            w1_fp16,
            model.b1,
            w2_fp16,
            model.b2,
            w3_fp16,
            model.b3,
            step.rays.origins,
            step.rays.directions,
            *step.background_rgb,
            render_m,
            render_s,
            render_q,
            rgba,
            model.k,
            image,
            image_gradient,
            dL_dRGB,
            dL_dA,
            dL_dk,
            dL_dw3,
            dL_db3,
            dL_dw2,
            dL_db2,
            dL_dw1,
            dL_db1,
            dL_d_conditioning_native,
            dL_d_features,
            dL_dm,
            dL_ds,
            dL_dq,
            step.ray_termination_threshold,
            auxiliary.depth,
            prefix,
            step.depth_lambda / batch_size,
            step.reg_depth_a,
            step.reg_depth_b,
            auxiliary.depth_and_index,
            surface_normal,
            auxiliary.normal,
            step.normal_lambda / batch_size,
        )
        (
            dL_dw3,
            dL_db3,
            dL_dw2,
            dL_db2,
            dL_dw1,
            dL_db1,
        ) = _reduce_full_mlp_gradients(
            dL_dw3, dL_db3, dL_dw2, dL_db2, dL_dw1, dL_db1
        )
        weight = step.warmup_lambda
        blended.RGB.add_(weight * dL_dRGB)
        blended.A.add_(weight * dL_dA)
        blended.k.add_(weight * dL_dk)
        blended.w3.add_(weight * dL_dw3)
        blended.b3.add_(weight * dL_db3)
        blended.w2.add_(weight * dL_dw2)
        blended.b2.add_(weight * dL_db2)
        blended.w1.add_(weight * dL_dw1)
        blended.b1.add_(weight * dL_db1)
        blended.conditioning_variable.add_(
            weight
            * dL_d_conditioning_native[
                :, : model.conditioning_variable.shape[1]
            ]
        )
        blended.features.add_(weight * dL_d_features)
        blended.m.add_(weight * dL_dm)
        blended.s.add_(weight * dL_ds)
        blended.q.add_(weight * dL_dq)

    return _metrics(loss)


def run_facet_training_step(
    step: FACETTrainingStepInput,
) -> FACETTrainingStepResult:
    """Render, backpropagate native gradients, and perform one Adam step."""

    model = step.model
    rgba = torch.cat([model.RGB, model.A], 1)
    blended = _BlendedGradients.zeros(model)

    base_metrics = None
    if step.warmup_lambda < 1.0:
        base_metrics = _run_base_phase(step, blended, rgba)

    facet_metrics = None
    if step.warmup_lambda > 0.0:
        facet_metrics = _run_full_phase(step, blended, rgba)
        for view in step.auxiliary_views:
            _run_full_phase(
                dataclasses.replace(
                    step,
                    rays=view.rays,
                    foreground=view.target,
                    alpha=view.alpha,
                    loss_weights=view.loss_weights,
                    warmup_lambda=step.warmup_lambda * view.weight,
                    depth_lambda=0.0,
                    normal_lambda=0.0,
                    camera_forward=None,
                ),
                blended,
                rgba,
            )

    gradients = blended.parameter_bundle(model, step.scale_regularization)
    gradients.assign_to(model)
    if step.freeze_view_weights and model.w1_v.grad is not None:
        model.w1_v.grad.zero_()
    step.optimizer.step()
    return FACETTrainingStepResult(
        base=base_metrics,
        facet=facet_metrics,
        gradients=gradients,
    )
