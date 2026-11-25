"""Reusable orchestration for one FACET training iteration."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import torch

from facet.training.optimizer_state import (
    update_planar_learning_rates,
)
from facet.model.rays import RayBundle
from facet.training.step import (
    FACETTrainingStepInput,
    FACETTrainingStepResult,
    run_facet_training_step,
)


LEARNING_RATE_FIELDS = {
    "RGB": "RGB",
    "A": "A",
    "k": "k",
    "w1_uv": "w1_uv",
    "w1_v": "w1_v",
    "w1_conditioning": "w1_conditioning",
    "b1": "b1",
    "w2": "w2",
    "b2": "b2",
    "w3": "w3",
    "b3": "b3",
    "conditioning_variable": "conditioning",
    "features": "features",
    "m": "m",
    "s": "s",
    "q": "q",
}


def scheduled_warmup_lambda(
    iteration: int, start_iteration: int, end_iteration: int
) -> float:
    """Return the linear warm-up blend coefficient between the base renderer and the full decoder."""
    return float(
        np.clip(
            (iteration - start_iteration) / (end_iteration - start_iteration),
            0,
            1,
        )
    )


def density_adaptive_scale_fraction(
    base_fraction: float,
    primitive_count: int,
    maximum_gaussians: int,
    exponent: float,
) -> float:
    """Relax the scale cap for sparse topology and recover it at full budget."""
    if exponent < 0.0:
        raise ValueError("density-adaptive scale exponent cannot be negative")
    if exponent == 0.0 or maximum_gaussians == -1:
        return float(base_fraction)
    if primitive_count <= 0 or maximum_gaussians <= 0:
        raise ValueError("primitive counts must be positive")
    density_ratio = max(float(maximum_gaussians) / primitive_count, 1.0)
    return float(base_fraction) * density_ratio**exponent


def scheduled_learning_rates(learning_config, iteration: int) -> dict[str, float]:
    """Evaluate the exponential learning-rate schedules."""
    values: dict[str, float] = {}
    for parameter_name, field_name in LEARNING_RATE_FIELDS.items():
        initial = getattr(learning_config, "lr_" + field_name)
        coefficient = getattr(
            learning_config, "lr_" + field_name + "_exp_decay_coef"
        )
        final = getattr(learning_config, "lr_" + field_name + "_final")
        decayed = initial * np.exp(coefficient * iteration)
        if coefficient <= 0.0:
            values[parameter_name] = float(np.maximum(decayed, final))
        else:
            values[parameter_name] = float(np.minimum(decayed, final))
    return values


@dataclass(frozen=True)
class TrainingBatch:
    rays: RayBundle
    foreground: torch.Tensor
    alpha: torch.Tensor
    image_height: int
    image_width: int
    camera_forward: torch.Tensor | None
    loss_weights: torch.Tensor | None = None


class Image2DFullBatchSampler:
    """Return every image pixel in stable row-major order on every step."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        if dataset.config.alpha_mode == "composite":
            self.foreground = dataset.foreground * dataset.alpha
            self.alpha = dataset.alpha
        else:
            self.foreground = dataset.foreground
            self.alpha = torch.ones_like(dataset.alpha)
        self._indices = dataset.flat_indices

    def next(self) -> TrainingBatch:
        return TrainingBatch(
            rays=self.dataset.rays(self._indices),
            foreground=self.foreground,
            alpha=self.alpha,
            image_height=self.dataset.height,
            image_width=self.dataset.width,
            camera_forward=None,
            loss_weights=self.dataset.loss_weights,
        )

    def advance(self) -> None:
        pass


class PlanarTopologyController:
    """Production 2D topology policy with structural XY/angle storage."""

    strict_2d = True

    def __init__(
        self,
        plane_extent_x: float,
        plane_extent_y: float,
        *,
        pruning_enabled: bool = True,
        freeze_at_maximum: bool = False,
        selection_signal: str = "movement",
        selection_fraction: float | None = None,
        spatial_balance_power: float = 0.0,
        spatial_balance_grid: int = 32,
        split_policy: str = "snapshot_current",
        split_offset: float = 0.5,
        split_scale: float = 0.625,
        split_optimizer_state: str = "legacy",
        density_adaptive_scale_exponent: float = 0.0,
        contrast_adaptive_scale_cap=None,
        maximum_rgb: float | None = None,
        maximum_rgb_final: float | None = None,
        maximum_rgb_relax_iteration: int | None = None,
    ) -> None:
        if (
            not np.isfinite(plane_extent_x)
            or not np.isfinite(plane_extent_y)
            or plane_extent_x <= 0.0
            or plane_extent_y <= 0.0
        ):
            raise ValueError("planar extents must be positive and finite")
        self.plane_extent_x = float(plane_extent_x)
        self.plane_extent_y = float(plane_extent_y)
        self.pruning_enabled = bool(pruning_enabled)
        self.freeze_at_maximum = bool(freeze_at_maximum)
        self.selection_signal = selection_signal
        self.selection_fraction = selection_fraction
        self.spatial_balance_power = spatial_balance_power
        self.spatial_balance_grid = spatial_balance_grid
        self.split_policy = split_policy
        self.split_offset = split_offset
        self.split_scale = split_scale
        self.split_optimizer_state = split_optimizer_state
        if density_adaptive_scale_exponent < 0.0:
            raise ValueError("density-adaptive scale exponent cannot be negative")
        self.density_adaptive_scale_exponent = density_adaptive_scale_exponent
        self.contrast_adaptive_scale_cap = contrast_adaptive_scale_cap
        if maximum_rgb is not None and not maximum_rgb > 0.0:
            raise ValueError("maximum RGB must be positive")
        # None: base colours are bounded below only.
        self.maximum_rgb = maximum_rgb
        # Optional step schedule: the bound becomes maximum_rgb_final from
        # maximum_rgb_relax_iteration on. Hot pixels form early, while the
        # ~80 dB images want a looser bound only for late refinement.
        if (maximum_rgb_final is None) != (maximum_rgb_relax_iteration is None):
            raise ValueError("maximum_rgb_final and its relax iteration go together")
        if maximum_rgb_final is not None and maximum_rgb is None:
            raise ValueError("a relaxed RGB bound needs an initial one")
        self.maximum_rgb_final = maximum_rgb_final
        self.maximum_rgb_relax_iteration = maximum_rgb_relax_iteration
        self.current_iteration = 0
        self._topology_frozen = False

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration = int(iteration)

    def _rgb_bound(self) -> float | None:
        if (self.maximum_rgb_relax_iteration is not None
                and self.current_iteration >= self.maximum_rgb_relax_iteration):
            return self.maximum_rgb_final
        return self.maximum_rgb

    def _maximum_log_scale(self, model, performance_config, extent: float):
        """Scalar cap, or a per-primitive one when a cap policy is attached."""
        fraction = self._maximum_scale_fraction(model, performance_config)
        cap = self.contrast_adaptive_scale_cap
        if cap is None or not cap.enabled:
            return np.log(fraction * (extent / 2.0))
        fractions = cap.fractions(model.mean_xy.detach(), fraction)
        return torch.log(fractions * (extent / 2.0))

    def _maximum_scale_fraction(self, model, performance_config) -> float:
        return density_adaptive_scale_fraction(
            performance_config.max_s_coef_clipping_threshold,
            model.number_of_Gaussians,
            performance_config.max_Gaussians_per_model,
            self.density_adaptive_scale_exponent,
        )

    def extent(self, model) -> float:
        with torch.no_grad():
            model.mean_xy[:, 0].clamp_(
                -0.5 * self.plane_extent_x, 0.5 * self.plane_extent_x
            )
            model.mean_xy[:, 1].clamp_(
                -0.5 * self.plane_extent_y, 0.5 * self.plane_extent_y
            )
        return self.plane_extent_x

    @staticmethod
    def _learning_rates(learning_rates: dict[str, float]) -> dict[str, float]:
        result = dict(learning_rates)
        result["mean_xy"] = result.pop("m")
        result["theta"] = result.pop("q")
        return result

    def constrain(
        self,
        model,
        optimizer,
        performance_config,
        extent: float,
        learning_rates: dict[str, float],
    ) -> None:
        model.s.data.clamp_(
            min=np.log(performance_config.min_s_coef_clipping_threshold)
        )
        model.s.data.copy_(
            torch.minimum(
                model.s.data,
                torch.as_tensor(
                    self._maximum_log_scale(model, performance_config, extent),
                    device=model.s.device,
                    dtype=model.s.dtype,
                ),
            )
        )
        model.RGB.data.clamp_(min=0.0, max=self._rgb_bound())
        self.extent(model)
        update_planar_learning_rates(
            optimizer, self._learning_rates(learning_rates)
        )


@dataclass(frozen=True)
class TrainerStepReport:
    phases: FACETTrainingStepResult
    elapsed_seconds: float
    extent: float
    kappa_min: float
    kappa_average: float
    kappa_max: float


class FACETTrainer:
    """Coordinate sampling, one isolated step, LR updates, and topology."""

    def __init__(
        self,
        *,
        model,
        renderer,
        optimizer,
        sampler: "Image2DFullBatchSampler",
        essential_config,
        performance_config,
        learning_config,
        sm_count: int,
        training_time_seconds: float = 0.0,
        topology_controller=None,
        step_function: Callable = run_facet_training_step,
        restore_clamped_gradient: bool = False,
        fast_zero_conditioning: bool = False,
        fast_descriptor: bool = False,
        background_seed: int = 0,
    ) -> None:
        self.model = model
        # Random background colors (--random_background) come from this generator, seeded from --seed; the
        # [seed, 1] entropy keeps it a separate stream from the auxiliary-view generator default_rng(seed).
        self.background_generator = np.random.default_rng([background_seed, 1])
        self.restore_clamped_gradient = restore_clamped_gradient
        self.fast_zero_conditioning = fast_zero_conditioning
        self.fast_descriptor = fast_descriptor
        self.renderer = renderer
        self.optimizer = optimizer
        self.sampler = sampler
        self.essential = essential_config
        self.performance = performance_config
        self.learning = learning_config
        self.sm_count = sm_count
        self.topology = topology_controller
        self.step_function = step_function
        self.extent = self.topology.extent(model)
        self.training_time_seconds = training_time_seconds
        self.fixed_background = torch.tensor(
            [
                essential_config.bg_color_R,
                essential_config.bg_color_G,
                essential_config.bg_color_B,
            ],
            dtype=torch.float32,
            device=sampler.foreground.device,
        ).reshape(1, 3)
        self.random_background_enabled = (
            performance_config.random_background
            and bool(torch.any(sampler.alpha < 1.0).item())
        )
        self.reg_depth_a = -(
            essential_config.t_near * essential_config.t_far
        ) / (essential_config.t_far - essential_config.t_near)
        self.reg_depth_b = essential_config.t_far / (
            essential_config.t_far - essential_config.t_near
        )

    def _background(self):
        if not self.random_background_enabled:
            return self.fixed_background, (
                self.essential.bg_color_R,
                self.essential.bg_color_G,
                self.essential.bg_color_B,
            )
        values = self.background_generator.random(3).astype(np.float32)
        margin = getattr(self, "random_background_margin", 0.0)
        if margin > 0.0:
            values = (values * (1.0 + 2.0 * margin) - margin).astype(np.float32)
        background = torch.as_tensor(
            values, device=self.sampler.foreground.device
        ).reshape(1, 3)
        return background, tuple(values.tolist())

    def step(self, iteration: int, warmup_lambda: float) -> TrainerStepReport:
        set_iteration = getattr(self.topology, "set_iteration", None)
        if set_iteration is not None:
            set_iteration(iteration)
        started = time.perf_counter()
        strict_2d = bool(getattr(self.topology, "strict_2d", False))
        depth_lambda = (
            self.performance.reg_depth_lambda
            if iteration > self.performance.reg_depth_start_iter
            else 0.0
        )
        normal_ramp = np.clip(
            (iteration - self.performance.reg_normal_start_iter)
            / max(float(self.performance.reg_normal_ramp_iters), 1.0),
            0.0,
            1.0,
        )
        normal_lambda = self.performance.reg_normal_lambda * normal_ramp
        if strict_2d:
            depth_lambda = 0.0
            normal_lambda = 0.0
        background, background_rgb = self._background()
        batch = self.sampler.next()
        phases = self.step_function(
            FACETTrainingStepInput(
                model=self.model,
                renderer=self.renderer,
                optimizer=self.optimizer,
                rays=batch.rays,
                foreground=batch.foreground,
                alpha=batch.alpha,
                background=background,
                background_rgb=background_rgb,
                warmup_lambda=warmup_lambda,
                ray_termination_threshold=(
                    self.performance.ray_termination_T_threshold_training
                ),
                depth_lambda=depth_lambda,
                normal_lambda=normal_lambda,
                reg_depth_a=self.reg_depth_a,
                reg_depth_b=self.reg_depth_b,
                scale_regularization=self.performance.reg_scale_lambda,
                sm_count=self.sm_count,
                image_height=batch.image_height,
                image_width=batch.image_width,
                normal_depth_edge_threshold=(
                    self.performance.reg_normal_depth_edge_threshold
                ),
                camera_forward=(
                    batch.camera_forward if normal_lambda > 0.0 else None
                ),
                loss_weights=batch.loss_weights,
                restore_clamped_gradient=self.restore_clamped_gradient,
                fast_zero_conditioning=self.fast_zero_conditioning,
                fast_descriptor=self.fast_descriptor,
            )
        )
        self.sampler.advance()

        learning_rates = scheduled_learning_rates(self.learning, iteration)
        self.topology.constrain(
            self.model,
            self.optimizer,
            self.performance,
            self.extent,
            learning_rates,
        )
        self.extent = self.topology.extent(self.model)
        elapsed = time.perf_counter() - started
        self.training_time_seconds += elapsed

        return TrainerStepReport(
            phases=phases,
            elapsed_seconds=elapsed,
            extent=self.extent,
            kappa_min=(
                1.0 + torch.nn.functional.softplus(torch.min(self.model.k)).item()
            ),
            kappa_average=(
                1.0 + torch.nn.functional.softplus(torch.mean(self.model.k)).item()
            ),
            kappa_max=(
                1.0 + torch.nn.functional.softplus(torch.max(self.model.k)).item()
            ),
        )
