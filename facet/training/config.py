"""Typed configuration schema for the current FACET training interface."""

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping


class ConfigGroup:
    """Shared construction helpers for one typed configuration group."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        return cls(**{field.name: values[field.name] for field in fields(cls)})

    @classmethod
    def from_namespace(cls, namespace):
        return cls.from_mapping(vars(namespace))


@dataclass(frozen=True)
class Image2DConfig(ConfigGroup):
    """Validated image-domain contract for the orthographic FACET 2D path."""

    source_path: str = ""
    mode: str = "image_2d"
    projection: str = "orthographic"
    plane_extent_x: float = 2.0
    plane_z: float = 0.0
    camera_distance: float = 1.0
    alpha_mode: str = "composite"
    bg_color_R: float = 0.0
    bg_color_G: float = 0.0
    bg_color_B: float = 0.0

    def __post_init__(self) -> None:
        if self.mode != "image_2d":
            raise ValueError("Image2DConfig requires mode='image_2d'")
        if self.projection != "orthographic":
            raise ValueError(
                "image_2d currently supports only projection='orthographic'"
            )
        if not math.isfinite(self.plane_extent_x) or self.plane_extent_x <= 0.0:
            raise ValueError("plane_extent_x must be a positive finite value")
        if not math.isfinite(self.plane_z):
            raise ValueError("plane_z must be finite")
        if not math.isfinite(self.camera_distance) or self.camera_distance <= 0.0:
            raise ValueError("camera_distance must be a positive finite value")
        if self.alpha_mode not in ("composite", "mask"):
            raise ValueError("alpha_mode must be 'composite' or 'mask'")
        background = (self.bg_color_R, self.bg_color_G, self.bg_color_B)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in background
        ):
            raise ValueError("2D background components must lie in [0, 1]")

    @property
    def background_rgb(self) -> tuple[float, float, float]:
        return (self.bg_color_R, self.bg_color_G, self.bg_color_B)


@dataclass
class EssentialConfig(ConfigGroup):
    source_path: str = ""
    model_path: str = ""
    start_iter: int = 0
    end_iter: int = 64_000
    # Endpoints of the linear warm-up from the base renderer to the full decoder.
    warmup_start_iter: int = 0
    warmup_end_iter: int = 1_000
    resolution: int = 1_000_000
    bg_color_R: float = 0.0
    bg_color_G: float = 0.0
    bg_color_B: float = 0.0
    t_near: float = 0.1
    t_far: float = 1_000.0
    images: str = "images"
    data_device: str = "cuda"


@dataclass
class PerformanceConfig(ConfigGroup):
    # Random-background transparency carving is applied only when the
    # training images contain a non-opaque alpha channel.
    random_background: bool = False
    number_of_sides: int = 8
    border_opacity: float = 0.003439
    initial_conditioning_std: float = 0.01
    initial_opacity: float = 0.01
    initial_k: float = 1.01
    ray_termination_T_threshold_training: float = 0.0001
    ray_termination_T_threshold_inference: float = 0.01
    opacity_threshold_for_Gauss_removal: float = 1.0 / 28.0
    densification_frequency: int = 100
    densification_start_iter: int = 800
    densification_end_iter: int = 24_000
    # Primitive budget and densification thresholds of the renderer interface; fitting
    # uses a fixed budget (fit.py sets max_Gaussians_per_model to --primitive_count).
    max_Gaussians_per_model: int = -1
    mu_grad_norm_threshold_for_densification: float = 0.001
    min_s_norm_threshold_for_Gauss_removal: float = 0.0001
    min_s_coef_clipping_threshold: float = 0.0001
    max_s_coef_clipping_threshold: float = 0.05
    # Depth and normal regularization of the renderer interface (fit.py sets both weights to 0).
    reg_depth_lambda: float = 0.0001
    reg_depth_start_iter: int = 0
    reg_normal_lambda: float = 0.0
    reg_normal_start_iter: int = 7_000
    reg_normal_ramp_iters: int = 3_000
    reg_normal_depth_edge_threshold: float = 0.02
    reg_scale_lambda: float = 0.001


@dataclass
class LearningConfig(ConfigGroup):
    lr_RGB: float = 0.005
    lr_RGB_exp_decay_coef: float = -0.000086643
    lr_RGB_final: float = 0.001

    lr_A: float = 0.05
    lr_A_exp_decay_coef: float = -0.000086643
    lr_A_final: float = 0.025

    lr_k: float = 0.1
    lr_k_exp_decay_coef: float = -0.000086643
    lr_k_final: float = 0.1

    lr_w1_uv: float = 0.0125
    lr_w1_uv_exp_decay_coef: float = -0.000086643
    lr_w1_uv_final: float = 0.001

    lr_w1_v: float = 0.03
    lr_w1_v_exp_decay_coef: float = -0.000086643
    lr_w1_v_final: float = 0.001

    lr_w1_conditioning: float = 0.0015
    lr_w1_conditioning_exp_decay_coef: float = -0.000086643
    lr_w1_conditioning_final: float = 0.001

    lr_b1: float = 0.00475
    lr_b1_exp_decay_coef: float = -0.000086643
    lr_b1_final: float = 0.001

    lr_w2: float = 0.005
    lr_w2_exp_decay_coef: float = -0.000086643
    lr_w2_final: float = 0.001

    lr_b2: float = 0.005
    lr_b2_exp_decay_coef: float = -0.000086643
    lr_b2_final: float = 0.001

    lr_w3: float = 0.005
    lr_w3_exp_decay_coef: float = -0.000086643
    lr_w3_final: float = 0.001

    lr_b3: float = 0.005
    lr_b3_exp_decay_coef: float = -0.000086643
    lr_b3_final: float = 0.001

    lr_conditioning: float = 0.005
    lr_conditioning_exp_decay_coef: float = -0.000086643
    lr_conditioning_final: float = 0.005

    lr_features: float = 0.005
    lr_features_exp_decay_coef: float = -0.000086643
    lr_features_final: float = 0.005

    lr_m: float = 0.0025
    lr_m_exp_decay_coef: float = -0.000150000
    lr_m_final: float = 0.000001189614075

    lr_s: float = 0.0396538025
    lr_s_exp_decay_coef: float = -0.000086643
    lr_s_final: float = 0.0396538025

    lr_q: float = 0.01
    lr_q_exp_decay_coef: float = -0.000086643
    lr_q_final: float = 0.001


@dataclass
class ApplicationConfig(ConfigGroup):
    real_time_preview: bool = False
    preview_resolution_scale: float = 1.0
    preview_frequency: int = 10


@dataclass
class FACETConfig:
    essential: EssentialConfig
    performance: PerformanceConfig
    learning: LearningConfig
    application: ApplicationConfig

    @classmethod
    def from_namespace(cls, namespace):
        return cls(
            essential=EssentialConfig.from_namespace(namespace),
            performance=PerformanceConfig.from_namespace(namespace),
            learning=LearningConfig.from_namespace(namespace),
            application=ApplicationConfig.from_namespace(namespace),
        )
