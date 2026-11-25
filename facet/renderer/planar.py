"""Native orthographic rendering helpers for the Slice 3.2 prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from facet.model.geometry import PlanarCompatibilityConstraint
from facet.model.rays import generate_indexed_orthographic_rays


@dataclass(frozen=True)
class RenderDeterminismReport:
    repeat_count: int
    bitwise_equal: bool
    maximum_absolute_difference: float


@dataclass(frozen=True)
class HitBatchingReport:
    capacity: int
    ray_count: int
    maximum: int
    median: float
    p95: float
    p99: float
    p99_9: float
    rays_exceeding_batch_capacity: int
    fraction_exceeding_batch_capacity: float


def order_hits_lexicographically(
    t: torch.Tensor,
    primitive_ids: torch.Tensor,
    *,
    capacity: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ordering for the native bounded ``(t, primitive_id)`` set."""

    if t.ndim != 1 or primitive_ids.ndim != 1 or t.shape != primitive_ids.shape:
        raise ValueError("t and primitive_ids must be equal-length vectors")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    order = sorted(
        range(t.numel()),
        key=lambda index: (float(t[index]), int(primitive_ids[index])),
    )[:capacity]
    indices = torch.tensor(order, dtype=torch.int64, device=t.device)
    return t[indices], primitive_ids[indices]


def summarize_hit_batching(
    candidate_counts: torch.Tensor,
    *,
    capacity: int = 16,
) -> HitBatchingReport:
    """Summarize first-pass candidate load relative to the batch capacity."""

    if candidate_counts.ndim != 1 or candidate_counts.numel() == 0:
        raise ValueError("candidate_counts must be a non-empty vector")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    counts = candidate_counts.detach().to(device="cpu", dtype=torch.float64)
    if bool((counts < 0).any()):
        raise ValueError("candidate counts cannot be negative")
    quantiles = torch.quantile(
        counts, torch.tensor([0.5, 0.95, 0.99, 0.999], dtype=torch.float64)
    )
    exceeding = int((counts > capacity).sum().item())
    return HitBatchingReport(
        capacity=capacity,
        ray_count=counts.numel(),
        maximum=int(counts.max().item()),
        median=float(quantiles[0]),
        p95=float(quantiles[1]),
        p99=float(quantiles[2]),
        p99_9=float(quantiles[3]),
        rays_exceeding_batch_capacity=exceeding,
        fraction_exceeding_batch_capacity=exceeding / counts.numel(),
    )


def _constrain_compatibility_model(model, plane_z: float) -> None:
    """Project only the intermediate XYZ/quaternion representation."""
    if "mean_xy" not in getattr(model, "_parameters", {}):
        PlanarCompatibilityConstraint(plane_z).apply(
            model, getattr(model, "optimizer", None)
        )


def diagnose_planar_hit_batching(
    renderer,
    model,
    dataset,
    *,
    capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, HitBatchingReport]:
    """Trace the first native hit batch and report its candidate load."""

    if capacity is None:
        capacity = int(getattr(renderer, "hit_buffer_size", 16))
    _constrain_compatibility_model(model, dataset.config.plane_z)
    renderer.sync_geometry(model)
    rays = dataset.rays(dataset.flat_indices)
    t_min = torch.zeros(
        dataset.ray_count, dtype=dataset.dtype, device=dataset.device
    )
    t, primitive_ids, counts = renderer.trace_hits(
        rays.origins, rays.directions, t_min, dataset.ray_count
    )
    return t, primitive_ids, summarize_hit_batching(counts, capacity=capacity)


def render_planar(
    renderer,
    model,
    dataset,
    *,
    mode: str = "FACET",
    background_rgb: tuple[float, float, float] | None = None,
    ray_termination_threshold: float = 0.01,
) -> torch.Tensor:
    """Render a row-major base or full-FACET image with explicit parallel rays."""

    if mode not in ("base", "FACET"):
        raise ValueError("planar render mode must be 'base' or 'FACET'")
    if ray_termination_threshold <= 0.0:
        raise ValueError("ray_termination_threshold must be positive")
    _constrain_compatibility_model(model, dataset.config.plane_z)
    renderer.sync_geometry(model)
    rays = dataset.rays(dataset.flat_indices)
    background = (
        dataset.config.background_rgb
        if background_rgb is None
        else background_rgb
    )
    image = torch.zeros(
        (dataset.ray_count, 3), dtype=dataset.dtype, device=dataset.device
    )
    rgba = torch.cat((model.RGB, model.A), dim=1).contiguous()
    common = (
        rays.origins,
        rays.directions,
        image,
        *background,
        model.m,
        model.s,
        model.q,
        rgba,
        model.k,
        ray_termination_threshold,
    )
    if mode == "base":
        renderer.forward_inference_base(*common)
    else:
        w1 = torch.cat(
            (model.w1_uv, model.w1_v, model.renderer_w1_conditioning()), dim=1
        ).detach().to(torch.float16)
        decoder = (
            model.native_conditioning().detach().to(torch.float16),
            model.features,
            w1,
            model.b1,
            model.w2.detach().to(torch.float16),
            model.b2,
            model.w3.detach().to(torch.float16),
            model.b3,
        )
        renderer.forward_inference(*decoder, *common)
    return image.reshape(dataset.height, dataset.width, 3)


def render_planar_resolution(
    renderer,
    model,
    dataset,
    width: int,
    height: int,
    *,
    mode: str = "FACET",
    background_rgb: tuple[float, float, float] | None = None,
    ray_termination_threshold: float = 0.01,
    tile_size: int | None = None,
) -> torch.Tensor:
    """Render the same continuous plane at an arbitrary resolution in tiles."""
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    if mode not in ("base", "FACET"):
        raise ValueError("planar render mode must be 'base' or 'FACET'")
    pixel_count = width * height
    tile_size = dataset.ray_count if tile_size is None else tile_size
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    _constrain_compatibility_model(model, dataset.config.plane_z)
    renderer.sync_geometry(model)
    image = torch.zeros(
        (pixel_count, 3), dtype=dataset.dtype, device=dataset.device
    )
    rgba = torch.cat((model.RGB, model.A), dim=1).contiguous()
    w1 = torch.cat(
        (model.w1_uv, model.w1_v, model.renderer_w1_conditioning()), dim=1
    ).detach().to(torch.float16)
    conditioning = model.native_conditioning().detach().to(torch.float16)
    w2 = model.w2.detach().to(torch.float16)
    w3 = model.w3.detach().to(torch.float16)
    background = (
        dataset.config.background_rgb
        if background_rgb is None
        else background_rgb
    )
    for start in range(0, pixel_count, tile_size):
        stop = min(start + tile_size, pixel_count)
        indices = torch.arange(
            start,
            stop,
            dtype=torch.int64,
            device=dataset.device,
        )
        rays = generate_indexed_orthographic_rays(
            indices,
            width,
            height,
            plane_extent_x=dataset.config.plane_extent_x,
            plane_z=dataset.config.plane_z,
            camera_distance=dataset.config.camera_distance,
            dtype=dataset.dtype,
        )
        tile = image[start:stop]
        common = (
            rays.origins,
            rays.directions,
            tile,
            *background,
            model.m,
            model.s,
            model.q,
            rgba,
            model.k,
            ray_termination_threshold,
        )
        if mode == "base":
            renderer.forward_inference_base(*common)
        else:
            decoder = (conditioning, model.features, w1, model.b1, w2, model.b2, w3, model.b3)
            renderer.forward_inference(*decoder, *common)
    return image.reshape(height, width, 3)




def render_planar_base(
    renderer,
    model,
    dataset,
    *,
    background_rgb: tuple[float, float, float] | None = None,
    ray_termination_threshold: float = 0.01,
) -> torch.Tensor:
    """Render the constant-colour warm-up representation."""

    return render_planar(
        renderer,
        model,
        dataset,
        mode="base",
        background_rgb=background_rgb,
        ray_termination_threshold=ray_termination_threshold,
    )


def diagnose_planar_render_determinism(
    render,
    *,
    repeat_count: int = 3,
) -> tuple[torch.Tensor, RenderDeterminismReport]:
    """Repeat a render and report both bitwise equality and worst deviation."""

    if repeat_count < 2:
        raise ValueError("repeat_count must be at least two")
    reference = render()
    bitwise_equal = True
    maximum_difference = 0.0
    for _ in range(repeat_count - 1):
        candidate = render()
        if candidate.shape != reference.shape:
            raise ValueError("repeated renders returned different shapes")
        bitwise_equal = bitwise_equal and torch.equal(candidate, reference)
        difference = float((candidate - reference).abs().max().item())
        maximum_difference = max(maximum_difference, difference)
    return reference, RenderDeterminismReport(
        repeat_count=repeat_count,
        bitwise_equal=bitwise_equal,
        maximum_absolute_difference=maximum_difference,
    )
