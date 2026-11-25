"""Full-image evaluation and artifacts for FACET 2D experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

from PIL import Image
import torch

from facet.evaluation.lpips.modules.lpips import LPIPS
from facet.renderer.planar import (
    diagnose_planar_hit_batching,
    diagnose_planar_render_determinism,
    render_planar_resolution,
)
from facet.evaluation.metrics import psnr
from facet.evaluation.ssim import ssim


@dataclass(frozen=True)
class PlanarEvaluationResult:
    metrics: dict[str, float | int | bool | None]
    hit_batching: dict[str, float | int]
    renders: dict[str, str]
    crops: dict[str, str]


def _save_rgb(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = image.detach().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


def _save_error(error: torch.Tensor, path: Path) -> None:
    maximum = float(error.max().item())
    normalized = error / maximum if maximum > 0.0 else error
    pixels = normalized.detach().mul(255.0).round().byte().cpu().numpy()
    Image.fromarray(pixels, mode="L").save(path)


def metric_image(image: torch.Tensor) -> torch.Tensor:
    """Match the clamped RGB domain optimized by the image loss and saved PNGs."""
    return image.clamp(0.0, 1.0)


def multi_scale_ssim(
    image: torch.Tensor,
    target: torch.Tensor,
    metric=None,
) -> torch.Tensor:
    """Compute MS-SSIM with the TorchMetrics implementation used by GaINeR."""
    if metric is None:
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

        metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(
            image.device
        )
    return metric(image, target)


def optional_multi_scale_ssim(
    image: torch.Tensor,
    target: torch.Tensor,
    metric=None,
) -> float | None:
    """Return MS-SSIM when the default five-scale window fits the image."""
    if min(image.shape[-2:]) <= 160:
        return None
    try:
        return float(multi_scale_ssim(image, target, metric).item())
    except ValueError as error:
        message = str(error)
        if not any(
            marker in message
            for marker in (
                "image height must be larger than",
                "image width must be larger than",
            )
        ):
            raise
        return None


def _diagnostic_crop_bounds(
    error: torch.Tensor, crop_size: int
) -> tuple[int, int, int, int]:
    height, width = error.shape
    size = min(crop_size, height, width)
    flat_index = int(torch.argmax(error).item())
    center_y, center_x = divmod(flat_index, width)
    left = min(max(center_x - size // 2, 0), width - size)
    top = min(max(center_y - size // 2, 0), height - size)
    return left, top, left + size, top + size


def _cuda_timed(callback):
    if not torch.cuda.is_available():
        started = time.perf_counter()
        result = callback()
        return result, time.perf_counter() - started
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callback()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end) / 1000.0


def evaluate_planar_reconstruction(
    renderer,
    model,
    dataset,
    output_directory: str | Path,
    *,
    ray_termination_threshold: float = 0.01,
    crop_size: int = 128,
    lpips_metric=None,
    ms_ssim_metric=None,
    render_mode: str = "FACET",
    save_upscaled_renders: bool = True,
    save_target: bool = True,
) -> PlanarEvaluationResult:
    """Report reconstruction quality, capacity, cost, and deterministic renders.

    The 2x and 4x renders are always produced, because the reported peak
    evaluation memory includes them; ``save_upscaled_renders`` only controls
    whether they are written. ``save_target`` controls the copy of the input.
    """
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    render = lambda scale: render_planar_resolution(
        renderer,
        model,
        dataset,
        dataset.width * scale,
        dataset.height * scale,
        mode=render_mode,
        ray_termination_threshold=ray_termination_threshold,
        tile_size=dataset.ray_count,
    )

    torch.cuda.reset_peak_memory_stats(dataset.device)
    image_1x, determinism = diagnose_planar_render_determinism(
        lambda: render(1), repeat_count=3
    )
    image_1x, render_seconds = _cuda_timed(lambda: render(1))
    image_2x = render(2)
    image_4x = render(4)
    torch.cuda.synchronize(dataset.device)
    peak_allocated_mb = torch.cuda.max_memory_allocated(dataset.device) / 2**20
    peak_reserved_mb = torch.cuda.max_memory_reserved(dataset.device) / 2**20

    target_hwc = dataset.target.reshape(dataset.height, dataset.width, 3)
    image_1x_metric = metric_image(image_1x)
    rendered = image_1x_metric.permute(2, 0, 1).unsqueeze(0).contiguous()
    target = (
        target_hwc.permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    if lpips_metric is None:
        lpips_metric = LPIPS("vgg", "0.1").to(dataset.device).eval()
    with torch.no_grad():
        metric_values = {
            "psnr": float(psnr(rendered, target).mean().item()),
            "ssim": float(ssim(rendered, target).item()),
            "ms_ssim": optional_multi_scale_ssim(
                rendered, target, ms_ssim_metric
            ),
            "lpips_vgg": float(lpips_metric(rendered, target).item()),
            "primitive_count": int(model.mean_xy.shape[0]),
            "render_time_ms_1x": 1000.0 * render_seconds,
            "peak_memory_allocated_mb": peak_allocated_mb,
            "peak_memory_reserved_mb": peak_reserved_mb,
            "deterministic_bitwise": determinism.bitwise_equal,
            "deterministic_max_abs": determinism.maximum_absolute_difference,
        }

    _, _, hit_report = diagnose_planar_hit_batching(renderer, model, dataset)
    error = torch.mean(torch.abs(image_1x_metric - target_hwc), dim=2)
    bounds = _diagnostic_crop_bounds(error, crop_size)
    left, top, right, bottom = bounds

    render_paths = {}
    saved = ((1, image_1x), (2, image_2x), (4, image_4x)) if save_upscaled_renders \
        else ((1, image_1x),)
    for scale, image in saved:
        path = output / f"render_{scale}x.png"
        _save_rgb(image, path)
        render_paths[f"{scale}x"] = str(path)
    if save_target:
        _save_rgb(target_hwc, output / "target.png")

    crop_paths = {
        "render": str(output / "crop_render.png"),
        "target": str(output / "crop_target.png"),
        "absolute_error": str(output / "crop_absolute_error.png"),
    }
    _save_rgb(image_1x_metric[top:bottom, left:right], Path(crop_paths["render"]))
    _save_rgb(target_hwc[top:bottom, left:right], Path(crop_paths["target"]))
    _save_error(error[top:bottom, left:right], Path(crop_paths["absolute_error"]))

    hit_batching = asdict(hit_report)
    return PlanarEvaluationResult(
        metrics=metric_values,
        hit_batching=hit_batching,
        renders=render_paths,
        crops=crop_paths,
    )
