"""Evaluate a FACET 2D checkpoint against a higher-resolution target image."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import time

from facet.evaluation.lpips.modules.lpips import LPIPS
import torch

from facet.training.config import Image2DConfig
from facet.io.checkpoint import load_training_checkpoint
from facet.evaluation.reconstruction import _save_rgb, metric_image, optional_multi_scale_ssim
from facet.renderer.planar import render_planar_resolution
from facet.renderer.facade import FACETRenderer
from facet.data.image_dataset import Image2DDataset
from facet.model.components import PlanarGaussianModel
from facet.evaluation.metrics import psnr
from facet.evaluation.ssim import ssim


def scaled_dimensions(width: int, height: int, scale: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or scale <= 0:
        raise ValueError("width, height, and scale must be positive")
    return width * scale, height * scale


def require_compatible_dimensions(
    target: tuple[int, int],
    source: tuple[int, int],
    scale: int,
) -> None:
    expected = scaled_dimensions(*source, scale)
    tolerance = scale - 1
    if any(
        abs(actual - wanted) > tolerance
        for actual, wanted in zip(target, expected)
    ):
        raise ValueError(
            f"target dimensions {target} are incompatible with {scale}x source "
            f"dimensions {source}; expected approximately {expected}"
        )


def _arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--target_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument(
        "--alignment", choices=("modcrop", "full_hr"), default="modcrop",
        type=lambda value: "full_hr" if value == "legacy_full_hr" else value,
        help="modcrop: crop the high-resolution target to SCALE x the source size; "
             "full_hr: compare against the uncropped target")
    parser.add_argument("--extra_scales", nargs="*", type=int, default=())
    parser.add_argument("--tile_size", type=int)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    state = load_training_checkpoint(args.checkpoint, device, require_rgb=True)
    if state.config is None or state.config.get("mode") != "image_2d":
        raise ValueError("checkpoint is not an image_2d checkpoint")
    image = state.config["image"]
    config_kwargs = {
        "plane_extent_x": image["plane_extent_x"],
        "plane_z": image["plane_z"],
        "camera_distance": image["camera_distance"],
        "alpha_mode": image["alpha_mode"],
        "bg_color_R": image["bg_color_R"],
        "bg_color_G": image["bg_color_G"],
        "bg_color_B": image["bg_color_B"],
    }
    source = Image2DDataset(
        Image2DConfig(source_path=str(args.source_path), **config_kwargs),
        device=device,
    )
    target = Image2DDataset(
        Image2DConfig(source_path=str(args.target_path), **config_kwargs),
        device=device,
    )
    require_compatible_dimensions(
        (target.width, target.height), (source.width, source.height), args.scale
    )
    original_target_dimensions = [target.width, target.height]
    if args.alignment == "modcrop":
        width, height = scaled_dimensions(source.width, source.height, args.scale)
        if target.width < width or target.height < height:
            raise ValueError("HR is smaller than the requested modcrop")
        target.target = (
            target.target.reshape(target.height, target.width, 3)[:height, :width]
            .contiguous().reshape(-1, 3)
        )
        target.width, target.height = width, height

    active_conditioning_dimensions = state.config.get("ablation", {}).get(
        "active_conditioning_dimensions", 96
    )
    model = PlanarGaussianModel(
        state.model,
        plane_z=image["plane_z"],
        requires_grad=False,
        active_conditioning_dimensions=active_conditioning_dimensions,
    )
    performance = state.config.get("performance", {})
    renderer = FACETRenderer(
        int(performance.get("number_of_sides", 8)),
        11.3449,
        args.tile_size or source.ray_count,
    )
    output = args.output_path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scales = tuple(dict.fromkeys((args.scale, *args.extra_scales)))
    renders: dict[str, str] = {}
    target_render = None
    render_times: dict[str, float] = {}
    with torch.no_grad():
        for scale in scales:
            if scale == args.scale:
                width, height = target.width, target.height
            else:
                width, height = scaled_dimensions(source.width, source.height, scale)
            started = time.perf_counter()
            rendered = render_planar_resolution(
                renderer,
                model,
                source,
                width,
                height,
                ray_termination_threshold=float(
                    performance.get("ray_termination_T_threshold_inference", 0.01)
                ),
                tile_size=args.tile_size or source.ray_count,
            )
            torch.cuda.synchronize(device)
            render_times[f"{scale}x"] = 1000.0 * (time.perf_counter() - started)
            path = output / f"render_{scale}x.png"
            _save_rgb(rendered, path)
            renders[f"{scale}x"] = str(path)
            if scale == args.scale:
                target_render = metric_image(rendered)

        assert target_render is not None
        rendered_nchw = target_render.permute(2, 0, 1).unsqueeze(0).contiguous()
        target_nchw = target.target.reshape(target.height, target.width, 3)
        target_nchw = target_nchw.permute(2, 0, 1).unsqueeze(0).contiguous()
        lpips_metric = LPIPS("vgg", "0.1").to(device).eval()
        metrics = {
            "psnr": float(psnr(rendered_nchw, target_nchw).mean().item()),
            "ssim": float(ssim(rendered_nchw, target_nchw).item()),
            "ms_ssim": optional_multi_scale_ssim(rendered_nchw, target_nchw),
            "lpips_vgg": float(lpips_metric(rendered_nchw * 2 - 1, target_nchw * 2 - 1).item()),
        }

    target_path = output / "target_hr.png"
    _save_rgb(target.target.reshape(target.height, target.width, 3), target_path)
    report = {
        "schema": "facet-image-2d-superresolution-v2",
        "alignment": args.alignment,
        "original_target_dimensions": original_target_dimensions,
        "lpips_input_range": [-1, 1],
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "source": str(source.path),
        "target": str(target.path),
        "source_dimensions": [source.width, source.height],
        "target_dimensions": [target.width, target.height],
        "evaluation_scale": args.scale,
        "metrics": metrics,
        "render_time_ms": render_times,
        "renders": renders,
        "target_render": str(target_path),
    }
    (output / "superresolution.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
