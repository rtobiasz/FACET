"""Optimize CLIPGaussian appearance style for a single FACET 2D image."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from facet.training.config import Image2DConfig
from facet.io.checkpoint import (
    create_checkpoint_payload,
    load_training_checkpoint,
    save_checkpoint_payload,
)
from facet.evaluation.reconstruction import _save_rgb
from facet.renderer.facade import FACETRenderer
from facet.data.image_dataset import Image2DDataset
from facet.model.components import (
    PlanarGaussianModel,
    materialize_renderer_conditioning,
)
from facet.stylization.losses import CLIPGaussianConfig, LatentOptimizationConfig, optimize_latent_style
from facet.stylization.decoder_backward import backward_model as perspective_backward_model, packed_weights


def _rays(dataset):
    rays = dataset.rays(dataset.flat_indices)
    return rays.origins, rays.directions


def forward_inference(renderer, checkpoint, _camera, conditioning, args, width, height):
    checkpoint["RGBA"] = torch.cat((checkpoint["RGB"], checkpoint["A"]), 1).contiguous()
    origins, directions = _rays(args.dataset)
    image = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda")
    w1, w2, w3 = packed_weights(checkpoint)
    renderer.forward_inference(
        materialize_renderer_conditioning(conditioning).to(torch.float16),
        checkpoint["features"], w1, checkpoint["b1"],
        w2, checkpoint["b2"], w3, checkpoint["b3"], origins, directions, image,
        args.bg_color_R, args.bg_color_G, args.bg_color_B, checkpoint["m"],
        checkpoint["s"], checkpoint["q"], checkpoint["RGBA"], checkpoint["k"],
        args.ray_termination_T_threshold_inference,
    )
    return image


def forward_training(renderer, checkpoint, _camera, conditioning, args, width, height):
    checkpoint["RGBA"] = torch.cat((checkpoint["RGB"], checkpoint["A"]), 1).contiguous()
    origins, directions = _rays(args.dataset)
    count = width * height
    image = torch.zeros(count, 3, dtype=torch.float32, device="cuda")
    depth = torch.zeros(count, 4, dtype=torch.float32, device="cuda")
    depth_and_index = torch.zeros(count, 2, dtype=torch.float32, device="cuda")
    surface_normal = torch.zeros(count, 3, dtype=torch.float32, device="cuda")
    normal = torch.zeros(count, 4, dtype=torch.float32, device="cuda")
    w1, w2, w3 = packed_weights(checkpoint)
    reg_depth_a = -(args.t_near * args.t_far) / (args.t_far - args.t_near)
    reg_depth_b = args.t_far / (args.t_far - args.t_near)
    renderer.forward_training(
        materialize_renderer_conditioning(conditioning).to(torch.float16),
        checkpoint["features"], w1, checkpoint["b1"], w2,
        checkpoint["b2"], w3, checkpoint["b3"], origins, directions, image,
        args.bg_color_R, args.bg_color_G, args.bg_color_B, checkpoint["m"],
        checkpoint["s"], checkpoint["q"], checkpoint["RGBA"], checkpoint["k"],
        args.ray_termination_T_threshold_training, depth, reg_depth_a, reg_depth_b,
        depth_and_index, surface_normal, normal,
    )
    return image, origins, directions, depth, reg_depth_a, reg_depth_b, depth_and_index, surface_normal, normal


# Style losses give image gradients far larger than reconstruction, enough to
# overflow the renderer's fixed-point accumulators. The backward pass is linear
# in the image gradient, so it is scaled down until it fits and the result is
# scaled back; the scale persists across calls and the result stays
# deterministic.
_GRADIENT_SCALE = [1.0]
_NON_FINITE_GRADIENT_STEPS = [0]


def _report_backward_inputs(renderer, args, kwargs):
    """Print what went into a backward pass that overflows at any gradient scale."""
    def describe(name, value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            finite = torch.isfinite(value)
            magnitude = value[finite].abs().max().item() if finite.any() else float("nan")
            print(f"  {name}: shape {tuple(value.shape)} non-finite {int((~finite).sum())} max {magnitude:.4g}")
    checkpoint, cache = args[0], args[2]
    for key, value in checkpoint.items():
        describe(f"checkpoint[{key}]", value)
    for index, value in enumerate(cache):
        describe(f"camera_cache[{index}]", value)
    for index in (1, 3, 4, 5):
        describe(f"argument {index}", args[index])
    trial = list(args)
    trial[5] = torch.zeros_like(args[5])
    try:
        perspective_backward_model(renderer._native_renderer, *trial, **kwargs)
        print("  backward with a zero image gradient: ok")
    except RuntimeError as error:
        print(f"  backward with a zero image gradient: {error}")


def backward_model(renderer, *args, **kwargs):
    args = list(args)
    active_dimensions = args[1].shape[1]
    args[1] = materialize_renderer_conditioning(args[1]).to(torch.float16)
    # A model that reproduces its image almost exactly makes CLIP(render) -
    # CLIP(source) vanish at the first step; its normalization then gives a
    # non-finite image gradient, which carries no direction. Those entries are
    # dropped (the patch term still has a direction) and counted.
    image_gradient = args[5]
    finite = torch.isfinite(image_gradient)
    if not bool(finite.all()):
        _NON_FINITE_GRADIENT_STEPS[0] += 1
        image_gradient = torch.where(finite, image_gradient, torch.zeros_like(image_gradient))
    # The native backward overwrites its image and cache buffers, so a retry
    # needs fresh copies.
    pristine = (args[2], args[3], args[4])
    while True:
        scale = _GRADIENT_SCALE[0]
        args[5] = image_gradient * scale
        args[2] = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in pristine[0])
        args[3], args[4] = pristine[1].clone(), pristine[2].clone()
        try:
            gradients = perspective_backward_model(renderer._native_renderer, *args, **kwargs)
            break
        except RuntimeError as error:
            if "overflowed" not in str(error):
                raise
            if scale < 1e-12:
                _report_backward_inputs(renderer, args, kwargs)
                raise
            _GRADIENT_SCALE[0] = scale / 16.0
    if scale != 1.0:
        gradients = {name: value / scale for name, value in gradients.items()}
    gradients["conditioning_variable"] = gradients["conditioning_variable"][
        :, :active_dimensions
    ].contiguous()
    return gradients


def backward_conditioning(renderer, *args, **kwargs):
    return backward_model(renderer, *args, **kwargs)["conditioning_variable"]


def update_planar_geometry(renderer, checkpoint, plane_z):
    """Keep tuned ellipses planar, then hand the new geometry to the renderer."""
    with torch.no_grad():
        q = checkpoint["q"]
        q[:, 1:3] = 0.0
        q.div_(q.norm(dim=1, keepdim=True).clamp_min(1e-12))
        checkpoint["m"][:, 2] = plane_z
    rgb = checkpoint["RGB"]
    checkpoint["RGBA"] = torch.cat((rgb, checkpoint["A"]), 1).contiguous()
    renderer.backend.set_geometry(
        renderer._native_renderer,
        checkpoint["m"].contiguous(),
        torch.exp(checkpoint["s"]).contiguous(),
        checkpoint["q"].contiguous(),
        torch.sigmoid(checkpoint["A"]).contiguous(),
        (1.0 + torch.nn.functional.softplus(checkpoint["k"])).contiguous(),
    )


def arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    style_source = parser.add_mutually_exclusive_group(required=True)
    style_source.add_argument("--style_image", type=Path)
    style_source.add_argument("--style_prompt", help="text condition, e.g. 'Starry Night by Vincent van Gogh'")
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--num_crops", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--style_lr", type=float, default=0.005)
    parser.add_argument("--model_lr", type=float, default=0.005)
    parser.add_argument("--model_reg_weight", type=float, default=0.0001)
    parser.add_argument("--no_model_finetune", action="store_true")
    parser.add_argument("--rgb_only", action="store_true")
    parser.add_argument(
        "--with_base_rgb",
        action="store_true",
        help="with the shared appearance (no --rgb_only), also tune the base colours",
    )
    parser.add_argument(
        "--finetune_geometry",
        action="store_true",
        help=(
            "also tune the ellipses as FACET 3D style transfer does (scales, opacity, "
            "kappa, in-plane angle and position); after every step the rotation is "
            "projected back to a turn about +z and the centre back onto the plane"
        ),
    )
    parser.add_argument("--rgb_bounds", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="clamp the tuned base colours to [LOW, HIGH] after every step (default: no clamp)")
    parser.add_argument("--geometry_grad_clip", type=float, default=0.1)
    parser.add_argument("--geometry_lr_m", type=float, default=1e-4)
    parser.add_argument("--geometry_lr_s", type=float, default=5e-3)
    parser.add_argument("--geometry_lr_q", type=float, default=1e-3)
    parser.add_argument("--geometry_lr_A", type=float, default=1e-3)
    parser.add_argument("--geometry_lr_k", type=float, default=1e-4)
    parser.add_argument("--lambda_dir", type=float, default=5.0)
    parser.add_argument("--lambda_patch", type=float, default=90.0)
    parser.add_argument("--lambda_content", type=float, default=0.8)
    parser.add_argument("--lambda_gram", type=float, default=0.0)
    parser.add_argument("--patch_threshold", type=float, default=0.0)
    parser.add_argument("--latent_reg_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_styled_checkpoint",
        action="store_true",
        help=(
            "write styled.checkpoint: the source image_2d checkpoint with the styled "
            "appearance tensors, loadable by every FACET 2D tool (edits, physics, lifting)"
        ),
    )
    return parser.parse_args()


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    state = load_training_checkpoint(args.checkpoint, "cuda", require_rgb=True)
    if state.config is None or state.config.get("mode") != "image_2d":
        raise ValueError("checkpoint is not an image_2d checkpoint")
    image_config = Image2DConfig(**state.config["image"])
    dataset = Image2DDataset(image_config, device="cuda")
    active = int(state.config.get("ablation", {}).get("active_conditioning_dimensions", 96))
    model = PlanarGaussianModel(
        state.model, plane_z=image_config.plane_z, requires_grad=False,
        active_conditioning_dimensions=active,
    )
    renderer = FACETRenderer(8, 11.3449, dataset.ray_count)
    renderer.sync_geometry(model)
    checkpoint = model.export_geometry()
    checkpoint["RGBA"] = torch.cat((checkpoint["RGB"], checkpoint["A"]), 1).contiguous()
    checkpoint["iteration"] = state.iteration
    checkpoint["training_time_seconds"] = state.training_time_seconds
    render_args = SimpleNamespace(
        dataset=dataset, bg_color_R=image_config.bg_color_R,
        bg_color_G=image_config.bg_color_G, bg_color_B=image_config.bg_color_B,
        ray_termination_T_threshold_inference=0.01,
        ray_termination_T_threshold_training=0.01, t_near=0.01, t_far=100.0,
    )
    args.output_path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        before = forward_inference(
            renderer, checkpoint, None, checkpoint["conditioning_variable"],
            render_args, dataset.width, dataset.height,
        ).reshape(dataset.height, dataset.width, 3)
    _save_rgb(before, args.output_path / "before.png")
    style_image_path = str(args.style_image.resolve()) if args.style_image is not None else ""
    if args.lambda_gram > 0.0 and not style_image_path:
        raise ValueError("--lambda_gram needs --style_image")
    style_config = CLIPGaussianConfig(
        style_image=style_image_path, style_prompt=args.style_prompt or "", lambda_dir=args.lambda_dir,
        lambda_patch=args.lambda_patch, lambda_content=args.lambda_content,
        lambda_gram=args.lambda_gram, patch_threshold=args.patch_threshold,
        lambda_bg=0.0, num_crops=args.num_crops, crop_size=args.crop_size,
        background=image_config.background_rgb,
    )
    latent_config = LatentOptimizationConfig(
        steps=args.steps, lr=args.style_lr, save_every=args.save_every,
        num_views=1, latent_reg_weight=args.latent_reg_weight,
        finetune_model=not args.no_model_finetune,
        finetune_shared_appearance=not args.rgb_only,
        finetune_base_rgb=args.rgb_only or args.with_base_rgb, model_lr=args.model_lr,
        model_reg_weight=args.model_reg_weight, freeze_view_branch=True,
        finetune_geometry=args.finetune_geometry,
        rgb_bounds=tuple(args.rgb_bounds) if args.rgb_bounds else None,
        geometry_grad_clip=args.geometry_grad_clip,
        geometry_lr_m=args.geometry_lr_m, geometry_lr_s=args.geometry_lr_s,
        geometry_lr_q=args.geometry_lr_q, geometry_lr_A=args.geometry_lr_A,
        geometry_lr_k=args.geometry_lr_k,
    )
    conditioning, styled = optimize_latent_style(
        style_config=style_config, latent_config=latent_config, renderer=renderer,
        checkpoint=checkpoint, base_conditioning=checkpoint["conditioning_variable"],
        cameras=[None], width=dataset.width, height=dataset.height,
        render_dir=str(args.output_path), forward_inference=forward_inference,
        forward_training=forward_training, backward_conditioning=backward_conditioning,
        backward_model=backward_model, render_args=render_args,
        update_geometry=lambda r, c: update_planar_geometry(r, c, image_config.plane_z),
        active_conditioning_dimensions=active, save_styled_model_checkpoint=False,
    )
    with torch.no_grad():
        after = forward_inference(
            renderer, styled, None, conditioning, render_args,
            dataset.width, dataset.height,
        ).reshape(dataset.height, dataset.width, 3)
    _save_rgb(after, args.output_path / "after.png")
    if args.save_styled_checkpoint:
        styled_model = {}
        planar = {
            "mean_xy": styled["m"][:, :2],
            "theta": 2.0 * torch.atan2(styled["q"][:, 3], styled["q"][:, 0]),
        }
        for name, value in state.model.items():
            replacement = planar[name] if name in planar else styled.get(name)
            if name == "conditioning_variable":
                replacement = conditioning
            if isinstance(replacement, torch.Tensor) and replacement.shape == value.shape:
                styled_model[name] = replacement.detach().to(value.dtype).contiguous()
            else:
                styled_model[name] = value.detach()
        config = dict(state.config)
        config["style"] = {
            "style_image": style_image_path or None,
            "style_prompt": args.style_prompt,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "steps": args.steps,
            "rgb_only": args.rgb_only,
            "with_base_rgb": args.with_base_rgb,
            "finetune_geometry": args.finetune_geometry,
            "num_crops": args.num_crops,
            "lambda_dir": args.lambda_dir,
            "lambda_patch": args.lambda_patch,
            "lambda_content": args.lambda_content,
            "lambda_gram": args.lambda_gram,
            "patch_threshold": args.patch_threshold,
            "model_lr": args.model_lr,
            "model_reg_weight": args.model_reg_weight,
            "seed": args.seed,
        }
        save_checkpoint_payload(
            args.output_path / "styled.checkpoint",
            create_checkpoint_payload(
                styled_model,
                optimizer_state=None,
                iteration=state.iteration,
                training_time_seconds=state.training_time_seconds,
                config=config,
                mode="image_2d",
            ),
        )
    report = {
        "schema": "facet-image-2d-style-gate-v1", "checkpoint": str(args.checkpoint.resolve()),
        "style_image": style_image_path or None, "style_prompt": args.style_prompt,
        "active_conditioning_dimensions": active,
        "steps": args.steps, "num_crops": args.num_crops, "seed": args.seed,
        "model_finetune": not args.no_model_finetune,
        "rgb_only": args.rgb_only,
        "model_lr": args.model_lr,
        "model_reg_weight": args.model_reg_weight,
        "mean_absolute_change": float((after - before).abs().mean().item()),
        "maximum_absolute_change": float((after - before).abs().max().item()),
        "backward_gradient_scale": _GRADIENT_SCALE[0],
        "non_finite_gradient_steps": _NON_FINITE_GRADIENT_STEPS[0],
    }
    (args.output_path / "style_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
