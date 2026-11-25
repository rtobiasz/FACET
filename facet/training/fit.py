"""Train and evaluate a structurally planar FACET model on one image."""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

from facet.training.config import (
    EssentialConfig,
    Image2DConfig,
    LearningConfig,
    PerformanceConfig,
)
from facet.io.checkpoint import (
    create_checkpoint_payload,
    load_training_checkpoint,
    save_checkpoint_payload,
)
from facet.training.optimizer_state import (
    build_planar_optimizer,
    compact_planar_conditioning_optimizer_state,
)
from facet.evaluation.reconstruction import evaluate_planar_reconstruction
from facet.renderer.planar import render_planar
from facet.renderer.facade import FACETRenderer
from facet.data.image_dataset import Image2DDataset
from facet.model.components import (
    PlanarGaussianModel,
    PlanarInitializationConfig,
    initialize_planar_model,
)
from facet.training.trainer import (
    FACETTrainer,
    Image2DFullBatchSampler,
    PlanarTopologyController,
    scheduled_learning_rates,
    scheduled_warmup_lambda,
)
from facet.io.history import write_history
from facet.evaluation.metrics import psnr


def _render_mode(appearance: str) -> str:
    return "base" if appearance == "base_colour" else "FACET"


def _post_step_psnr(renderer, model, dataset, *, render_mode="FACET") -> float:
    """Evaluate the synchronized model state that a checkpoint would contain."""
    with torch.no_grad():
        image = render_planar(
            renderer, model, dataset, mode=render_mode
        ).clamp(0.0, 1.0)
        rendered = image.permute(2, 0, 1).unsqueeze(0).contiguous()
        target = (
            dataset.target.reshape(dataset.height, dataset.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        return float(psnr(rendered, target).mean().item())


DISABLEABLE_COMPONENTS = ("kappa", "uv", "view", "opacity_modulation")
# The decoder emits four (R, G, B, opacity) groups; these are the opacity rows.
_OPACITY_OUTPUTS = (3, 7, 11, 15)


def _impose_disabled_components(model, components, raw_kappa, zero_conditioning=0) -> None:
    """Hold disabled components at their fixed values (after init and every step)."""
    with torch.no_grad():
        if zero_conditioning:
            model.conditioning_variable[:, -zero_conditioning:] = 0.0
            model.w1_conditioning[:, -zero_conditioning:] = 0.0
        if "kappa" in components:
            model.k.fill_(raw_kappa)
        if "uv" in components:
            model.features.zero_()
            model.w1_uv.zero_()
        if "view" in components:
            model.w1_v.zero_()
        if "opacity_modulation" in components:
            rows = list(_OPACITY_OUTPUTS)
            model.w3[rows] = 0.0
            model.b3[rows] = 6.0


DECODER_LEARNING_RATE_FIELDS = (
    "w1_uv", "w1_v", "w1_conditioning", "b1", "w2", "b2", "w3", "b3")


def _parse_args():
    parser = ArgumentParser(description="FACET 2D natural-image reconstruction")
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--primitive_count", type=int, default=500000)
    parser.add_argument("--initial_opacity", type=float, default=0.01)
    parser.add_argument(
        "--active_conditioning_dimensions",
        type=int,
        default=4,
        choices=(0, 4),
        help=("descriptor size: 4 (default) or 0 (the descriptor-free ablation); the "
              "renderer interface reserves 96 channels, of which the leading ones are trained"),
    )
    parser.add_argument(
        "--disable_component",
        action="append",
        default=[],
        choices=DISABLEABLE_COMPONENTS,
        help=(
            "ablation: hold an inherited FACET component fixed for the whole run "
            "(re-imposed after every optimiser step): kappa (kernel shape kept at "
            "its initial, near-Gaussian value), uv (no in-primitive coordinates "
            "into the decoder: hit-point encoding table and its weights at zero), "
            "view (no ray-direction input), opacity_modulation (decoder's opacity "
            "outputs held at the neutral sigmoid(6))"
        ),
    )
    parser.add_argument(
        "--orientation",
        choices=("scalar_angle", "fixed_identity"),
        default="scalar_angle",
        help="train scalar in-plane angles or hold every primitive at identity",
    )
    parser.add_argument(
        "--appearance",
        choices=("full", "base_colour"),
        default="full",
        help=(
            "use the neural appearance decoder or constant per-primitive RGB "
            "(the constant-colour ablation)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum_scale", type=float, default=0.00125)
    parser.add_argument("--maximum_scale_fraction", type=float, default=0.0025)
    parser.add_argument(
        "--lean_outputs",
        action="store_true",
        help=("write only what results need: no final-state checkpoint (it is "
              "hashed, evaluated and then removed; best.checkpoint stays), no "
              "saved 2x/4x renders and no copy of the input image"),
    )
    parser.add_argument(
        "--maximum_rgb",
        type=float,
        default=None,
        help=("upper bound on the base primitive colour; unset: base colours "
              "are bounded below only"),
    )
    parser.add_argument(
        "--restore_clamped_gradient",
        action=BooleanOptionalAction,
        default=True, help=("let the image-loss gradient pass the [0, 1] render clamp where it "
              "points back into range, so over-bright pixels can recover"),
    )
    parser.add_argument(
        "--fast_descriptor",
        action="store_true",
        help=("speed (off by default): with --active_conditioning_dimensions 1-16 "
              "pass the descriptor to the training renderer as a compact N x 4/8/16 "
              "tensor and request its gradient at that width instead of the 96D "
              "ABI; needs a renderer built with COMPACT_CONDITIONING=1, which then "
              "skips the remaining first-layer inputs, dW1 columns and gradient "
              "atomics (not bit-exact, like a different seed). With 0D it acts as "
              "--fast_zero_conditioning"),
    )
    parser.add_argument("--scale_regularization", type=float, default=0.0005)
    parser.add_argument("--diagnostic_frequency", type=int, default=100)
    parser.add_argument("--alpha_mode", choices=("composite", "mask"), default="composite")
    parser.add_argument("--background_rgb", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--random_background",
        action="store_true",
        help="composite RGBA training targets over a new random RGB background",
    )
    parser.add_argument(
        "--random_background_opaque",
        action="store_true",
        help=("random background also for opaque images (off by default): the render is"
              " colour + transmittance x background, so a random background penalises"
              " transmittance and the model cannot borrow colour from the background"
              " (see-through after 2D->3D lifting)"),
    )
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument(
        "--sr_training_scale",
        type=int,
        default=0,
        help=("super-resolution (off by default): render at K times the image "
              "resolution and fit the image through the antialiased bicubic "
              "K-fold downscale (facet/superres/degradation.py); best checkpoints are "
              "picked by the degraded render's PSNR"),
    )
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("--iterations cannot be negative")
    if not 0.0 < args.initial_opacity < 1.0:
        parser.error("--initial_opacity must be in (0, 1)")
    if args.diagnostic_frequency <= 0:
        parser.error("--diagnostic_frequency must be positive")
    if args.sr_training_scale == 1 or args.sr_training_scale < 0:
        parser.error("--sr_training_scale must be 0 (off) or at least 2")
    return args


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _planar_learning_rates(
    config, iteration: int, *, orientation: str = "scalar_angle"
) -> dict[str, float]:
    rates = scheduled_learning_rates(config, iteration)
    rates["mean_xy"] = rates.pop("m")
    rates["theta"] = rates.pop("q")
    if orientation == "fixed_identity":
        rates["theta"] = 0.0
    return rates


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
    }


def _model_diagnostics(model: PlanarGaussianModel) -> dict:
    gradients = {}
    for name in ("mean_xy", "s", "theta", "A", "RGB", "conditioning_variable"):
        gradient = getattr(model, name).grad
        if gradient is not None:
            gradients[name] = _tensor_summary(torch.linalg.vector_norm(
                gradient.reshape(gradient.shape[0], -1), dim=1
            ))
    return {
        "primitive_count": int(model.number_of_Gaussians),
        "scales": _tensor_summary(model.scales),
        "opacities": _tensor_summary(model.opacities),
        "gradient_norms": gradients,
    }


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FACET 2D training")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output = args.output_path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    image_config = Image2DConfig(
        source_path=str(args.source_path),
        plane_extent_x=2.0,
        plane_z=0.0,
        camera_distance=1.0,
        alpha_mode=args.alpha_mode,
        bg_color_R=args.background_rgb[0],
        bg_color_G=args.background_rgb[1],
        bg_color_B=args.background_rgb[2],
    )
    dataset = Image2DDataset(image_config, device=device)
    initialization = PlanarInitializationConfig(
        primitive_count=args.primitive_count,
        method="random",
        seed=args.seed,
        appearance_initialization="neutral",
        active_conditioning_dimensions=args.active_conditioning_dimensions,
        initial_opacity=args.initial_opacity,
    )
    learning = LearningConfig()
    # Scaling both ends keeps the decay curve's shape; the manifest records the
    # resulting rates, so a run always shows which ones it trained with.
    learning.lr_m *= 1.0
    learning.lr_m_final *= 1.0
    if args.orientation == "fixed_identity":
        learning.lr_q = 0.0
        learning.lr_q_final = 0.0
    if args.resume is None:
        model = initialize_planar_model(dataset, initialization)
        if args.orientation == "fixed_identity":
            model.theta.data.zero_()
        optimizer = build_planar_optimizer(
            model,
            _planar_learning_rates(learning, 0, orientation=args.orientation),
        )
        first_iteration = 1
        training_seconds = 0.0
    else:
        state = load_training_checkpoint(
            args.resume.expanduser().resolve(), device, require_rgb=True
        )
        if state.config is None or state.config.get("mode") != "image_2d":
            raise ValueError("resume checkpoint lacks image_2d experiment config")
        model = PlanarGaussianModel(
            state.model,
            plane_z=image_config.plane_z,
            requires_grad=True,
            active_conditioning_dimensions=args.active_conditioning_dimensions,
        )
        checkpoint_ablation = state.config.get("ablation", {})
        if (
            checkpoint_ablation.get("orientation", "scalar_angle")
            != args.orientation
        ):
            raise ValueError("resume checkpoint orientation ablation does not match")
        if checkpoint_ablation.get("appearance", "full") != args.appearance:
            raise ValueError("resume checkpoint appearance ablation does not match")
        if checkpoint_ablation.get("active_conditioning_dimensions", 96) != args.active_conditioning_dimensions:
            raise ValueError("resume checkpoint conditioning ablation does not match")
        optimizer = build_planar_optimizer(
            model,
            _planar_learning_rates(
                learning, state.iteration, orientation=args.orientation
            ),
        )
        optimizer.load_state_dict(
            compact_planar_conditioning_optimizer_state(
                state.optimizer_state, args.active_conditioning_dimensions
            )
        )
        first_iteration = state.iteration + 1
        training_seconds = state.training_time_seconds
    model.optimizer = optimizer

    essential = EssentialConfig(
        source_path=str(dataset.path),
        end_iter=args.iterations,
        warmup_start_iter=250,
        warmup_end_iter=(
            250 + 500
        ),
        bg_color_R=args.background_rgb[0],
        bg_color_G=args.background_rgb[1],
        bg_color_B=args.background_rgb[2],
    )
    performance = PerformanceConfig(
        densification_start_iter=500,
        densification_end_iter=0,
        densification_frequency=100,
        max_Gaussians_per_model=args.primitive_count,
        mu_grad_norm_threshold_for_densification=1.0,
        opacity_threshold_for_Gauss_removal=1.0 / 28.0,
        min_s_coef_clipping_threshold=args.minimum_scale,
        max_s_coef_clipping_threshold=args.maximum_scale_fraction,
        random_background=args.random_background,
        reg_depth_lambda=0.0,
        reg_normal_lambda=0.0,
        reg_scale_lambda=args.scale_regularization,
    )
    post_step_psnr = _post_step_psnr
    sr_view = None
    if args.sr_training_scale:
        import facet.superres.degradation as sr_degradation
        sr_view = sr_degradation.render_view(dataset, args.sr_training_scale)
        sr_operator = sr_degradation.BicubicDegradation(
            dataset.width, dataset.height, args.sr_training_scale, device)
        sr_degradation.install(sr_operator)

        def post_step_psnr(renderer, model, dataset, *, render_mode="FACET"):
            return sr_degradation.degraded_psnr(
                renderer, model, sr_view, dataset, sr_operator,
                render_mode=render_mode)
    renderer = FACETRenderer(8, 11.3449, (dataset if sr_view is None else sr_view).ray_count)
    renderer.sync_geometry(model)
    topology = PlanarTopologyController(
        image_config.plane_extent_x,
        image_config.plane_extent_x * dataset.height / dataset.width,
        pruning_enabled=True,
        freeze_at_maximum=False,
        selection_signal="movement",
        selection_fraction=None,
        spatial_balance_power=0.0,
        spatial_balance_grid=32,
        split_policy="snapshot_current",
        split_offset=0.5,
        split_scale=0.625,
        split_optimizer_state="legacy",
        density_adaptive_scale_exponent=0.0,
        contrast_adaptive_scale_cap=None,
        maximum_rgb=args.maximum_rgb,
        maximum_rgb_final=None,
        maximum_rgb_relax_iteration=None,
    )
    trainer = FACETTrainer(
        model=model,
        renderer=renderer,
        optimizer=optimizer,
        sampler=Image2DFullBatchSampler(dataset if sr_view is None else sr_view),
        essential_config=essential,
        performance_config=performance,
        learning_config=learning,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        training_time_seconds=training_seconds,
        topology_controller=topology,
        restore_clamped_gradient=args.restore_clamped_gradient,
        fast_zero_conditioning=False,
        fast_descriptor=args.fast_descriptor,
        background_seed=args.seed,
        )
    if args.random_background_opaque:
        trainer.random_background_enabled = True

    disabled_components = tuple(sorted(set(args.disable_component)))
    disabled_raw_kappa = float(model.k.detach().flatten()[0].item())
    if args.fast_descriptor and args.active_conditioning_dimensions > 16:
        raise ValueError("--fast_descriptor supports at most 16 active descriptor dimensions")
    if not 0 <= 0 < max(1, args.active_conditioning_dimensions):
        raise ValueError("zero_conditioning_dimensions must be below the active dimensions")
    if disabled_components:
        if args.resume is not None:
            raise ValueError("--disable_component and --zero_conditioning_dimensions are for fresh runs only")
        _impose_disabled_components(
            model, disabled_components, disabled_raw_kappa, 0
        )
        renderer.sync_geometry(model)
    experiment_config = {
        "mode": "image_2d",
        "ablation": {
            "disabled_components": list(disabled_components),
            "orientation": args.orientation,
            "appearance": args.appearance,
            "active_conditioning_dimensions": args.active_conditioning_dimensions,
            "compositing": "deterministic_alpha",
            "maximum_rgb": args.maximum_rgb,
            "restore_clamped_gradient": args.restore_clamped_gradient,
            "fast_descriptor": args.fast_descriptor,
        },
        "image": asdict(image_config),
        "data": {
            "path": str(dataset.path),
            "width": dataset.width,
            "height": dataset.height,
            "coordinate_convention": "row-major pixel centres; +Y down",
            "colour_initialization": "nearest target pixel",
        },
        "initialization": asdict(initialization),
        "essential": asdict(essential),
        "performance": asdict(performance),
        "learning": asdict(learning),
        "diagnostic_frequency": args.diagnostic_frequency,
    }
    if args.random_background_opaque:
        experiment_config["ablation"]["random_background_opaque"] = True
    if args.sr_training_scale:
        experiment_config["ablation"]["sr_training_scale"] = args.sr_training_scale
    history = []
    topology_events = []
    best_checkpoint_path = output / "best.checkpoint"
    initial_diagnostic = {
        "iteration": first_iteration - 1,
        **_model_diagnostics(model),
    }
    best = {"psnr": float("-inf"), "iteration": first_iteration - 1}
    if args.resume is not None:
        resume_psnr = post_step_psnr(
            renderer,
            model,
            dataset,
            render_mode=(
                _render_mode(args.appearance)
            ),
        )
        initial_diagnostic["checkpoint_psnr"] = resume_psnr
        best = {"psnr": resume_psnr, "iteration": first_iteration - 1}
        save_checkpoint_payload(
            best_checkpoint_path,
            create_checkpoint_payload(
                model.model_tensors(),
                optimizer_state=optimizer.state_dict(),
                iteration=first_iteration - 1,
                training_time_seconds=trainer.training_time_seconds,
                config=experiment_config,
                mode="image_2d",
            ),
        )
    diagnostics = [initial_diagnostic]
    for iteration in range(first_iteration, args.iterations + 1):
        count_before = int(model.number_of_Gaussians)
        if disabled_components:
            # Applied before the step too, so the renderer never sees a free value.
            _impose_disabled_components(
                model, disabled_components, disabled_raw_kappa, 0
            )
        report = trainer.step(
            iteration,
            0.0
            if args.appearance == "base_colour"
            else scheduled_warmup_lambda(
                iteration,
                250,
                250 + 500,
            ),
        )
        if disabled_components:
            _impose_disabled_components(
                model, disabled_components, disabled_raw_kappa, 0
            )
        renderer.sync_geometry(model)
        if sr_view is not None and sr_operator.loss_calls == 0:
            raise RuntimeError("--sr_training_scale: the degraded loss was not used")
        optimizer = trainer.optimizer
        phase = report.phases.facet or report.phases.base
        count_after = int(model.number_of_Gaussians)
        history.append(
            {
                "iteration": iteration,
                "loss": phase.loss,
                "psnr": phase.psnr,
                "primitive_count": model.number_of_Gaussians,
                "step_seconds": report.elapsed_seconds,
            }
        )
        if count_after != count_before:
            topology_events.append({
                "iteration": iteration,
                "before": count_before,
                "after": count_after,
                "net_change": count_after - count_before,
            })
        diagnostic_due = (
            iteration == first_iteration
            or iteration % args.diagnostic_frequency == 0
            or iteration == args.iterations
        )
        if diagnostic_due:
            checkpoint_psnr = post_step_psnr(
                renderer,
                model,
                dataset,
                render_mode=(
                    _render_mode(args.appearance)
                ),
            )
            diagnostics.append({
                "iteration": iteration,
                "checkpoint_psnr": checkpoint_psnr,
                **_model_diagnostics(model),
            })
            if checkpoint_psnr > best["psnr"]:
                best = {"psnr": checkpoint_psnr, "iteration": iteration}
                save_checkpoint_payload(
                    best_checkpoint_path,
                    create_checkpoint_payload(
                        model.model_tensors(),
                        optimizer_state=optimizer.state_dict(),
                        iteration=iteration,
                        training_time_seconds=trainer.training_time_seconds,
                        config=experiment_config,
                        mode="image_2d",
                    ),
                )
        if iteration == first_iteration or iteration % 25 == 0:
            print(
                f"{iteration}: loss={phase.loss:.8f}, psnr={phase.psnr:.3f}, "
                f"gaussians={model.number_of_Gaussians}",
                flush=True,
            )

    completed_iteration = max(first_iteration - 1, args.iterations)
    checkpoint_path = output / "checkpoint.checkpoint"
    payload = create_checkpoint_payload(
        model.model_tensors(),
        optimizer_state=optimizer.state_dict(),
        iteration=completed_iteration,
        training_time_seconds=trainer.training_time_seconds,
        config=experiment_config,
        mode="image_2d",
    )
    save_checkpoint_payload(checkpoint_path, payload)

    restored_state = load_training_checkpoint(
        checkpoint_path, device, require_rgb=True
    )
    restored = PlanarGaussianModel(
        restored_state.model,
        plane_z=image_config.plane_z,
        requires_grad=False,
        active_conditioning_dimensions=args.active_conditioning_dimensions,
    )
    restored_renderer = FACETRenderer(8, 11.3449, dataset.ray_count)
    with torch.no_grad():
        evaluation = evaluate_planar_reconstruction(
            restored_renderer,
            restored,
            dataset,
            output / "evaluation",
            crop_size=args.crop_size,
            render_mode=(
                _render_mode(args.appearance)
            ),
            save_upscaled_renders=not args.lean_outputs,
            save_target=not args.lean_outputs,
        )

    manifest = {
        "schema": "facet-image-2d-experiment-v1",
        "git_revision": _git_revision(),
        "command": sys.argv,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "input_sha256": _file_sha256(dataset.path),
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_retained": not args.lean_outputs,
        "config": experiment_config,
        "training": {
            "iteration": completed_iteration,
            "time_seconds": trainer.training_time_seconds,
            # Peak VRAM is part of the cost comparison against GaINeR, and it
            # is otherwise unrecoverable once the process exits.
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None
            ),
            "peak_vram_reserved_bytes": (
                torch.cuda.max_memory_reserved()
                if torch.cuda.is_available()
                else None
            ),
            # The per-iteration history is ~93% of a 30000-iteration manifest
            # and no result reader needs it; it goes to a lossless sidecar.
            **write_history(output, history),
            "best_sampled_psnr": best,
            "topology_events": topology_events,
            "diagnostics": diagnostics,
        },
        "evaluation": asdict(evaluation),
    }
    manifest_path = output / "manifest.json"
    temporary_path = output / "manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    if args.lean_outputs:
        # Evaluated and hashed above; best.checkpoint is the model to keep.
        checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(manifest["evaluation"], indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
