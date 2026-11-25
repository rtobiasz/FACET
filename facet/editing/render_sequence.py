#!/usr/bin/env python3
"""Render a sidecar or procedural FACET 2D edit sequence."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import math
import time

import numpy as np
from PIL import Image
import torch

from facet.training.config import Image2DConfig
from facet.io.checkpoint import load_training_checkpoint
from facet.editing.transport import (
    edit_planar_state,
    initial_neighbours,
    neighbourhood_deformation_gradient,
    transport_by_deformation_gradient,
)
from facet.editing.state_io import file_sha256, load_planar_edit_state, ordered_edit_tensors
from facet.editing.sequence import (
    animation_strength,
    collect_edit_state_paths,
    encode_video,
    progressive_cut_selection,
)
from facet.renderer.planar import render_planar, render_planar_resolution
from facet.renderer.facade import FACETRenderer
from facet.data.image_dataset import Image2DDataset
from facet.model.components import PlanarGaussianModel


def _frame(angle, logs) -> torch.Tensor:
    """R(angle) diag(exp(logs)) in float64, the map from in-primitive uv to the plane."""
    work = torch.float64
    c, s = torch.cos(angle.to(work)), torch.sin(angle.to(work))
    rotation = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2)
    return rotation @ torch.diag_embed(torch.exp(logs.to(work)))


def frame_residual_degrees(theta, log_scales, deformation, new_theta, new_log_scales, query_map=None) -> torch.Tensor:
    """|angle| of the rotation A'^-1 A between the exact local map A = F R(theta) S and the
    representable R(theta') S' (both map the unit disc to the same ellipse). With an
    optional query_map M the decoder reads c = M A'^-1 p, so the map from its
    coordinates to the plane is A' M^-1 instead of A'."""
    work = torch.float64
    representable = _frame(new_theta, new_log_scales)
    if query_map is not None:
        representable = representable @ torch.linalg.inv(query_map.to(work).reshape(-1, 2, 2))
    residual = torch.linalg.solve(representable, deformation.to(work) @ _frame(theta, log_scales))
    return torch.rad2deg(torch.atan2(residual[:, 1, 0] - residual[:, 0, 1], residual[:, 0, 0] + residual[:, 1, 1])).abs()


def save_rgb(image: torch.Tensor, path: Path) -> None:
    pixels = image.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source_path", type=Path)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--edit_state_dir", type=Path)
    source_group.add_argument(
        "--edit", choices=("bend", "stretch", "cut", "displace")
    )
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--amount", type=float, default=0.15)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument(
        "--selection",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument("--frame_count", type=int, default=120)
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--resolution_scale", type=int, default=1)
    parser.add_argument(
        "--background_rgb", type=float, nargs=3, metavar=("R", "G", "B")
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--transport_deformation",
        action="store_true",
        help=(
            "Turn and stretch each simulated ellipse by the deformation gradient in "
            "the NNNNNN_F.npz written next to each state by facet/physics/simulate_mpm.py "
            "--save_deformation_gradient; states without one keep their shapes"
        ),
    )
    parser.add_argument(
        "--transport_components",
        choices=("full", "rotation", "scale"),
        default="full",
        help=(
            "with --transport_deformation, apply only the transported angle (rotation: "
            "the frame turns, the footprint keeps its scales) or only the transported "
            "scales (scale: the footprint stretches, the frame keeps its angle); a control"
        ),
    )
    parser.add_argument(
        "--frame_residual_report",
        action="store_true",
        help=(
            "with --transport_deformation, report per frame the angle by which the "
            "representable frame R(theta') diag(s') differs from the exact local map "
            "F R(theta) diag(s): A'^-1 A is a rotation (same ellipse), the shear the SVD drops"
        ),
    )
    parser.add_argument(
        "--render_timing",
        action="store_true",
        help="report per frame the median CUDA-synchronised render time of 5 repeats",
    )
    parser.add_argument(
        "--deformation_source",
        choices=("solver", "neighbours"),
        default="solver",
        help=(
            "solver: the solver's F from the _F.npz file. neighbours: the total "
            "deformation fitted from each primitive's initial neighbours, which "
            "keeps plastic flow that the solver's elastic F drops; the _F.npz "
            "file then only names the simulated primitives"
        ),
    )
    parser.add_argument("--neighbour_count", type=int, default=8)
    parser.add_argument("--maximum_stretch", type=float, default=4.0)
    parser.add_argument(
        "--simulated_scale",
        type=float,
        default=1.0,
        help=(
            "multiply the scales of simulated primitives (those named in the "
            "_F.npz files) by this factor, reached linearly over "
            "--simulated_scale_ramp frames: mixed fluid otherwise shows its "
            "semi-transparent primitives as dots"
        ),
    )
    parser.add_argument("--simulated_scale_ramp", type=int, default=30)
    parser.add_argument(
        "--neighbour_cutoff",
        type=float,
        help=(
            "neighbours: a neighbour counts only while its distance is at most this "
            "times its initial distance, so torn-off spray keeps its droplets"
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.fps <= 0:
        raise ValueError("fps must be greater than zero")
    if args.resolution_scale <= 0:
        raise ValueError("resolution_scale must be greater than zero")
    background_rgb = None
    if args.background_rgb is not None:
        if any(value < 0.0 or value > 1.0 for value in args.background_rgb):
            raise ValueError("background_rgb values must be in [0, 1]")
        background_rgb = tuple(args.background_rgb)
    if (
        args.edit is not None
        and args.edit in ("cut", "displace")
        and args.selection is None
    ):
        raise ValueError(f"{args.edit} requires --selection")
    edit_paths = (
        collect_edit_state_paths(args.edit_state_dir, args.pattern)
        if args.edit_state_dir is not None
        else []
    )
    # Deformation-gradient files share the directory but are not edit states.
    edit_paths = [path for path in edit_paths if not path.name.endswith("_F.npz")]
    if args.transport_deformation and args.edit_state_dir is None:
        raise ValueError("--transport_deformation needs --edit_state_dir")
    device = torch.device("cuda:0")
    state = load_training_checkpoint(args.checkpoint, device, require_rgb=True)
    if state.config is None or state.config.get("mode") != "image_2d":
        raise ValueError("checkpoint is not an image_2d checkpoint")
    image_values = dict(state.config["image"])
    if args.source_path is not None:
        image_values["source_path"] = str(args.source_path.resolve())
    config = Image2DConfig(**image_values)
    dataset = Image2DDataset(config, device=device)
    conditioning_dimensions = state.config.get("ablation", {}).get(
        "active_conditioning_dimensions", 96
    )
    model = PlanarGaussianModel(
        state.model,
        plane_z=config.plane_z,
        active_conditioning_dimensions=conditioning_dimensions,
    )
    renderer = FACETRenderer(8, 11.3449, dataset.ray_count)
    checkpoint_hash = file_sha256(args.checkpoint)
    original_means = model.mean_xy.detach().clone()
    original_opacity = model.A.detach().clone()
    original_theta = model.theta.detach().clone()
    original_log_scales = model.s.detach().clone()
    frames = args.output_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    frame_reports = []
    neighbour_cache = None

    if args.edit_state_dir is not None:
        jobs = [(path, None) for path in edit_paths]
    else:
        if args.frame_count < 2:
            raise ValueError("frame_count must be at least two")
        jobs = [
            (None, animation_strength(index, args.frame_count, cycle=args.cycle))
            for index in range(args.frame_count)
        ]

    with torch.no_grad():
        for index, (edit_path, strength) in enumerate(jobs):
            if edit_path is not None:
                edit_state = load_planar_edit_state(edit_path, device=device)
                expected_hash = edit_state.metadata.get("source_checkpoint_sha256")
                if expected_hash is not None and expected_hash != checkpoint_hash:
                    raise ValueError(
                        "planar edit sidecar belongs to a different checkpoint: "
                        f"{edit_path}"
                    )
                edited_means, activity = ordered_edit_tensors(
                    edit_state, model.number_of_Gaussians
                )
                edited_means = edited_means.to(model.mean_xy.dtype)
                metadata = edit_state.metadata
            else:
                selection = None
                if args.selection is not None:
                    xmin, xmax, ymin, ymax = args.selection
                    selection = (
                        (original_means[:, 0] >= xmin)
                        & (original_means[:, 0] <= xmax)
                        & (original_means[:, 1] >= ymin)
                        & (original_means[:, 1] <= ymax)
                    )
                effective_selection = selection
                if args.edit == "cut":
                    effective_selection = progressive_cut_selection(
                        original_means,
                        selection,
                        strength,
                        axis=args.axis,
                    )
                effective_amount = args.amount * strength
                if args.edit == "stretch":
                    effective_amount = 1.0 + (args.amount - 1.0) * strength
                edited_means, activity = edit_planar_state(
                    original_means,
                    args.edit,
                    effective_amount,
                    axis=args.axis,
                    selection=effective_selection,
                )
                metadata = {
                    "edit": args.edit,
                    "strength": strength,
                    "amount": effective_amount,
                    "axis": args.axis,
                    "selection": args.selection,
                }
            model.mean_xy.copy_(edited_means)
            model.theta.copy_(original_theta)
            model.s.copy_(original_log_scales)
            transported = 0
            residual_degrees = None
            if args.transport_deformation and edit_path is not None:
                gradient_path = edit_path.with_name(f"{edit_path.stem}_F.npz")
                if gradient_path.exists():
                    with np.load(gradient_path, allow_pickle=False) as payload:
                        ids = torch.as_tensor(payload["primitive_ids"], device=device)
                        deformation = torch.as_tensor(payload["deformation"], device=device)
                    if args.deformation_source == "neighbours":
                        if neighbour_cache is None or not torch.equal(neighbour_cache[0], ids):
                            neighbour_cache = (
                                ids,
                                initial_neighbours(original_means[ids], args.neighbour_count),
                            )
                        deformation = neighbourhood_deformation_gradient(
                            original_means[ids],
                            edited_means[ids],
                            neighbour_cache[1],
                            maximum_stretch=args.maximum_stretch,
                            neighbour_cutoff=args.neighbour_cutoff,
                        )
                    new_theta, new_log_scales = transport_by_deformation_gradient(
                        original_theta[ids], original_log_scales[ids], deformation
                    )
                    if args.transport_components != "scale":
                        model.theta[ids] = new_theta
                    if args.transport_components != "rotation":
                        model.s[ids] = new_log_scales
                    transported = int(ids.numel())
                    query_map = None
                    if args.frame_residual_report:
                        residual_degrees = frame_residual_degrees(
                            original_theta[ids], original_log_scales[ids], deformation, new_theta, new_log_scales,
                            query_map,
                        )
            if args.simulated_scale != 1.0 and edit_path is not None:
                gradient_path = edit_path.with_name(f"{edit_path.stem}_F.npz")
                if gradient_path.exists():
                    with np.load(gradient_path, allow_pickle=False) as payload:
                        ids = torch.as_tensor(payload["primitive_ids"], device=device)
                    ramp = min(1.0, index / max(1, args.simulated_scale_ramp))
                    model.s[ids] += ramp * math.log(args.simulated_scale)
            model.A.copy_(original_opacity)
            model.A[~activity] = -torch.inf
            # The decoder is queried with each component's transported frame.
            affine = {}

            def render():
                if args.resolution_scale == 1:
                    return render_planar(
                        renderer, model, dataset, background_rgb=background_rgb, **affine
                    )
                return render_planar_resolution(
                    renderer,
                    model,
                    dataset,
                    dataset.width * args.resolution_scale,
                    dataset.height * args.resolution_scale,
                    background_rgb=background_rgb,
                    **affine,
                )

            rendered = render()
            render_seconds = None
            if args.render_timing:
                times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    render()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - started)
                render_seconds = sorted(times)[2]
            frame_path = frames / f"{index:06d}.png"
            save_rgb(rendered, frame_path)
            changed = (~activity) | torch.any(edited_means != original_means, dim=1)
            frame_reports.append(
                {
                    "index": index,
                    "edit_state": str(edit_path.resolve()) if edit_path else None,
                    "frame": str(frame_path.resolve()),
                    "edited_primitives": int(changed.sum().item()),
                    "active_primitives": int(activity.sum().item()),
                    "deformation_transported_primitives": transported,
                    "metadata": metadata,
                }
            )
            if render_seconds is not None:
                frame_reports[-1]["render_seconds"] = render_seconds
            if residual_degrees is not None:
                frame_reports[-1]["frame_residual_degrees"] = {
                    "mean": float(residual_degrees.mean()),
                    "median": float(residual_degrees.median()),
                    "p95": float(torch.quantile(residual_degrees.float(), 0.95)),
                    "max": float(residual_degrees.max()),
                }

    if args.video is not None:
        encode_video(frames, args.video, args.fps)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "active_conditioning_dimensions": conditioning_dimensions,
        "primitive_count": model.number_of_Gaussians,
        "frame_count": len(frame_reports),
        "mode": "sidecar" if args.edit_state_dir is not None else "procedural",
        "edit": args.edit,
        "resolution_scale": args.resolution_scale,
        "background_rgb": background_rgb,
        "transport_deformation": args.transport_deformation,
        **({"transport_components": args.transport_components} if args.transport_components != "full" else {}),
        "deformation_source": args.deformation_source if args.transport_deformation else None,
        "neighbour_count": args.neighbour_count,
        "maximum_stretch": args.maximum_stretch,
        "neighbour_cutoff": args.neighbour_cutoff,
        "simulated_scale": args.simulated_scale,
        "simulated_scale_ramp": args.simulated_scale_ramp,
        "fps": args.fps,
        "video": str(args.video.resolve()) if args.video is not None else None,
        "frames": frame_reports,
    }
    manifest = args.output_path / "sequence_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
