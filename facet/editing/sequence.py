"""Utilities shared by planar edit-sequence command-line tools."""
from __future__ import annotations

from pathlib import Path
import math
import re
import shutil
import subprocess

import torch


def natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """Sort numbered frame names numerically while preserving text ordering."""
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_edit_state_paths(directory: Path, pattern: str) -> list[Path]:
    """Return naturally sorted sidecars and fail clearly for an empty sequence."""
    paths = sorted(directory.glob(pattern), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"No planar edit sidecars matching {pattern!r} in {directory}"
        )
    return paths


def animation_strength(index: int, frame_count: int, *, cycle: bool) -> float:
    """Return a smooth normalized edit strength for one animation frame."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    if not 0 <= index < frame_count:
        raise IndexError("animation frame index is out of range")
    if index == 0:
        return 0.0
    if index == frame_count - 1:
        return 0.0 if cycle else 1.0
    phase = index / (frame_count - 1)
    if cycle:
        return math.sin(math.pi * phase) ** 2
    return phase * phase * (3.0 - 2.0 * phase)


def progressive_cut_selection(
    mean_xy: torch.Tensor,
    selection: torch.Tensor,
    strength: float,
    *,
    axis: str,
) -> torch.Tensor:
    """Reveal a cut monotonically along one axis within a selected region."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if selection.shape != (mean_xy.shape[0],) or selection.dtype != torch.bool:
        raise ValueError("selection must be a boolean vector with one entry per primitive")
    result = torch.zeros_like(selection)
    if strength == 0.0 or not bool(selection.any()):
        return result
    if strength == 1.0:
        return selection.detach().clone()
    coordinate = mean_xy[:, 0 if axis == "x" else 1]
    selected_coordinate = coordinate[selection]
    lower = selected_coordinate.min()
    upper = selected_coordinate.max()
    threshold = lower + strength * (upper - lower)
    return selection & (coordinate <= threshold)


def encode_video(frames: Path, output: Path, fps: int) -> None:
    """Encode numbered PNG frames into a broadly compatible H.264 video."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required when --video is provided")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
