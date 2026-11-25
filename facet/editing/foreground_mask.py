#!/usr/bin/env python3
"""Create a binary foreground mask by comparing RGB with the image border."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def colour_key_mask(image: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("input must have at least three colour channels")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    rgb = pixels[..., :3].astype(np.float32)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    difference = np.max(np.abs(rgb - background), axis=2)
    return difference >= threshold, background


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or mask_array.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    if radius == 0:
        return mask_array.copy()
    padded = np.pad(mask_array, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask_array)
    height, width = mask_array.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def alpha_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("alpha mask input must have exactly four channels")
    if not 0 <= threshold <= 255:
        raise ValueError("alpha threshold must be in [0, 255]")
    return pixels[..., 3] >= threshold


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=18.0)
    parser.add_argument("--dilation_radius", type=int, default=0)
    parser.add_argument(
        "--use_alpha",
        action="store_true",
        help="Use alpha instead of estimating foreground from the border colour",
    )
    args = parser.parse_args()

    loaded = Image.open(args.input)
    if args.use_alpha:
        image = np.asarray(loaded.convert("RGBA"))
        mask = alpha_mask(image, int(args.threshold))
        background = None
    else:
        image = np.asarray(loaded.convert("RGB"))
        mask, background = colour_key_mask(image, args.threshold)
    mask = dilate_mask(mask, args.dilation_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(args.output)
    print(
        f"Wrote {args.output.resolve()}: selected={int(mask.sum())}/{mask.size}, "
        f"background_rgb={background.tolist() if background is not None else None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
