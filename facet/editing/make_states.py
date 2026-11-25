"""Write a FACET edit sequence for a geometric warp (facet/editing/warps.py).

The primitives whose centres fall inside the object mask (optionally dilated
by --mask_dilation pixels) are warped, the others keep their place; with
--selection all every primitive goes through the warp, which is local by
construction. Primitives on the object's edge can have their centre just
outside the mask while their ellipse still covers the edge, so the default
mask selection leaves a faint outline behind. Frame k uses t = smoothstep(k / (frames - 1)), so the
last frame is the full edit. Next to each NNNNNN.npz state goes NNNNNN_F.npz
with the warp's Jacobian, which facet/editing/render_sequence.py
--transport_deformation uses to rotate and stretch the ellipses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from facet.editing.warps import warp_with_jacobian  # noqa: E402
from facet.editing.state_format import PlanarEditArrays, load_planar_edit_arrays, save_planar_edit_arrays  # noqa: E402
from facet.physics.mpm import selection_from_image_mask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True, help="facet/editing/export_state.py output")
    parser.add_argument("--mask", type=Path, required=True, help="object mask at the image resolution")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--selection", choices=("mask", "all"), default="mask")
    parser.add_argument("--mask_dilation", type=int, default=0, help="pixels")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    mask = np.asarray(Image.open(args.mask).convert("L")) > 127
    if args.mask_dilation > 0:
        from scipy import ndimage

        mask = ndimage.binary_dilation(mask, iterations=args.mask_dilation)
    height, width = mask.shape
    aspect = height / width
    source = load_planar_edit_arrays(args.identity)
    plane = source.mean_xy.astype(np.float64)
    selected = selection_from_image_mask(plane, mask, (-1.0, 1.0, -aspect, aspect)) & source.active
    if args.selection == "all":
        selected = source.active.copy()
    ids = np.flatnonzero(selected)
    pixels = np.stack(((plane[:, 0] + 1.0) * width / 2.0, (plane[:, 1] + aspect) * width / 2.0), axis=1)
    points = torch.as_tensor(pixels[ids], dtype=torch.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(args.frames):
        x = frame / max(1, args.frames - 1)
        t = x * x * (3.0 - 2.0 * x)
        warped, jacobian = warp_with_jacobian(points, spec["warp"], t)
        moved = pixels.copy()
        moved[ids] = warped.numpy()
        mean_xy = np.stack((moved[:, 0] * 2.0 / width - 1.0, moved[:, 1] * 2.0 / width - aspect), axis=1)
        state = PlanarEditArrays(
            primitive_ids=source.primitive_ids,
            mean_xy=mean_xy.astype(source.mean_xy.dtype),
            active=source.active,
            metadata={**source.metadata, "geometric_edit": spec["warp"], "t": t, "selected": int(ids.size)},
        )
        save_planar_edit_arrays(args.output_dir / f"{frame:06d}.npz", state)
        # The pixel-to-plane map is a uniform scale, so the Jacobian carries over.
        np.savez_compressed(
            args.output_dir / f"{frame:06d}_F.npz",
            primitive_ids=source.primitive_ids[ids],
            deformation=jacobian.numpy().astype(np.float32),
        )
    determinant = np.linalg.det(jacobian.numpy())
    summary = {
        "spec": spec, "frames": args.frames, "selection": args.selection,
        "mask_dilation": args.mask_dilation, "selected_primitives": int(ids.size),
        "primitives": int(source.primitive_count), "jacobian_determinant_min": float(determinant.min()),
        "jacobian_determinant_max": float(determinant.max()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "spec"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
