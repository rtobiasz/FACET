#!/usr/bin/env python3
"""Training image and Depth Pro depth for a single-image 2D->3D scene.

Resizes IMAGE (Lanczos) to --size for FACET training (OUT/NAME.png), runs
Depth Pro (Apple ml-depth-pro, fp16, CUDA) on the full-resolution image and
writes
  OUT/depth_pro_full.npz  depth (H0, W0) and focallength_px at full resolution,
  OUT/depth_pro.npz       depth bilinearly resized to --size, focallength_px
                          scaled by the width ratio,
  OUT/meta.json           source, sizes, focal length, camera_angle_x
                          (horizontal FOV in radians, what facet/lifting/lift.py
                          --camera-angle-x takes for a perspective lift),
  OUT/depth_vis.png       near = bright.
Run with a Python environment that has Depth Pro installed
(https://github.com/apple/ml-depth-pro), DEPTH_PRO_REPOSITORY pointing to its clone:
  python -m facet.lifting.depth_pro \\
      --image SRC --size W H --out DIR [--name NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_PRO_REPOSITORY = Path(os.environ.get("DEPTH_PRO_REPOSITORY", "ml-depth-pro"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--size", type=int, nargs=2, required=True, metavar=("W", "H"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", help="training image name (default: the output directory's name)")
    p.add_argument("--camera-angle-x", type=float,
                   help="fix the horizontal FOV (radians) instead of Depth Pro's estimate (an edited photo of a known scene)")
    a = p.parse_args()
    a.image, a.out = a.image.resolve(), a.out.resolve()
    import os
    import torch
    os.chdir(DEPTH_PRO_REPOSITORY)  # the model loads ./checkpoints/depth_pro.pt
    import depth_pro

    a.out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.out.name
    w, h = a.size
    image = Image.open(a.image).convert("RGB")
    w0, h0 = image.size
    image.resize((w, h), Image.LANCZOS).save(a.out / f"{name}.png")

    model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"), precision=torch.half)
    model.eval()
    rgb, _, f_px = depth_pro.load_rgb(str(a.image))
    if a.camera_angle_x is not None:
        f_px = torch.tensor(0.5 * w0 / math.tan(0.5 * a.camera_angle_x), device="cuda")
    with torch.no_grad():
        pred = model.infer(transform(rgb), f_px=f_px)
    depth = pred["depth"].float().cpu().numpy()
    focal = float(pred["focallength_px"])
    np.savez(a.out / "depth_pro_full.npz", depth=depth.astype(np.float32), focallength_px=focal)
    small = np.asarray(Image.fromarray(depth).resize((w, h), Image.BILINEAR), dtype=np.float32)
    scale = w / w0
    np.savez(a.out / "depth_pro.npz", depth=small, focallength_px=focal * scale)
    fov = 2 * math.atan(0.5 * w / (focal * scale))
    meta = {"source": str(a.image), "source_sha256": hashlib.sha256(a.image.read_bytes()).hexdigest(),
            "source_size": [w0, h0], "size": [w, h], "focallength_px": focal * scale,
            "camera_angle_x": fov, "depth_range": [float(small.min()), float(small.max())]}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    v = (small - small.min()) / max(float(small.max() - small.min()), 1e-12)
    Image.fromarray((255 * (1 - v)).astype(np.uint8)).save(a.out / "depth_vis.png")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
