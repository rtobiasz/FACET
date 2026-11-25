"""Fill invalid depth (<= 0 or non-finite) inside an object mask from the nearest valid pixel.

Outside the mask the depth is left as is, so background pixels stay invalid and their
primitives are still deactivated by facet/lifting/lift.py --invalid deactivate. Used for DTU GT
depth, whose scans leave holes on the object (DTU 110: 4.7% of the source mask).
  python facet/lifting/fill_depth.py --depth ref_depth_gt.npz --mask target_mask_25.png --threshold 10 --out filled.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--threshold", type=int, default=10, help="mask pixels > THRESHOLD are the object")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = dict(np.load(a.depth))
    depth = data["depth"].astype(np.float64)
    mask = np.asarray(Image.open(a.mask).convert("L")) > a.threshold
    assert mask.shape == depth.shape, (mask.shape, depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    fill = mask & ~valid
    out = depth.copy()
    out[fill] = depth[iy[fill], ix[fill]]
    data["depth"] = out.astype(data["depth"].dtype)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **data)
    print(f"filled {int(fill.sum())} of {int(mask.sum())} mask pixels; {int((~valid & ~mask).sum())} invalid outside")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
