#!/usr/bin/env python3
"""Extended 2D->3D evaluation beyond masked PSNR.

Masked PSNR on the target mask ignores everything a method paints outside the
target silhouette and rewards filling holes inside it. Per target view this
reports, for renders on a white background (and, when given, the same renders
on black, from which alpha = 1 - (white - black)):

  psnr_masked     PSNR inside the target mask
  psnr_full       whole image on white (targets are white outside the mask)
  psnr_covis      target mask AND pixels co-visible with the source view
  psnr_disocc     target mask minus co-visible pixels (not seen by the source)
  ssim_full, lpips_full   whole image
  iou / precision / recall   render silhouette (alpha > 0.5) vs target mask;
                  FACET 2D models are semi-transparent (mean alpha 0.7-0.8 even
                  in 2D), so recall at 0.5 understates their coverage
  hole_fraction   target-mask pixels with alpha < 0.1 (holes, transparency-robust)
  outside_coverage   alpha mass outside the target mask / mask area (bloat)
  alpha_covis     mean alpha on co-visible target pixels
  luma_ratio_covis   mean render luma / mean target luma on co-visible pixels

Co-visibility is geometric and method independent: source pixels inside the
source mask with valid GT depth are projected into the target camera with a
z-buffer (pixel centres, protocol intrinsics), then closed with a 3x3 kernel.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def psnr(a, b, m=None) -> float:
    d = (a - b) ** 2
    mse = d[m].mean() if m is not None else d.mean()
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def luma(im):
    return im @ np.array([0.2126, 0.7152, 0.0722])


def covisible(depth, src_mask, w, h, f_src, c2w_gl, f_tgt) -> np.ndarray:
    hs, ws = depth.shape
    vv, uu = np.nonzero(src_mask & np.isfinite(depth) & (depth > 0))
    z = depth[vv, uu]
    p = np.stack([(uu + 0.5 - ws / 2) / f_src * z, (vv + 0.5 - hs / 2) / f_src * z, z, np.ones_like(z)])
    ref = np.eye(4); ref[2, 3] = 1.0
    rel = CV2GL @ np.linalg.inv(ref) @ c2w_gl @ CV2GL  # target c2w in the source OpenCV frame
    q = np.linalg.inv(rel) @ p
    front = q[2] > 1e-6
    u = np.floor(f_tgt * q[0, front] / q[2, front] + w / 2).astype(int)
    v = np.floor(f_tgt * q[1, front] / q[2, front] + h / 2).astype(int)
    zz = q[2, front]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, v[inb] * w + u[inb], zz[inb])
    hit = np.isfinite(zbuf).reshape(h, w).astype(np.uint8)
    return cv2.morphologyEx(hit, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True, help="eval dir with eval_path.json, targets, masks")
    p.add_argument("--gt-depth", type=Path, required=True)
    p.add_argument("--source-mask", type=Path, required=True)
    p.add_argument("--white", type=Path, required=True, help="renders on white, NNNNN.png")
    p.add_argument("--black", type=Path, help="the same renders on black")
    p.add_argument("--mask-threshold", type=int, required=True)
    p.add_argument("--exclude-index", type=int, action="append", default=[])
    p.add_argument("--source-fov-x", type=float, required=True, help="protocol source camera, radians")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import lpips
    import torch
    from skimage.metrics import structural_similarity
    net = lpips.LPIPS(net="vgg", verbose=False)

    meta = json.loads((args.scene / "eval_path.json").read_text())
    w, h = int(meta["render_width"]), int(meta["render_height"])
    depth = np.load(args.gt_depth)["depth"].astype(np.float64)
    src_mask = cv2.imread(str(args.source_mask), cv2.IMREAD_GRAYSCALE) > 127
    f_src = 0.5 * depth.shape[1] / math.tan(0.5 * args.source_fov_x)
    frames = []
    for idx, fr in enumerate(meta["camera_path"]):
        if idx in args.exclude_index:
            continue
        tgt = rgb(args.scene / fr["file_path"])
        mask = cv2.imread(str(args.scene / fr["file_path"].replace("rgb", "mask")), cv2.IMREAD_GRAYSCALE) > args.mask_threshold
        f_tgt = 0.5 * h / math.tan(0.5 * math.radians(fr["fov"]))
        cov = covisible(depth, src_mask, w, h, f_src, np.asarray(fr["camera_to_world"]).reshape(4, 4), f_tgt)
        a_cov, a_dis = mask & cov, mask & ~cov
        white = rgb(args.white / f"{idx:05d}.png")
        with torch.no_grad():
            t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float() * 2 - 1  # noqa: E731
            lp = float(net(t(white), t(tgt)).item())
        rec = {"index": idx, "target": fr["file_path"],
               "psnr_masked": psnr(white, tgt, mask), "psnr_full": psnr(white, tgt),
               "psnr_covis": psnr(white, tgt, a_cov),
               "psnr_disocc": psnr(white, tgt, a_dis) if a_dis.any() else None,
               "disocc_fraction": float(a_dis.sum() / max(mask.sum(), 1)),
               "ssim_full": float(structural_similarity(white, tgt, data_range=1.0, channel_axis=2)),
               "lpips_full": lp,
               "luma_ratio_covis": float(luma(white)[a_cov].mean() / luma(tgt)[a_cov].mean())}
        if args.black is not None:
            black = rgb(args.black / f"{idx:05d}.png")
            alpha = np.clip(1.0 - (white - black).mean(2), 0.0, 1.0)
            sil = alpha > 0.5
            inter = (sil & mask).sum()
            rec.update(iou=float(inter / max((sil | mask).sum(), 1)),
                       precision=float(inter / max(sil.sum(), 1)),
                       recall=float(inter / max(mask.sum(), 1)),
                       alpha_covis=float(alpha[a_cov].mean()),
                       hole_fraction=float((mask & (alpha < 0.1)).sum() / max(mask.sum(), 1)),
                       outside_coverage=float(alpha[~mask].sum() / max(mask.sum(), 1)))
        frames.append(rec)
    keys = [k for k in frames[0] if k not in ("index", "target")]
    mean = {k: float(np.mean([f[k] for f in frames if f[k] is not None])) for k in keys}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mean": mean, "frames": frames}, indent=2) + "\n")
    print(json.dumps({k: round(v, 4) for k, v in mean.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
