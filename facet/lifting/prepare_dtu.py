#!/usr/bin/env python3
"""DTU novel-view inputs for lifting (scan 114 by default; reference 25, targets 24/26).

Resizes and crops the rectified DTU images and masks like GaINeR's DTU protocol and builds:

* depth: STL points projected with pixel centres at i+0.5 (pixel = floor(u)),
  a true minimum z-buffer, no 5x5 max-dilation (which picked the farther
  surface), holes filled only from *valid* depth (nearest valid pixel inside
  the reference mask), then extrapolated outside the mask;
* intrinsics: full K after resize/crop is saved (fx != fy for DTU); the
  eval_path vertical FOV of each target comes from that target's fy instead of
  the reference fx;
* Depth Pro: raw Z-depth aligned to the normalised GT by least-squares scale
  and shift on mask & valid-GT pixels, without clipping to a background
  plane. A z-score-aligned variant restricted to the same pixels is written
  for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
TARGET_W, TARGET_H = 768, 512


def read_cam_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float64, sep=" ").reshape(4, 4)
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float64, sep=" ").reshape(3, 3)
    return intrinsics, extrinsics


def zbuffer_depth(points: np.ndarray, k: np.ndarray, w2c: np.ndarray, w: int, h: int) -> np.ndarray:
    cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
    z = cam[:, 2]
    keep = z > 0
    cam, z = cam[keep], z[keep]
    u = np.floor(k[0, 0] * cam[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.floor(k[1, 1] * cam[:, 1] / z + k[1, 2]).astype(np.int64)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.full(h * w, np.inf)
    np.minimum.at(depth, v[inb] * w + u[inb], z[inb])
    depth = depth.reshape(h, w)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def nearest_fill(values: np.ndarray, source: np.ndarray) -> np.ndarray:
    _, idx = ndimage.distance_transform_edt(~source, return_indices=True)
    return values[idx[0], idx[1]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--cam-dir", type=Path, required=True)
    p.add_argument("--stl-path", type=Path, required=True)
    p.add_argument("--pro-raw", type=Path, required=True, help="depth-pro-run output for ref_image.png")
    p.add_argument("--ref-id", type=int, default=25)
    p.add_argument("--src-ids", type=int, nargs="+", default=[24, 26])
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    img_ref = cv2.imread(str(args.img_dir / f"{args.ref_id:06d}.png"))
    mask_ref = cv2.imread(str(args.mask_dir / f"{args.ref_id:03d}.png"), cv2.IMREAD_GRAYSCALE)
    k_ref, e_ref = read_cam_file(args.cam_dir / f"{args.ref_id:08d}_cam.txt")
    h, w = img_ref.shape[:2]
    ds = lambda a: cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)  # noqa: E731

    # Crop origin from K/2 of the reference camera.
    start_x = int(round(k_ref[0, 2] / 2.0 - TARGET_W / 2.0))
    start_y = int(round(k_ref[1, 2] / 2.0 - TARGET_H / 2.0))
    crop = lambda a: a[start_y:start_y + TARGET_H, start_x:start_x + TARGET_W]  # noqa: E731

    def crop_k(k: np.ndarray) -> np.ndarray:
        out_k = k.copy()
        out_k[:2, :] *= 0.5
        out_k[0, 2] -= start_x
        out_k[1, 2] -= start_y
        return out_k

    k_crop = crop_k(k_ref)
    img = crop(ds(img_ref)).copy()
    mask = crop(ds(mask_ref)) > 127

    points = np.asarray(o3d.io.read_point_cloud(str(args.stl_path)).points, dtype=np.float64)
    raw = zbuffer_depth(points, k_crop, e_ref, TARGET_W, TARGET_H)
    valid = raw > 0
    in_mask_valid = mask & valid
    filled = nearest_fill(raw, in_mask_valid)  # inside and outside the mask
    scale = 1.0 / float(np.median(raw[in_mask_valid]))
    depth_gt = (filled * scale).astype(np.float32)
    np.savez(out / "ref_depth_gt.npz", depth=depth_gt, valid=in_mask_valid)

    img[~mask] = 255
    cv2.imwrite(str(out / "ref_image.png"), img)

    pro = np.load(args.pro_raw)["depth"].astype(np.float64)
    pro = cv2.resize(pro, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    fit = in_mask_valid
    a_mat = np.column_stack([pro[fit], np.ones(fit.sum())])
    (s_ls, t_ls), *_ = np.linalg.lstsq(a_mat, depth_gt[fit].astype(np.float64), rcond=None)
    pro_ls = (s_ls * pro + t_ls).astype(np.float32)
    np.savez(out / "ref_depth_pro.npz", depth=pro_ls)
    g, q = depth_gt[fit], pro[fit]
    pro_z = ((pro - q.mean()) / (q.std() + 1e-8) * g.std() + g.mean()).astype(np.float32)
    np.savez(out / "ref_depth_pro_zscore.npz", depth=pro_z)

    frames = []
    mirage_ref_c2w = np.eye(4)
    mirage_ref_c2w[2, 3] = 1.0
    targets = {}
    for vid in [args.ref_id] + args.src_ids:
        t_img = cv2.imread(str(args.img_dir / f"{vid:06d}.png"))
        t_mask = cv2.imread(str(args.mask_dir / f"{vid:03d}.png"), cv2.IMREAD_GRAYSCALE)
        k_t, e_t = read_cam_file(args.cam_dir / f"{vid:08d}_cam.txt")
        t_img_c = crop(ds(t_img)).copy()
        t_mask_c = crop(ds(t_mask))
        t_img_c[t_mask_c < 127] = 255
        cv2.imwrite(str(out / f"target_rgb_{vid}.png"), t_img_c)
        cv2.imwrite(str(out / f"target_mask_{vid}.png"), t_mask_c)
        k_t_crop = crop_k(k_t)
        rel = e_ref @ np.linalg.inv(e_t)
        rel[:3, 3] *= scale
        c2w = mirage_ref_c2w @ (CV2GL @ rel @ CV2GL)
        fov_y = float(np.rad2deg(2.0 * np.arctan(0.5 * TARGET_H / k_t_crop[1, 1])))
        frames.append({"camera_to_world": c2w.flatten().tolist(), "fov": fov_y,
                       "aspect": TARGET_W / TARGET_H, "file_path": f"target_rgb_{vid}.png"})
        targets[str(vid)] = {"K": k_t_crop.tolist(), "c2w_mirage_gl": c2w.tolist(),
                             "principal_point_offset_px": [float(k_t_crop[0, 2] - TARGET_W / 2),
                                                           float(k_t_crop[1, 2] - TARGET_H / 2)]}
    (out / "eval_path.json").write_text(json.dumps(
        {"camera_type": "perspective", "render_height": TARGET_H, "render_width": TARGET_W,
         "camera_path": frames, "fps": 1, "seconds": len(frames), "is_cycle": False,
         "smoothness_value": 0, "default_fov": frames[0]["fov"]}, indent=4))

    pro_err = np.abs(pro_ls[fit] - g)
    meta = {
        "script": "facet/lifting/prepare_dtu.py",
        "conventions": "OpenCV K, pixel centres at i+0.5, Z-depth, mirage scale = 1/median(valid GT in ref mask)",
        "crop_origin_xy": [start_x, start_y], "K_ref": k_crop.tolist(), "mirage_scale": scale,
        "ref_camera_angle_x": float(2.0 * np.arctan(0.5 * TARGET_W / k_crop[0, 0])),
        "gt_depth": {"mask_pixels": int(mask.sum()), "valid_in_mask": int(in_mask_valid.sum()),
                     "holes_in_mask_filled": int((mask & ~valid).sum())},
        "depth_pro": {"raw": str(args.pro_raw), "raw_sha256": sha(args.pro_raw),
                      "ls_scale": float(s_ls), "ls_shift": float(t_ls),
                      "ls_abs_err_in_mask_median": float(np.median(pro_err)),
                      "aligned_range_in_mask": [float(pro_ls[mask].min()), float(pro_ls[mask].max())]},
        "targets": targets,
        "inputs_sha256": {"stl": sha(args.stl_path)},
    }
    (out / "corrected_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("crop_origin_xy", "mirage_scale", "ref_camera_angle_x",
                                             "gt_depth", "depth_pro")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
