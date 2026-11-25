"""Textured-mesh lifting baseline (review point 2): the source image as a texture on the
triangle mesh of its own depth map, rendered from the protocol's target cameras.

The geometry is facet/lifting/lift.py's: in the reference camera's OpenCV frame, the vertex of pixel
(i, j) is at its centre u = i + 0.5, v = j + 0.5 back-projected to D(j, i) with cx = W/2,
cy = H/2, fx = fy = W/2 / tan(camera_angle_x / 2); the target cameras come from
eval_path.json exactly as in facet_lift.target_cameras, with pixel-centred pinhole rays.
Each pixel quad gives two triangles, kept when its three depths are valid (finite, > 0)
and its normal is within --max-slant degrees of the source ray (the mesh form of
facet/lifting/lift.py --edge-max-slant). Colour and alpha (RGBA sources; 1 otherwise) are the
source pixels at the vertices, interpolated barycentrically; the first hit is composited
over the background, a miss is background. Open3D ray casting (CPU).
Writes OUTPUT/NNNNN.png (one per camera_path entry) and OUTPUT/report.json.
  python facet/lifting/mesh_baseline.py --scene S --image S/ref_image.png --depth D.npz \
    --camera-angle-x FOV --background 1 1 1 --output OUT
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from facet.lifting.lift import target_cameras  # noqa: E402


def build_mesh(depth: np.ndarray, rgba: np.ndarray, camera_angle_x: float, max_slant: float):
    h, w = depth.shape
    fx = 0.5 * w / math.tan(0.5 * camera_angle_x)
    cx, cy = w / 2.0, h / 2.0
    v, u = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    ok = np.isfinite(depth) & (depth > 0)
    d = np.where(ok, depth, 1.0)
    vertices = np.stack(((u - cx) / fx * d, (v - cy) / fx * d, d), -1).reshape(-1, 3)
    index = np.arange(h * w).reshape(h, w)
    a, b, c, e = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3), np.stack((b, c, e), -1).reshape(-1, 3)))
    valid = ok.reshape(-1)[faces].all(1)
    p0, p1, p2 = (vertices[faces[:, k]] for k in range(3))
    normal = np.cross(p1 - p0, p2 - p0)
    centroid = (p0 + p1 + p2) / 3.0
    cosine = np.abs((normal * centroid).sum(1)) / np.maximum(
        np.linalg.norm(normal, axis=1) * np.linalg.norm(centroid, axis=1), 1e-30)
    steep = cosine < math.cos(math.radians(max_slant)) if max_slant > 0 else np.zeros(len(faces), bool)
    keep = valid & ~steep
    return vertices, faces[keep], rgba.reshape(-1, 4), {"faces_valid": int(valid.sum()),
                                                        "faces_steep_dropped": int((valid & steep).sum()),
                                                        "faces_kept": int(keep.sum())}


def bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear lookup of image (H, W, C) at continuous pixel-index coordinates xy (N, 2), clamped."""
    h, w = image.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1); y = np.clip(xy[:, 1], 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2); y0 = np.minimum(np.floor(y).astype(np.int64), h - 2)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * image[y0, x0] + fx * (1 - fy) * image[y0, x0 + 1]
            + (1 - fx) * fy * image[y0 + 1, x0] + fx * fy * image[y0 + 1, x0 + 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True, help="dir with eval_path.json")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--camera-angle-x", type=float, required=True)
    p.add_argument("--max-slant", type=float, default=85.0)
    p.add_argument("--background", type=float, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--uv-texture", action="store_true",
                   help="sample the source image bilinearly at the hit's interpolated image coordinate "
                        "(a UV texture) instead of interpolating vertex colours; default off")
    p.add_argument("--ray-offset", type=float, default=0.0,
                   help="shift every target ray by this many pixels in x and y (default 0). Rays from the source "
                        "camera pass exactly through the mesh vertices (pixel centres) and Open3D's non-watertight "
                        "ray casting then misses some of them; 1e-3 removes the speckles in videos")
    a = p.parse_args()
    import open3d as o3d

    image = Image.open(a.image)
    rgba = np.asarray(image.convert("RGBA"), np.float64) / 255.0
    if image.mode not in ("RGBA", "LA"):
        rgba[..., 3] = 1.0
    with np.load(a.depth) as payload:
        depth = payload["depth"].astype(np.float64)
    assert depth.shape == rgba.shape[:2], (depth.shape, rgba.shape)
    vertices, faces, colours, stats = build_mesh(depth, rgba, a.camera_angle_x, a.max_slant)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    width, height, cams = target_cameras(json.loads((a.scene / "eval_path.json").read_text()))
    a.output.mkdir(parents=True, exist_ok=True)
    background = np.asarray(a.background, np.float64)
    cols, rows = np.meshgrid(np.arange(width) + 0.5 + a.ray_offset, np.arange(height) + 0.5 + a.ray_offset)
    sh, sw = depth.shape
    pixel_xy = np.stack(np.meshgrid(np.arange(sw, dtype=np.float64), np.arange(sh, dtype=np.float64)), -1).reshape(-1, 2)
    coverage = []
    for n, cam in enumerate(cams):
        fxt = 0.5 * width / math.tan(0.5 * cam["fov_x"])
        fyt = 0.5 * height / math.tan(0.5 * cam["fov_y"])
        local = np.stack(((cols - width / 2) / fxt, (rows - height / 2) / fyt, np.ones_like(cols)), -1).reshape(-1, 3)
        c2w = cam["c2w_cv"]
        directions = local @ c2w[:3, :3].T
        origins = np.broadcast_to(c2w[:3, 3], directions.shape)
        hit = scene.cast_rays(o3d.core.Tensor(np.concatenate((origins, directions), 1).astype(np.float32)))
        ids = hit["primitive_ids"].numpy().astype(np.int64)
        bary = hit["primitive_uvs"].numpy().astype(np.float64)
        found = ids != o3d.t.geometry.RaycastingScene.INVALID_ID
        out = np.broadcast_to(background, (len(ids), 3)).copy()
        tri = faces[ids[found]]
        w1, w2 = bary[found, 0:1], bary[found, 1:2]
        if a.uv_texture:
            sample = bilinear(rgba, (1 - w1 - w2) * pixel_xy[tri[:, 0]] + w1 * pixel_xy[tri[:, 1]] + w2 * pixel_xy[tri[:, 2]])
        else:
            sample = (1 - w1 - w2) * colours[tri[:, 0]] + w1 * colours[tri[:, 1]] + w2 * colours[tri[:, 2]]
        out[found] = sample[:, :3] * sample[:, 3:] + background * (1 - sample[:, 3:])
        Image.fromarray((np.clip(out, 0, 1).reshape(height, width, 3) * 255).round().astype(np.uint8)).save(
            a.output / f"{n:05d}.png")
        coverage.append(float(found.mean()))
    report = {"options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "mesh": {"vertices": int(len(vertices)), **stats}, "hit_fraction": coverage}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mesh"]), "hit fraction", [round(c, 4) for c in coverage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
