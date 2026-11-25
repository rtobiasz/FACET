#!/usr/bin/env python3
"""Novel-view inputs for lifting from a NeRF-Synthetic scene (GaINeR's Hotdog protocol for any
scene): the reference view --ref-frame, the target views --targets, the target cameras, and the
GT depth of the reference view rendered from the mesh (zero off the mesh; --nn-fill fills it
from the nearest valid pixel). Also copies the reference RGBA to ref_image.png. The Hotdog
setting of the paper is --ref-frame r_14 --targets r_32 ... r_39. Needs open3d, trimesh and
opencv.
"""
import json
import cv2
import numpy as np
import open3d as o3d
import trimesh
import argparse
import scipy.ndimage as ndimage
from pathlib import Path

CAMERA_ANGLE_X = 0.6911112070083618

def get_c2w_from_nerf(transform_matrix):
    return np.array(transform_matrix, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nerf-dir", type=str, default="data/nerf_synthetic/hotdog")
    parser.add_argument("--train-json", type=str, default="transforms_train.json")
    parser.add_argument("--test-json", type=str, default="transforms_test.json")
    parser.add_argument("--ref-frame", type=str, default="r_14")
    parser.add_argument("--out-dir", type=str, default="data/hotdog_eval")
    parser.add_argument("--targets", nargs="+", default=["r_36", "r_35", "r_37", "r_34", "r_38", "r_33", "r_39", "r_32"])
    parser.add_argument("--nn-fill", action="store_true", help="fill the GT depth off the mesh from the nearest valid pixel (not used for the reported results)")
    args = parser.parse_args()

    nerf_dir = Path(args.nerf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_mesh = trimesh.load(str(nerf_dir / "mesh.obj"), force='mesh')
    pts_3d = np.asarray(t_mesh.vertices, dtype=np.float32)

    blender_to_nerf = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float32)
    pts_3d = pts_3d @ blender_to_nerf.T

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts_3d)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(t_mesh.faces))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    with open(nerf_dir / args.train_json, 'r') as f:
        train_meta = json.load(f)

    ref_c2w = None
    for frame in train_meta['frames']:
        if args.ref_frame in frame['file_path']:
            ref_c2w = get_c2w_from_nerf(frame['transform_matrix'])
            break

    nerf_cam_dist = np.linalg.norm(ref_c2w[:3, 3])
    mirage_scale = 1.0 / nerf_cam_dist

    w, h = 800, 800
    fx = 0.5 * w / np.tan(0.5 * CAMERA_ANGLE_X)

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dir_x = (u - w / 2) / fx
    dir_y = (v - h / 2) / fx
    dir_z = np.ones_like(dir_x)
    dirs = np.stack([dir_x, dir_y, dir_z], axis=-1).reshape(-1, 3)

    c2w_cv = ref_c2w.copy()
    c2w_cv[:, 1:3] *= -1

    dirs_world = (c2w_cv[:3, :3] @ dirs.T).T
    origins_world = np.broadcast_to(c2w_cv[:3, 3], dirs_world.shape)

    rays = np.concatenate([origins_world, dirs_world], axis=-1).astype(np.float32)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays_tensor)
    t_hit = ans['t_hit'].numpy()
    valid = np.isfinite(t_hit)

    z_depth = t_hit * dirs[:, 2]

    depth_mirage = np.zeros(h * w, dtype=np.float32)
    depth_mirage[valid] = z_depth[valid] * mirage_scale
    depth_mirage = depth_mirage.reshape((h, w))

    # Optional nearest-neighbour extrapolation (--nn-fill): every background pixel takes the
    # depth of the closest surface pixel, so background components stick to the object's edge.
    mask = depth_mirage > 0
    if args.nn_fill and mask.any():
        _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
        depth_final = depth_mirage[indices[0], indices[1]]
    else:
        depth_final = depth_mirage

    np.savez(str(out_dir / "ref_depth_gt.npz"), depth=depth_final)
    print(f"✅ Z-Depth GT wygenerowane! Min(hit): {depth_final.min():.4f}, Max: {depth_final.max():.4f}")

    selected_frames = args.targets
    import shutil
    shutil.copy(nerf_dir / "train" / f"{args.ref_frame}.png", out_dir / "ref_image.png")
    with open(nerf_dir / args.test_json, 'r') as f:
        test_meta = json.load(f)

    mirage_ref_c2w = np.eye(4, dtype=np.float32)
    mirage_ref_c2w[2, 3] = 1.0
    ref_c2w_inv = np.linalg.inv(ref_c2w)
    frames = []

    for frame in test_meta['frames']:
        frame_name = frame['file_path'].split('/')[-1]
        if frame_name not in selected_frames:
            continue

        test_c2w = get_c2w_from_nerf(frame['transform_matrix'])
        T_rel = ref_c2w_inv @ test_c2w
        T_rel[:3, 3] *= mirage_scale
        mirage_test_c2w = mirage_ref_c2w @ T_rel

        target_rgb_name = f"target_rgb_{frame_name}.png"
        target_mask_name = f"target_mask_{frame_name}.png"

        img_rel_path = frame['file_path'][2:] if frame['file_path'].startswith("./") else frame['file_path']
        img_path = nerf_dir / (img_rel_path + ".png")

        img_rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_rgba is not None and img_rgba.shape[2] == 4:
            alpha = img_rgba[:, :, 3] / 255.0
            img_rgb = img_rgba[:, :, :3]
            bg = np.ones_like(img_rgb) * 255
            img_white_bg = (img_rgb * alpha[:, :, None] + bg * (1 - alpha[:, :, None])).astype(np.uint8)
            mask_out = (alpha * 255).astype(np.uint8)
        else:
            img_white_bg = img_rgba
            mask_out = np.ones((h, w), dtype=np.uint8) * 255

        cv2.imwrite(str(out_dir / target_rgb_name), img_white_bg)
        cv2.imwrite(str(out_dir / target_mask_name), mask_out)

        fov_y = float(np.rad2deg(2.0 * np.arctan((0.5 * h) / fx)))
        frames.append({
            "camera_to_world": mirage_test_c2w.flatten().tolist(),
            "fov": fov_y, "aspect": w / h, "file_path": target_rgb_name
        })

    with open(out_dir / "eval_path.json", "w") as f:
        json.dump({"camera_type": "perspective", "render_height": h, "render_width": w, "camera_path": frames,
                   "fps": 1, "seconds": len(frames), "is_cycle": False, "smoothness_value": 0, "default_fov": fov_y}, f,
                  indent=4)
    print(f"✅ Zapisano eval_path.json z {len(frames)} kamerami.")

if __name__ == "__main__":
    main()