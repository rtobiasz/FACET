#!/usr/bin/env python3
"""E2+E3: lift a FACET 2D checkpoint with a reference Z-depth and render new views.

E2  read-only adapter: checkpoint -> pickle-free sidecar with the real
    mean_xy, theta, log_scales_xy, opacity activity and stable primitive ids
    (row index). The checkpoint's SHA-256 is checked before and after.
E3  render: the lifted primitives are true 2D surfels in 3D (mean m, two log
    scales s, quaternion q), rendered by the unchanged FACET OptiX renderer
    with pixel-centred pinhole rays from each eval_path.json camera.

Geometry, in the reference camera's OpenCV frame (+X right, +Y down, +Z fwd):
the FACET plane x in [-1, 1], y in [-H/W, H/W] maps to continuous pixel
coordinates u = (x+1) W/2, v = H/2 + y W/2 (pixel i covers [i, i+1)), and
P = Z [(u-cx)/fx, (v-cy)/fy, 1] with cx = W/2, cy = H/2, fx = fy =
W/2 / tan(camera_angle_x/2); this is exactly GaINeR's Mirage camera at z=1
over the z=0 plane. eval_path.json poses are Mirage OpenGL camera-to-world;
the reference camera is the identity rotation at (0, 0, 1).

Options (GaINeR's protocol is the default):
  --sampling round|floor|bilinear   depth lookup; round is GaINeR's integer
                                    round(u), floor the pixel-centre fix
  --transport depth_scaled|jacobian depth_scaled: in-plane arms x Z (GaINeR's XY x D);
                                    jacobian: local backprojection Jacobian
                                    including the depth gradient, surfel frame
                                    from the polar factor, scales = arm lengths
  --invalid deactivate|plane        D <= 0 / non-finite: opacity off, or D = 1
  --id-depth-order DELTA            keep the 2D training composite order of
                                    coplanar primitives (see lift())
  --freeze-view-direction           fold the view-direction term of the
                                    appearance MLP at the training direction
                                    (0, 0, 1) into b1 and zero w1_v: identical
                                    on training rays, view-independent after
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- geometry

def edge_guard(depth, valid_map, iu, iv, d, grad_u, grad_v, tau):
    """Keep depth gradients from crossing depth discontinuities.

    At the nearest pixel, a one-sided difference above tau*D per pixel is a
    discontinuity, not a slope; lift() sets tau = tan(max slant)/fx, the
    per-pixel slope of a surface tilted by the max slant at the image centre.
    There the gradient becomes the minmod of the two one-sided differences, or
    0 if that is still above the bound.
    """
    h, w = depth.shape
    out = []
    for grad, (du, dv) in ((grad_u, (1, 0)), (grad_v, (0, 1))):
        fu, fv = np.clip(iu + du, 0, w - 1), np.clip(iv + dv, 0, h - 1)
        bu, bv = np.clip(iu - du, 0, w - 1), np.clip(iv - dv, 0, h - 1)
        fwd = np.where(valid_map[fv, fu], depth[fv, fu] - d, np.inf)
        bwd = np.where(valid_map[bv, bu], d - depth[bv, bu], np.inf)
        bound = tau * np.abs(d)
        edge = (np.abs(fwd) > bound) | (np.abs(bwd) > bound)
        mm = np.where(np.sign(fwd) == np.sign(bwd), np.sign(fwd) * np.minimum(np.abs(fwd), np.abs(bwd)), 0.0)
        mm = np.where(np.isfinite(mm) & (np.abs(mm) <= bound), mm, 0.0)
        out.append(np.where(edge, mm, grad))
    return out[0], out[1], None


def sample_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray, mode: str,
                 edge_tau: float = 0.0):
    """Return D, dD/du, dD/dv (pixel units) and a validity mask."""
    h, w = depth.shape
    valid_map = np.isfinite(depth) & (depth > 0)
    if mode in ("round", "floor"):
        snap = np.round if mode == "round" else np.floor
        iu = np.clip(snap(u).astype(np.int64), 0, w - 1)
        iv = np.clip(snap(v).astype(np.int64), 0, h - 1)
        d = depth[iv, iu].astype(np.float64)
        ok = valid_map[iv, iu]
        # Central differences where both neighbours are valid, else zero.
        def diff(axis_u: bool):
            if axis_u:
                a, b = np.clip(iu - 1, 0, w - 1), np.clip(iu + 1, 0, w - 1)
                va, vb = depth[iv, a], depth[iv, b]
                good = valid_map[iv, a] & valid_map[iv, b] & (b > a)
                return np.where(good, (vb - va) / np.maximum(b - a, 1), 0.0)
            a, b = np.clip(iv - 1, 0, h - 1), np.clip(iv + 1, 0, h - 1)
            va, vb = depth[a, iu], depth[b, iu]
            good = valid_map[a, iu] & valid_map[b, iu] & (b > a)
            return np.where(good, (vb - va) / np.maximum(b - a, 1), 0.0)
        gu, gv = diff(True), diff(False)
        if edge_tau > 0:
            gu, gv, _ = edge_guard(depth, valid_map, iu, iv, d, gu, gv, edge_tau)
        return d, gu, gv, ok
    # Bilinear on pixel centres: sample position in index space is (u-0.5, v-0.5).
    fu, fv = np.clip(u - 0.5, 0, w - 1), np.clip(v - 0.5, 0, h - 1)
    u0 = np.clip(np.floor(fu).astype(np.int64), 0, w - 2)
    v0 = np.clip(np.floor(fv).astype(np.int64), 0, h - 2)
    au, av = fu - u0, fv - v0
    q00, q10 = depth[v0, u0], depth[v0, u0 + 1]
    q01, q11 = depth[v0 + 1, u0], depth[v0 + 1, u0 + 1]
    ok = (valid_map[v0, u0] & valid_map[v0, u0 + 1]
          & valid_map[v0 + 1, u0] & valid_map[v0 + 1, u0 + 1])
    d = (1 - av) * ((1 - au) * q00 + au * q10) + av * ((1 - au) * q01 + au * q11)
    du = (1 - av) * (q10 - q00) + av * (q11 - q01)
    dv = (1 - au) * (q01 - q00) + au * (q11 - q10)
    du, dv = np.where(ok, du, 0.0), np.where(ok, dv, 0.0)
    if edge_tau > 0:
        iu = np.clip(np.floor(u).astype(np.int64), 0, w - 1)
        iv = np.clip(np.floor(v).astype(np.int64), 0, h - 1)
        du, dv, _ = edge_guard(depth, valid_map, iu, iv, d, du, dv, edge_tau)
    return d, du, dv, ok


def rotmat_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Scalar-first unit quaternions from (N, 3, 3) rotations."""
    q = np.empty((r.shape[0], 4))
    tr = np.trace(r, axis1=1, axis2=2)
    cand = np.stack([tr, r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], 1)
    k = np.argmax(cand, 1)
    for case in range(4):
        i = k == case
        m = r[i]
        if case == 0:
            s = np.sqrt(1.0 + tr[i]) * 2
            q[i] = np.stack([0.25 * s, (m[:, 2, 1] - m[:, 1, 2]) / s,
                             (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s], 1)
        elif case == 1:
            s = np.sqrt(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]) * 2
            q[i] = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, 0.25 * s,
                             (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s], 1)
        elif case == 2:
            s = np.sqrt(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]) * 2
            q[i] = np.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s,
                             0.25 * s, (m[:, 1, 2] + m[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]) * 2
            q[i] = np.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s,
                             (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def lift(mean_xy, theta, log_s, depth, camera_angle_x, sampling, transport, invalid,
         id_depth_order=0.0, edge_max_slant_deg=0.0, k=None, edge_max_jump=0.0):
    """k = (fx, fy, cx, cy) of the source image overrides the square, centred
    camera derived from camera_angle_x."""
    h, w = depth.shape
    x, y = mean_xy[:, 0].astype(np.float64), mean_xy[:, 1].astype(np.float64)
    u, v = (x + 1.0) * w / 2.0, h / 2.0 + y * w / 2.0
    if k is None:
        fx = 0.5 * w / math.tan(0.5 * camera_angle_x)
        fy, cx, cy = fx, w / 2.0, h / 2.0
    else:
        fx, fy, cx, cy = (float(value) for value in k)
    edge_tau = math.tan(math.radians(edge_max_slant_deg)) / fx if edge_max_slant_deg > 0 else 0.0
    d, d_u, d_v, ok = sample_depth(depth, u, v, sampling, edge_tau)
    if invalid == "plane":
        d = np.where(ok, d, 1.0)
        d_u, d_v = np.where(ok, d_u, 0.0), np.where(ok, d_v, 0.0)
    n_jump = 0
    if edge_max_jump > 0:
        # Monocular depth smooths an occlusion edge into a short steep ramp that
        # the slant bound cannot tell from a grazing plane; the relative depth
        # range across a 5x5 window can (a grazing table: a few %, a chair
        # against the wall: tens of %). There the surfel is not stretched and
        # snaps to the nearer of the two layers.
        from scipy import ndimage
        lo = ndimage.minimum_filter(depth, size=5)
        hi = ndimage.maximum_filter(depth, size=5)
        iu = np.clip(np.floor(u).astype(int), 0, w - 1)
        iv = np.clip(np.floor(v).astype(int), 0, h - 1)
        a_lo, a_hi = lo[iv, iu], hi[iv, iu]
        jump = (a_hi - a_lo) / np.maximum(a_lo, 1e-6) > edge_max_jump
        d = np.where(jump, np.where(d - a_lo < a_hi - d, a_lo, a_hi), d)
        d_u, d_v = np.where(jump, 0.0, d_u), np.where(jump, 0.0, d_v)
        n_jump = int(jump.sum())
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], 1)
    mean = ray * d[:, None]

    c, s = np.cos(theta.astype(np.float64)), np.sin(theta.astype(np.float64))
    scales = np.exp(log_s.astype(np.float64))
    # In-plane arms in FACET plane units: columns R(theta) diag(scales).
    arms2 = np.stack([np.stack([c * scales[:, 0], s * scales[:, 0]], 1),
                      np.stack([-s * scales[:, 1], c * scales[:, 1]], 1)], 2)  # (N, 2, 2)
    half_w = w / 2.0
    if transport in ("depth_scaled", "legacy"):
        jac = np.zeros((len(x), 3, 2))
        jac[:, 0, 0] = half_w / fx * d
        jac[:, 1, 1] = half_w / fy * d
    else:
        jac = np.zeros((len(x), 3, 2))
        jac[:, 0, 0] = half_w / fx * d
        jac[:, 1, 1] = half_w / fy * d
        jac[:, :, 0] += ray * (d_u * half_w)[:, None]
        jac[:, :, 1] += ray * (d_v * half_w)[:, None]
    arms3 = jac @ arms2  # (N, 3, 2)
    uu, ss, vt = np.linalg.svd(arms3, full_matrices=False)
    if transport == "jacobian_exact":
        # Exact Gaussian: principal axes U and lengths S of J A. The whitened
        # local coordinates (the appearance LUT input) then turn by V^T, so
        # order and sign the axes to keep V^T as close to identity as possible.
        swap = np.abs(vt[:, 0, 0]) < np.abs(vt[:, 0, 1])
        uu = np.where(swap[:, None, None], uu[:, :, ::-1], uu)
        ss = np.where(swap[:, None], ss[:, ::-1], ss)
        vt = np.where(swap[:, None, None], vt[:, ::-1, :], vt)
        sign = np.sign(np.stack([vt[:, 0, 0], vt[:, 1, 1]], 1))
        sign[sign == 0] = 1.0
        frame = uu * sign[:, None, :]
        new_log_s = np.log(np.maximum(ss, 1e-12))
    else:
        frame = uu @ vt  # polar factor, keeps the local axis labels
        new_log_s = np.log(np.maximum(np.linalg.norm(arms3, axis=1), 1e-12))
    normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    rot = np.concatenate([frame, normal[:, :, None]], 2)
    quat = rotmat_to_quaternion(rot)
    if id_depth_order > 0.0:
        # Coplanar primitives were composited in (t, id) order during 2D
        # training because t ties exactly on orthographic rays. Moving each
        # primitive along its own reference ray by (1 + delta id/N) leaves the
        # reference projection unchanged and turns rounding-noise ties into
        # the training id order.
        factor = 1.0 + id_depth_order * np.arange(len(x)) / len(x)
        mean = mean * factor[:, None]
        new_log_s = new_log_s + np.log(factor)[:, None]
    active = ok if invalid == "deactivate" else np.ones_like(ok)
    stats = {"primitives": int(len(x)), "invalid_depth": int((~ok).sum()),
             "fx": fx, "fy": fy, "cx": cx, "cy": cy, "edge_jump_snapped": n_jump,
             "z_range": [float(mean[:, 2].min()), float(mean[:, 2].max())]}
    return mean, quat, new_log_s, active, stats


def _git_commit() -> str | None:
    import subprocess
    try:
        here = Path(__file__).resolve().parent
        head = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", str(Path(__file__).resolve())],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def halton(index: int, base: int) -> float:
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


def pinhole_rays(c2w_cv: np.ndarray, k, width: int, height: int, torch, device):
    """Pixel-centred pinhole rays for a full K (fx, fy, cx, cy); OpenCV frame."""
    fx, fy, cx, cy = (float(value) for value in k)
    cols = torch.arange(width, device=device, dtype=torch.float64) + 0.5
    rows = torch.arange(height, device=device, dtype=torch.float64) + 0.5
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    local = torch.stack(((xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)), -1).reshape(-1, 3)
    c2w = torch.as_tensor(c2w_cv, dtype=torch.float64, device=device)
    directions = local @ c2w[:3, :3].T
    origins = c2w[:3, 3].expand_as(directions)
    return origins.float().contiguous(), directions.float().contiguous()


def lift_lattice(mean_xy, theta, log_s, depth, scale, max_stretch, invert, camera_angle_x,
                 id_depth_order=0.0, stretch_clamp=False, drop_steep=0.0, norm_from=None):
    """GaINeR's qualitative lifting (its depth-lattice deformation).

    The plane point moves only along Z by scale * (normalised depth - mean),
    read from a bicubic B-spline over the pixel grid; GaINeR shifts every
    tetrahedron vertex separately, so arms tilt by finite differences. Here
    the arms are transported by the spline derivative (the Jacobian of the
    height field), with GaINeR's max-stretch fallback to a rigid shift
    (stretch_clamp: the arms keep their slope up to max_stretch instead of
    lying flat, so a steep depth edge stays covered; drop_steep > 0: surfels
    whose depth rise across them exceeds drop_steep x their size are removed,
    so a disocclusion stays an empty hole instead of a sprinkle of thin
    surfels). norm_from: normalise with another depth's range and mean (a
    background layer lifted in its front's frame).
    Returned in the reference camera's OpenCV frame (Mirage z=0 -> depth 1).
    """
    from scipy.interpolate import RectBivariateSpline
    d = depth.astype(np.float64)
    if norm_from is None:
        if invert:
            d = d.max() - d
        h, w = d.shape
        d = (d - d.min()) / (d.max() - d.min())
        delta = (d - d.mean()) * scale
    else:
        ref = norm_from.astype(np.float64)
        if invert:
            d, ref = ref.max() - d, ref.max() - ref
        h, w = d.shape
        d, ref = (d - ref.min()) / (ref.max() - ref.min()), (ref - ref.min()) / (ref.max() - ref.min())
        delta = (d - ref.mean()) * scale
    spline = RectBivariateSpline(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), delta, kx=3, ky=3)
    t = math.tan(0.5 * camera_angle_x)
    x, y = mean_xy[:, 0].astype(np.float64), mean_xy[:, 1].astype(np.float64)
    aspect = h / w
    # GaINeR's pixel lookup: col = x_norm (W-1), row = (1 - y_norm)(H-1).
    col = np.clip((x + 1.0) / 2.0 * (w - 1), 0, w - 1)
    row = np.clip((y + aspect) / (2.0 * aspect) * (h - 1), 0, h - 1)
    dz = spline.ev(row, col)
    d_col = spline.ev(row, col, dy=1) * (w - 1) / 2.0           # d delta / d x
    d_row = spline.ev(row, col, dx=1) * (h - 1) / (2.0 * aspect)  # d delta / d y
    mean = np.stack([x * t, y * t, 1.0 - dz], 1)
    c, s = np.cos(theta.astype(np.float64)), np.sin(theta.astype(np.float64))
    sc = np.exp(log_s.astype(np.float64))
    arms2 = np.stack([np.stack([c * sc[:, 0], s * sc[:, 0]], 1),
                      np.stack([-s * sc[:, 1], c * sc[:, 1]], 1)], 2)
    jac = np.zeros((len(x), 3, 2))
    jac[:, 0, 0] = t
    jac[:, 1, 1] = t
    jac[:, 2, 0] = -d_col
    jac[:, 2, 1] = -d_row
    arms3 = jac @ arms2
    if max_stretch > 0:
        old_len = np.linalg.norm(arms2, axis=1).max(1) * t
        steep = np.abs(arms3[:, 2, :]).max(1)
        stretched = steep > max_stretch * old_len
        dropped = steep > drop_steep * old_len if drop_steep > 0 else np.zeros(len(x), bool)
        if stretch_clamp:
            arms3[stretched, 2, :] *= (max_stretch * old_len[stretched] / steep[stretched])[:, None]
        else:
            arms3[stretched, 2, :] = 0.0
    else:
        stretched = np.zeros(len(x), bool)
        dropped = np.zeros(len(x), bool)
    uu, _, vt = np.linalg.svd(arms3, full_matrices=False)
    frame = uu @ vt
    normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    quat = rotmat_to_quaternion(np.concatenate([frame, normal[:, :, None]], 2))
    new_log_s = np.log(np.maximum(np.linalg.norm(arms3, axis=1), 1e-12))
    if id_depth_order > 0.0:
        factor = 1.0 + id_depth_order * np.arange(len(x)) / len(x)
        mean = mean * factor[:, None]
        new_log_s = new_log_s + np.log(factor)[:, None]
    stats = {"primitives": int(len(x)), "rigid_fallback": int(stretched.sum()),
             "dz_range": [float(dz.min()), float(dz.max())], "scale": scale}
    if drop_steep > 0:
        stats["dropped_steep"] = int(dropped.sum())
    return mean, quat, new_log_s, ~dropped, stats


def sine_deform(mean, quat, log_s, x_range, amplitude, frequency, head, neck, phase):
    """GaINeR's snake animation (a travelling sine wave) on lifted surfels.

    Each centre moves up by A w(x) sin(-F x + phase) (Mirage y up, so the
    OpenCV y goes down by it), w = 0 on the head (the first HEAD of the body's
    x range), a smoothstep over the next NECK, 1 on the rest. GaINeR moves
    the four tetrahedron vertices of a Gaussian by its centre's offset; here
    the arms are carried by the deformation's Jacobian (y' = y - delta(x):
    dy'/dx = -delta'(x)) and re-orthonormalised by SVD, as the lift does.
    """
    x = mean[:, 0]
    lo, hi = x_range

    def delta(xx):
        t = np.clip((xx - lo) / (hi - lo + 1e-9), 0, 1)
        s = np.clip((t - head) / max(neck, 1e-9), 0, 1)
        w = s * s * (3 - 2 * s)
        return amplitude * w * np.sin(-frequency * xx + phase)

    h = 1e-4 * (hi - lo)
    dd = (delta(x + h) - delta(x - h)) / (2 * h)
    new_mean = mean.copy()
    new_mean[:, 1] -= delta(x)
    rot = quaternion_to_rotmat(quat)
    arms = rot[:, :, :2] * np.exp(log_s[:, None, :2])
    arms[:, 1, :] -= dd[:, None] * arms[:, 0, :]
    uu, _, vt = np.linalg.svd(arms, full_matrices=False)
    frame = uu @ vt
    normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    new_quat = rotmat_to_quaternion(np.concatenate([frame, normal[:, :, None]], 2))
    new_log_s = log_s.copy()
    new_log_s[:, :2] = np.log(np.maximum(np.linalg.norm(arms, axis=1), 1e-12))
    return new_mean, new_quat, new_log_s


def ripple_displacement(points, elapsed, amplitude, wavelength, speed, origin, direction, axis, front, ramp, s0=0.0,
                        along_ratio=1.0, edge_taper=None):
    """A travelling sine wave that starts at time 0 (elapsed = t - T0, seconds).

    Wave coordinate s: the distance from ORIGIN (x, y in the reference OpenCV
    frame; a circular ripple) or, with no origin, the coordinate along
    DIRECTION minus S0 (main() passes the smallest one over the surfels: a
    plane wave entering from that edge). The front leaves s = 0 at SPEED units per second;
    behind it each point moves by A sin(2 pi (s - SPEED elapsed) / WAVELENGTH),
    the leading edge feathered over FRONT wavelengths and the amplitude eased
    in over RAMP seconds (both smoothsteps, so the motion starts from rest).
    AXIS 'normal': along -z (off a flat print, towards the reference camera);
    'along': in the plane, along the travel direction (a compression wave);
    'both': the normal one plus ALONG_RATIO times the in-plane one.
    EDGE_TAPER (xmin, xmax, ymin, ymax, width): the amplitude falls to zero
    towards the print's border over WIDTH (smoothstep), so the edges stay put.
    """
    xy = points[:, :2]
    if origin is not None:
        rel = xy - np.asarray(origin, float)
        s = np.hypot(rel[:, 0], rel[:, 1])
        unit = rel / np.maximum(s, 1e-9)[:, None]
        s = s - s0  # s0: the ring the front starts from (--ripple-start-radius; 0 = the origin)
    else:
        dvec = np.asarray(direction, float) / np.linalg.norm(direction)
        s = xy @ dvec - s0
        unit = np.broadcast_to(dvec, xy.shape)
    reach = speed * elapsed
    edge = np.clip((reach - s) / max(front * wavelength, 1e-9), 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    r = np.clip(elapsed / ramp, 0, 1) if ramp > 0 else 1.0
    r = r * r * (3 - 2 * r)
    u = amplitude * r * edge * np.sin(2 * math.pi * (s - reach) / wavelength)
    if origin is not None and s0 > 0:
        # Pinned at the start ring (a prop's foot): still inside, eased in over half a wavelength.
        w = np.clip(s / (0.5 * wavelength), 0, 1)
        u = u * w * w * (3 - 2 * w)
    if edge_taper is not None:
        x0, x1, y0, y1, width = edge_taper
        d = np.minimum.reduce([xy[:, 0] - x0, x1 - xy[:, 0], xy[:, 1] - y0, y1 - xy[:, 1]])
        w = np.clip(d / width, 0, 1)
        u = u * w * w * (3 - 2 * w)
    out = np.zeros_like(points)
    if axis in ("normal", "both"):
        out[:, 2] = -u
    if axis == "along":
        out[:, :2] = u[:, None] * unit
    elif axis == "both":
        out[:, :2] = along_ratio * u[:, None] * unit
    return out


def mesh_hop_height(t, height, start, rise, hold, fall):
    """--mesh-hop: 0 until START, smoothstep up to HEIGHT over RISE, HOLD, then a free fall
    (h = HEIGHT (1 - (s / FALL)^2)) over FALL, 0 after landing."""
    s = t - start
    if s <= 0:
        return 0.0
    if s < rise:
        x = s / rise
        return height * x * x * (3 - 2 * x)
    s -= rise
    if s < hold:
        return height
    s -= hold
    return height * max(0.0, 1.0 - (s / fall) ** 2) if fall > 0 else 0.0


def ripple_deform(mean, quat, log_s, elapsed, spec, arms_mode="jacobian"):
    """Move the surfels by ripple_displacement; the arms by its Jacobian (central differences), as the lift does.

    ARMS_MODE 'keep' (diagnostic): translate only, the arms keep their rest orientation and length.
    """
    disp = lambda p: ripple_displacement(p, elapsed, *spec)  # noqa: E731
    if arms_mode == "keep":
        return mean + disp(mean), quat.copy(), log_s.copy()
    h = 1e-4
    jac = np.zeros((len(mean), 3, 3))
    for c in range(3):
        step = np.zeros(3)
        step[c] = h
        jac[:, :, c] = (disp(mean + step) - disp(mean - step)) / (2 * h)
    jac += np.eye(3)
    new_mean = mean + disp(mean)
    rot = quaternion_to_rotmat(quat)
    arms = jac @ (rot[:, :, :2] * np.exp(log_s[:, None, :2]))
    uu, _, vt = np.linalg.svd(arms, full_matrices=False)
    frame = uu @ vt
    # Keep the surfel's normal on the same side as before (frame columns keep their order).
    normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    new_quat = rotmat_to_quaternion(np.concatenate([frame, normal[:, :, None]], 2))
    new_log_s = log_s.copy()
    new_log_s[:, :2] = np.log(np.maximum(np.linalg.norm(arms, axis=1), 1e-12))
    return new_mean, new_quat, new_log_s


def cage_deform(mean, quat, log_s, cage, frame):
    """--cage: the surfels ride a deformed triangle mesh (a cloth simulation, any mesh tool).

    The cage is a regular grid over the flat print (reference frame, plane z = Z0): ORIGIN (x, y)
    of vertex (0, 0), CELL (dx, dy), FRAMES (T, ny + 1, nx + 1, 3) its deformed vertices. Each cell
    is split into two triangles; a surfel keeps its barycentric coordinates in its rest triangle,
    and its arms are carried by the interpolated vertex tangents (and their normal),
    re-orthonormalised by SVD as the ripple does. Frame index FRAME (clamped to the last).
    """
    verts = np.asarray(cage["frames"][min(frame, len(cage["frames"]) - 1)], np.float64)
    ny, nx = verts.shape[0] - 1, verts.shape[1] - 1
    (x0, y0), (dx, dy), z0 = cage["origin"], cage["cell"], float(cage["z0"])
    gx, gy = (mean[:, 0] - x0) / dx, (mean[:, 1] - y0) / dy
    i, j = np.clip(np.floor(gx).astype(int), 0, nx - 1), np.clip(np.floor(gy).astype(int), 0, ny - 1)
    fx, fy = np.clip(gx - i, 0, 1), np.clip(gy - j, 0, 1)
    pa, pb, pc, pd = verts[j, i], verts[j, i + 1], verts[j + 1, i], verts[j + 1, i + 1]
    lower = (fx + fy <= 1.0)[:, None]
    # Lower triangle (a, b, c): p = a + fx (b - a) + fy (c - a); upper (d, c, b): p = d + (1 - fx)(c - d) + (1 - fy)(b - d).
    pos = np.where(lower, pa + fx[:, None] * (pb - pa) + fy[:, None] * (pc - pa),
                   pd + (1 - fx)[:, None] * (pc - pd) + (1 - fy)[:, None] * (pb - pd))
    # Tangents: per-vertex central differences, interpolated like the position, so neighbouring
    # triangles share them (constant per-triangle maps open cracks at the vertices of a curved cage).
    tx, ty = np.gradient(verts, dx, axis=1), np.gradient(verts, dy, axis=0)

    def interp(field):
        fa, fb, fc, fd = field[j, i], field[j, i + 1], field[j + 1, i], field[j + 1, i + 1]
        return np.where(lower, fa + fx[:, None] * (fb - fa) + fy[:, None] * (fc - fa),
                        fd + (1 - fx)[:, None] * (fc - fd) + (1 - fy)[:, None] * (fb - fd))

    ex, ey = interp(tx), interp(ty)
    nrm = np.cross(ex, ey)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    new_mean = pos + (mean[:, 2] - z0)[:, None] * nrm
    jac = np.stack([ex, ey, nrm], 2)  # the rest frame is the identity (x, y, normal z)
    rot = quaternion_to_rotmat(quat)
    arms = jac @ (rot[:, :, :2] * np.exp(log_s[:, None, :2]))
    uu, _, vt = np.linalg.svd(arms, full_matrices=False)
    frame_r = uu @ vt
    normal = np.cross(frame_r[:, :, 0], frame_r[:, :, 1])
    new_quat = rotmat_to_quaternion(np.concatenate([frame_r, normal[:, :, None]], 2))
    new_log_s = log_s.copy()
    new_log_s[:, :2] = np.log(np.maximum(np.linalg.norm(arms, axis=1), 1e-12))
    return new_mean, new_quat, new_log_s


def target_cameras(eval_path: dict, target_k: dict | None = None):
    """target_k maps an eval_path file_path to its full K (fx, fy, cx, cy)."""
    ref = np.eye(4)
    ref[2, 3] = 1.0
    width, height = int(eval_path["render_width"]), int(eval_path["render_height"])
    cams = []
    for frame in eval_path["camera_path"]:
        c2w_gl = np.asarray(frame["camera_to_world"], dtype=np.float64).reshape(4, 4)
        rel_cv = CV2GL @ np.linalg.inv(ref) @ c2w_gl @ CV2GL
        fov_y = math.radians(float(frame["fov"]))
        fov_x = 2.0 * math.atan(math.tan(0.5 * fov_y) * width / height)
        cams.append({"file_path": frame.get("file_path", f"frame_{len(cams):05d}"), "c2w_cv": rel_cv,
                     "fov_x": fov_x, "fov_y": fov_y,
                     "K": (target_k or {}).get(frame.get("file_path")),
                     # Optional thin lens (--dof-samples): along the view axis.
                     "focus_distance": frame.get("focus_distance"),
                     "aperture_radius": frame.get("aperture_radius")})
    return width, height, cams


# ---------------------------------------------------------------- rendering

class LiftedModel:
    """Duck-typed stand-in for PlanarGaussianModel with 3D surfel geometry."""

    def __init__(self, planar, mean, quat, log_s, active, freeze_view, torch, opacity_gamma=1.0):
        dev, dt = planar.mean_xy.device, planar.mean_xy.dtype
        t = lambda a: torch.as_tensor(a, dtype=dt, device=dev).contiguous()  # noqa: E731
        self.m, self.q, self.s = t(mean), t(quat), t(log_s)
        self.A = planar.A.detach().clone()
        if opacity_gamma != 1.0:
            # alpha' = 1 - (1 - alpha)^gamma: each surfel as opaque as gamma
            # stacked copies of itself; the 2D image changes, the lift ghosts less.
            one_minus = torch.sigmoid(-self.A).clamp_min(1e-12) ** opacity_gamma
            self.A = torch.log1p(-one_minus.clamp(max=1 - 1e-7)) - torch.log(one_minus)
        self.A[~torch.as_tensor(active, device=dev)] = -torch.inf
        self.RGB, self.k, self.features = planar.RGB.detach(), planar.k.detach(), planar.features.detach()
        self.w1_uv, self.b1 = planar.w1_uv.detach(), planar.b1.detach().clone()
        self.w1_v = planar.w1_v.detach().clone()
        self.w2, self.b2, self.w3, self.b3 = (planar.w2.detach(), planar.b2.detach(),
                                              planar.w3.detach(), planar.b3.detach())
        self._planar = planar
        if freeze_view:
            self.b1 = self.b1 + (self.w1_v @ t(view_encoding_training_direction())).reshape(self.b1.shape)
            self.w1_v = torch.zeros_like(self.w1_v)

    def renderer_w1_conditioning(self):
        return self._planar.renderer_w1_conditioning().detach()

    def native_conditioning(self):
        return self._planar.native_conditioning().detach()

    def renderer_geometry(self):
        import torch
        return (self.m, torch.exp(self.s), self.q, torch.sigmoid(self.A),
                1.0 + torch.nn.functional.softplus(self.k))


def view_encoding_training_direction() -> np.ndarray:
    """gamma((0,0,1)) in w1_v column order: component-major, 4 octaves, (sin, cos).

    The order is an assumption checked by the identity test: on the training
    rays the folded network must reproduce the original render.
    """
    out = []
    for comp in (0.0, 0.0, 1.0):
        for octave in range(4):
            angle = math.pi * (1 << octave) * comp
            out += [math.sin(angle), math.cos(angle)]
    return np.asarray(out)


def render(renderer, model, rays_o, rays_d, width, height, background, torch):
    image = torch.zeros((rays_o.shape[0], 3), dtype=rays_o.dtype, device=rays_o.device)
    # One BVH build per model: rebuilding it on every call (three per frame
    # with a mirror) fails in SetGeometry after a few hundred builds.
    if getattr(renderer, "_facet_lift_synced", None) is not model:
        renderer.sync_geometry(model)
        renderer._facet_lift_synced = model
    rgba = torch.cat((model.RGB, model.A), 1).contiguous()
    w1 = torch.cat((model.w1_uv, model.w1_v, model.renderer_w1_conditioning()), 1).to(torch.float16)
    renderer.forward_inference(
        model.native_conditioning().to(torch.float16), model.features, w1, model.b1,
        model.w2.to(torch.float16), model.b2, model.w3.to(torch.float16), model.b3,
        rays_o.contiguous(), rays_d.contiguous(), image, *background,
        model.m, model.s, model.q, rgba, model.k, 0.01)
    return image.reshape(height, width, 3)


def geometry_pass(renderer, model, rays_o, rays_d, torch):
    """Alpha, expected hit distance along the unit ray and the composited normal (explicit opacity only)."""
    n = rays_o.shape[0]
    if getattr(renderer, "_facet_lift_synced", None) is not model:
        renderer.sync_geometry(model)
        renderer._facet_lift_synced = model
    rgb = torch.zeros((n, 3), dtype=rays_o.dtype, device=rays_o.device)
    normal = torch.zeros_like(rgb)
    alpha = torch.zeros((n,), dtype=rays_o.dtype, device=rays_o.device)
    numerator = torch.zeros_like(alpha)
    rgba = torch.cat((model.RGB, model.A), 1).contiguous()
    renderer._native_renderer.Forward_inference_base_with_geometry(
        rays_o.contiguous(), rays_d.contiguous(), rgb, normal, alpha, numerator, 0.0, 0.0, 0.0,
        model.m, model.s, model.q, rgba, model.k, 0.01)
    dist = numerator / alpha.clamp_min(1e-8) * rays_d.norm(dim=1)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return alpha, dist, normal


INSET_MARGIN = 0.01  # fraction of the image; global surfels deeper inside the inset go


def load_inset(checkpoint, crop_json, global_size, lattice_depth, args, torch, device):
    """A crop model moved into the global planar frame and lattice-lifted."""
    from facet.io.checkpoint import load_training_checkpoint
    from facet.model.components import PlanarGaussianModel
    state = load_training_checkpoint(checkpoint, device, require_rgb=True)
    cfg = state.config["image"]
    planar = PlanarGaussianModel(state.model, plane_z=cfg.get("plane_z", 0.0),
                                 active_conditioning_dimensions=state.config.get("ablation", {}).get(
                                     "active_conditioning_dimensions", 96))
    u0, v0, u1, v1 = json.loads(crop_json.read_text())["uv"]
    wg, hg = global_size
    wc, hc = state.config["data"]["width"], state.config["data"]["height"]
    ag, ac = hg / wg, hc / wc
    xy = planar.mean_xy.detach().cpu().numpy().astype(np.float64)
    # Crop planar x in [-1, 1], y in [-ac, ac] -> global; a similarity (u1 - u0).
    xg = 2 * u0 - 1 + (xy[:, 0] + 1) * (u1 - u0)
    yg = (2 * v0 - 1 + (xy[:, 1] / ac + 1) * (v1 - v0)) * ag
    log_s = planar.s.detach().cpu().numpy() + math.log(u1 - u0)
    theta = planar.theta.detach().cpu().numpy()
    mean_xy = np.stack([xg, yg], 1)
    mean, quat, new_log_s, active, _ = lift_lattice(
        mean_xy, theta, log_s, lattice_depth, args.lattice_scale, args.lattice_max_stretch,
        not args.lattice_no_invert, args.camera_angle_x)
    alive = np.isfinite(planar.A.detach().cpu().numpy()[:, 0])
    print(f"inset: {int(alive.sum())} surfels, box {u0:.4f} {v0:.4f} {u1:.4f} {v1:.4f}")
    model = LiftedModel(planar, mean, quat, new_log_s, active & alive, args.freeze_view_direction,
                        torch, args.opacity_gamma)
    lifted = {"planar": planar, "mean_xy": mean_xy, "mean": mean, "quat": quat, "log_s": new_log_s,
              "active": active & alive}
    return model, (u0, v0, u1, v1), lifted


BAND_DILATE = 2  # pixels: surfels centred just outside the band still cover its edge


def load_background_layer(checkpoint, depth_npz, band_png, args, torch, device, front_lattice=None):
    """The background layer's model, lifted like the main one (perspective, or the
    lattice in the front's normalisation), band surfels only."""
    from PIL import Image
    from scipy import ndimage
    from facet.io.checkpoint import load_training_checkpoint
    from facet.model.components import PlanarGaussianModel
    state = load_training_checkpoint(checkpoint, device, require_rgb=True)
    planar = PlanarGaussianModel(state.model, plane_z=state.config["image"].get("plane_z", 0.0),
                                 active_conditioning_dimensions=state.config.get("ablation", {}).get(
                                     "active_conditioning_dimensions", 96))
    depth = np.load(depth_npz)["depth"]
    band = ndimage.binary_dilation(np.asarray(Image.open(band_png).convert("L")) > 127, iterations=BAND_DILATE)
    mean_xy = planar.mean_xy.detach().cpu().numpy()
    h, w = depth.shape
    if front_lattice is not None:
        mean, quat, new_log_s, active, stats = lift_lattice(
            mean_xy, planar.theta.detach().cpu().numpy(), planar.s.detach().cpu().numpy(), depth,
            args.lattice_scale, args.lattice_max_stretch, not args.lattice_no_invert, args.camera_angle_x,
            args.id_depth_order, args.lattice_stretch_clamp, args.lattice_drop_steep, norm_from=front_lattice)
        # GaINeR's pixel lookup (lift_lattice).
        aspect = h / w
        iu = np.clip(np.round((mean_xy[:, 0] + 1.0) / 2.0 * (w - 1)).astype(int), 0, w - 1)
        iv = np.clip(np.round((mean_xy[:, 1] + aspect) / (2.0 * aspect) * (h - 1)).astype(int), 0, h - 1)
    else:
        mean, quat, new_log_s, active, stats = lift(
            mean_xy, planar.theta.detach().cpu().numpy(), planar.s.detach().cpu().numpy(), depth,
            args.camera_angle_x, args.sampling, args.transport, args.invalid, args.id_depth_order,
            args.edge_max_slant, args.lift_k, args.edge_max_jump)
        iu = np.clip(np.floor((mean_xy[:, 0] + 1.0) * w / 2.0).astype(int), 0, w - 1)
        iv = np.clip(np.floor(h / 2.0 + mean_xy[:, 1] * w / 2.0).astype(int), 0, h - 1)
    if args.lift_scale_inflate != 1.0:
        new_log_s = new_log_s + math.log(args.lift_scale_inflate)
    alive = np.isfinite(planar.A.detach().cpu().numpy()[:, 0]) & active & band[iv, iu]
    print(f"background layer: {int(alive.sum())} surfels in the band ({band.mean():.3f} of the image)")
    model = LiftedModel(planar, mean, quat, new_log_s, alive, args.freeze_view_direction, torch,
                        args.opacity_gamma)
    return model, depth, {"surfels": int(alive.sum()), "band_fraction": float(band.mean()),
                          "lift": stats}


def quaternion_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations from scalar-first unit quaternions."""
    w, x, y, z = (q[:, i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)


def lattice_surface(depth, scale, invert, camera_angle_x):
    """z(X, Y) of the lattice-lifted painting in the reference OpenCV frame."""
    from scipy.interpolate import RectBivariateSpline
    d = depth.astype(np.float64)
    if invert:
        d = d.max() - d
    h, w = d.shape
    d = (d - d.min()) / (d.max() - d.min())
    spline = RectBivariateSpline(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64),
                                 (d - d.mean()) * scale, kx=3, ky=3)
    t, aspect = math.tan(0.5 * camera_angle_x), h / w

    def surface(xc, yc):
        x, y = xc / t, yc / t
        inside = (np.abs(x) <= 1) & (np.abs(y) <= aspect)
        col = np.clip((x + 1.0) / 2.0 * (w - 1), 0, w - 1)
        row = np.clip((y + aspect) / (2.0 * aspect) * (h - 1), 0, h - 1)
        return 1.0 - spline.ev(row, col), inside
    return surface


MIRROR_ROOM_Z = -1.0  # a wall behind every camera path, for reflected rays that leave the painting


def mirror_backs(mirror, lifted, disc, surface, max_stretch=8.0):
    """Surfels of the painted reflection moved to where it came from.

    Each surfel inside the mirror disc (seen from the reference camera, the
    painter's view) is followed along its reference ray to the cap, reflected
    and marched through the lifted relief; the first crossing of a surface is
    where the painted reflection saw the scene from behind (the backs of the
    figures), otherwise the ray lands on a wall behind the viewer. Arms map
    by finite differences; the surfel faces the reflected ray. These surfels
    are traced for reflected rays only.
    """
    xy, mean = lifted["mean_xy"], lifted["mean"]
    aspect = float(np.ptp(xy[:, 1]) / np.ptp(xy[:, 0]))
    u, v, r = disc
    inside = (np.hypot(xy[:, 0] - (2 * u - 1), xy[:, 1] - (2 * v - 1) * aspect) < 2 * r) & lifted["active"]
    idx = np.flatnonzero(inside)
    rot = quaternion_to_rotmat(lifted["quat"][idx])
    sc = np.exp(lifted["log_s"][idx])
    arms = rot[:, :, :2] * sc[:, None, :]
    centre = mean[idx]
    frac = 0.25
    points = np.concatenate([centre, centre + frac * arms[:, :, 0], centre + frac * arms[:, :, 1]])
    hit, kind = _mirror_map(mirror, points, surface)
    n = len(idx)
    hc, h1, h2 = hit[:n], hit[n:2 * n], hit[2 * n:]
    ok = kind[:n] >= 0
    new_arms = np.stack([(h1 - hc) / frac, (h2 - hc) / frac], 2)
    # Missed rays (not ok, inactive) can leave non-finite arms, which the SVD
    # below cannot take (the dense --inset model has some).
    finite = np.isfinite(new_arms).all((1, 2))
    new_arms[~finite] = arms[~finite]
    hc = np.where(np.isfinite(hc), hc, centre)
    ok &= finite
    # An arm that crosses to another surface, or blows up, keeps the other's gain.
    gain = np.linalg.norm(new_arms, axis=1) / np.maximum(np.linalg.norm(arms, axis=1), 1e-12)
    bad = np.stack([kind[n:2 * n] != kind[:n], kind[2 * n:] != kind[:n]], 1)
    # The room behind the viewer magnifies far more than the figures: one
    # median per surface.
    med = np.ones(n)
    for surf in (0, 1):
        sel = (kind[:n] == surf)[:, None] & ~bad
        if sel.any():
            med[kind[:n] == surf] = np.median(gain[sel])
    bad |= gain > max_stretch * med[:, None]
    for k in range(2):
        other = 1 - k
        use = np.where(bad[:, other], med, gain[:, other])
        fix = bad[:, k]
        new_arms[fix, :, k] = arms[fix, :, k] * use[fix, None]
    uu, _, vt = np.linalg.svd(new_arms, full_matrices=False)
    frame = uu @ vt
    normal = np.cross(frame[:, :, 0], frame[:, :, 1])
    quat = rotmat_to_quaternion(np.concatenate([frame, normal[:, :, None]], 2))
    log_s = np.log(np.maximum(np.linalg.norm(new_arms, axis=1), 1e-12))
    mean_new = mean.copy()
    quat_all, log_all = lifted["quat"].copy(), lifted["log_s"].copy()
    mean_new[idx], quat_all[idx], log_all[idx] = hc, quat, log_s
    active = np.zeros(len(mean), bool)
    active[idx[ok]] = True
    stats = {"surfels": int(n), "on_scene": int((kind[:n] == 1).sum()),
             "on_room": int((kind[:n] == 0).sum()), "missed": int((~ok).sum()),
             "median_gain_scene": float(np.median(med[kind[:n] == 1])) if (kind[:n] == 1).any() else None,
             "median_gain_room": float(np.median(med[kind[:n] == 0])) if (kind[:n] == 0).any() else None}
    print("mirror backs:", stats)
    return mean_new, quat_all, log_all, active, stats


def _mirror_map(mirror, points, surface, steps=800, chunk=20000):
    """Reference ray to each point -> cap -> reflect -> first surface or the room."""
    c = mirror.center.cpu().numpy().astype(np.float64)
    radius = mirror.radius
    d = points / np.linalg.norm(points, axis=1, keepdims=True)
    b = d @ -c
    disc = b * b - (c @ c) + radius ** 2
    t = -b - np.sqrt(np.maximum(disc, 0))
    q = t[:, None] * d
    n = (q - c) / radius
    r = d - 2 * (d * n).sum(1, keepdims=True) * n
    hit = np.full_like(points, np.nan)
    kind = np.full(len(points), -1)  # 1 scene, 0 room, -1 none
    for s0 in range(0, len(points), chunk):
        sl = slice(s0, s0 + chunk)
        qq, rr = q[sl], r[sl]
        room = np.where(rr[:, 2] < -1e-6, (MIRROR_ROOM_Z - qq[:, 2]) / np.minimum(rr[:, 2], -1e-6), np.inf)
        s_max = np.minimum(room, 3.0)
        prev = None
        found = np.zeros(len(qq), bool)
        for i in range(1, steps + 1):
            s = s_max * i / steps
            pts = qq + s[:, None] * rr
            zs, inside = surface(pts[:, 0], pts[:, 1])
            f = np.where(inside, pts[:, 2] - zs, -1.0)
            if prev is not None:
                cross = ~found & (prev < 0) & (f >= 0)
                if cross.any():
                    w = prev[cross] / (prev[cross] - f[cross])
                    s_hit = s_max[cross] * (i - 1 + w) / steps
                    hit[sl][cross] = qq[cross] + (s_hit - 1e-3)[:, None] * rr[cross]
                    kind[sl][cross] = 1
                    found |= cross
            prev = f
        to_room = ~found & np.isfinite(room)
        hit[sl][to_room] = qq[to_room] + room[to_room, None] * rr[to_room]
        kind[sl][to_room] = 0
    return hit, kind


class ConvexMirror:
    """A spherical cap over a painted mirror, traced with secondary rays.

    The cap sits on the lifted wall (median depth of the surfels inside the
    disc) and bulges towards the viewer. Pixels whose primary ray hits it get
    (1 - rho) * the painted mirror + rho * the lifted scene along the reflected
    ray, weighted by what that ray covers; the surfels are two-sided, so the
    reflection shows the backs of the lifted figures and moves with the camera.
    Reference camera OpenCV frame, like the lifted means.
    """

    def __init__(self, center, radius, wall_z, rho, torch, device):
        self.center = torch.as_tensor(center, dtype=torch.float32, device=device)
        self.radius, self.wall_z, self.rho = float(radius), float(wall_z), float(rho)
        self.torch = torch
        self.coverage = []  # mean reflection coverage per frame, for the report
        self.backs = None  # (renderer, model) of mirror_backs(), traced before the scene
        self.miss = "painted"  # what an uncovered reflected ray shows (--mirror-miss)

    @classmethod
    def from_disc(cls, mean_xy, mean, disc, camera_angle_x, bulge, rho, torch, device):
        u, v, r = disc
        aspect = float(np.ptp(mean_xy[:, 1]) / np.ptp(mean_xy[:, 0]))
        cx, cy, rp = 2.0 * u - 1.0, (2.0 * v - 1.0) * aspect, 2.0 * r
        inside = np.hypot(mean_xy[:, 0] - cx, mean_xy[:, 1] - cy) < rp
        t = math.tan(0.5 * camera_angle_x)
        a, wall = rp * t, float(np.median(mean[inside, 2]))
        h = bulge * a
        big_r = (a * a + h * h) / (2.0 * h)
        print(f"mirror: {int(inside.sum())} surfels, wall z {wall:.4f}, disc {a:.4f}, sphere R {big_r:.4f}")
        return cls((cx * t, cy * t, wall - h + big_r), big_r, wall, rho, torch, device)

    def apply(self, img, origins, directions, trace, background):
        d = directions / directions.norm(dim=1, keepdim=True)
        oc = origins - self.center
        b = (oc * d).sum(1)
        disc = b * b - (oc * oc).sum(1) + self.radius ** 2
        t = -b - disc.clamp_min(0).sqrt()  # nearest root
        p = origins + t[:, None] * d
        hit = (disc > 0) & (t > 0) & (p[:, 2] <= self.wall_z)  # the cap in front of the wall
        if not bool(hit.any()):
            return img
        n = (p[hit] - self.center) / self.radius
        dh = d[hit]
        refl = (dh - 2.0 * (dh * n).sum(1, keepdim=True) * n).contiguous()
        o2 = (p[hit] + 1e-4 * refl).contiguous()
        # On black and on white: premultiplied colour and coverage of the
        # reflection. Where it leaves the lifted scene (towards the viewer,
        # nothing was painted there) the painted reflection shows through.
        n_ray = o2.shape[0]
        black = trace(o2, refl, 1, n_ray, (0.0, 0.0, 0.0)).reshape(-1, 3)
        white = trace(o2, refl, 1, n_ray, (1.0, 1.0, 1.0)).reshape(-1, 3)
        alpha = (1.0 - (white - black).mean(1, keepdim=True)).clamp(0, 1)
        if self.backs is not None:
            # The painted reflection, moved to the backs, in front of the scene.
            renderer, model = self.backs
            b_black = render(renderer, model, o2, refl, 1, n_ray, (0.0, 0.0, 0.0), self.torch).reshape(-1, 3)
            b_white = render(renderer, model, o2, refl, 1, n_ray, (1.0, 1.0, 1.0), self.torch).reshape(-1, 3)
            b_alpha = (1.0 - (b_white - b_black).mean(1, keepdim=True)).clamp(0, 1)
            black = b_black + (1.0 - b_alpha) * black
            alpha = 1.0 - (1.0 - b_alpha) * (1.0 - alpha)
        flat = img.reshape(-1, 3).clone()
        if self.miss == "mean":
            # Nothing of the painted mirror stays: uncovered reflection gets its mean tone.
            tone = flat[hit].mean(0, keepdim=True)
            flat[hit] = (1.0 - self.rho) * flat[hit] + self.rho * (black + (1.0 - alpha) * tone)
        else:
            flat[hit] = (1.0 - self.rho * alpha) * flat[hit] + self.rho * black
        self.coverage.append(float(alpha.mean()))
        return flat.reshape(img.shape)


class PlaneLight:
    """A distant light (the window) shading a plane of the lifted scene.

    Only where a primary ray meets the fitted plane (fit_table_plane.py) and
    the reference view saw that plane there: the pixel keeps (1 - s) of its
    colour and gets s * E, where E is the irradiance relative to open light:
    a soft shadow (the source spans ANGLE degrees) behind every ball, plus,
    for glass balls, the caustic of photons refracted in and out of the ball
    (Schlick Fresnel, total internal reflection dropped) and splatted on the
    plane. The painted light in the photo is left as it is.
    """

    def __init__(self, direction, strength, angle_deg, plane, depth, camera_angle_x, balls, torch,
                 device, photons=4_000_000, cell=0.002, extent=0.6, seed=0, caustic_gain=1.0):
        self.torch, self.device = torch, device
        self.l = torch.as_tensor(np.asarray(direction, float) / np.linalg.norm(direction),
                                 dtype=torch.float32, device=device)
        self.strength, self.half = float(strength), math.radians(angle_deg) / 2
        self.n = torch.as_tensor(plane["normal"], dtype=torch.float32, device=device)
        self.d = float(plane["offset"])
        self.balls = balls
        h, w = depth.shape
        self.w, self.h = w, h
        self.f = 0.5 * w / math.tan(0.5 * camera_angle_x)
        self.depth = torch.as_tensor(depth, dtype=torch.float32, device=device)
        # Plane chart around the first ball's contact point.
        n = np.asarray(plane["normal"], float)
        e1 = np.cross(n, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        c0 = balls[0].base - balls[0].radius * n if balls else -self.d * n
        self.chart = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (c0, e1, e2)]
        self.cell, self.extent = cell, extent
        self.cells = int(round(2 * extent / cell))
        self.caustic = torch.zeros((self.cells, self.cells), device=device)
        gen = torch.Generator(device=device).manual_seed(seed)
        for ball in balls:
            if ball.material == "glass":
                self.caustic += self._photons(ball, photons, gen)
        self.caustic *= caustic_gain
        if balls:
            k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device)
            k = (k[:, None] * k[None]) / 256.0
            self.caustic = torch.nn.functional.conv2d(self.caustic[None, None], k[None, None],
                                                      padding=2)[0, 0]

    def _cell_index(self, p):
        torch = self.torch
        c0, e1, e2 = self.chart
        q = p - c0
        i = torch.floor(((q @ e1) + self.extent) / self.cell).long()
        j = torch.floor(((q @ e2) + self.extent) / self.cell).long()
        ok = (i >= 0) & (i < self.cells) & (j >= 0) & (j < self.cells)
        return i, j, ok

    def _jittered_light(self, count, gen):
        torch = self.torch
        l = self.l
        a = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        if abs(float(l[0])) > 0.9:
            a = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        b1 = torch.linalg.cross(l, a); b1 = b1 / b1.norm()
        b2 = torch.linalg.cross(l, b1)
        r = self.half * torch.sqrt(torch.rand(count, device=self.device, generator=gen))
        phi = 2 * math.pi * torch.rand(count, device=self.device, generator=gen)
        v = l + torch.tan(r)[:, None] * (torch.cos(phi)[:, None] * b1 + torch.sin(phi)[:, None] * b2)
        return v / v.norm(dim=1, keepdim=True), b1, b2

    def _photons(self, ball, count, gen):
        torch = self.torch
        centre = torch.as_tensor(ball.base, dtype=torch.float32, device=self.device)
        radius, ior = ball.radius, ball.ior
        lv, b1, b2 = self._jittered_light(count, gen)
        rr = radius * torch.sqrt(torch.rand(count, device=self.device, generator=gen))
        ph = 2 * math.pi * torch.rand(count, device=self.device, generator=gen)
        start = centre + (torch.cos(ph) * rr)[:, None] * b1 + (torch.sin(ph) * rr)[:, None] * b2 \
            + 3 * radius * lv
        d = -lv
        ok, t0, _ = VirtualSphere._roots(start, d, centre, radius)
        p = start + t0[:, None] * d
        nrm = (p - centre) / radius
        f0 = ((ior - 1) / (ior + 1)) ** 2
        cos1 = -(d * nrm).sum(1, keepdim=True)
        f1 = f0 + (1 - f0) * (1 - cos1) ** 5
        v_in, _ = VirtualSphere._refract(d, nrm, 1.0 / ior, torch)
        _, _, t1 = VirtualSphere._roots(p, v_in, centre, radius)
        q = p + t1[:, None] * v_in
        n2 = (q - centre) / radius
        v_out, out_ok = VirtualSphere._refract(v_in, -n2, ior, torch)
        cos2 = (v_in * n2).sum(1, keepdim=True).clamp(0, 1)
        f2 = f0 + (1 - f0) * (1 - cos2) ** 5
        weight = ((1 - f1) * (1 - f2)).squeeze(1) * (ok & out_ok).float()
        denom = v_out @ self.n
        t = -((q @ self.n) + self.d) / denom
        hit_p = q + t[:, None] * v_out
        i, j, inside = self._cell_index(hit_p)
        keep = inside & (denom < 0) & (t > 0) & (weight > 0)
        grid = torch.zeros(self.cells * self.cells, device=self.device)
        grid.index_add_(0, (i * self.cells + j)[keep], weight[keep])
        # Relative irradiance: photon flux / (cell area * open-light flux on the plane).
        flux = math.pi * radius ** 2 / count
        cos_plane = float(self.l @ self.n)
        return grid.reshape(self.cells, self.cells) * flux / (self.cell ** 2 * max(cos_plane, 1e-3))

    def _shadow(self, p):
        """Open fraction of the source at p (soft: the source spans 2*half)."""
        torch = self.torch
        open_frac = torch.ones(p.shape[0], device=self.device)
        for ball in self.balls:
            c = torch.as_tensor(ball.base, dtype=torch.float32, device=self.device)
            oc = c - p
            t = oc @ self.l
            dist = (oc - t[:, None] * self.l[None]).norm(dim=1)
            pen = (t.clamp_min(0) * math.tan(self.half)).clamp_min(1e-4)
            cover = ((ball.radius + pen - dist) / (2 * pen)).clamp(0, 1) * (t > 0).float()
            if ball.material == "bubble":
                cover = 0.15 * cover  # a thin shell lets most of the light through
            open_frac = open_frac * (1 - cover)
        for mesh in getattr(self, "meshes", []):
            open_frac = open_frac * mesh.light_visibility(p + 1e-4 * self.n, samples=8)
        for mirror in getattr(self, "mirrors", []):
            open_frac = open_frac * (1 - self._rect_cover(mirror, p, self.l))
        return open_frac

    def _rect_cover(self, mirror, p, direction):
        """How much of the source, seen from p along direction, the framed mirror covers (soft edges)."""
        denom = direction @ mirror.normal
        t = ((mirror.centre - p) @ mirror.normal) / (denom if abs(float(denom)) > 1e-8 else 1e-8)
        q = p + t[:, None] * direction[None]
        pen = (t.clamp_min(0) * math.tan(self.half)).clamp_min(1e-4)
        x, y = ((q - mirror.centre) @ mirror.e1).abs(), ((q - mirror.centre) @ mirror.e2).abs()
        cx = ((mirror.half[0] - x) / (2 * pen) + 0.5).clamp(0, 1)
        cy = ((mirror.half[1] - y) / (2 * pen) + 0.5).clamp(0, 1)
        return cx * cy * (t > 0).float()

    def _mirror_sun(self, p):
        """Sunlight the framed mirrors reflect onto p, relative to open light (0.92 reflectance)."""
        torch = self.torch
        e = torch.zeros(p.shape[0], device=self.device)
        for mirror in getattr(self, "mirrors", []):
            if float(self.l @ mirror.normal) <= 0:
                continue  # the sun is behind the mirror
            # Light arrives along -l and leaves the mirror along r; p is lit where -r from p meets the mirror.
            r = -self.l - 2.0 * float(-self.l @ mirror.normal) * mirror.normal
            if float(r @ self.n) >= 0:
                continue  # reflected away from the plane
            border = 1 - 0.06 * min(mirror.half) / np.array(mirror.half)
            glass = type("Glass", (), {"centre": mirror.centre, "normal": mirror.normal, "e1": mirror.e1,
                                       "e2": mirror.e2, "half": (mirror.half[0] * border[0],
                                                                 mirror.half[1] * border[1])})
            e = e + 0.92 * self._rect_cover(glass, p, -r)
        return e

    def apply(self, img, origins, directions, skip):
        torch = self.torch
        d = directions / directions.norm(dim=1, keepdim=True)
        denom = d @ self.n
        t = -((origins @ self.n) + self.d) / torch.where(denom.abs() < 1e-8, torch.full_like(denom, -1e-8), denom)
        p = origins + t[:, None] * d
        # Seen as that plane from the reference camera too (same pixel depth).
        u = torch.floor(self.f * p[:, 0] / p[:, 2] + self.w / 2).long()
        v = torch.floor(self.f * p[:, 1] / p[:, 2] + self.h / 2).long()
        ok = (t > 0) & (p[:, 2] > 1e-3) & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        ref = torch.full_like(t, float("nan"))
        ref[ok] = self.depth[v[ok], u[ok]]
        ok &= (ref - p[:, 2]).abs() < 0.02 * p[:, 2]
        if skip is not None:
            ok &= ~skip
        if not bool(ok.any()):
            return img
        pp = p[ok]
        e = self._shadow(pp)
        i, j, inside = self._cell_index(pp)
        caustic = torch.zeros_like(e)
        caustic[inside] = self.caustic[i[inside], j[inside]]
        e = e + caustic
        if getattr(self, "mirror_sun", False):
            e = e + self._mirror_sun(pp)
        flat = img.reshape(-1, 3).clone()
        flat[ok] = flat[ok] * ((1 - self.strength) + self.strength * e)[:, None]
        return flat.reshape(img.shape)


class PlaneMirror:
    """A flat framed mirror standing in front of the lifted painting.

    Placed so that, from the mean camera position, it shows the lifted
    surface around pixel (U, V) at GRAZE degrees from the surface normal,
    looking along the in-image axis AXIS (degrees, counter-clockwise from +u)
    (or from --plane-mirror-eye):
    the mirror stands DIST from that point on the grazing ray, facing halfway
    between it and the camera. For Holbein's anamorphic skull the grazing view
    along the skull turns the smear into a skull. Reference camera OpenCV frame.
    """

    FRAME = (0.30, 0.22, 0.10)

    def __init__(self, spec, mean_xy, mean, cams, camera_angle_x, torch, device, eye=None, fixed=None):
        u, v, axis, graze, dist, half_w, half_h = (float(x) for x in spec)
        if fixed is not None:
            # The same mirror as another run (its report.json), e.g. to
            # compare a flat painting with the lift under one mirror.
            self._set(fixed["surface_point"], fixed["surface_normal"], fixed["centre"], fixed["normal"],
                      cams, (half_w, half_h), axis, torch, device)
            return
        aspect = float(np.ptp(mean_xy[:, 1]) / np.ptp(mean_xy[:, 0]))
        cx, cy = 2.0 * u - 1.0, (2.0 * v - 1.0) * aspect
        r = np.hypot(mean_xy[:, 0] - cx, mean_xy[:, 1] - cy)
        point = np.median(mean[r < 0.01], axis=0)
        # Local surface normal of the lift, towards the viewer.
        near = mean[r < 0.06]
        _, _, vt = np.linalg.svd(near - near.mean(0), full_matrices=False)
        n = vt[2] if vt[2][2] < 0 else -vt[2]
        a = np.array([math.cos(math.radians(axis)), -math.sin(math.radians(axis)), 0.0])
        a = a - (a @ n) * n
        a /= np.linalg.norm(a)
        g = -math.sin(math.radians(graze)) * a + math.cos(math.radians(graze)) * n
        centre = point + dist * g
        camera = np.mean([c["c2w_cv"][:3, 3] for c in cams], axis=0) if eye is None else np.asarray(eye, float)
        d1 = centre - camera
        d1 /= np.linalg.norm(d1)
        m = d1 - (-g)
        m /= np.linalg.norm(m)
        if m @ d1 > 0:
            m = -m
        self._set(point, n, centre, m, cams, (half_w, half_h), axis, torch, device)

    def _set(self, point, n, centre, m, cams, half, axis, torch, device):
        point, n, centre, m = (np.asarray(x, float) for x in (point, n, centre, m))
        # Level with the cameras: the frame's vertical is their mean up.
        up = -np.mean([c["c2w_cv"][:3, 1] for c in cams], axis=0)
        e2 = up - (up @ m) * m
        e2 /= np.linalg.norm(e2)
        e1 = np.cross(e2, m)
        print(f"plane mirror: surface point {np.round(point, 4)}, normal {np.round(n, 3)}, "
              f"centre {np.round(centre, 4)}, facing {np.round(m, 3)}")
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.centre, self.normal, self.e1, self.e2 = t(centre), t(m), t(e1), t(e2)
        self.half = half
        self.torch = torch
        self.geometry = {"surface_point": point.tolist(), "surface_normal": n.tolist(),
                         "centre": centre.tolist(), "normal": m.tolist()}

    def apply(self, img, origins, directions, trace, background, zbuf=None):
        d = directions / directions.norm(dim=1, keepdim=True)
        denom = d @ self.normal
        t = ((self.centre - origins) @ self.normal) / denom.where(denom.abs() > 1e-8, self.torch.ones_like(denom))
        p = origins + t[:, None] * d
        x, y = (p - self.centre) @ self.e1 / self.half[0], (p - self.centre) @ self.e2 / self.half[1]
        hit = (denom < 0) & (t > 0) & (x.abs() <= 1) & (y.abs() <= 1)
        if zbuf is not None:
            hit &= t < zbuf
            zbuf[hit] = t[hit]
        if not bool(hit.any()):
            return img
        border = 1 - 0.06 * min(self.half) / np.array(self.half)
        frame = hit & ((x.abs() > border[0]) | (y.abs() > border[1]))
        glass = hit & ~frame
        flat = img.reshape(-1, 3).clone()
        if bool(glass.any()):
            dg = d[glass]
            refl = dg - 2.0 * (dg @ self.normal)[:, None] * self.normal
            o2 = (p[glass] + 1e-4 * refl).contiguous()
            flat[glass] = 0.92 * trace(o2, refl.contiguous(), 1, o2.shape[0], background).reshape(-1, 3)
        flat[frame] = self.torch.as_tensor(self.FRAME, dtype=flat.dtype, device=flat.device)
        return flat.reshape(img.shape)


class VirtualSphere:
    """A chrome or glass ball placed in front of the lifted painting.

    Rays that hit it are bent (one mirror bounce, or Snell refraction in and
    out with Schlick's Fresnel mix of the reflection) and traced through the
    lifted surfels, which the rasteriser of a mesh or of GaINeR's
    tetrahedra cannot do from one render. (U, V) is the reference-view pixel
    it covers (fractions), Z its depth in the reference camera (0: just in
    front of the surfels under it), R its radius
    as a fraction of the image width at that depth. The ball must sit in front
    of the surfels: a hit replaces the pixel. ORBIT moves it on a circle of
    that radius (fraction of width, and of height vertically) over the path.
    """

    def __init__(self, spec, camera_angle_x, aspect, torch, device, mean_xy, mean, ior=1.5):
        u, v, z, r, self.material = spec[0], spec[1], spec[2], spec[3], spec[4]
        u, v, z, r = float(u), float(v), float(z), float(r)
        t = math.tan(0.5 * camera_angle_x)
        if z <= 0:
            # Auto: just in front of the nearest surfel under its footprint.
            near = np.hypot(mean_xy[:, 0] - (2 * u - 1), mean_xy[:, 1] - (2 * v - 1) * aspect) < 3 * r
            z = float(np.min(mean[near, 2])) / (1 + 2.4 * r * t)
            print(f"sphere at ({u}, {v}): z {z:.4f}")
        self.base = np.array([(2 * u - 1) * t * z, (2 * v - 1) * aspect * t * z, z])
        self.radius, self.unit, self.aspect = 2 * r * t * z, 2 * t * z, aspect
        self.ior, self.torch, self.device = ior, torch, device
        self.orbit = float(spec[5]) if len(spec) > 5 else 0.0
        assert self.material in ("chrome", "glass", "bubble", "emissive"), self.material
        self.light = None  # (unit direction towards the light, strength) for a glint

    @classmethod
    def at(cls, centre, radius, material, torch, device, ior=1.5):
        """A ball given by its centre (reference OpenCV frame) and radius."""
        ball = cls.__new__(cls)
        ball.base, ball.radius, ball.material = np.asarray(centre, float), float(radius), material
        ball.unit = ball.aspect = 1.0
        ball.ior, ball.torch, ball.device, ball.orbit, ball.light = ior, torch, device, 0.0, None
        return ball

    def center(self, frame, n_frames):
        if getattr(self, "path", None) is not None:
            return self.torch.as_tensor(self.path[frame % len(self.path)], dtype=self.torch.float32,
                                        device=self.device)
        a = 2 * math.pi * frame / max(n_frames, 1)
        off = self.orbit * self.unit * np.array([math.cos(a), self.aspect * math.sin(a), 0.0])
        return self.torch.as_tensor(self.base + off, dtype=self.torch.float32, device=self.device)

    @staticmethod
    def _roots(o, d, c, r):
        oc = o - c
        b = (oc * d).sum(1)
        disc = b * b - (oc * oc).sum(1) + r * r
        sq = disc.clamp_min(0).sqrt()
        return disc > 0, -b - sq, -b + sq

    @staticmethod
    def _refract(d, n, eta, torch):
        cos_i = -(d * n).sum(1, keepdim=True)
        k = 1 - eta * eta * (1 - cos_i * cos_i)
        out = eta * d + (eta * cos_i - k.clamp_min(0).sqrt()) * n
        return out / out.norm(dim=1, keepdim=True), (k > 0).squeeze(1)

    def transmittance(self, origins, directions, tmax):
        """Shadow rays (unit directions) for facet/lifting/lights.py: glass lets 60 % through, chrome none."""
        torch = self.torch
        c = self.center(getattr(self, "frame_now", 0), getattr(self, "n_frames_now", 1))
        ok, t0, _ = self._roots(origins, directions, c, self.radius)
        blocked = ok & (t0 > 0) & (t0 < tmax)
        keep = {"glass": 0.6, "bubble": 0.9}.get(self.material, 0.0)
        return torch.where(blocked, torch.full_like(t0, keep), torch.ones_like(t0))

    def apply(self, img, origins, directions, trace, background, frame, n_frames, zbuf=None):
        torch = self.torch
        c = self.center(frame, n_frames)
        d = directions / directions.norm(dim=1, keepdim=True)
        ok, t0, _ = self._roots(origins, d, c, self.radius)
        hit = ok & (t0 > 0)
        if zbuf is not None:
            # --bounces: only where this ball is the nearest virtual object so far.
            hit &= t0 < zbuf
            zbuf[hit] = t0[hit]
        self.last_hit = hit
        if not bool(hit.any()):
            return img
        dh = d[hit]
        p = origins[hit] + t0[hit, None] * dh
        n = (p - c) / self.radius
        refl = dh - 2 * (dh * n).sum(1, keepdim=True) * n

        def trace_bg(o, v, bg):
            return trace(o.contiguous(), v.contiguous(), 1, o.shape[0], bg).reshape(-1, 3)

        def trace_scene(o, v):
            return trace_bg(o, v, background)

        if self.material == "emissive":
            # The bulb of --point-light: flat bright core, slightly darker rim.
            cos_v = -(dh * n).sum(1, keepdim=True)
            col = self.torch.as_tensor(self.emit, dtype=dh.dtype, device=dh.device) * (0.75 + 0.25 * cos_v)
            flat = img.reshape(-1, 3).clone()
            flat[hit] = col
            return flat.reshape(img.shape)
        if self.material == "chrome":
            # Towards the viewer nothing was painted: a grey studio gradient
            # behind whatever the reflected ray covers (premultiplied on black).
            o2 = p + 1e-4 * refl
            black = trace_bg(o2, refl, (0.0, 0.0, 0.0))
            alpha = (1 - (trace_bg(o2, refl, (1.0, 1.0, 1.0)) - black).mean(1, keepdim=True)).clamp(0, 1)
            env = 0.12 + 0.45 * (-refl[:, 1:2]).clamp(0, 1) + 0.08 * (-refl[:, 2:3]).clamp(0, 1)
            col = 0.9 * (black + (1 - alpha) * env)
        elif self.material == "bubble":
            # Thin glass shell: the two walls cancel the bending, so the scene
            # shows through upright (lightly tinted); Fresnel reflection on top,
            # doubled for the two walls.
            cos_i = -(dh * n).sum(1, keepdim=True)
            f0 = ((self.ior - 1) / (self.ior + 1)) ** 2
            fres = (2 * (f0 + (1 - f0) * (1 - cos_i) ** 5)).clamp(max=1)
            _, _, t_far = self._roots(origins[hit], dh, c, self.radius)
            exit_p = origins[hit] + (t_far[:, None] + 1e-4) * dh
            col = fres * trace_scene(p + 1e-4 * refl, refl) + (1 - fres) * 0.96 * trace_scene(exit_p, dh)
        else:
            cos_i = -(dh * n).sum(1, keepdim=True)
            # A glass wall (surface_ior) around a lower-index filling: the
            # wall sets the reflection, the filling the bending.
            surface = getattr(self, "surface_ior", None) or self.ior
            f0 = ((surface - 1) / (surface + 1)) ** 2
            fres = f0 + (1 - f0) * (1 - cos_i) ** 5
            v_in, _ = self._refract(dh, n, 1.0 / self.ior, torch)
            _, _, t1 = self._roots(p, v_in, c, self.radius)
            q = p + t1[:, None] * v_in
            n2 = (q - c) / self.radius
            v_out, ok_out = self._refract(v_in, -n2, self.ior, torch)
            # Total internal reflection: leave along the internal mirror ray.
            v_tir = v_in - 2 * (v_in * n2).sum(1, keepdim=True) * n2
            v_out = torch.where(ok_out[:, None], v_out, v_tir)
            c_refr = trace_scene(q + 1e-4 * v_out, v_out)
            col = fres * trace_scene(p + 1e-4 * refl, refl) + (1 - fres) * c_refr
        if self.light is not None:
            # The window as a small bright source seen in the reflection.
            direction, strength = self.light
            glint = (refl @ direction[:, None]).clamp_min(0) ** 300
            col = col + strength * 2.5 * glint
        flat = img.reshape(-1, 3).clone()
        flat[hit] = col
        self.last_hit = hit
        return flat.reshape(img.shape)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=Path, help="reference Z-depth npz (key 'depth')")
    p.add_argument("--scene-dir", type=Path, help="directory with eval_path.json")
    p.add_argument("--camera-angle-x", type=float, default=0.6911112070083618)
    p.add_argument("--sampling", choices=["round", "floor", "bilinear"], default="round")
    p.add_argument("--transport", choices=["depth_scaled", "jacobian", "jacobian_exact"], default="depth_scaled",
                   type=lambda value: "depth_scaled" if value == "legacy" else value)
    p.add_argument("--invalid", choices=["deactivate", "plane"], default="deactivate")
    p.add_argument("--freeze-view-direction", action="store_true")
    p.add_argument("--lift-k", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                   help="full source K in pixels, instead of --camera-angle-x")
    p.add_argument("--target-k-json", type=Path,
                   help="corrected_meta.json: render each target with its own full K")
    p.add_argument("--edge-max-slant", type=float, default=0.0, metavar="DEG",
                   help="cut depth gradients steeper than a surface tilted by DEG (0: off)")
    p.add_argument("--id-depth-order", type=float, default=0.0, metavar="DELTA",
                   help="push primitive i along its reference ray by (1 + DELTA i/N)")
    p.add_argument("--lattice-depth", type=Path,
                   help="GaINeR qualitative lifting (depth_lattice_deform.py) with this depth npz")
    p.add_argument("--lattice-scale", type=float, default=0.2)
    p.add_argument("--lattice-max-stretch", type=float, default=2.0)
    p.add_argument("--lattice-gainer-units", action="store_true",
                   help="read --lattice-scale in GaINeR's units: depth_lattice_deform.py maps the image"
                        " plane to a third of the reference camera's (x in [(-right+1.5)/3, (right+1.5)/3])"
                        " but adds the depth shift unscaled, so the same scale extrudes 3x more there")
    p.add_argument("--lattice-drop-steep", type=float, default=0.0, metavar="S",
                   help="remove surfels whose depth rise across them exceeds S x their size (0: off);"
                        " disocclusions stay clean holes (for inpainting) instead of thin sprinkles")
    p.add_argument("--lattice-stretch-clamp", action="store_true",
                   help="surfels past --lattice-max-stretch keep the slope at the limit instead of"
                        " lying flat (GaINeR's rigid shift leaves gaps on steep depth edges)")
    p.add_argument("--lift-scale-inflate", type=float, default=1.0,
                   help="multiply the lifted surfels' in-plane scales (closes the gaps the lift opens)")
    p.add_argument("--lattice-no-invert", action="store_true",
                   help="the submission pipeline passes --invert-depth; this disables it")
    p.add_argument("--mirror", type=float, nargs=3, metavar=("U", "V", "R"),
                   help="ray-traced convex mirror over the painted disc centred at (U, V)"
                        " (fractions of width, height), radius R (fraction of width)")
    p.add_argument("--mirror-backs", action="store_true",
                   help="move the painted reflection onto the backs of the lifted scene"
                        " (mirror_backs; from the --inset model when given)")
    p.add_argument("--mirror-miss", choices=["painted", "mean"], default="painted",
                   help="where the reflection covers nothing: the painted mirror, or its mean tone")
    p.add_argument("--mirror-backs-opacity-gamma", type=float, default=3.0,
                   help="opacity gamma of the magnified back surfels (they are only a texture)")
    p.add_argument("--mirror-bulge", type=float, default=0.35, help="cap height / disc radius")
    p.add_argument("--mirror-reflectance", type=float, default=0.7,
                   help="weight of the traced reflection against the painted mirror")
    p.add_argument("--sphere", nargs="+", action="append", default=[],
                   metavar="U V Z R MATERIAL [ORBIT]",
                   help="virtual chrome|glass ball (see VirtualSphere); repeatable")
    p.add_argument("--opacity-gamma", type=float, default=1.0,
                   help="raise each surfel's opacity to 1 - (1 - alpha)^GAMMA after the lift")
    p.add_argument("--inset", nargs=2, metavar=("CHECKPOINT", "CROP_JSON"),
                   help="higher-resolution FACET model of a crop (crop.json 'uv' box of the"
                        " source image), lifted by the same lattice and composited over it")
    p.add_argument("--inset-keep-under", action="store_true",
                   help="keep the global surfels under the inset: its holes show the global model,"
                        " not the background")
    p.add_argument("--background-layer", nargs=3, metavar=("CHECKPOINT", "DEPTH_NPZ", "BAND_PNG"),
                   help="FACET model of prepare_background_layer.py's layer.png, lifted (perspective or lattice)"
                        " with its depth.npz, only its surfels inside band.png; seen where the main"
                        " model lets light through (disocclusions) instead of the background")
    p.add_argument("--sphere-at", nargs=5, action="append", default=[],
                   metavar=("X", "Y", "Z", "R", "MATERIAL"),
                   help="ball by centre and radius in the reference OpenCV frame; repeatable")
    p.add_argument("--sphere-path", nargs=3, action="append", default=[], metavar=("NPY", "R", "MATERIAL"),
                   help="moving ball: centre per frame from an (F, 3) .npy; repeatable")
    p.add_argument("--sphere-ior", type=float, default=1.5,
                   help="refractive index of --sphere-at/--sphere-path balls (1.33: water in a thin glass shell);"
                        " MATERIAL glass:IOR sets one ball's own")
    p.add_argument("--dof-samples", type=int, default=0,
                   help="thin-lens depth of field: lens samples per pixel (0: pinhole)")
    p.add_argument("--aperture", type=float, default=0.0,
                   help="lens radius (reference units); eval_path frames may set aperture_radius")
    p.add_argument("--focus-distance", type=float, default=1.0,
                   help="focal plane along the view axis; eval_path frames may set focus_distance")
    p.add_argument("--plane-mirror", type=float, nargs=7,
                   metavar=("U", "V", "AXIS", "GRAZE", "DIST", "HALF_W", "HALF_H"),
                   help="flat framed mirror showing pixel (U, V) at GRAZE degrees along AXIS (see PlaneMirror)")
    p.add_argument("--plane-mirror-eye", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="camera position the plane mirror is aimed for (default: the mean camera)")
    p.add_argument("--bounces", type=int, default=0,
                   help="0: virtual objects drawn in order over the scene, secondary rays see the scene"
                        " only; N >= 1: a z-buffer over them and N levels of secondary"
                        " rays that see every virtual object")
    p.add_argument("--mesh", nargs=6, action="append", default=[],
                   metavar=("NPZ", "X", "Y", "Z", "YAW", "LENGTH"),
                   help="triangle-mesh prop (prepare_fiat.py npz, facet/lifting/meshes.py) standing on point"
                        " (X, Y, Z), up = the --light-plane normal, turned by YAW degrees, LENGTH long;"
                        " needs --bounces; repeatable")
    p.add_argument("--point-light", nargs=4, metavar=("PATH_NPY", "POWER", "RADIUS", "DIM"),
                   help="moving point light (facet/lifting/lights.py): (F, 3) positions per frame, reference"
                        " OpenCV frame; the photo dimmed to DIM plus the light, with shadows of the"
                        " lifted scene (a proxy mesh of --depth) and the virtual objects; needs --bounces")
    p.add_argument("--point-light-tint", type=float, nargs=3, default=(1.0, 0.86, 0.68))
    p.add_argument("--point-light-samples", type=int, default=8, help="points on the light's sphere")
    p.add_argument("--deform-sine", type=float, nargs=4, metavar=("AMPLITUDE", "FREQUENCY", "HEAD", "NECK"),
                   help="animate the lifted surfels with GaINeR's snake wave (sine_deform), one period over"
                        " the camera path; off by default")
    p.add_argument("--ripple", type=float, nargs=4, metavar=("AMPLITUDE", "WAVELENGTH", "SPEED", "T0"),
                   help="time-varying travelling sine wave on the lifted surfels (ripple_displacement): still"
                        " until T0 seconds (frame index / --ripple-fps), then a wave front leaves the origin"
                        " at SPEED units/s; lengths in the reference OpenCV frame; off by default")
    p.add_argument("--ripple-fps", type=float, default=30.0, help="frames per second of the camera path")
    p.add_argument("--ripple-origin", type=float, nargs=2, metavar=("X", "Y"), default=None,
                   help="circular ripple from this point (reference frame x, y); default a plane wave")
    p.add_argument("--ripple-direction", type=float, nargs=2, metavar=("DX", "DY"), default=(1.0, 0.0),
                   help="plane wave travel direction (reference frame x, y)")
    p.add_argument("--ripple-axis", choices=["normal", "along", "both"], default="normal",
                   help="displacement along -z (out of a flat print), in the plane along the travel, or both")
    p.add_argument("--ripple-along-ratio", type=float, default=1.0,
                   help="with --ripple-axis both: in-plane amplitude / out-of-plane amplitude")
    p.add_argument("--ripple-front", type=float, default=1.0, help="leading edge width, in wavelengths")
    p.add_argument("--ripple-edge-taper", type=float, default=0.0,
                   help="fade the ripple to zero towards the print's border (bounding box of the active surfels)"
                        " over this fraction of its smaller side, so the edges stay put; 0 (default) = off")
    p.add_argument("--ripple-arms", choices=["jacobian", "keep"], default="jacobian",
                   help="jacobian: the arms follow the wave's Jacobian; keep (diagnostic): translate the surfels only")
    p.add_argument("--mesh-hop", type=float, nargs=5, default=None,
                   metavar=("HEIGHT", "START", "RISE", "HOLD", "FALL"),
                   help="the --mesh props lift off along up (the --light-plane normal) at START seconds"
                        " (frame index / --ripple-fps), ease up to HEIGHT over RISE, hover for HOLD, then"
                        " drop back over FALL with constant acceleration; landing at START+RISE+HOLD+FALL"
                        " (set --ripple's T0 there for the impact wave)")
    p.add_argument("--ripple-start-radius", type=float, default=0.0,
                   help="with --ripple-origin: the front starts from this ring (e.g. a prop's foot), still inside")
    p.add_argument("--facet-occludes-meshes", action="store_true",
                   help="z-buffer the --mesh props against the FACET surface (its expected depth where alpha > 0.5)")
    p.add_argument("--shade-facet", type=float, default=None, metavar="AMBIENT",
                   help="shade the FACET surface by its composited normal: AMBIENT + (1 - AMBIENT)|n.l|, l = --light")
    p.add_argument("--cage", type=Path, default=None,
                   help="npz (origin, cell, z0, frames): the surfels follow a deformed grid mesh, frame i of the"
                        " camera path using cage frame i (cage_deform; e.g. facet/physics/simulate_cloth.py)")
    p.add_argument("--ripple-ramp", type=float, default=0.5, help="amplitude ease-in after T0, seconds")
    p.add_argument("--point-light-bulb", type=float, default=0.03,
                   help="radius of the glowing ball drawn at the light (0: none)")
    p.add_argument("--proxy-max-jump", type=float, default=0.1,
                   help="proxy mesh: no triangles over a relative depth jump above this")
    p.add_argument("--proxy-stride", type=int, default=2,
                   help="proxy mesh: one vertex every STRIDE depth pixels (1: full resolution)")
    p.add_argument("--point-light-surface", choices=["proxy", "facet"], default="proxy",
                   help="point lit on each ray: the proxy mesh's first hit, or FACET's median depth"
                        " (the surface the render shows; normal from the proxy next to it)")
    p.add_argument("--plane-mirror-from", type=Path, metavar="REPORT_JSON",
                   help="reuse the plane mirror of another run (its report.json) instead of placing one")
    p.add_argument("--plane-mirror-shadow", action="store_true",
                   help="the --plane-mirror casts a shadow on the --light-plane")
    p.add_argument("--plane-mirror-sun", action="store_true",
                   help="with --plane-mirror-shadow, also the patch of sunlight the mirror reflects onto that plane")
    p.add_argument("--miss-background", action="store_true",
                   help="render on white as trained; --background only where rays miss the painting")
    p.add_argument("--miss-sky-ground", type=float, nargs=6, metavar=("SR", "SG", "SB", "GR", "GG", "GB"),
                   help="colour of rays that miss the lifted photo: sky upwards, ground downwards (default: --background)")
    p.add_argument("--miss-image", type=Path,
                   help="rays that miss the lifted photo take this image's colour along their direction through the"
                        " reference camera, mirrored at its edges (default: --background)")
    p.add_argument("--miss-padding", choices=("reflection", "smooth"), default="reflection",
                   help="--miss-image beyond its frame: mirrored (reflection) or a smooth extrapolation of its edges"
                        " (smooth: no mirrored copies of the photo in refractions)")
    p.add_argument("--flat-depth", type=float, default=None, metavar="Z",
                   help="baseline: replace --depth by the constant Z (a flat billboard)")
    p.add_argument("--sphere-surface-ior", type=float, default=None,
                   help="glass balls: index of a thin wall that sets the Fresnel reflection"
                        " (brighter, less transmissive rim); default: the ball's own index")
    p.add_argument("--light", type=float, nargs=3, metavar=("LX", "LY", "LZ"),
                   help="direction towards a distant light (reference OpenCV frame)")
    p.add_argument("--light-plane", type=Path, help="fit_table_plane.py json: the plane it shades")
    p.add_argument("--light-strength", type=float, default=0.55,
                   help="share of a plane pixel's colour that comes from this light")
    p.add_argument("--light-angle", type=float, default=3.0, help="angular size of the source, deg")
    p.add_argument("--caustic-gain", type=float, default=1.0,
                   help="scale of the glass balls' caustic on the --light-plane (1: as traced)")
    p.add_argument("--edge-max-jump", type=float, default=0.0, metavar="REL",
                   help="snap surfels where the depth jumps by more than REL across 5x5 px (0: off)")
    p.add_argument("--supersample", type=int, default=1,
                   help="trace SS x SS rays per pixel and box-filter them (antialiasing)")
    p.add_argument("--time-frames", action="store_true",
                   help="report each frame's render time (CUDA-synchronised, image write excluded) in report.json")
    p.add_argument("--identity-test", action="store_true",
                   help="lift onto the constant plane D=1 and render the reference camera")
    p.add_argument("--source-path", type=Path, help="training image, if moved since training")
    p.add_argument("--background", type=float, nargs=3, default=None,
                   help="default: the training background from the checkpoint; FACET leaves"
                        " residual transmittance, so a different one changes the image")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.lattice_gainer_units:
        args.lattice_scale *= 3.0

    import torch
    from PIL import Image
    from facet.io.checkpoint import load_training_checkpoint
    from facet.renderer.facade import FACETRenderer
    from facet.model.components import PlanarGaussianModel
    from facet.model.rays import generate_indexed_orthographic_rays, generate_indexed_perspective_rays

    device = torch.device("cuda:0")
    ckpt_sha_before = sha256(args.checkpoint)
    state = load_training_checkpoint(args.checkpoint, device, require_rgb=True)
    image_cfg = state.config["image"]
    cond = state.config.get("ablation", {}).get("active_conditioning_dimensions", 96)
    planar = PlanarGaussianModel(state.model, plane_z=image_cfg.get("plane_z", 0.0),
                                 active_conditioning_dimensions=cond)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.background is None:
        args.background = tuple(float(image_cfg.get(f"bg_color_{c}", 0.0)) for c in "RGB")

    mean_xy = planar.mean_xy.detach().cpu().numpy()
    theta = planar.theta.detach().cpu().numpy()
    log_s = planar.s.detach().cpu().numpy()
    alive = np.isfinite(planar.A.detach().cpu().numpy()[:, 0])
    ids = np.arange(len(mean_xy), dtype=np.int64)
    np.savez(args.output / "planar_sidecar.npz", primitive_ids=ids, mean_xy=mean_xy,
             theta=theta, log_scales_xy=log_s, active=alive,
             checkpoint_sha256=np.asarray(ckpt_sha_before))

    if args.identity_test:
        # Reference image size from the checkpoint's own dataset geometry.
        with Image.open(args.source_path or image_cfg["source_path"]) as source_image:
            width, height = source_image.size
        depth = np.ones((height, width), dtype=np.float32)
        cams = [{"file_path": "identity", "c2w_cv": np.eye(4),
                 "fov_x": args.camera_angle_x,
                 "fov_y": 2.0 * math.atan(math.tan(0.5 * args.camera_angle_x) * height / width)}]
    else:
        depth = np.load(args.depth)["depth"] if args.depth is not None else None
        if args.flat_depth is not None:
            # Baseline: the photo as a flat billboard at one depth.
            depth = np.full_like(depth, args.flat_depth)
        target_k = None
        if args.target_k_json is not None:
            meta = json.loads(args.target_k_json.read_text())["targets"]
            target_k = {f"target_rgb_{vid}.png": (t["K"][0][0], t["K"][1][1], t["K"][0][2], t["K"][1][2])
                        for vid, t in meta.items()}
        width, height, cams = target_cameras(json.loads((args.scene_dir / "eval_path.json").read_text()),
                                             target_k)
    # Supersampling: trace SS x SS rays per output pixel, box-filter down.
    ss = 1 if args.identity_test else max(1, int(args.supersample))
    out_w, out_h = width, height
    if ss > 1:
        width, height = width * ss, height * ss
        for cam in cams:
            if cam.get("K") is not None:
                cam["K"] = tuple(float(v) * ss for v in cam["K"])
    if args.lattice_depth is not None:
        lat = np.load(args.lattice_depth)
        lat = lat["depth"] if "depth" in lat.files else lat[lat.files[0]]
        mean, quat, new_log_s, active, stats = lift_lattice(
            mean_xy, theta, log_s, lat, args.lattice_scale, args.lattice_max_stretch,
            not args.lattice_no_invert, args.camera_angle_x, args.id_depth_order,
            args.lattice_stretch_clamp, args.lattice_drop_steep)
    else:
      mean, quat, new_log_s, active, stats = lift(mean_xy, theta, log_s, depth, args.camera_angle_x,
                                                args.sampling, args.transport, args.invalid,
                                                args.id_depth_order, args.edge_max_slant, args.lift_k,
                                                args.edge_max_jump)
    if args.lift_scale_inflate != 1.0:
        new_log_s = new_log_s + math.log(args.lift_scale_inflate)
    model = LiftedModel(planar, mean, quat, new_log_s, active & alive,
                        args.freeze_view_direction, torch, args.opacity_gamma)
    renderer = FACETRenderer(8, 11.3449, width * height)
    inset = None
    if args.inset is not None:
        if args.lattice_depth is None:
            raise SystemExit("--inset needs --lattice-depth")
        data = state.config["data"]
        inset, box, inset_lifted = load_inset(Path(args.inset[0]), Path(args.inset[1]), (data["width"], data["height"]),
                                lat, args, torch, device)
        # The global surfels under the inset (a thin overlap band stays).
        u, v = (mean_xy[:, 0] + 1) / 2, (mean_xy[:, 1] / (data["height"] / data["width"]) + 1) / 2
        m = INSET_MARGIN
        under = (u > box[0] + m) & (u < box[2] - m) & (v > box[1] + m) & (v < box[3] - m)
        if not args.inset_keep_under:
            model.A[torch.as_tensor(under, device=device)] = -torch.inf
        inset_renderer = FACETRenderer(8, 11.3449, width * height)
    back = None
    if args.background_layer is not None:
        if (depth is None and args.lattice_depth is None) or args.inset is not None:
            raise SystemExit("--background-layer needs a --depth or --lattice-depth lift, without --inset")
        back, back_depth, back_stats = load_background_layer(
            *(Path(x) for x in args.background_layer), args, torch, device,
            lat if args.lattice_depth is not None else None)
        stats["background_layer"] = back_stats
        back_renderer = FACETRenderer(8, 11.3449, width * height)

    miss_img = None
    if args.miss_image is not None:
        if depth is None:
            raise SystemExit("--miss-image needs a --depth lift (its reference intrinsics)")
        from PIL import Image as _Image
        miss_img = torch.as_tensor(np.asarray(_Image.open(args.miss_image).convert("RGB"), np.float32) / 255.0,
                                   device=device).permute(2, 0, 1).contiguous()
        if args.miss_padding == "smooth":
            # Three times the frame in each direction. Beyond each edge, the edge's outer rows (columns)
            # averaged, blurred along the edge the more the farther out: the sky goes on above the photo
            # and the pavement below it, without mirrored copies or streaks.
            def extend_rows(img, n):
                c, _, w = img.shape
                band = img[:, :8].mean(1)
                sig = torch.arange(n, 0, -1, dtype=img.dtype, device=img.device) + 4.0
                blurred = []
                for s in (2.0 ** torch.arange(2, 13)).tolist():
                    r = min(int(3 * s), 4 * w); x = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
                    k = torch.exp(-x * x / (2 * s * s)); k = (k / k.sum())[None, None].repeat(c, 1, 1)
                    blurred.append(torch.nn.functional.conv1d(torch.nn.functional.pad(band[None], (r, r), mode="replicate"),
                                                              k, groups=c)[0])
                blurred = torch.stack(blurred)
                pos = (torch.log2(sig) - 2).clamp(0, len(blurred) - 1)
                i0 = pos.floor().long(); i1 = (i0 + 1).clamp(max=len(blurred) - 1); f = (pos - i0)[None, :, None]
                return blurred[i0].permute(1, 0, 2) * (1 - f) + blurred[i1].permute(1, 0, 2) * f
            h0, w0 = miss_img.shape[1:]
            ext = torch.cat([extend_rows(miss_img, h0), miss_img, extend_rows(miss_img.flip(1), h0).flip(1)], 1)
            ext = ext.transpose(1, 2)
            miss_img = torch.cat([extend_rows(ext, w0), ext, extend_rows(ext.flip(1), w0).flip(1)], 1).transpose(1, 2).contiguous()

    def trace(o, d, w, h, bg):
        if args.miss_sky_ground is not None and back is None and inset is None and tuple(bg) == tuple(args.background):
            # Rays that leave the lifted photo: sky colour upwards, ground colour
            # downwards (OpenCV: up is -y), blended over a few degrees of elevation.
            white = render(renderer, model, o, d, w, h, (1.0, 1.0, 1.0), torch)
            black = render(renderer, model, o, d, w, h, (0.0, 0.0, 0.0), torch)
            t = (white - black).mean(-1, keepdim=True).clamp(0, 1)
            up = (-(d / d.norm(dim=1, keepdim=True))[:, 1:2] / 0.1 + 0.5).clamp(0, 1)
            up = (up * up * (3 - 2 * up)).reshape(*t.shape[:-1], 1)
            sky, ground = (torch.as_tensor(c, dtype=white.dtype, device=white.device)
                           for c in (args.miss_sky_ground[:3], args.miss_sky_ground[3:]))
            return black + t * (up * sky + (1 - up) * ground)
        if args.miss_image is not None and back is None and inset is None and tuple(bg) == tuple(args.background):
            # Rays that leave the lifted photo see the photo itself, looked up along
            # the ray direction through the reference camera, mirrored at its edges
            # (sky texture above the frame, pavement below it).
            white = render(renderer, model, o, d, w, h, (1.0, 1.0, 1.0), torch)
            black = render(renderer, model, o, d, w, h, (0.0, 0.0, 0.0), torch)
            t = (white - black).mean(-1, keepdim=True).clamp(0, 1)
            dz = d[:, 2:3].clamp_min(0.05)
            uv = torch.cat([(stats["fx"] * d[:, 0:1] / dz + stats["cx"]) / miss_img.shape[2],
                            (stats["fy"] * d[:, 1:2] / dz + stats["cy"]) / miss_img.shape[1]], 1) * 2 - 1
            if args.miss_padding == "smooth":
                uv = uv / 3  # the padded image spans [-3, 3] of the frame
            env = torch.nn.functional.grid_sample(miss_img[None], uv.clamp(-3, 3)[None, :, None, :].to(miss_img.dtype),
                                                  mode="bilinear",
                                                  padding_mode="reflection" if args.miss_padding == "reflection" else "border",
                                                  align_corners=False)[0, :, :, 0].T
            return black + t * env.reshape(*t.shape[:-1], 3).to(white.dtype)
        if (args.miss_background and tuple(bg) == tuple(args.background)
                and tuple(bg) not in ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))):
            # The model as trained, on white (its residual transmittance keeps
            # white under the painting); --background only where a ray misses
            # the painting altogether.
            white = trace(o, d, w, h, (1.0, 1.0, 1.0))
            black = trace(o, d, w, h, (0.0, 0.0, 0.0))
            t = (white - black).mean(-1, keepdim=True).clamp(0, 1)
            miss = ((t - 0.5) / 0.4).clamp(0, 1)
            miss = miss * miss * (3 - 2 * miss)
            bg_t = torch.as_tensor(bg, dtype=white.dtype, device=white.device)
            return white + miss * t * (bg_t - 1.0)
        if back is not None:
            # Front over back: the front's own transmittance (white - black)
            # lets the background layer through, the layer lets bg through.
            black = render(renderer, model, o, d, w, h, (0.0, 0.0, 0.0), torch)
            white = render(renderer, model, o, d, w, h, (1.0, 1.0, 1.0), torch)
            t = (white - black).mean(-1, keepdim=True).clamp(0, 1)
            return black + t * render(back_renderer, back, o, d, w, h, bg, torch)
        base = render(renderer, model, o, d, w, h, bg, torch)
        if inset is None:
            return base
        # The inset as trained (on white); blended out by its coverage, so its
        # holes and the band around it show the global model.
        white = render(inset_renderer, inset, o, d, w, h, (1.0, 1.0, 1.0), torch)
        black = render(inset_renderer, inset, o, d, w, h, (0.0, 0.0, 0.0), torch)
        cover = (1.0 - (white - black).mean(-1, keepdim=True)).clamp(0, 1)
        weight = ((cover - 0.5) / 0.35).clamp(0, 1)
        weight = weight * weight * (3 - 2 * weight)
        return weight * white + (1 - weight) * base

    aspect = float(np.ptp(mean_xy[:, 1]) / np.ptp(mean_xy[:, 0]))
    spheres = [VirtualSphere(spec, args.camera_angle_x, aspect, torch, device, mean_xy, mean)
               for spec in args.sphere]
    def material_ior(material):
        # MATERIAL[:IOR] overrides --sphere-ior for one ball.
        name, _, ior = material.partition(":")
        return name, float(ior) if ior else args.sphere_ior

    spheres += [VirtualSphere.at([float(x) for x in spec[:3]], float(spec[3]), material_ior(spec[4])[0],
                                 torch, device, ior=material_ior(spec[4])[1])
                for spec in args.sphere_at]
    for spec in args.sphere_path:
        # One centre per frame (reference OpenCV frame), from an (F, 3) .npy.
        path = np.load(spec[0])
        name, ior = material_ior(spec[2])
        ball = VirtualSphere.at(path[0], float(spec[1]), name, torch, device, ior=ior)
        ball.path = path
        spheres.append(ball)
    for ball in spheres:
        ball.surface_ior = args.sphere_surface_ior
    plane_light = None
    light_dir = None if args.light is None else torch.as_tensor(
        np.asarray(args.light) / np.linalg.norm(args.light), dtype=torch.float32, device=device)
    if args.light is not None:
        l_dir = torch.as_tensor(np.asarray(args.light) / np.linalg.norm(args.light),
                                dtype=torch.float32, device=device)
        for ball in spheres:
            ball.light = (l_dir, 1.0)
        if args.light_plane is not None:
            if depth is None:
                raise SystemExit("--light-plane needs --depth (the reference depth)")
            plane_light = PlaneLight(args.light, args.light_strength, args.light_angle,
                                     json.loads(args.light_plane.read_text()), depth,
                                     args.camera_angle_x,
                                     # The shadow and caustic are baked for fixed balls.
                                     [ball for ball in spheres if getattr(ball, "path", None) is None],
                                     torch, device, caustic_gain=args.caustic_gain)
    mirror = None
    if args.mirror is not None:
        mirror = ConvexMirror.from_disc(mean_xy, mean, args.mirror, args.camera_angle_x,
                                        args.mirror_bulge, args.mirror_reflectance, torch, device)
        mirror.miss = args.mirror_miss
        if args.mirror_backs:
            if args.lattice_depth is None:
                raise SystemExit("--mirror-backs needs --lattice-depth")
            source = inset_lifted if inset is not None else {
                "planar": planar, "mean_xy": mean_xy, "mean": mean, "quat": quat, "log_s": new_log_s,
                "active": active & alive}
            surface = lattice_surface(lat, args.lattice_scale, not args.lattice_no_invert,
                                      args.camera_angle_x)
            b_mean, b_quat, b_log_s, b_active, backs_stats = mirror_backs(
                mirror, source, args.mirror, surface)
            backs_model = LiftedModel(source["planar"], b_mean, b_quat, b_log_s, b_active,
                                      args.freeze_view_direction, torch,
                                      args.mirror_backs_opacity_gamma)
            mirror.backs = (FACETRenderer(8, 11.3449, width * height), backs_model)
    meshes = []
    for spec in args.mesh:
        if args.bounces <= 0:
            raise SystemExit("--mesh needs --bounces >= 1")
        from facet.lifting.meshes import VirtualMesh
        up = (np.asarray(json.loads(args.light_plane.read_text())["normal"]) if args.light_plane is not None
              else np.array([0.0, -1.0, 0.0]))
        meshes.append(VirtualMesh(spec[0], [float(x) for x in spec[1:4]], up, float(spec[4]), float(spec[5]),
                                  args.light, torch, device, args.light_angle))
    if plane_light is not None:
        plane_light.meshes = meshes
    point_light = None
    if args.point_light is not None:
        if args.bounces <= 0 or depth is None or args.lattice_depth is not None:
            raise SystemExit("--point-light needs --bounces >= 1 and a --depth (perspective) lift")
        from facet.lifting.lights import PointLight
        light_path = np.load(args.point_light[0])
        point_light = PointLight(light_path, float(args.point_light[1]), float(args.point_light[2]),
                                 float(args.point_light[3]), args.point_light_tint, args.point_light_samples,
                                 np.load(args.depth)["depth"] if args.flat_depth is None else depth,
                                 args.camera_angle_x, args.proxy_max_jump, torch, device, args.proxy_stride)
        point_light.occluders = list(spheres) + list(meshes)
        if args.point_light_surface == "facet":
            def median_t(rend, mdl, o, d):
                if getattr(rend, "_facet_lift_synced", None) is not mdl:
                    rend.sync_geometry(mdl)
                    rend._facet_lift_synced = mdl
                out = torch.empty((o.shape[0], 2), dtype=torch.float32, device=device)
                rgba = torch.cat((mdl.RGB, mdl.A), 1).contiguous()
                rend._native_renderer.GetMedianDepth_base(
                    o.float().contiguous(), d.float().contiguous(), mdl.m, mdl.s, mdl.q, rgba, mdl.k,
                    0.5, out)
                return out[:, 0]

            def surface_t(o, d):
                t = median_t(renderer, model, o, d)
                if back is not None:
                    # Through a disocclusion (no front surface): the layer behind.
                    t = torch.where(torch.isfinite(t), t, median_t(back_renderer, back, o, d))
                return t
            point_light.surface_fn = surface_t
            if back is not None:
                point_light.add_back_proxy(back_depth)
        for mesh in meshes:
            mesh.point_light = point_light
        if args.point_light_bulb > 0:
            bulb = VirtualSphere.at(light_path[0], args.point_light_bulb, "emissive", torch, device)
            bulb.path = light_path
            bulb.emit = [min(1.0, 1.1 * c) for c in args.point_light_tint]
            spheres.append(bulb)
    plane_mirror = None
    if args.plane_mirror is not None:
        plane_mirror = PlaneMirror(args.plane_mirror, mean_xy, mean, cams, args.camera_angle_x, torch, device,
                                   args.plane_mirror_eye,
                                   None if args.plane_mirror_from is None else
                                   json.loads(args.plane_mirror_from.read_text())["plane_mirror"])
        if args.plane_mirror_shadow:
            if plane_light is None:
                raise SystemExit("--plane-mirror-shadow needs --light and --light-plane")
            plane_light.mirrors = [plane_mirror]
            plane_light.mirror_sun = args.plane_mirror_sun
    indices = torch.arange(width * height, device=device)
    renders = []
    if args.deform_sine is not None:
        # The body's x range: surfels that are clearly there (opacity > 0.3).
        body = (active & alive) & (1 / (1 + np.exp(-planar.A.detach().cpu().numpy().reshape(-1))) > 0.3)
        deform_range = (float(mean[body, 0].min()), float(mean[body, 0].max()))
        deform_base = (mean.copy(), quat.copy(), new_log_s.copy())
    if args.ripple is not None:
        if args.deform_sine is not None:
            raise SystemExit("--ripple and --deform-sine are exclusive")
        r_amp, r_len, r_speed, r_t0 = args.ripple
        r_s0 = args.ripple_start_radius
        if args.ripple_origin is None:
            dvec = np.asarray(args.ripple_direction, float) / np.linalg.norm(args.ripple_direction)
            r_s0 = float((mean[active & alive, :2] @ dvec).min())
        r_taper = None
        if args.ripple_edge_taper > 0:
            body_xy = mean[active & alive, :2]
            r_taper = (float(body_xy[:, 0].min()), float(body_xy[:, 0].max()), float(body_xy[:, 1].min()),
                       float(body_xy[:, 1].max()))
            r_taper += (args.ripple_edge_taper * min(r_taper[1] - r_taper[0], r_taper[3] - r_taper[2]),)
        ripple_spec = (r_amp, r_len, r_speed, args.ripple_origin, args.ripple_direction, args.ripple_axis,
                       args.ripple_front, args.ripple_ramp, r_s0, args.ripple_along_ratio, r_taper)
        ripple_base = (mean.copy(), quat.copy(), new_log_s.copy())
        static_model = model
    cage = cage_surface = None
    if args.cage is not None:
        cage = dict(np.load(args.cage))
        cage_base = (mean.copy(), quat.copy(), new_log_s.copy())
        if args.facet_occludes_meshes or args.shade_facet is not None:
            from facet.lifting.meshes import CageSurface
            cage_surface = CageSurface(cage["frames"].shape[1] - 1, cage["frames"].shape[2] - 1, torch, device)
    frame_seconds = []
    for index, cam in enumerate(cams):
        if args.time_frames:
            import time
            torch.cuda.synchronize()
            t_frame = time.perf_counter()
        if args.ripple is not None:
            elapsed = index / args.ripple_fps - r_t0
            if elapsed <= 0:
                model = static_model
            else:
                r_mean, r_quat, r_log_s = ripple_deform(*ripple_base, elapsed, ripple_spec, args.ripple_arms)
                model = LiftedModel(planar, r_mean, r_quat, r_log_s, active & alive,
                                    args.freeze_view_direction, torch, args.opacity_gamma)
        if cage is not None:
            if cage_surface is not None:
                cage_surface.update(cage["frames"][min(index, len(cage["frames"]) - 1)])
            c_mean, c_quat, c_log_s = cage_deform(*cage_base, cage, index)
            model = LiftedModel(planar, c_mean, c_quat, c_log_s, active & alive,
                                args.freeze_view_direction, torch, args.opacity_gamma)
        if args.deform_sine is not None:
            amplitude, frequency, head, neck = args.deform_sine
            d_mean, d_quat, d_log_s = sine_deform(*deform_base, deform_range, amplitude, frequency, head, neck,
                                                  2 * math.pi * index / len(cams))
            model = LiftedModel(planar, d_mean, d_quat, d_log_s, active & alive,
                                args.freeze_view_direction, torch, args.opacity_gamma)
        if point_light is not None:
            point_light.frame = index
        for ball in spheres:
            ball.frame_now, ball.n_frames_now = index, len(cams)
        if args.mesh_hop is not None and meshes:
            hop = mesh_hop_height(index / args.ripple_fps, *args.mesh_hop)
            for mesh in meshes:
                mesh.translate(hop * np.asarray(mesh.geometry["up"], float))
        if cam.get("K") is not None:
            origins, directions = pinhole_rays(cam["c2w_cv"], cam["K"], width, height, torch, device)
        else:
            c2w = torch.as_tensor(cam["c2w_cv"], dtype=torch.float32, device=device)
            rays = generate_indexed_perspective_rays(
                indices, c2w[None, :3, 3], c2w[None, :3, 0], c2w[None, :3, 1], c2w[None, :3, 2],
                width, height, cam["fov_x"], cam["fov_y"])
            origins, directions = rays.origins, rays.directions
        def trace_all(o, d, w, h, bg, depth):
            # --bounces: the nearest virtual object wins (a z-buffer over the
            # mirror, balls and meshes), and their secondary rays see all of
            # them again, DEPTH - 1 levels deep; the lifted scene is behind.
            img = trace(o, d, w, h, bg)
            geo = None
            if args.facet_occludes_meshes or args.shade_facet is not None:
                geo = geometry_pass(renderer, model, o, d, torch)
                if cage_surface is not None:
                    # A cage: its exact surface replaces the speckled expected depth and normal.
                    dn = d / d.norm(dim=1, keepdim=True)
                    t_c, f_c, u_c, v_c = cage_surface.intersect(o, dn)
                    hit_c = torch.isfinite(t_c)
                    n_c = torch.zeros_like(geo[2])
                    if bool(hit_c.any()):
                        n_c[hit_c] = cage_surface.normal(f_c[hit_c], u_c[hit_c], v_c[hit_c]).to(n_c.dtype)
                    geo = (geo[0], torch.where(hit_c, t_c.to(geo[1].dtype), torch.full_like(geo[1], float("inf"))),
                           torch.where(hit_c[:, None], n_c, geo[2]))
            if args.shade_facet is not None and light_dir is not None:
                # Two-sided Lambert on the composited normal, weighted by coverage.
                lam = (geo[2] @ light_dir).abs()[:, None]
                factor = 1.0 - geo[0][:, None] * (1.0 - (args.shade_facet + (1.0 - args.shade_facet) * lam))
                img = (img.reshape(-1, 3) * factor).reshape(img.shape)
            if mirror is not None:
                img = mirror.apply(img, o, d, trace, bg)
            if depth <= 0:
                if point_light is not None:
                    img = point_light.apply(img, o, d, torch.ones(o.shape[0], dtype=torch.bool, device=o.device))
                return img
            zbuf = torch.full((o.shape[0],), float("inf"), device=o.device)
            if args.facet_occludes_meshes:
                # The FACET surface hides meshes behind it (default: meshes are drawn over it).
                zbuf = torch.where(geo[0] > 0.5, geo[1].float(), zbuf)
                if os.environ.get("FACET_LIFT_DUMP_GEOMETRY") and depth == args.bounces:
                    np.savez_compressed(os.environ["FACET_LIFT_DUMP_GEOMETRY"], alpha=geo[0].cpu().numpy(),
                                        dist=geo[1].cpu().numpy(), w=w, h=h, o=o.cpu().numpy(), d=d.cpu().numpy())

            def sub(o2, d2, w2, h2, bg2):
                return trace_all(o2, d2, w2, h2, bg2, depth - 1)

            if plane_mirror is not None:
                img = plane_mirror.apply(img, o, d, sub, bg, zbuf)
            for ball in spheres:
                img = ball.apply(img, o, d, sub, bg, index, len(cams), zbuf)
            for mesh in meshes:
                img = mesh.apply(img, o, d, sub, bg, zbuf)
            if plane_light is not None:
                img = plane_light.apply(img, o, d, torch.isfinite(zbuf))
            if point_light is not None:
                img = point_light.apply(img, o, d, ~torch.isfinite(zbuf))
            return img

        def shade(origins, directions):
            return shade_scene(origins, directions)

        def shade_scene(origins, directions):
            if args.bounces > 0:
                return trace_all(origins, directions, width, height, tuple(args.background), args.bounces)
            img = trace(origins, directions, width, height, tuple(args.background))
            if mirror is not None:
                img = mirror.apply(img, origins, directions, trace, tuple(args.background))
            if plane_mirror is not None:
                img = plane_mirror.apply(img, origins, directions, trace, tuple(args.background))
            skip = None
            for ball in spheres:
                img = ball.apply(img, origins, directions, trace, tuple(args.background),
                                 index, len(cams))
                skip = ball.last_hit if skip is None else (skip | ball.last_hit)
            if plane_light is not None:
                img = plane_light.apply(img, origins, directions, skip)
            return img

        aperture = cam.get("aperture_radius")
        aperture = args.aperture if aperture is None else float(aperture)
        with torch.no_grad():
            if args.dof_samples > 0 and aperture > 0:
                # Thin lens: every ray keeps its point on the focal plane
                # (FOCUS along the view axis) and starts from a lens sample.
                focus = cam.get("focus_distance")
                focus = args.focus_distance if focus is None else float(focus)
                c2w = torch.as_tensor(cam["c2w_cv"], dtype=torch.float32, device=device)
                right, down, forward = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]
                unit_z = directions / (directions @ forward)[:, None]
                px_x = 2 * math.tan(0.5 * cam["fov_x"]) / width
                px_y = 2 * math.tan(0.5 * cam["fov_y"]) / height
                img = 0
                n = args.dof_samples
                for k in range(n):
                    # Subpixel jitter (Halton 2, 3) antialiases the in-focus part.
                    jx, jy = halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5
                    focal = origins + focus * (unit_z + jx * px_x * right + jy * px_y * down)
                    # Golden-angle (Vogel) disc: even coverage, deterministic.
                    r = aperture * math.sqrt((k + 0.5) / n)
                    phi = k * math.pi * (3 - math.sqrt(5))
                    o = origins + r * math.cos(phi) * right + r * math.sin(phi) * down
                    d = focal - o
                    img = img + shade(o.contiguous(), (d / d.norm(dim=1, keepdim=True)).contiguous())
                img = img / n
            else:
                img = shade(origins, directions)
        if ss > 1:
            img = img.reshape(out_h, ss, out_w, ss, 3).mean((1, 3))
        if args.time_frames:
            torch.cuda.synchronize()
            frame_seconds.append(time.perf_counter() - t_frame)
        pixels = img.clamp(0, 1).mul(255).round().byte().cpu().numpy()
        Image.fromarray(pixels).save(args.output / f"{index:05d}.png")
        renders.append(img)

    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": ckpt_sha_before,
              "checkpoint_unchanged": sha256(args.checkpoint) == ckpt_sha_before,
              "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "lift": stats, "active_after_lift": int((active & alive).sum()),
              "targets": [c["file_path"] for c in cams],
              # What replay_render.py needs besides the options.
              "renderer_directory": os.environ.get("FACET_RENDERER_DIRECTORY"),
              "git_commit": _git_commit()}
    if args.time_frames:
        report["frame_seconds"] = frame_seconds
    if mirror is not None:
        report["mirror_reflection_coverage"] = mirror.coverage
    if plane_mirror is not None:
        report["plane_mirror"] = plane_mirror.geometry
    if meshes:
        report["meshes"] = [m.geometry for m in meshes]
    if point_light is not None:
        report["point_light"] = point_light.geometry
    if args.identity_test:
        # Original orthographic 2D render of the untouched planar model.
        ortho = generate_indexed_orthographic_rays(
            indices, width, height, plane_extent_x=image_cfg.get("plane_extent_x", 2.0),
            plane_z=image_cfg.get("plane_z", 0.0), camera_distance=image_cfg.get("camera_distance", 1.0))
        flat = LiftedModel(planar, planar.m.detach().cpu().numpy(), planar.q.detach().cpu().numpy(),
                           log_s, alive, False, torch)
        with torch.no_grad():
            base = render(renderer, flat, ortho.origins, ortho.directions, width, height,
                          tuple(args.background), torch)
        Image.fromarray(base.clamp(0, 1).mul(255).round().byte().cpu().numpy()).save(
            args.output / "orthographic_reference.png")
        diff = (renders[0] - base).abs()
        mse = float(((renders[0].clamp(0, 1) - base.clamp(0, 1)) ** 2).mean())
        report["identity"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                              "psnr_vs_orthographic": -10 * math.log10(max(mse, 1e-20))}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "options"}, indent=2))
    return 0 if report["checkpoint_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
