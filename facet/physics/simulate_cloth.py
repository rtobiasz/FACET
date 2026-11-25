"""A cloth simulation that deforms a FACET print (facet/lifting/lift.py --cage).

The flat print (reference frame: x right, y down, the plane z = 1 seen by the
reference camera at the origin) becomes a cloth grid in NVIDIA Warp's
simulator (warp.sim, semi-implicit Euler, y up): the sheet is held flat,
released at --release seconds and falls away from the reference camera onto
a static sphere (--drape) or, pinned along its left edge, waves in a wind
blowing through it (--flag). Every --fps frame's vertices go back to the
reference frame and are saved as the cage npz (origin, cell, z0, frames)
plus the sphere (centre, radius) for the renderer.

Sim frame (y up, gravity -y) <- reference frame: (x, y, z)_sim =
S (x, 1 - z, y)_ref + (0, H0, 0), so the rest sheet lies in the plane y = H0
with its image rows along +z.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="cage npz")
    p.add_argument("--x-range", type=float, nargs=2, required=True, help="the print's x extent (reference frame)")
    p.add_argument("--y-range", type=float, nargs=2, required=True, help="the print's y extent (reference frame)")
    p.add_argument("--cells", type=int, nargs=2, default=(48, 64), help="grid cells along x, y")
    p.add_argument("--mode", choices=["drape", "flag"], default="drape")
    p.add_argument("--sphere", type=float, nargs=4, default=(0.0, 0.0, 0.35, 0.25),
                   metavar=("X", "Y", "DEPTH", "RADIUS"),
                   help="drape: the sphere's centre (x, y) under the print, DEPTH behind its plane, RADIUS")
    p.add_argument("--wind", type=float, nargs=3, default=(0.0, 0.0, -3.0),
                   help="flag: wind velocity in the reference frame (x right, y down, z away from the camera)")
    p.add_argument("--flutter", type=float, default=0.0,
                   help="flag: amplitude of a travelling cross-wind normal to the sheet (reference units/s)")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--release", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--substeps", type=int, default=64)
    p.add_argument("--scale", type=float, default=4.0, help="sim units per reference unit")
    p.add_argument("--stiffness", type=float, nargs=3, default=(1.0e3, 1.0e3, 10.0),
                   metavar=("TRI_KE", "TRI_KA", "EDGE_KE"), help="membrane, area and bending stiffness")
    a = p.parse_args()
    import warp as wp
    import warp.sim

    wp.init()
    S, H0 = a.scale, 4.0
    nx, ny = a.cells
    (x0, x1), (y0, y1) = a.x_range, a.y_range
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    prad = 0.25 * min(dx, dy) * S
    builder = wp.sim.ModelBuilder()
    upright = a.mode == "flag"
    if upright:
        # Flag: the sheet hangs in its own plane, gravity along the image's down direction:
        # (x, y, z)_sim = S (x, -y, z - 1)_ref + (0, H0, 0); grid rows run up from the image bottom.
        pos, rot = wp.vec3(S * x0, H0 - S * y1, 0.0), wp.quat_identity()
    else:
        # rot +90 deg about x: the grid's local (u, v, 0) -> (u, 0, v), image rows along +z.
        pos, rot = wp.vec3(S * x0, H0, S * y0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    builder.add_cloth_grid(
        pos=pos, rot=rot,
        vel=wp.vec3(0.0, 0.0, 0.0), dim_x=nx, dim_y=ny, cell_x=S * dx, cell_y=S * dy, mass=0.1,
        fix_left=(a.mode == "flag"), tri_ke=a.stiffness[0], tri_ka=a.stiffness[1], tri_kd=10.0,
        edge_ke=a.stiffness[2], edge_kd=0.0, particle_radius=prad)
    sphere = None
    if a.mode == "drape":
        sx, sy, depth, radius = a.sphere
        sphere = {"centre_ref": [sx, sy, 1.0 + depth], "radius_ref": radius}
        builder.add_shape_sphere(body=-1, pos=wp.vec3(S * sx, H0 - S * depth, S * sy), radius=S * radius - prad,
                                 ke=1.0e4, kd=1.0e2, kf=1.0e2, mu=1.0)
    model = builder.finalize()
    model.ground = False
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1.0e4, 1.0e2, 1.0
    rest_q = model.particle_q.numpy().copy()
    # The grid's particle order: row-major, x fastest.
    rest = rest_q.reshape(ny + 1, nx + 1, 3)
    assert np.allclose(rest[0, 1, 0] - rest[0, 0, 0], S * dx, atol=1e-4), "unexpected particle order"
    assert np.allclose(rest[1, 0, 1 if upright else 2] - rest[0, 0, 1 if upright else 2], S * dy, atol=1e-4), \
        "unexpected particle order"

    def to_ref(q):
        if upright:  # rows bottom-up in the sim, top-down in the image
            return np.stack([q[..., 0] / S, -(q[..., 1] - H0) / S, 1.0 + q[..., 2] / S], -1)[::-1]
        return np.stack([q[..., 0] / S, q[..., 2] / S, 1.0 - (q[..., 1] - H0) / S], -1)

    assert np.allclose(to_ref(rest)[0, 0, :2], [x0, y0], atol=1e-4), "unexpected grid placement"
    integrator = wp.sim.SemiImplicitIntegrator()
    state0, state1 = model.state(), model.state()
    gravity = model.gravity
    frame_dt = 1.0 / a.fps
    sim_dt = frame_dt / a.substeps
    wind_sim = (np.array([a.wind[0], -a.wind[1], a.wind[2]]) if upright
                else np.array([a.wind[0], -a.wind[2], a.wind[1]])) * S
    rest_x = rest_q[:, 0]
    frames = []
    n_frames = int(round(a.seconds * a.fps))
    for f in range(n_frames):
        q = state0.particle_q.numpy().reshape(ny + 1, nx + 1, 3)
        frames.append(to_ref(q).astype(np.float32))
        t = f * frame_dt
        if t < a.release:
            continue
        model.gravity = gravity
        if a.mode == "flag":
            # Wind: a drag towards the wind velocity on every free particle, gusting in time.
            gust = 1.0 + 0.5 * math.sin(2.0 * math.pi * 0.7 * t)
            v = state0.particle_qd.numpy()
            target = np.tile(wind_sim * gust, (len(v), 1))
            # A travelling cross-wind ripple along the flag (normal to the sheet) makes it flutter.
            target[:, 2] += a.flutter * S * np.sin(2.0 * math.pi * (rest_x / (S * (x1 - x0)) * 1.5 - 1.2 * t))
            v += (target - v) * min(1.0, 0.8 * frame_dt)
            v[model.particle_inv_mass.numpy() == 0] = 0.0  # the pinned edge
            state0.particle_qd.assign(v)
        for _ in range(a.substeps):
            wp.sim.collide(model, state0)
            state0.clear_forces()
            integrator.simulate(model, state0, state1, sim_dt)
            state0, state1 = state1, state0
    frames = np.stack(frames)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, origin=np.array([x0, y0]), cell=np.array([dx, dy]), z0=np.float64(1.0),
                        frames=frames)
    info = {"mode": a.mode, "frames": n_frames, "fps": a.fps, "sphere": sphere, "cells": [nx, ny],
            "max_displacement": float(np.abs(frames - frames[:1]).max())}
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
