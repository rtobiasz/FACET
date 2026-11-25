"""A moving point light with cast shadows in a lifted photo (facet/lifting/lift.py --point-light).

The colour stays FACET's; the geometry for the lighting is a proxy triangle
mesh built from the same depth map the lift uses (reference camera, pixel
centres, OpenCV frame), without the triangles that span a depth jump (the
lift snaps those surfels to one side, so there is no surface between a
chair and the wall behind it). Traced with NVIDIA Warp like facet/lifting/meshes.py.

Every ray that ends on the lifted scene gets
    colour * (DIM + E),   E = POWER * max(0, n.l) / (d^2 + r^2) * V * tint,
the photo dimmed to DIM (its own lighting stays in it, dimmed) plus the
light at distance d; V is the fraction of SAMPLES points of the light's
sphere (radius r) seen from the hit, past the proxy (the lifted scene's own
shadows: chair on the wall, plant on the window frame), the virtual balls
(glass lets 60 % through) and meshes. The same terms light the meshes, so
the car is lit and shadowed by the moving light too.
"""
from __future__ import annotations

import math

import numpy as np


def proxy_mesh(depth, camera_angle_x, max_jump, stride=2, normal_sigma=3.0):
    """Vertices (V, 3), faces (F, 3) over the depth map (reference OpenCV frame), and
    vertices of a smoothed copy for the shading normals.

    Depth Pro's depth is fine for shadows but its per-pixel noise gives a
    wrinkled wall under a close light; the normals come from the depth
    smoothed by NORMAL_SIGMA proxy pixels, except near depth edges (where the
    smoothed depth moves by more than 2 %), which keep the raw normal.
    """
    from scipy import ndimage

    h, w = depth.shape
    f = 0.5 * w / math.tan(0.5 * camera_angle_x)
    if stride > 1:
        depth = ndimage.zoom(depth, 1.0 / stride, order=1)
    hs, ws = depth.shape
    sy, sx = h / hs, w / ws
    vv, uu = np.mgrid[0:hs, 0:ws]
    u = (uu + 0.5) * sx
    v = (vv + 0.5) * sy
    z = depth.astype(np.float64)
    smooth = ndimage.gaussian_filter(z, normal_sigma)
    smooth = np.where(np.abs(smooth - z) / z > 0.02, z, smooth)

    def back_project(zz):
        return np.stack([(u - w / 2) / f * zz, (v - h / 2) / f * zz, zz], -1).reshape(-1, 3)

    verts, verts_smooth = back_project(z), back_project(smooth)
    idx = np.arange(hs * ws).reshape(hs, ws)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    zf = z.reshape(-1)[faces]
    keep = (zf.max(1) - zf.min(1)) / zf.min(1) <= max_jump
    return verts.astype(np.float32), faces[keep].astype(np.int32), verts_smooth.astype(np.float32)


class PointLight:
    def __init__(self, path, power, radius, dim, tint, samples, depth, camera_angle_x, max_jump,
                 torch, device, stride=2):
        from facet.lifting.meshes import _kernels
        k = _kernels()
        wp = k["wp"]
        self.torch, self.device = torch, device
        self.path = np.asarray(path, np.float64)
        self.power, self.radius, self.dim = float(power), float(radius), float(dim)
        self.tint = torch.as_tensor(tint, dtype=torch.float32, device=device)
        self.samples = int(samples)
        self.frame = 0
        verts, faces, verts_smooth = proxy_mesh(np.asarray(depth, np.float64), camera_angle_x, max_jump, stride)
        self.wp_points = wp.array(verts, dtype=wp.vec3, device="cuda:0")
        self.wp_faces = wp.array(faces.reshape(-1), dtype=int, device="cuda:0")
        self.mesh = wp.Mesh(points=self.wp_points, indices=self.wp_faces)
        self.v = torch.as_tensor(verts, device=device)
        self.faces = torch.as_tensor(faces.astype(np.int64), device=device)
        tri = torch.as_tensor(verts_smooth, device=device)[self.faces]
        fn = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.face_normal = fn / fn.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # Points on the light's sphere (Fibonacci), fixed: no flicker between frames.
        i = np.arange(self.samples) + 0.5
        phi = np.arccos(1 - 2 * i / self.samples)
        theta = math.pi * (1 + 5 ** 0.5) * i
        pts = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
        self.offsets = torch.as_tensor(pts * self.radius * 0.8, dtype=torch.float32, device=device)
        self.occluders = []  # balls and meshes, set by facet/lifting/lift.py
        self.emitter = None  # the ball drawn at the light, never an occluder
        # --point-light-surface facet: (o, d) -> distance to the surface FACET
        # shows on each ray (its median depth); None: the proxy's first hit.
        self.surface_fn = None
        self._proxy_args = (camera_angle_x, max_jump, stride)
        self.back_mesh = None
        self.geometry = {"proxy_vertices": int(len(verts)), "proxy_faces": int(len(faces)),
                         "power": power, "radius": radius, "dim": dim, "tint": list(tint)}
        print(f"point light: proxy {len(verts)} vertices, {len(faces)} faces (depth jumps > {max_jump} cut)")

    def position(self):
        return self.torch.as_tensor(self.path[self.frame % len(self.path)], dtype=self.torch.float32,
                                    device=self.device)

    def add_back_proxy(self, depth):
        """facet/lifting/lift.py --background-layer: the layer's own proxy, for the
        normals of the points seen through a disocclusion (shadows still come
        from the main proxy only)."""
        from facet.lifting.meshes import _kernels
        wp = _kernels()["wp"]
        verts, faces, verts_smooth = proxy_mesh(np.asarray(depth, np.float64), self._proxy_args[0],
                                                self._proxy_args[1], self._proxy_args[2])
        self.back_wp = (wp.array(verts, dtype=wp.vec3, device="cuda:0"),
                        wp.array(faces.reshape(-1), dtype=int, device="cuda:0"))
        self.back_mesh = wp.Mesh(points=self.back_wp[0], indices=self.back_wp[1])
        tri = self.torch.as_tensor(verts_smooth, device=self.device)[self.torch.as_tensor(
            faces.astype(np.int64), device=self.device)]
        fn = self.torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.back_normal = fn / fn.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.geometry["back_proxy_faces"] = int(len(faces))

    def _cast(self, o, d, tmax, mesh=None):
        from facet.lifting.meshes import _kernels
        k = _kernels()
        wp, torch = k["wp"], self.torch
        n = o.shape[0]
        t_out = torch.empty(n, device=self.device)
        f_out = torch.empty(n, dtype=torch.int32, device=self.device)
        u_out = torch.zeros(n, device=self.device)
        v_out = torch.zeros(n, device=self.device)
        torch.cuda.synchronize()
        wp.launch(k["cast"], dim=n, device="cuda:0",
                  inputs=[(mesh or self.mesh).id, wp.from_torch(o.float().contiguous(), dtype=wp.vec3),
                          wp.from_torch(d.float().contiguous(), dtype=wp.vec3),
                          wp.from_torch(tmax.float().contiguous()), wp.from_torch(t_out), wp.from_torch(f_out),
                          wp.from_torch(u_out), wp.from_torch(v_out)])
        wp.synchronize_device("cuda:0")
        return torch.where(f_out >= 0, t_out, torch.full_like(t_out, float("inf"))), f_out.long()

    def terms(self, p, n, view, shine=None):
        """Irradiance (N, 3) from the light at points p with normals n, and a highlight (N, 1)."""
        torch = self.torch
        centre = self.position()
        eps = 2e-3 * p[:, 2:3].abs().clamp_min(0.2)
        start = p + eps * n
        spec = torch.zeros(p.shape[0], 1, device=self.device)
        to_c = centre - p
        dist2 = (to_c * to_c).sum(1)
        lam = ((n * to_c).sum(1) / dist2.sqrt().clamp_min(1e-6)).clamp_min(0)
        vis = torch.zeros(p.shape[0], device=self.device)
        for s in range(self.samples):
            target = centre + self.offsets[s]
            dvec = target - start
            length = dvec.norm(dim=1)
            dirn = dvec / length[:, None].clamp_min(1e-9)
            t, _ = self._cast(start, dirn, length)
            trans = (~torch.isfinite(t)).float()
            for obj in self.occluders:
                trans = trans * obj.transmittance(start, dirn, length)
            vis += trans
        vis /= self.samples
        irr = self.power * lam * vis / (dist2 + self.radius ** 2)
        if shine is not None:
            half = to_c / dist2.sqrt()[:, None].clamp_min(1e-6) - view
            half = half / half.norm(dim=1, keepdim=True).clamp_min(1e-9)
            spec = (n * half).sum(1, keepdim=True).clamp_min(0) ** shine[:, None] * (vis * self.power / (
                dist2 + self.radius ** 2))[:, None]
        return irr[:, None] * self.tint, spec

    def apply(self, img, origins, directions, mask):
        """Relight rays (mask) that end on the lifted scene."""
        torch = self.torch
        flat = img.reshape(-1, 3).clone()
        idx = torch.nonzero(mask).squeeze(1)
        if idx.numel() == 0:
            return img
        o = origins[idx]
        d = directions[idx] / directions[idx].norm(dim=1, keepdim=True)
        if self.surface_fn is None:
            t, face = self._cast(o, d, torch.full((idx.numel(),), 1e6, device=self.device))
        else:
            # The point FACET shows, not the proxy's first hit: next to an edge
            # the proxy ray can slip through a cut (onto the wall, in the
            # object's shadow) or hit the stride-quantised mesh off the surface.
            # The normal is the proxy face met within a short span around it.
            t = self.surface_fn(o, d)
            span = 0.03 * torch.nan_to_num(t, posinf=1.0).clamp_min(0.2)
            _, face = self._cast(o + (t - span).clamp_min(0)[:, None] * d, d, 2 * span)
            near = torch.isfinite(t) & (face >= 0)
            normals = self.face_normal[face.clamp_min(0)]
            if self.back_mesh is not None and not bool(near.all()):
                # FACET's surface is the background layer: its proxy's face there.
                miss = torch.nonzero(~near & torch.isfinite(t)).squeeze(1)
                if miss.numel():
                    _, face_b = self._cast(o[miss] + (t[miss] - span[miss]).clamp_min(0)[:, None] * d[miss],
                                           d[miss], 2 * span[miss], self.back_mesh)
                    ok = face_b >= 0
                    normals = normals.clone()
                    normals[miss[ok]] = self.back_normal[face_b[ok]]
                    near = near.clone()
                    near[miss[ok]] = True
                    face = face.clone()
                    face[miss[ok]] = 0  # any valid index: the normal is taken from normals
            if not bool(near.all()):
                # No proxy face at FACET's surface (a cut, a hole): the proxy's own first hit.
                t_p, face_p = self._cast(o[~near], d[~near], torch.full(((~near).sum(),), 1e6, device=self.device))
                t, face = t.clone(), face.clone()
                t[~near], face[~near] = t_p, face_p
                normals = normals.clone()
                normals[~near] = self.face_normal[face_p.clamp_min(0)]
            t = torch.where(face >= 0, t, torch.full_like(t, float("inf")))
        hit = torch.isfinite(t)
        scale = torch.full((idx.numel(), 3), self.dim, device=self.device)
        if bool(hit.any()):
            p = o[hit] + t[hit, None] * d[hit]
            n = (self.face_normal[face.clamp_min(0)] if self.surface_fn is None else normals)[hit]
            n = torch.where(((n * d[hit]).sum(1, keepdim=True) > 0), -n, n)
            irr, _ = self.terms(p, n, d[hit])
            scale[hit] = self.dim + irr
        flat[idx] = flat[idx] * scale
        return flat.reshape(img.shape)
