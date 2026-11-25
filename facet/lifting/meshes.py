"""A triangle-mesh prop in the lifted scene (facet/lifting/lift.py --mesh), traced with NVIDIA Warp.

The mesh (npz with vertices, faces and per-part materials: x right, y up, z forward, metres,
ground at y = 0) is scaled to LENGTH along z, turned by YAW about the up
vector and stood on a point of the scene; its BVH lives in Warp, rays come
from torch. Shading, per part material (the material modes of the mesh file):
  metal with kd > 0.5  car paint: diffuse colour, clear-coat Fresnel (f0 0.04)
                       reflection and a highlight;
  metal                chrome (rims): tinted mirror;
  ceramic              dark plastic and rubber: diffuse, weak reflection;
  glass                Fresnel reflection plus tinted straight transmission.
Diffuse light is ambient plus the --light direction, shadowed by the mesh
itself; secondary rays go through the caller's trace, so the prop sees the
lifted scene and the other virtual objects (facet/lifting/lift.py --bounces).
Reference camera OpenCV frame (y down).
"""
from __future__ import annotations

import json
import math

import numpy as np

_KERNELS = {}


def _kernels():
    if _KERNELS:
        return _KERNELS
    import warp as wp

    wp.init()

    @wp.kernel
    def cast(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
             tmax: wp.array(dtype=float), t_out: wp.array(dtype=float), face_out: wp.array(dtype=int),
             u_out: wp.array(dtype=float), v_out: wp.array(dtype=float)):
        i = wp.tid()
        q = wp.mesh_query_ray(mesh, origins[i], directions[i], tmax[i])
        if q.result:
            t_out[i] = q.t
            face_out[i] = q.face
            u_out[i] = q.u
            v_out[i] = q.v
        else:
            t_out[i] = 1.0e30
            face_out[i] = -1

    _KERNELS["cast"] = cast
    _KERNELS["wp"] = wp
    return _KERNELS


class VirtualMesh:
    def __init__(self, npz_path, contact, up, yaw_deg, length, light, torch, device, light_angle_deg=3.0):
        k = _kernels()
        wp = k["wp"]
        data = np.load(npz_path)
        v, n = data["vertices"].astype(np.float64), data["normals"].astype(np.float64)
        self.materials = json.loads(str(data["materials"]))
        scale = length / float(np.ptp(v[:, 2]))
        up = np.asarray(up, float)
        up /= np.linalg.norm(up)
        # Car forward: the camera's +z (away from it) turned by YAW about up.
        fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
        if np.linalg.norm(fwd) < 1e-6:
            # Up along the view axis (a prop standing on a print seen from above):
            # forward is the image's down direction instead.
            fwd = np.array([0.0, 1.0, 0.0]) - up[1] * up
        fwd /= np.linalg.norm(fwd)
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        side = np.cross(up, fwd)
        fwd = c * fwd + s * side
        right = np.cross(up, fwd)  # car x, so that (x, y, z) = (right, up, forward) stays right-handed
        rot = np.stack([right, up, fwd], 1)  # columns: car axes in the scene
        self.points = (v * scale) @ rot.T + np.asarray(contact, float)
        normals = n @ rot.T
        self.torch, self.device = torch, device
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
        self.v = t(self.points)
        self.vn = t(normals)
        self.faces = t(data["faces"].astype(np.int64), torch.long)
        self.part = t(data["part"].astype(np.int64), torch.long)
        self.wp_points = wp.array(self.points.astype(np.float32), dtype=wp.vec3, device="cuda:0")
        self.wp_faces = wp.array(data["faces"].astype(np.int32).reshape(-1), dtype=int, device="cuda:0")
        self.mesh = wp.Mesh(points=self.wp_points, indices=self.wp_faces)
        self.light = None if light is None else t(np.asarray(light, float) / np.linalg.norm(light))
        self.half = math.radians(light_angle_deg) / 2
        mats = self.materials
        self.rgb = t([m["rgb"] for m in mats])
        self.kd = t([m.get("kd", 0.9) for m in mats])
        self.eta = t([m.get("eta", 1.45) for m in mats])
        self.shine = t([m.get("n", 16.0) for m in mats])
        mode = [m["mode"] for m in mats]
        self.is_paint = t([md == "metal" and m.get("kd", 0) > 0.5 for md, m in zip(mode, mats)], torch.bool)
        self.is_chrome = t([md == "metal" and m.get("kd", 0) <= 0.5 for md, m in zip(mode, mats)], torch.bool)
        # Parts named "rims" are "metal" with kd 0.9: chrome by name.
        for i, m in enumerate(mats):
            if m["id"] == "rims":
                self.is_paint[i], self.is_chrome[i] = False, True
        self.is_glass = t([md == "glass" for md in mode], torch.bool)
        self.geometry = {"contact": list(map(float, contact)), "up": up.tolist(), "yaw_deg": yaw_deg,
                         "length": length, "scale": scale, "bounds_min": self.points.min(0).tolist(),
                         "bounds_max": self.points.max(0).tolist()}
        print(f"mesh: {len(self.points)} vertices, {self.faces.shape[0]} faces, scale {scale:.4f}, "
              f"bounds {np.round(self.points.min(0), 3)} .. {np.round(self.points.max(0), 3)}")

    def translate(self, offset):
        """Move the mesh rigidly to its rest position + OFFSET (scene units); refits the BVH."""
        offset = np.asarray(offset, float)
        if np.array_equal(offset, getattr(self, "_offset", np.zeros(3))):
            return
        self._offset = offset
        moved = (self.points + offset).astype(np.float32)
        self.wp_points.assign(moved)
        self.mesh.refit()
        self.v = self.torch.as_tensor(moved, device=self.device)

    def intersect(self, origins, directions, tmax=None):
        """Nearest hit along unit directions: t (inf on a miss), face, barycentrics u, v."""
        k = _kernels()
        wp, torch = k["wp"], self.torch
        count = origins.shape[0]
        o = origins.float().contiguous()
        d = directions.float().contiguous()
        tm = torch.full((count,), 1.0e6, device=self.device) if tmax is None else tmax.float().contiguous()
        t_out = torch.empty(count, device=self.device)
        f_out = torch.empty(count, dtype=torch.int32, device=self.device)
        u_out = torch.zeros(count, device=self.device)
        v_out = torch.zeros(count, device=self.device)
        torch.cuda.synchronize()
        wp.launch(k["cast"], dim=count, device="cuda:0",
                  inputs=[self.mesh.id, wp.from_torch(o, dtype=wp.vec3), wp.from_torch(d, dtype=wp.vec3),
                          wp.from_torch(tm), wp.from_torch(t_out), wp.from_torch(f_out),
                          wp.from_torch(u_out), wp.from_torch(v_out)])
        wp.synchronize_device("cuda:0")
        t_out = torch.where(f_out >= 0, t_out, torch.full_like(t_out, float("inf")))
        return t_out, f_out.long(), u_out, v_out

    def _normal(self, face, u, v):
        tri = self.faces[face]
        # Warp's barycentrics: p = u p0 + v p1 + (1 - u - v) p2.
        n = (u[:, None] * self.vn[tri[:, 0]] + v[:, None] * self.vn[tri[:, 1]]
             + (1 - u - v)[:, None] * self.vn[tri[:, 2]])
        return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def light_visibility(self, points, samples=6):
        """Fraction of the (soft) light seen from points past the mesh."""
        torch = self.torch
        if self.light is None:
            return torch.ones(points.shape[0], device=self.device)
        l = self.light
        a = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        if abs(float(l[0])) > 0.9:
            a = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        b1 = torch.linalg.cross(l, a)
        b1 = b1 / b1.norm()
        b2 = torch.linalg.cross(l, b1)
        seen = torch.zeros(points.shape[0], device=self.device)
        for k in range(samples):
            r = self.half * math.sqrt((k + 0.5) / samples)
            phi = k * math.pi * (3 - math.sqrt(5))
            dirn = l + math.tan(r) * (math.cos(phi) * b1 + math.sin(phi) * b2)
            dirn = dirn / dirn.norm()
            t, _, _, _ = self.intersect(points, dirn.expand_as(points))
            seen += (~torch.isfinite(t)).float()
        return seen / samples

    def transmittance(self, origins, directions, tmax):
        """Shadow rays (unit directions) for a point light: 0 where the mesh is in the way."""
        t, _, _, _ = self.intersect(origins, directions, tmax)
        return (~self.torch.isfinite(t)).float()

    def apply(self, img, origins, directions, trace, background, zbuf):
        torch = self.torch
        d = directions / directions.norm(dim=1, keepdim=True)
        t, face, u, v = self.intersect(origins, d)
        hit = torch.isfinite(t) & (t < zbuf)
        if not bool(hit.any()):
            return img
        zbuf[hit] = t[hit]
        dh = d[hit]
        p = origins[hit] + t[hit, None] * dh
        n = self._normal(face[hit], u[hit], v[hit])
        n = torch.where(((n * dh).sum(1, keepdim=True) > 0), -n, n)  # towards the ray
        part = self.part[face[hit]]
        rgb, kd, eta, shine = self.rgb[part], self.kd[part], self.eta[part], self.shine[part]
        cos_i = (-(dh * n).sum(1, keepdim=True)).clamp(0, 1)
        refl = dh - 2 * (dh * n).sum(1, keepdim=True) * n
        eps = 1e-4 * max(1.0, float(p.abs().max()))

        def sub(o, dirn):
            return trace(o.contiguous(), dirn.contiguous(), 1, o.shape[0], background).reshape(-1, 3)

        # Diffuse light: ambient + the window, shadowed by the car itself.
        ambient = 0.38
        if self.light is not None:
            lam = (n @ self.light).clamp_min(0)[:, None]
            vis = self.light_visibility(p + eps * n, samples=4)[:, None]
            half_vec = self.light - dh
            half_vec = half_vec / half_vec.norm(dim=1, keepdim=True)
            spec = (n * half_vec).sum(1, keepdim=True).clamp_min(0) ** shine[:, None] * vis
        else:
            lam, vis, spec = torch.zeros_like(cos_i), torch.zeros_like(cos_i), torch.zeros_like(cos_i)
        diffuse = rgb * (ambient + 0.75 * lam * vis)
        light = getattr(self, "point_light", None)
        if light is not None:
            # facet/lifting/lights.py: the moving light instead of the window, the room dimmed.
            irr, spec = light.terms(p, n, dh, shine)
            diffuse = rgb * (light.dim + irr)
        paint, chrome, glass = self.is_paint[part], self.is_chrome[part], self.is_glass[part]
        f0 = torch.where(glass, ((eta - 1) / (eta + 1)) ** 2, torch.full_like(eta, 0.04))[:, None]
        fres = f0 + (1 - f0) * (1 - cos_i) ** 5
        col = diffuse * kd[:, None] + 0.15 * spec  # ceramic / rubber
        need_refl = paint | chrome | glass
        reflection = torch.zeros_like(col)
        if bool(need_refl.any()):
            reflection[need_refl] = sub(p[need_refl] + eps * n[need_refl], refl[need_refl])
        col = torch.where(paint[:, None], diffuse * kd[:, None] * (1 - fres) + fres * reflection + 0.6 * spec, col)
        col = torch.where(chrome[:, None], 0.8 * rgb * reflection + 0.1 * diffuse + 0.8 * spec, col)
        if bool(glass.any()):
            through = sub(p[glass] + eps * dh[glass], dh[glass])
            tint = 0.35 + 0.65 * rgb[glass]
            col[glass] = (fres[glass] * reflection[glass] + (1 - fres[glass]) * tint * through
                          + 0.8 * spec[glass])
        # Plastic parts also get a faint clear reflection.
        plastic = ~need_refl
        if bool(plastic.any()):
            col[plastic] = col[plastic] + fres[plastic] * sub(p[plastic] + eps * n[plastic], refl[plastic]) * 0.5
        flat = img.reshape(-1, 3).clone()
        flat[hit] = col
        return flat.reshape(img.shape)


class CageSurface:
    """The deformed cage grid of facet/lifting/lift.py --cage as a ray-traceable surface: the cloth's exact
    depth for z-buffering props (planar surfels stick out of a curved cage by ~ size^2 / 2r, so the
    FACET expected depth is speckled there) and its smooth normal for shading."""

    intersect = VirtualMesh.intersect

    def __init__(self, ny, nx, torch, device):
        k = _kernels()
        self.wp, self.torch, self.device = k["wp"], torch, device
        idx = np.arange((ny + 1) * (nx + 1)).reshape(ny + 1, nx + 1)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        # The split of facet_lift.cage_deform: (a, b, c) and (d, c, b).
        faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([d, c, b], -1).reshape(-1, 3)])
        self.faces = torch.as_tensor(faces.astype(np.int64), device=device)
        self.wp_faces = self.wp.array(faces.astype(np.int32).reshape(-1), dtype=int, device="cuda:0")
        self.mesh, self.shape = None, (ny + 1, nx + 1)

    def update(self, verts):
        verts = np.asarray(verts, np.float64)
        tx, ty = np.gradient(verts, axis=1), np.gradient(verts, axis=0)
        n = np.cross(tx, ty).reshape(-1, 3)
        self.vn = self.torch.as_tensor(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12),
                                       dtype=self.torch.float32, device=self.device)
        pts = verts.reshape(-1, 3).astype(np.float32)
        if self.mesh is None:
            self.wp_points = self.wp.array(pts, dtype=self.wp.vec3, device="cuda:0")
            self.mesh = self.wp.Mesh(points=self.wp_points, indices=self.wp_faces)
        else:
            self.wp_points.assign(pts)
            self.mesh.refit()

    def normal(self, face, u, v):
        return VirtualMesh._normal(self, face, u, v)
