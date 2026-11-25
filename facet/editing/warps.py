"""Geometric image edits defined in pixel coordinates (x right, y down).

One warp is applied unchanged to FACET's primitive centres and to GaINeR's
means, so both methods get the same deformation; the Jacobian of the warp is
the deformation gradient that FACET uses to carry its ellipses along.

Each warp takes `t` in [0, 1] (0 = identity, 1 = the full edit) so that an
edit can also be rendered as an animation. Specs are JSON objects:

  {"type": "bend", "base": [x, y], "length": L, "angle": degrees}
      Bends what lies above `base` like a beam: the axis through `base`
      (pointing up) becomes a circular arc of length L that turns by `angle`
      (positive = to the right); points above the arc continue rigidly.
  {"type": "joint", "pivot": [x, y], "direction": [dx, dy], "falloff": f,
   "width": w, "angle": degrees}
      Rotates a limb about a joint: the weight rises smoothly from 0 at the
      pivot to 1 at distance f along `direction` and falls to 0 between w and 1.6 w from
      the limb's axis; each point rotates by weight * angle about the pivot
      (positive = clockwise on screen).
  {"type": "cut", "point": [x, y], "normal": [nx, ny], "translate": [tx, ty],
   "angle": degrees}
      Moves the side of the line through `point` that `normal` points to as a
      rigid body: rotation by `angle` about `point`, then translation.
  {"type": "chain", "warps": [spec, ...]}
      Applies the warps in order.
  {"type": "smudge", "strokes": [{"path": [[x, y], ...], "radius": R, "strength": s}, ...],
   "step": d}
      A finger-paint brush: the brush centre moves along each stroke's polyline in
      steps of d pixels (default R / 4), and every step moves the points near it by
      s * (1 - (r / R)^2)^2 times the step (r = distance to the centre), so the
      points it picks up are dragged along. With s < 1 and d <= R / 4 each step is
      one-to-one. t is the fraction of all strokes the brush has travelled. unwarp_points
      inverts it step by step (fixed-point iteration per step).

Three global warps with a closed-form inverse (unwarp_points), so that the
warped source image can serve as a reference for the transport control:

  {"type": "rotate", "centre": [x, y], "angle": degrees}
      Rotation about `centre` (positive = clockwise on screen).
  {"type": "scale", "centre": [x, y], "factors": [sx, sy]}
      Axis-aligned stretch about `centre`.
  {"type": "shear", "centre": [x, y], "factor": k}
      Horizontal shear about `centre`: x' = x + k (y - cy), y' = y.
  {"type": "swirl", "centre": [x, y], "radius": R, "angle": degrees}
      Rotation about `centre` by angle * (1 - r / R)^2 inside radius R; the
      radius of every point is kept, so the inverse turns back by the same angle.
"""

from __future__ import annotations

import math

import torch


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _rotate(points: torch.Tensor, centre: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    # Clockwise on screen with y pointing down.
    offset = points - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    return centre + torch.stack(
        (cos * offset[:, 0] - sin * offset[:, 1], sin * offset[:, 0] + cos * offset[:, 1]), dim=1
    )


def _bend(points: torch.Tensor, spec: dict, t: float) -> torch.Tensor:
    bx, by = spec["base"]
    length = float(spec["length"])
    total = math.radians(float(spec["angle"])) * t
    lateral = points[:, 0] - bx
    height = by - points[:, 1]  # up is positive
    if abs(total) < 1e-9:
        return points.clone()
    curvature = total / length
    along = height.clamp(0.0, length)
    rest = (height - length).clamp_min(0.0)
    phi = curvature * along
    # Arc axis in (right, up) coordinates, then the lateral offset along the
    # rotated normal and the rigid continuation along the end tangent.
    axis_right = (1.0 - torch.cos(phi)) / curvature
    axis_up = torch.sin(phi) / curvature
    right = axis_right + lateral * torch.cos(phi) + rest * math.sin(total)
    up = axis_up - lateral * torch.sin(phi) + rest * math.cos(total)
    bent = torch.stack((bx + right, by - up), dim=1)
    above = (height > 0.0).unsqueeze(1)
    return torch.where(above, bent, points)


def _joint(points: torch.Tensor, spec: dict, t: float) -> torch.Tensor:
    pivot = points.new_tensor(spec["pivot"])
    direction = points.new_tensor(spec["direction"])
    direction = direction / direction.norm()
    offset = points - pivot
    along = offset @ direction
    across = (offset[:, 0] * direction[1] - offset[:, 1] * direction[0]).abs()
    width = float(spec["width"])
    weight = _smoothstep(along / float(spec["falloff"])) * _smoothstep((1.6 * width - across) / (0.6 * width))
    return _rotate(points, pivot, weight * math.radians(float(spec["angle"])) * t)


def _cut(points: torch.Tensor, spec: dict, t: float) -> torch.Tensor:
    point = points.new_tensor(spec["point"])
    normal = points.new_tensor(spec["normal"])
    side = ((points - point) @ normal) > 0.0
    moved = _rotate(points, point, points.new_tensor(math.radians(float(spec.get("angle", 0.0))) * t))
    moved = moved + points.new_tensor(spec.get("translate", [0.0, 0.0])) * t
    return torch.where(side.unsqueeze(1), moved, points)


def _brush_steps(spec: dict) -> list[tuple[list[float], list[float], float, float]]:
    """(centre, step, radius, strength) for every brush step of a smudge spec, in order."""
    steps = []
    for stroke in spec["strokes"]:
        radius, strength = float(stroke["radius"]), float(stroke.get("strength", 0.8))
        spacing = float(spec.get("step", radius / 4.0))
        path = stroke["path"]
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            count = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing))
            dx, dy = (x1 - x0) / count, (y1 - y0) / count
            steps += [([x0 + k * dx, y0 + k * dy], [dx, dy], radius, strength) for k in range(count)]
    return steps


def _smudge(points: torch.Tensor, spec: dict, t: float) -> torch.Tensor:
    steps = _brush_steps(spec)
    travelled = t * len(steps)
    for index, (centre, step, radius, strength) in enumerate(steps):
        fraction = min(1.0, travelled - index)
        if fraction <= 0.0:
            break
        r2 = ((points - points.new_tensor(centre)) ** 2).sum(1) / (radius * radius)
        weight = strength * (1.0 - r2).clamp_min(0.0) ** 2
        points = points + weight.unsqueeze(1) * points.new_tensor(step) * fraction
    return points


def _swirl_angle(points: torch.Tensor, spec: dict, t: float) -> torch.Tensor:
    radius = (points - points.new_tensor(spec["centre"])).norm(dim=1)
    falloff = (1.0 - radius / float(spec["radius"])).clamp_min(0.0)
    return math.radians(float(spec["angle"])) * t * falloff * falloff


def _global(points: torch.Tensor, spec: dict, t: float, sign: float) -> torch.Tensor:
    centre = points.new_tensor(spec["centre"])
    kind = spec["type"]
    if kind == "rotate":
        return _rotate(points, centre, points.new_tensor(sign * math.radians(float(spec["angle"])) * t))
    if kind == "scale":
        factors = 1.0 + (points.new_tensor(spec["factors"]) - 1.0) * t
        return centre + (points - centre) * (factors if sign > 0 else 1.0 / factors)
    if kind == "shear":
        k = sign * float(spec["factor"]) * t
        return torch.stack((points[:, 0] + k * (points[:, 1] - centre[1]), points[:, 1]), 1)
    return _rotate(points, centre, sign * _swirl_angle(points, spec, t))


def _unsmudge(points: torch.Tensor, spec: dict, t: float, iterations: int = 30) -> torch.Tensor:
    # Undo the brush steps in reverse order; each step p' = p + w(p) d is inverted by the
    # fixed-point iteration p = p' - w(p) d, a contraction for strength < 1 and step <= R / 4.
    steps = _brush_steps(spec)
    travelled = t * len(steps)
    for index in reversed(range(len(steps))):
        fraction = min(1.0, travelled - index)
        if fraction <= 0.0:
            continue
        centre, step, radius, strength = steps[index]
        centre, step = points.new_tensor(centre), points.new_tensor(step) * fraction
        target, source = points, points
        for _ in range(iterations):
            r2 = ((source - centre) ** 2).sum(1) / (radius * radius)
            source = target - (strength * (1.0 - r2).clamp_min(0.0) ** 2).unsqueeze(1) * step
        points = source
    return points


def unwarp_points(points: torch.Tensor, spec: dict, t: float = 1.0) -> torch.Tensor:
    """Inverse of warp_points for the rotate, scale, swirl, shear and smudge warps."""
    if spec["type"] == "smudge":
        return _unsmudge(points, spec, t)
    if spec["type"] not in ("rotate", "scale", "swirl", "shear"):
        raise ValueError(f"no closed-form inverse for warp type: {spec['type']}")
    return _global(points, spec, t, -1.0)


def warp_points(points: torch.Tensor, spec: dict, t: float = 1.0) -> torch.Tensor:
    """Warp (N, 2) pixel positions; differentiable in `points`."""
    kind = spec["type"]
    if kind in ("rotate", "scale", "swirl", "shear"):
        return _global(points, spec, t, 1.0)
    if kind == "bend":
        return _bend(points, spec, t)
    if kind == "joint":
        return _joint(points, spec, t)
    if kind == "cut":
        return _cut(points, spec, t)
    if kind == "smudge":
        return _smudge(points, spec, t)
    if kind == "chain":
        for item in spec["warps"]:
            points = warp_points(points, item, t)
        return points
    raise ValueError(f"unknown warp type: {kind}")


def warp_with_jacobian(points: torch.Tensor, spec: dict, t: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Warped positions and the per-point 2x2 Jacobian d warp / d point."""
    source = points.detach().clone().requires_grad_(True)
    warped = warp_points(source, spec, t)
    rows = []
    for component in range(2):
        (gradient,) = torch.autograd.grad(warped[:, component].sum(), source, retain_graph=component == 0)
        rows.append(gradient)
    jacobian = torch.stack(rows, dim=1)  # (N, 2 outputs, 2 inputs)
    return warped.detach(), jacobian.detach()
