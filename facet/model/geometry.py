"""Pure geometry helpers for the constrained FACET 2D reference layer."""

from __future__ import annotations

import math

import torch


def angles_to_planar_quaternions(theta: torch.Tensor) -> torch.Tensor:
    """Materialize scalar-first rotations around world ``+Z``."""

    if not theta.dtype.is_floating_point:
        raise TypeError("planar angles must use a floating-point dtype")
    half_angle = 0.5 * theta
    zeros = torch.zeros_like(theta)
    return torch.stack(
        (torch.cos(half_angle), zeros, zeros, torch.sin(half_angle)),
        dim=-1,
    )


def quaternion_gradient_to_angle(
    theta: torch.Tensor,
    quaternion_gradient: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact chain rule from native ``dL/dq`` to ``dL/dtheta``."""

    expected_shape = theta.shape + (4,)
    if quaternion_gradient.shape != expected_shape:
        raise ValueError(
            "quaternion gradient shape must equal theta.shape + (4,)"
        )
    if quaternion_gradient.device != theta.device:
        raise ValueError("theta and quaternion gradient must share a device")
    if quaternion_gradient.dtype != theta.dtype:
        raise ValueError("theta and quaternion gradient must share a dtype")
    half_angle = 0.5 * theta
    return 0.5 * (
        -torch.sin(half_angle) * quaternion_gradient[..., 0]
        + torch.cos(half_angle) * quaternion_gradient[..., 3]
    )


def project_quaternions_to_image_plane(
    quaternions: torch.Tensor,
    *,
    previous: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    """Project scalar-first quaternions onto rotations around world ``+Z``."""

    if quaternions.ndim < 1 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have shape (..., 4)")
    if not quaternions.dtype.is_floating_point:
        raise TypeError("quaternions must use a floating-point dtype")
    if previous is not None:
        if previous.shape != quaternions.shape:
            raise ValueError("previous quaternions must match the input shape")
        if (
            previous.device != quaternions.device
            or previous.dtype != quaternions.dtype
        ):
            raise ValueError("previous quaternions must match device and dtype")
    if eps is None:
        eps = 8.0 * torch.finfo(quaternions.dtype).eps
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    zeros = torch.zeros_like(quaternions[..., 0])
    allowed = torch.stack(
        (quaternions[..., 0], zeros, zeros, quaternions[..., 3]),
        dim=-1,
    )
    norm = torch.linalg.vector_norm(allowed, dim=-1, keepdim=True)
    projected = allowed / norm.clamp_min(eps)

    identity = torch.zeros_like(quaternions)
    identity[..., 0] = 1.0
    if previous is None:
        fallback = identity
    else:
        previous_allowed = torch.stack(
            (previous[..., 0], zeros, zeros, previous[..., 3]),
            dim=-1,
        )
        previous_norm = torch.linalg.vector_norm(
            previous_allowed, dim=-1, keepdim=True
        )
        previous_projected = previous_allowed / previous_norm.clamp_min(eps)
        fallback = torch.where(previous_norm > eps, previous_projected, identity)

    projected = torch.where(norm > eps, projected, fallback)
    if previous is not None:
        reverse = (projected * fallback).sum(dim=-1, keepdim=True) < 0.0
        projected = torch.where(reverse, -projected, projected)
    return projected.contiguous()


class PlanarCompatibilityConstraint:
    """Project the renderer's XYZ/quaternion parameters and their Adam state onto the image plane."""

    def __init__(self, plane_z: float = 0.0) -> None:
        if not math.isfinite(plane_z):
            raise ValueError("plane_z must be finite")
        self.plane_z = float(plane_z)
        self._previous_quaternions: torch.Tensor | None = None

    @staticmethod
    def _optimizer_state(optimizer, parameter):
        if optimizer is None:
            return {}
        return optimizer.state.get(parameter, {})

    def apply(self, model, optimizer=None) -> None:
        if model.m.ndim != 2 or model.m.shape[1] != 3:
            raise ValueError("compatibility means must have shape (N, 3)")
        if model.q.shape != (model.m.shape[0], 4):
            raise ValueError("compatibility quaternions must have shape (N, 4)")
        with torch.no_grad():
            previous = self._previous_quaternions
            if previous is not None and (
                previous.shape != model.q.shape
                or previous.device != model.q.device
                or previous.dtype != model.q.dtype
            ):
                previous = None
            projected = project_quaternions_to_image_plane(
                model.q.detach(), previous=previous
            )
            model.m[:, 2].fill_(self.plane_z)
            model.q.copy_(projected)
            self._previous_quaternions = projected.clone()

            mean_state = self._optimizer_state(optimizer, model.m)
            for name in ("exp_avg", "exp_avg_sq"):
                value = mean_state.get(name)
                if (
                    isinstance(value, torch.Tensor)
                    and value.shape == model.m.shape
                ):
                    value[:, 2].zero_()

            quaternion_state = self._optimizer_state(optimizer, model.q)
            first_moment = quaternion_state.get("exp_avg")
            if (
                isinstance(first_moment, torch.Tensor)
                and first_moment.shape == model.q.shape
            ):
                first_moment[:, 1:3].zero_()
                radial = (first_moment * projected).sum(dim=1, keepdim=True)
                first_moment.sub_(radial * projected)
            second_moment = quaternion_state.get("exp_avg_sq")
            if (
                isinstance(second_moment, torch.Tensor)
                and second_moment.shape == model.q.shape
            ):
                second_moment[:, 1:3].zero_()
