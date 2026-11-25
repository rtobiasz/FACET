"""Deterministic, means-only geometry edits for FACET 2D."""
from __future__ import annotations

import math
import torch


EDIT_MODES = ("identity", "translate", "stretch", "bend", "displace", "cut")


def edit_mean_xy(
    mean_xy: torch.Tensor,
    mode: str,
    amount: float,
    *,
    axis: str = "x",
    selection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return edited means without changing primitive order or appearance."""
    if mean_xy.ndim != 2 or mean_xy.shape[1] != 2:
        raise ValueError("mean_xy must have shape (N, 2)")
    if mode not in ("identity", "translate", "stretch", "bend"):
        raise ValueError("mode must be identity, translate, stretch, or bend")
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    result = mean_xy.detach().clone()
    if selection is None:
        selection = torch.ones(mean_xy.shape[0], dtype=torch.bool, device=mean_xy.device)
    if selection.shape != (mean_xy.shape[0],) or selection.dtype != torch.bool:
        raise ValueError("selection must be a boolean vector with one entry per primitive")
    if not bool(selection.any()):
        return result
    primary = 0 if axis == "x" else 1
    secondary = 1 - primary
    if mode == "translate":
        result[selection, primary] = result[selection, primary] + amount
    elif mode == "stretch":
        result[selection, primary] = result[selection, primary] * amount
    elif mode == "bend":
        coordinate = result[selection, primary]
        span = (coordinate.max() - coordinate.min()).clamp_min(1e-8)
        phase = 2.0 * math.pi * (coordinate - coordinate.min()) / span
        result[selection, secondary] = result[selection, secondary] + amount * torch.sin(phase)
    return result


def edit_planar_state(
    mean_xy: torch.Tensor,
    mode: str,
    amount: float,
    *,
    axis: str = "x",
    selection: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edit positions and activity without deleting or reordering primitives."""
    if mode not in EDIT_MODES:
        raise ValueError(f"mode must be one of {EDIT_MODES}")
    count = mean_xy.shape[0] if mean_xy.ndim == 2 else 0
    if active is None:
        active = torch.ones(count, dtype=torch.bool, device=mean_xy.device)
    if active.shape != (count,) or active.dtype != torch.bool:
        raise ValueError("active must be a boolean vector with one entry per primitive")
    edited_active = active.detach().clone()
    if mode in ("cut", "displace") and selection is None:
        raise ValueError(f"{mode} requires an explicit selection")
    if mode == "cut":
        edited_means = edit_mean_xy(mean_xy, "identity", amount)
        if selection.shape != (count,) or selection.dtype != torch.bool:
            raise ValueError("selection must be a boolean vector with one entry per primitive")
        edited_active[selection] = False
    else:
        geometry_mode = "translate" if mode == "displace" else mode
        edited_means = edit_mean_xy(
            mean_xy, geometry_mode, amount, axis=axis, selection=selection
        )
    return edited_means, edited_active


def transport_planar_ellipses(
    mean_xy: torch.Tensor,
    theta: torch.Tensor,
    log_scales: torch.Tensor,
    mode: str,
    amount: float,
    *,
    axis: str = "x",
    selection: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport ellipse orientation and scales by the edit local Jacobian."""
    count = mean_xy.shape[0]
    if (
        mean_xy.shape != (count, 2)
        or theta.shape != (count,)
        or log_scales.shape != (count, 2)
    ):
        raise ValueError("expected mean_xy (N,2), theta (N,), and log_scales (N,2)")
    if mode not in ("identity", "translate", "stretch", "bend"):
        raise ValueError(
            "ellipse transport supports identity, translate, stretch, and bend"
        )
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    if selection is None:
        selection = torch.ones(count, dtype=torch.bool, device=mean_xy.device)
    if selection.shape != (count,) or selection.dtype != torch.bool:
        raise ValueError("selection must be a boolean vector with one entry per primitive")
    result_theta = theta.detach().clone()
    result_log_scales = log_scales.detach().clone()
    if mode in ("identity", "translate") or not bool(selection.any()):
        return result_theta, result_log_scales

    selected_theta = theta[selection]
    cosine, sine = torch.cos(selected_theta), torch.sin(selected_theta)
    rotation = torch.stack(
        (torch.stack((cosine, -sine), -1), torch.stack((sine, cosine), -1)), -2
    )
    variances = torch.exp(2.0 * log_scales[selection])
    covariance = rotation @ torch.diag_embed(variances) @ rotation.transpose(-1, -2)
    jacobian = torch.eye(2, device=mean_xy.device, dtype=mean_xy.dtype).expand(
        selected_theta.shape[0], 2, 2
    ).clone()
    primary = 0 if axis == "x" else 1
    secondary = 1 - primary
    if mode == "stretch":
        jacobian[:, primary, primary] = amount
    else:
        coordinate = mean_xy[selection, primary]
        span = (coordinate.max() - coordinate.min()).clamp_min(1e-8)
        phase = 2.0 * math.pi * (coordinate - coordinate.min()) / span
        jacobian[:, secondary, primary] = (
            amount * (2.0 * math.pi / span) * torch.cos(phase)
        )
    transported = jacobian @ covariance @ jacobian.transpose(-1, -2)
    eigenvalues, eigenvectors = torch.linalg.eigh(transported)
    eigenvalues = eigenvalues.flip(-1).clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    eigenvectors = eigenvectors.flip(-1)
    major_axis = eigenvectors[:, :, 0]
    result_theta[selection] = torch.atan2(major_axis[:, 1], major_axis[:, 0])
    result_log_scales[selection] = 0.5 * torch.log(eigenvalues)
    return result_theta, result_log_scales


def transport_by_deformation_gradient(
    theta: torch.Tensor,
    log_scales: torch.Tensor,
    deformation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry each ellipse through a per-primitive deformation gradient F.

    The ellipse frame R(theta) diag(exp(log_scales)) maps to A = F R S, whose
    singular vectors and values are the new axes and scales. Of the four
    proper frames that fit A's axes, the one closest to R_p R(theta) is kept,
    where R_p is the rotation of F's polar decomposition. The primitive's
    local appearance frame then turns with the material instead of snapping
    to the major axis, which transport_planar_ellipses does. Shear inside the
    material frame (the right singular vectors of A) cannot be represented by
    an ellipse and is dropped. F must have a positive determinant.
    """
    count = theta.shape[0]
    if theta.shape != (count,) or log_scales.shape != (count, 2) or deformation.shape != (count, 2, 2):
        raise ValueError("expected theta (N,), log_scales (N,2), and deformation (N,2,2)")
    if count == 0:
        return theta.detach().clone(), log_scales.detach().clone()
    work = torch.float64
    deformation = deformation.to(work)
    if bool((torch.linalg.det(deformation) <= 0.0).any()):
        raise ValueError("deformation gradients must have a positive determinant")
    angle = theta.detach().to(work)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack((torch.stack((cosine, -sine), -1), torch.stack((sine, cosine), -1)), -2)
    frame = deformation @ rotation @ torch.diag_embed(torch.exp(log_scales.detach().to(work)))
    axes, singular, _ = torch.linalg.svd(frame)
    left, _, right_t = torch.linalg.svd(deformation)
    polar = left @ right_t
    target = (polar @ rotation)[:, :, 0]
    candidates = torch.stack((axes[:, :, 0], -axes[:, :, 0], axes[:, :, 1], -axes[:, :, 1]), 1)
    choice = torch.argmax((candidates * target[:, None, :]).sum(-1), 1)
    first = candidates[torch.arange(count), choice]
    swapped = choice >= 2
    scales = torch.where(swapped[:, None], singular.flip(-1), singular)
    new_theta = torch.atan2(first[:, 1], first[:, 0]).to(theta.dtype)
    new_log_scales = torch.log(scales.clamp_min(torch.finfo(work).tiny)).to(log_scales.dtype)
    return new_theta, new_log_scales


def initial_neighbours(initial_xy: torch.Tensor, count: int) -> torch.Tensor:
    """Return the indices of each point's ``count`` nearest initial neighbours."""
    from scipy.spatial import cKDTree

    if initial_xy.ndim != 2 or initial_xy.shape[1] != 2:
        raise ValueError("initial_xy must have shape (N, 2)")
    if count < 2 or initial_xy.shape[0] <= count:
        raise ValueError("need at least two neighbours and more points than neighbours")
    points = initial_xy.detach().to("cpu", torch.float64).numpy()
    _, indices = cKDTree(points).query(points, k=count + 1)
    return torch.as_tensor(indices[:, 1:], device=initial_xy.device)


def neighbourhood_deformation_gradient(
    initial_xy: torch.Tensor,
    current_xy: torch.Tensor,
    neighbours: torch.Tensor,
    *,
    maximum_stretch: float = 4.0,
    neighbour_cutoff: float | None = None,
) -> torch.Tensor:
    """Total deformation gradient of each point from its initial neighbours.

    F is the least-squares affine map of the neighbour offsets,
    F = (sum dx dX^T)(sum dX dX^T)^-1, so it measures how the material around
    a point has actually moved since frame zero, including plastic flow that
    an MPM solver's elastic F forgets. F is rebuilt from its SVD with the
    singular values clamped to [1/maximum_stretch, maximum_stretch] and, where
    the fit folds the neighbourhood (det F <= 0), the smaller one made
    positive, so the result always has a positive determinant.

    With neighbour_cutoff c, a neighbour counts only while its distance is at
    most c times its initial distance: material that has torn apart (spray)
    keeps its droplets instead of being stretched to cover the gap. A point
    left with fewer than two such neighbours keeps F = I.
    """
    if initial_xy.shape != current_xy.shape or initial_xy.ndim != 2 or initial_xy.shape[1] != 2:
        raise ValueError("initial_xy and current_xy must have the same shape (N, 2)")
    if neighbours.ndim != 2 or neighbours.shape[0] != initial_xy.shape[0]:
        raise ValueError("neighbours must have shape (N, K)")
    if not maximum_stretch >= 1.0:
        raise ValueError("maximum_stretch must be at least 1")
    work = torch.float64
    initial = initial_xy.detach().to(work)
    current = current_xy.detach().to(work)
    reference = initial[neighbours] - initial[:, None, :]
    moved = current[neighbours] - current[:, None, :]
    attached = None
    if neighbour_cutoff is not None:
        if not neighbour_cutoff >= 1.0:
            raise ValueError("neighbour_cutoff must be at least 1")
        attached = moved.norm(dim=-1) <= neighbour_cutoff * reference.norm(dim=-1)
        reference = reference * attached[..., None]
        moved = moved * attached[..., None]
    cross = moved.transpose(1, 2) @ reference
    gram = reference.transpose(1, 2) @ reference
    regulariser = 1e-9 * torch.diagonal(gram, dim1=1, dim2=2).sum(-1).clamp_min(1e-30)
    gram = gram + regulariser[:, None, None] * torch.eye(2, dtype=work, device=gram.device)
    deformation = cross @ torch.linalg.inv(gram)
    left, singular, right_t = torch.linalg.svd(deformation)
    # A reflection in U V^T is moved into the smaller singular value, which the
    # clamp then makes positive: the neighbourhood is treated as squeezed flat.
    reflected = torch.linalg.det(left @ right_t) < 0.0
    left = left.clone()
    left[reflected, :, 1] = -left[reflected, :, 1]
    singular = singular.clamp(1.0 / maximum_stretch, maximum_stretch)
    result = left @ torch.diag_embed(singular) @ right_t
    if attached is not None:
        detached = attached.sum(-1) < 2
        result[detached] = torch.eye(2, dtype=work, device=result.device)
    return result
