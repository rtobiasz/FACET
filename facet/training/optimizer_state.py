#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping
import torch

from facet.io.checkpoint import MODEL_TENSOR_NAMES

if TYPE_CHECKING:
    pass


OPTIMIZER_GROUP_NAMES = MODEL_TENSOR_NAMES
PLANAR_OPTIMIZER_GROUP_NAMES = MODEL_TENSOR_NAMES[:-3] + ("mean_xy", "theta", "s")
PER_PRIMITIVE_PARAMETER_NAMES = (
    "RGB",
    "A",
    "k",
    "conditioning_variable",
    "m",
    "s",
    "q",
)
PLANAR_PER_PRIMITIVE_PARAMETER_NAMES = (
    "RGB",
    "A",
    "k",
    "conditioning_variable",
    "mean_xy",
    "theta",
    "s",
)


@dataclass(frozen=True)
class DensificationConfig:
    opacity_threshold: float
    minimum_scale_norm: float
    movement_threshold: float
    maximum_gaussians: int
    minimum_scale: float
    maximum_scale_fraction: float
    pruning_enabled: bool = True
    selection_signal: str = "movement"
    selection_fraction: float | None = None
    spatial_balance_power: float = 0.0
    spatial_balance_grid: int = 32
    split_policy: str = "snapshot_current"
    split_offset: float = 0.5
    split_scale: float = 0.625
    split_optimizer_state: str = "legacy"


def named_optimizer_groups(
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict]:
    if len(optimizer.param_groups) != len(OPTIMIZER_GROUP_NAMES):
        raise ValueError(
            "FACET optimizer must have exactly "
            f"{len(OPTIMIZER_GROUP_NAMES)} parameter groups"
        )
    groups: dict[str, dict] = {}
    for expected_name, group in zip(
        OPTIMIZER_GROUP_NAMES, optimizer.param_groups
    ):
        existing_name = group.get("name")
        if existing_name is not None and existing_name != expected_name:
            raise ValueError(
                f"optimizer group {existing_name!r} occupies the "
                f"{expected_name!r} compatibility slot"
            )
        group["name"] = expected_name
        groups[expected_name] = group
    return groups


def build_planar_optimizer(model, learning_rates: Mapping[str, float]) -> torch.optim.Adam:
    """Build named Adam groups for structural planar parameters."""
    missing = set(PLANAR_OPTIMIZER_GROUP_NAMES) - set(learning_rates)
    if missing:
        raise ValueError(f"missing planar learning rates: {sorted(missing)}")
    return torch.optim.Adam([
        {"params": [getattr(model, name)], "lr": float(learning_rates[name]), "name": name}
        for name in PLANAR_OPTIMIZER_GROUP_NAMES
    ])


def compact_planar_conditioning_optimizer_state(
    optimizer_state: object,
    active_conditioning_dimensions: int,
) -> object:
    """Slice fixed-96D Adam moments to the compact planar width."""
    if not 0 <= active_conditioning_dimensions <= 96:
        raise ValueError("active_conditioning_dimensions must be in [0, 96]")
    if not isinstance(optimizer_state, Mapping):
        return optimizer_state
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list):
        return optimizer_state

    result = dict(optimizer_state)
    result["state"] = dict(state)
    result["param_groups"] = [dict(group) for group in groups]
    compact_names = {"conditioning_variable", "w1_conditioning"}
    for group in result["param_groups"]:
        if group.get("name") not in compact_names:
            continue
        parameters = group.get("params")
        if not isinstance(parameters, list) or len(parameters) != 1:
            continue
        parameter_id = parameters[0]
        parameter_state = result["state"].get(parameter_id)
        if not isinstance(parameter_state, Mapping):
            continue
        compact_state = dict(parameter_state)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = compact_state.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                compact_state[key] = value[
                    :, :active_conditioning_dimensions
                ].clone().contiguous()
        result["state"][parameter_id] = compact_state
    return result


def named_planar_optimizer_groups(optimizer: torch.optim.Optimizer) -> dict[str, dict]:
    if len(optimizer.param_groups) != len(PLANAR_OPTIMIZER_GROUP_NAMES):
        raise ValueError(
            f"planar optimizer must have exactly {len(PLANAR_OPTIMIZER_GROUP_NAMES)} parameter groups"
        )
    groups = {}
    for expected, group in zip(PLANAR_OPTIMIZER_GROUP_NAMES, optimizer.param_groups):
        if group.get("name", expected) != expected:
            raise ValueError(f"optimizer group does not match planar slot {expected!r}")
        group["name"] = expected
        groups[expected] = group
    return groups

def update_planar_learning_rates(
    optimizer: torch.optim.Optimizer,
    learning_rates: Mapping[str, float],
) -> None:
    for name, group in named_planar_optimizer_groups(optimizer).items():
        group["lr"] = float(learning_rates[name])
