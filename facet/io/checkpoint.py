"""Versioned FACET checkpoints: saving, loading and conversion of the tuple layout."""

from __future__ import annotations

import gc
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT = "FACET"
# Format name of checkpoints written before the code release, identified by its SHA-256.
PRE_RELEASE_FORMAT_SHA256 = "74f2278145014f91c45e6a2b87f267485a35ddc804410d9519167a9c1aca1a6b"


def is_supported_format(name: object) -> bool:
    """True for the checkpoint formats this code reads."""
    return name == CHECKPOINT_FORMAT or (
        isinstance(name, str) and hashlib.sha256(name.encode()).hexdigest() == PRE_RELEASE_FORMAT_SHA256
    )
CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, CHECKPOINT_VERSION)

MODEL_TENSOR_NAMES = (
    "RGB",
    "A",
    "k",
    "w1_uv",
    "w1_v",
    "w1_conditioning",
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
    "conditioning_variable",
    "features",
    "m",
    "s",
    "q",
)
LEGACY_MODEL_TENSOR_NAMES = MODEL_TENSOR_NAMES[1:]
PLANAR_GEOMETRY_NAMES = ("mean_xy", "theta")
PLANAR_MODEL_TENSOR_NAMES = MODEL_TENSOR_NAMES[:-3] + PLANAR_GEOMETRY_NAMES + ("s",)
PLANAR_LEGACY_MODEL_TENSOR_NAMES = PLANAR_MODEL_TENSOR_NAMES[1:]


@dataclass(frozen=True)
class CheckpointState:
    model: dict[str, torch.Tensor]
    optimizer_state: object
    iteration: int
    training_time_seconds: float
    config: dict[str, Any] | None
    source_version: int


def checkpoint_tensor_names(
    checkpoint: tuple[object, ...] | list[object],
) -> tuple[str, ...]:
    """Return tensor names for a supported tuple schema."""
    length = len(checkpoint)
    if length == 18:
        return MODEL_TENSOR_NAMES
    if length == 17:
        return LEGACY_MODEL_TENSOR_NAMES
    raise ValueError(
        f"Unsupported FACET checkpoint with {length} entries; expected "
        "17 (FACET-only) or 18 (RGB+FACET)"
    )


def _model_to_device(
    model_payload: Mapping[str, object],
    device: torch.device | str,
    *,
    require_rgb: bool,
) -> dict[str, torch.Tensor]:
    if not isinstance(model_payload, Mapping):
        raise ValueError("Checkpoint model must be a mapping of named tensors")

    is_planar = all(name in model_payload for name in PLANAR_GEOMETRY_NAMES)
    required = PLANAR_LEGACY_MODEL_TENSOR_NAMES if is_planar else LEGACY_MODEL_TENSOR_NAMES
    missing = [name for name in required if name not in model_payload]
    if missing:
        raise ValueError(f"Checkpoint is missing model tensors: {', '.join(missing)}")
    if require_rgb and "RGB" not in model_payload:
        raise ValueError("This checkpoint has no base RGB tensor")

    if is_planar:
        names = PLANAR_MODEL_TENSOR_NAMES if "RGB" in model_payload else PLANAR_LEGACY_MODEL_TENSOR_NAMES
    else:
        names = MODEL_TENSOR_NAMES if "RGB" in model_payload else LEGACY_MODEL_TENSOR_NAMES
    model: dict[str, torch.Tensor] = {}
    for index, name in enumerate(names):
        value = model_payload[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Checkpoint entry {index} ({name}) is not a torch.Tensor"
            )
        model[name] = value.detach().to(device).contiguous()
    return model


def _legacy_iteration(optimizer_state: object) -> int:
    if not isinstance(optimizer_state, Mapping):
        return -1
    state = optimizer_state.get("state")
    if not isinstance(state, Mapping):
        return -1
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            continue
        step = parameter_state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                continue
            step = step.item()
        try:
            return int(step)
        except (TypeError, ValueError):
            continue
    return -1


def _legacy_checkpoint_state(
    checkpoint: tuple[object, ...] | list[object],
    device: torch.device | str,
    *,
    require_rgb: bool,
) -> CheckpointState:
    names = checkpoint_tensor_names(checkpoint)
    model_payload = {name: checkpoint[index] for index, name in enumerate(names)}
    model = _model_to_device(model_payload, device, require_rgb=require_rgb)
    optimizer_state = checkpoint[-2]
    try:
        training_time = float(checkpoint[-1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The final checkpoint entry must contain training time in seconds"
        ) from error
    return CheckpointState(
        model=model,
        optimizer_state=optimizer_state,
        iteration=_legacy_iteration(optimizer_state),
        training_time_seconds=training_time,
        config=None,
        source_version=0,
    )


def _versioned_checkpoint_state(
    checkpoint: Mapping[str, object],
    device: torch.device | str,
    *,
    require_rgb: bool,
) -> CheckpointState:
    if not is_supported_format(checkpoint.get("format")):
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint.get('format')!r}; "
            f"expected {CHECKPOINT_FORMAT!r}"
        )
    version = checkpoint.get("version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"Unsupported FACET checkpoint version {version!r}; "
            f"expected one of {SUPPORTED_CHECKPOINT_VERSIONS}"
        )
    config = checkpoint.get("config")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("Checkpoint config must be a mapping or None")
    if version == 1:
        if isinstance(config, Mapping) and config.get("mode") == "image_2d":
            raise ValueError("v1 checkpoints cannot declare image_2d geometry")
    else:
        mode = checkpoint.get("mode")
        orientation = checkpoint.get("orientation")
        payload_model = checkpoint.get("model")
        planar_payload = isinstance(payload_model, Mapping) and all(
            name in payload_model for name in PLANAR_GEOMETRY_NAMES
        )
        if mode not in ("reconstruction_3d", "image_2d"):
            raise ValueError("v2 checkpoint mode must be reconstruction_3d or image_2d")
        expected_orientation = "scalar_angle" if mode == "image_2d" else "quaternion"
        if orientation != expected_orientation:
            raise ValueError(f"{mode} checkpoint requires orientation={expected_orientation!r}")
        if planar_payload != (mode == "image_2d"):
            raise ValueError("checkpoint mode does not match stored geometry")
    model = _model_to_device(
        checkpoint.get("model"), device, require_rgb=require_rgb
    )

    training = checkpoint.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Checkpoint training state must be a mapping")
    if "iteration" not in training:
        raise ValueError("Checkpoint training state is missing iteration")
    if "time_seconds" not in training:
        raise ValueError("Checkpoint training state is missing time_seconds")
    try:
        iteration = int(training["iteration"])
        training_time = float(training["time_seconds"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint iteration and time_seconds must be numeric"
        ) from error

    return CheckpointState(
        model=model,
        optimizer_state=checkpoint.get("optimizer"),
        iteration=iteration,
        training_time_seconds=training_time,
        config=dict(config) if config is not None else None,
        source_version=version,
    )


def checkpoint_state_from_payload(
    checkpoint: object,
    device: torch.device | str,
    *,
    require_rgb: bool = False,
) -> CheckpointState:
    """Normalize a v1 mapping or a tuple checkpoint into one typed state."""
    if isinstance(checkpoint, (tuple, list)):
        return _legacy_checkpoint_state(
            checkpoint, device, require_rgb=require_rgb
        )
    if isinstance(checkpoint, Mapping) and "format" in checkpoint:
        return _versioned_checkpoint_state(
            checkpoint, device, require_rgb=require_rgb
        )
    raise ValueError(
        "Unsupported FACET checkpoint object; expected a tuple or list "
        "or a versioned checkpoint mapping"
    )


def model_from_checkpoint(
    checkpoint: object,
    device: torch.device | str,
    *,
    require_rgb: bool = False,
) -> dict[str, object]:
    """Return the model-only view for any supported schema."""
    state = checkpoint_state_from_payload(
        checkpoint, device, require_rgb=require_rgb
    )
    model: dict[str, object] = dict(state.model)
    model["iteration"] = state.iteration
    model["training_time_seconds"] = state.training_time_seconds
    return model


def create_checkpoint_payload(
    model: Mapping[str, torch.Tensor],
    *,
    optimizer_state: object,
    iteration: int,
    training_time_seconds: float,
    config: Mapping[str, Any] | None = None,
    mode: str = "reconstruction_3d",
) -> dict[str, object]:
    """Build the current portable checkpoint mapping without moving tensors."""
    if mode not in ("reconstruction_3d", "image_2d"):
        raise ValueError("mode must be reconstruction_3d or image_2d")
    required = PLANAR_LEGACY_MODEL_TENSOR_NAMES if mode == "image_2d" else LEGACY_MODEL_TENSOR_NAMES
    missing = [name for name in required if name not in model]
    if missing:
        raise ValueError(f"Checkpoint is missing model tensors: {', '.join(missing)}")
    names = (
        PLANAR_MODEL_TENSOR_NAMES if "RGB" in model else PLANAR_LEGACY_MODEL_TENSOR_NAMES
    ) if mode == "image_2d" else (
        MODEL_TENSOR_NAMES if "RGB" in model else LEGACY_MODEL_TENSOR_NAMES
    )
    model_payload: dict[str, torch.Tensor] = {}
    for index, name in enumerate(names):
        value = model[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Checkpoint entry {index} ({name}) is not a torch.Tensor"
            )
        model_payload[name] = value.detach()
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "mode": mode,
        "orientation": "scalar_angle" if mode == "image_2d" else "quaternion",
        "model": model_payload,
        "optimizer": optimizer_state,
        "training": {
            "iteration": int(iteration),
            "time_seconds": float(training_time_seconds),
        },
        "config": dict(config) if config is not None else None,
    }


def save_checkpoint_payload(
    checkpoint_path: Path | str,
    payload: Mapping[str, object],
) -> None:
    """Atomically write a checkpoint while preserving an existing destination."""
    path = Path(checkpoint_path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str,
    *,
    require_rgb: bool = False,
) -> CheckpointState:
    """Load and normalize a checkpoint, then release its CPU container."""
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint_state_from_payload(
        checkpoint, device, require_rgb=require_rgb
    )
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def load_model_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str,
    *,
    require_rgb: bool = False,
) -> dict[str, object]:
    """Load the model-only view of any supported checkpoint."""
    state = load_training_checkpoint(
        checkpoint_path, device, require_rgb=require_rgb
    )
    model: dict[str, object] = dict(state.model)
    model["iteration"] = state.iteration
    model["training_time_seconds"] = state.training_time_seconds
    return model
