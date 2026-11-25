"""Export a model-only FACET 2D checkpoint for static inference."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from facet.io.checkpoint import (
    create_checkpoint_payload,
    load_training_checkpoint,
    save_checkpoint_payload,
)
from facet.model.components import PlanarGaussianModel


def export_inference_checkpoint(source: Path, destination: Path) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().absolute()
    if source == destination:
        raise ValueError("input and output checkpoint paths must differ")
    state = load_training_checkpoint(source, "cpu", require_rgb=True)
    if state.config is None or state.config.get("mode") != "image_2d":
        raise ValueError("only image_2d checkpoints can be exported")
    active = int(state.config.get("ablation", {}).get(
        "active_conditioning_dimensions", 96
    ))
    model = PlanarGaussianModel(
        state.model,
        plane_z=state.config["image"]["plane_z"],
        requires_grad=False,
        active_conditioning_dimensions=active,
    )
    payload = create_checkpoint_payload(
        model.model_tensors(),
        optimizer_state=None,
        iteration=state.iteration,
        training_time_seconds=state.training_time_seconds,
        config=state.config,
        mode="image_2d",
    )
    payload["purpose"] = "static_inference"
    save_checkpoint_payload(destination, payload)
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.model_tensors().values()
    )
    return {
        "schema": "facet-image-2d-inference-export-v1",
        "input": str(source),
        "output": str(destination.resolve()),
        "active_conditioning_dimensions": active,
        "primitive_count": model.number_of_Gaussians,
        "model_values": sum(t.numel() for t in model.model_tensors().values()),
        "model_tensor_bytes": tensor_bytes,
        "input_bytes": source.stat().st_size,
        "output_bytes": destination.stat().st_size,
        "optimizer_included": False,
    }


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = export_inference_checkpoint(args.input, args.output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
