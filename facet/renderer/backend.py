from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import sys
from typing import Any


class NativeFACETBackend:
    """Thin binding around the compiled FACET renderer extension."""

    def __init__(self, native_module: Any | None = None) -> None:
        if native_module is None:
            renderer_dir = Path(
                os.environ.get(
                    "FACET_RENDERER_DIRECTORY",
                    Path(__file__).resolve().parents[2] / "build" / "renderer",
                )
            ).expanduser().resolve()
            if str(renderer_dir) not in sys.path:
                sys.path.insert(0, str(renderer_dir))
            native_module = import_module("PYOPTIXFACETRENDERER")
        self.native_module = native_module

    @property
    def hit_buffer_size(self) -> int:
        return int(getattr(self.native_module, "HIT_BUFFER_SIZE", 16))

    def create_renderer(
        self,
        number_of_sides: int,
        chi_square_squared_radius: float,
        max_batch_size: int,
    ) -> Any:
        return self.native_module.CPyOptiXFACETRenderer(
            number_of_sides,
            chi_square_squared_radius,
            max_batch_size,
        )

    def set_geometry(self, renderer: Any, *geometry: object) -> None:
        renderer.SetGeometry(*geometry)

    def generate_rays(self, *args: object) -> Any:
        return self.native_module.GenerateRays(*args)

    def trace_hits(self, renderer: Any, *args: object) -> Any:
        return renderer.TraceHits(*args)

    def forward_training_base(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_training_base(*args)

    def backward_base(self, renderer: Any, *args: object) -> Any:
        return renderer.Backward_base(*args)

    def forward_training(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_training(*args)

    def backward(self, renderer: Any, *args: object) -> Any:
        return renderer.Backward(*args)

    def forward_inference_base(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_inference_base(*args)

    def forward_inference(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_inference(*args)
