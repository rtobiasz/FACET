#!/usr/bin/env python3
"""Run a 2D MPM simulation while preserving FACET primitive identities."""
from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from facet.editing.state_format import (
    PlanarEditArrays,
    load_planar_edit_arrays,
    save_planar_edit_arrays,
)
from facet.physics.mpm import (
    plane_deformation_gradient,
    isolate_planar_selection,
    prepare_mpm_particles,
    restore_planar_state,
    selection_from_image_mask,
)


def summarize_simulation(
    initial: PlanarEditArrays,
    final: PlanarEditArrays,
    simulated_rows: np.ndarray,
) -> dict[str, object]:
    if not np.array_equal(initial.primitive_ids, final.primitive_ids):
        raise ValueError("MPM simulation changed primitive identity or ordering")
    if not np.array_equal(initial.active, final.active):
        raise ValueError("MPM simulation changed the activity mask")
    displacement = np.linalg.norm(final.mean_xy - initial.mean_xy, axis=1)
    changed = displacement > 0.0
    simulated_mask = np.zeros(initial.primitive_count, dtype=np.bool_)
    simulated_mask[simulated_rows] = True
    if np.any(changed & ~simulated_mask):
        raise ValueError("MPM simulation changed an unselected primitive")
    selected_displacement = displacement[simulated_rows]
    selected_delta = final.mean_xy[simulated_rows] - initial.mean_xy[simulated_rows]
    return {
        "primitive_count": initial.primitive_count,
        "simulated_primitive_count": int(simulated_rows.size),
        "changed_primitive_count": int(np.count_nonzero(changed)),
        "mean_selected_displacement": float(selected_displacement.mean()),
        "maximum_selected_displacement": float(selected_displacement.max()),
        "mean_selected_delta_xy": selected_delta.mean(axis=0).tolist(),
        "primitive_ids_preserved": True,
        "activity_mask_preserved": True,
        "unselected_primitives_preserved": True,
    }


def load_mpm_runtime(path: Path | None, arch_name: str):
    if path is not None:
        sys.path.insert(0, str(path.resolve()))
    try:
        import taichi as ti
        from engine.mpm_solver import MPMSolver
    except ImportError as error:
        raise RuntimeError(
            "MPM execution requires Taichi Elements; install the dependency or "
            "provide --taichi_elements_path. Use --steps 0 to validate interchange."
        ) from error
    arch = getattr(ti, arch_name)
    ti.init(arch=arch)
    return MPMSolver


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=99)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--offset", type=float, nargs=2, default=(0.25, 0.4))
    parser.add_argument("--elasticity_scale", type=float, default=4.0)
    parser.add_argument(
        "--floor_height",
        type=float,
        default=0.25,
        help="Height of the slip floor in MPM coordinates; GaINeR's sand uses 0.35.",
    )
    parser.add_argument(
        "--collider",
        nargs=5,
        action="append",
        metavar=("X", "Y", "NX", "NY", "SURFACE"),
        help=(
            "Surface collider (point, normal, sticky|slip|separate); repeatable. "
            "When given, replaces the default slip floor at --floor_height"
        ),
    )
    parser.add_argument(
        "--horizontal_force_cosine",
        type=float,
        help=(
            "Replace gravity before step f by (A cos(pi f / 100), 0), the wind of "
            "GaINeR's bouquet (A = 75)"
        ),
    )
    parser.add_argument("--maximum_particles", type=int)
    parser.add_argument(
        "--selection",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Simulate only active primitives inside this planar box",
    )
    parser.add_argument(
        "--selection_mask",
        type=Path,
        help="Simulate primitives whose planar positions sample a nonzero mask pixel",
    )
    parser.add_argument(
        "--selection_mask_domain",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Planar bounds covered by the row-major selection mask",
    )
    parser.add_argument("--selection_mask_threshold", type=int, default=127)
    parser.add_argument(
        "--selection_mask_dilation",
        type=int,
        default=0,
        help="Dilate the thresholded selection mask by this many pixels (default off)",
    )
    parser.add_argument(
        "--isolate_selection",
        action="store_true",
        help="Hide active primitives outside the simulation selection",
    )
    parser.add_argument(
        "--material", choices=("elastic", "sand", "water", "snow"), default="elastic"
    )
    parser.add_argument(
        "--secondary_material",
        choices=("elastic", "sand", "water", "snow"),
        help="Material for particles above --secondary_above; the rest use --material",
    )
    parser.add_argument(
        "--secondary_above",
        type=float,
        help=(
            "MPM height above which particles get --secondary_material; GaINeR's "
            "sand-and-water scene splits at 0.55 (the upper half after scale 0.7, "
            "offset 0.2)"
        ),
    )
    parser.add_argument("--arch", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--taichi_elements_path", type=Path)
    parser.add_argument(
        "--save_deformation_gradient",
        action="store_true",
        help=(
            "Also write NNNNNN_F.npz per frame: the primitive IDs of the simulated "
            "particles and their deformation gradients in plane coordinates, for "
            "facet/editing/render_sequence.py --transport_deformation"
        ),
    )
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if args.dt <= 0.0:
        raise ValueError("dt must be greater than zero")
    if args.resolution <= 0:
        raise ValueError("resolution must be greater than zero")
    if not 0 <= args.selection_mask_threshold <= 255:
        raise ValueError("selection_mask_threshold must be in [0, 255]")
    if (args.secondary_material is None) != (args.secondary_above is None):
        raise ValueError("secondary_material and secondary_above must be provided together")
    if (args.selection_mask is None) != (args.selection_mask_domain is None):
        raise ValueError(
            "selection_mask and selection_mask_domain must be provided together"
        )

    source = load_planar_edit_arrays(args.input)
    selection = None
    if args.selection is not None:
        xmin, xmax, ymin, ymax = args.selection
        if xmin > xmax or ymin > ymax:
            raise ValueError("selection bounds must be ordered")
        selection = (
            (source.mean_xy[:, 0] >= xmin)
            & (source.mean_xy[:, 0] <= xmax)
            & (source.mean_xy[:, 1] >= ymin)
            & (source.mean_xy[:, 1] <= ymax)
        )
    if args.selection_mask is not None:
        mask_pixels = np.asarray(Image.open(args.selection_mask).convert("L"))
        mask_binary = mask_pixels > args.selection_mask_threshold
        if args.selection_mask_dilation > 0:
            from scipy import ndimage

            mask_binary = ndimage.binary_dilation(
                mask_binary, iterations=args.selection_mask_dilation
            )
        mask_selection = selection_from_image_mask(
            source.mean_xy,
            mask_binary,
            tuple(args.selection_mask_domain),
        )
        selection = mask_selection if selection is None else selection & mask_selection
    if args.isolate_selection:
        if selection is None:
            raise ValueError("isolate_selection requires a box or image mask")
        source = isolate_planar_selection(source, selection)
        selection = None
    particles = prepare_mpm_particles(
        source,
        scale=args.scale,
        offset=tuple(args.offset),
        maximum_particles=args.maximum_particles,
        selection=selection,
    )
    MPMSolver = None
    if args.steps > 0:
        MPMSolver = load_mpm_runtime(args.taichi_elements_path, args.arch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_zero = restore_planar_state(
        source,
        particles,
        particles.positions,
        metadata={
            "simulation": {
                "engine": "taichi_elements",
                "frame": 0,
                "dt": args.dt,
                "material": args.material,
                "transform": particles.transform.as_metadata(),
            }
        },
    )
    # Preserve the source positions exactly for the identity gate. A simulator's
    # float32 normalization round-trip must not perturb the zero-time frame.
    frame_zero = PlanarEditArrays(
        primitive_ids=frame_zero.primitive_ids,
        mean_xy=source.mean_xy.copy(),
        active=frame_zero.active,
        metadata=frame_zero.metadata,
    )
    save_planar_edit_arrays(args.output_dir / "000000.npz", frame_zero)
    if args.steps == 0:
        print(f"Wrote zero-step MPM frame to {args.output_dir.resolve()}")
        return 0

    solver = MPMSolver(
        res=(args.resolution, args.resolution),
        E_scale=args.elasticity_scale,
        max_num_particles=max(1024, particles.positions.shape[0]),
    )
    if args.collider is None:
        solver.add_surface_collider(
            point=(0.5, args.floor_height), normal=(0, 1), surface=solver.surface_slip
        )
    else:
        for x, y, nx, ny, surface in args.collider:
            if surface not in ("sticky", "slip", "separate"):
                raise ValueError("collider surface must be sticky, slip, or separate")
            solver.add_surface_collider(
                point=(float(x), float(y)),
                normal=(float(nx), float(ny)),
                surface=getattr(solver, f"surface_{surface}"),
            )
    material = getattr(solver, f"material_{args.material}")
    # The solver returns particles in the order they were added; with two
    # materials, restore the primitive order after every step.
    secondary = np.zeros(particles.positions.shape[0], dtype=bool)
    if args.secondary_material is not None:
        secondary = particles.positions[:, 1] > args.secondary_above
    order = np.concatenate((np.flatnonzero(~secondary), np.flatnonzero(secondary)))
    restore_order = np.argsort(order)
    primary_count = int((~secondary).sum())
    if primary_count:
        solver.add_particles(
            particles=particles.positions[order[:primary_count]], material=material
        )
    if secondary.any():
        solver.add_particles(
            particles=particles.positions[order[primary_count:]],
            material=getattr(solver, f"material_{args.secondary_material}"),
        )
    copy_deformation = None
    if args.save_deformation_gradient:
        import taichi as ti

        def copy_deformation(destination, count):
            for p in range(count):
                for i, j in ti.static(ti.ndrange(2, 2)):
                    destination[p, i, j] = solver.F[p][i, j]

        # Postponed annotations (from __future__) are strings that Taichi
        # cannot resolve, so the kernel signature is set explicitly.
        copy_deformation.__annotations__ = {
            "destination": ti.types.ndarray(),
            "count": ti.i32,
        }
        copy_deformation = ti.kernel(copy_deformation)

    for frame in range(1, args.steps + 1):
        if args.horizontal_force_cosine is not None:
            solver.set_gravity(
                [args.horizontal_force_cosine * math.cos(frame / 100.0 * math.pi), 0.0]
            )
        solver.step(args.dt)
        simulated = np.asarray(solver.particle_info()["position"])[restore_order]
        if copy_deformation is not None:
            count = simulated.shape[0]
            gradients = np.zeros((count, 2, 2), dtype=np.float32)
            copy_deformation(gradients, count)
            gradients = gradients[restore_order]
            np.savez_compressed(
                args.output_dir / f"{frame:06d}_F.npz",
                primitive_ids=particles.primitive_ids,
                deformation=plane_deformation_gradient(gradients, particles.transform).astype(np.float32),
            )
        state = restore_planar_state(
            source,
            particles,
            simulated,
            metadata={
                "simulation": {
                    "engine": "taichi_elements",
                    "frame": frame,
                    "dt": args.dt,
                    "material": args.material,
                    "transform": particles.transform.as_metadata(),
                }
            },
        )
        save_planar_edit_arrays(args.output_dir / f"{frame:06d}.npz", state)
        print(".", end="", flush=True)
    summary = summarize_simulation(source, state, particles.source_rows)
    summary.update(
        {
            "steps": args.steps,
            "frame_dt": args.dt,
            "resolution": args.resolution,
            "arch": args.arch,
            "floor_height": args.floor_height if args.collider is None else None,
            "colliders": args.collider,
            "horizontal_force_cosine": args.horizontal_force_cosine,
            "material": args.material,
            "secondary_material": args.secondary_material,
            "secondary_above": args.secondary_above,
            "secondary_particle_count": int(secondary.sum()),
            "isolated_selection": args.isolate_selection,
            "selection": args.selection,
            "selection_mask": (
                str(args.selection_mask.resolve()) if args.selection_mask else None
            ),
            "selection_mask_domain": args.selection_mask_domain,
            "selection_mask_dilation": args.selection_mask_dilation,
        }
    )
    (args.output_dir / "simulation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSimulation finished in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
