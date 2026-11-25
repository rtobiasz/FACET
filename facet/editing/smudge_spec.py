#!/usr/bin/env python3
"""Write a smudge warp spec (facet/editing/warps.py) from a stroke recipe.

  random   --count N --seed S: smooth strokes through four random control points
           (Catmull-Rom), radius and length drawn per stroke, kept MARGIN pixels inside the image
  spiral   --centre X Y (repeatable) --spiral-radius R --turns T: inward spirals, like painted swirls
  streaks  --start X Y (repeatable) --direction DX DY --length L --wave A: parallel strokes, each a
           sine wave of amplitude A along DIRECTION
  python facet/editing/smudge_spec.py random --image kodim03.png --size 768 512 --count 6 --seed 0 --output spec.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def catmull_rom(points: np.ndarray, samples: int = 12) -> list[list[float]]:
    padded = np.vstack([points[0], points, points[-1]])
    path = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1:i + 3]
        for s in np.linspace(0.0, 1.0, samples, endpoint=False):
            path.append(0.5 * (2 * p1 + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2
                               + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3))
    path.append(padded[-2])
    return [[round(float(x), 2), round(float(y), 2)] for x, y in path]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("random", "spiral", "streaks"))
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, nargs=2, default=[768, 512], metavar=("W", "H"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--radius", type=float, nargs=2, default=[28.0, 48.0], help="brush radius range")
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=60.0)
    p.add_argument("--centre", type=float, nargs=2, action="append", default=[])
    p.add_argument("--spiral-radius", type=float, default=60.0)
    p.add_argument("--turns", type=float, default=1.5)
    p.add_argument("--start", type=float, nargs=2, action="append", default=[])
    p.add_argument("--direction", type=float, nargs=2, default=[1.0, 0.0])
    p.add_argument("--length", type=float, default=250.0)
    p.add_argument("--wave", type=float, default=0.0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    w, h = a.size
    strokes = []
    if a.mode == "random":
        for _ in range(a.count):
            start = rng.uniform([a.margin, a.margin], [w - a.margin, h - a.margin])
            heading, length = rng.uniform(0, 2 * math.pi), rng.uniform(120, 260)
            points = [start]
            for _ in range(3):
                heading += rng.uniform(-0.9, 0.9)
                step = points[-1] + length / 3 * np.array([math.cos(heading), math.sin(heading)])
                points.append(np.clip(step, [a.margin, a.margin], [w - a.margin, h - a.margin]))
            strokes.append({"path": catmull_rom(np.array(points)), "radius": round(float(rng.uniform(*a.radius)), 1),
                            "strength": a.strength})
    elif a.mode == "spiral":
        for k, (cx, cy) in enumerate(a.centre):
            phase = rng.uniform(0, 2 * math.pi)
            n = max(8, int(a.turns * 24))
            path = [[round(cx + a.spiral_radius * (1 - 0.8 * i / n) * math.cos(phase + 2 * math.pi * a.turns * i / n), 2),
                     round(cy + a.spiral_radius * (1 - 0.8 * i / n) * math.sin(phase + 2 * math.pi * a.turns * i / n), 2)]
                    for i in range(n + 1)]
            strokes.append({"path": path, "radius": a.radius[0] if k % 2 == 0 else a.radius[1], "strength": a.strength})
    else:
        d = np.array(a.direction, float)
        d /= np.linalg.norm(d)
        normal = np.array([-d[1], d[0]])
        for sx, sy in a.start:
            n = max(2, int(a.length / 10))
            path = [np.array([sx, sy]) + d * a.length * i / n + normal * a.wave * math.sin(2 * math.pi * i / n * 2)
                    for i in range(n + 1)]
            strokes.append({"path": [[round(float(x), 2), round(float(y), 2)] for x, y in path],
                            "radius": round(float(rng.uniform(*a.radius)), 1), "strength": a.strength})
    spec = {"image": a.image, "label": f"smudge {a.mode}", "warp": {"type": "smudge", "strokes": strokes}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(spec, indent=1) + "\n")
    print(a.output, len(strokes), "strokes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
