#!/usr/bin/env python3
"""Download the evaluation images: Kodak-24 and the DIV2K validation set.

    python scripts/download_data.py [--root data] [--datasets kodak div2k]

Writes
  data/kodak/kodim01.png ... kodim24.png              (768 x 512, Kodak-24)
  data/div2k/DIV2K_valid_HR/0801.png ... 0900.png     (high resolution, the SR targets)
  data/div2k/DIV2K_valid_LR_bicubic/X2/0801x2.png ... (DIV2K x2: the reconstruction table)
  data/div2k/DIV2K_valid_LR_x8/0801x8.png ...         (the 8x super-resolution inputs)
The DIV2K archives are about 0.5 GB (HR), 0.1 GB (x2) and 10 MB (x8).
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

KODAK = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
DIV2K = "https://data.vision.ee.ethz.ch/cvl/DIV2K/{}.zip"
DIV2K_ARCHIVES = ("DIV2K_valid_HR", "DIV2K_valid_LR_bicubic_X2", "DIV2K_valid_LR_x8")


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=("kodak", "div2k"), default=["kodak", "div2k"])
    args = parser.parse_args()
    if "kodak" in args.datasets:
        for index in range(1, 25):
            fetch(KODAK.format(index), args.root / "kodak" / f"kodim{index:02d}.png")
    if "div2k" in args.datasets:
        for name in DIV2K_ARCHIVES:
            archive = args.root / "div2k" / f"{name}.zip"
            fetch(DIV2K.format(name), archive)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(args.root / "div2k")
    print(f"done: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
