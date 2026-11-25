"""Command-line plumbing shared by the entry points in scripts/.

Every entry point accepts ``--config FILE.toml``. A config file has one table per command; the keys
of a table are that command's options without the leading dashes, written exactly as on the command
line (``maximum_scale_fraction`` for training, ``edge-max-slant`` for lifting)::

    [train]
    maximum_scale_fraction = 0.002
    restore_clamped_gradient = true

Values become command-line arguments placed before the ones typed by the user, so options given on
the command line override the config. ``true`` adds a flag, ``false`` adds its ``--no-`` form, and a
list passes several values (``background_rgb = [1, 1, 1]``) or, for repeatable options, one value per
occurrence (``disable_component = ["uv"]``).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from pathlib import Path

REPEATABLE = {"disable_component", "collider", "mesh", "sphere", "plane-mirror", "point-light"}


def config_arguments(path: Path, section: str) -> list[str]:
    """The command-line arguments that table ``section`` of a config file stands for."""
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if section not in config:
        raise SystemExit(f"{path} has no [{section}] table (tables: {', '.join(config) or 'none'})")
    arguments: list[str] = []
    for key, value in config[section].items():
        flag = f"--{key}"
        if isinstance(value, bool):
            arguments.append(flag if value else f"--no-{key}")
        elif isinstance(value, list):
            if key in REPEATABLE:
                for item in value:
                    arguments += [flag, *([str(v) for v in item] if isinstance(item, list) else [str(item)])]
            else:
                arguments += [flag, *(str(item) for item in value)]
        else:
            arguments += [flag, str(value)]
    return arguments


def run(module: str, section: str, argv: list[str] | None = None) -> int:
    """Run ``module.main()`` with the config table ``section`` expanded in front of ``argv``."""
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, rest = pre.parse_known_args(argv)
    expanded = config_arguments(known.config, section) if known.config else []
    sys.argv = [f"{module}", *expanded, *rest]
    result = importlib.import_module(module).main()
    return int(result or 0)


def dispatch(commands: dict[str, tuple[str, str, str]], description: str) -> int:
    """Subcommand front end: ``commands`` maps a name to (module, config table, help)."""
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        lines = [description, "", "commands:"]
        lines += [f"  {name:<14} {help_text}" for name, (_, _, help_text) in commands.items()]
        lines += ["", "Run a command with --help for its options."]
        print("\n".join(lines))
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2
    module, section, _ = commands[sys.argv[1]]
    return run(module, section, sys.argv[2:])
