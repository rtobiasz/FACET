"""Every table of every file in configs/ must parse as options of the command it belongs to."""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from facet.cli import config_arguments

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
COMMANDS = {
    "train": "facet.training.fit",
    "evaluate": "facet.evaluation.evaluate_checkpoint",
    "render": "facet.evaluation.render",
    "mask": "facet.editing.foreground_mask",
    "export_state": "facet.editing.export_state",
    "states": "facet.editing.make_states",
    "render_edit": "facet.editing.render_sequence",
    "simulate": "facet.physics.simulate_mpm",
    "cloth": "facet.physics.simulate_cloth",
    "lift": "facet.lifting.lift",
    "lift_evaluate": "facet.lifting.evaluate",
    "stylize": "facet.stylization.stylize",
    "super_resolve": "facet.superres.evaluate",
}
DOCUMENTATION_TABLES = {"expected"}


class _Parsed(Exception):
    def __init__(self, namespace):
        super().__init__()
        self.namespace = namespace


def parse_only(module_name: str, argv: list[str]) -> argparse.Namespace:
    """Run a command's argument parser on argv without running the command."""
    original = argparse.ArgumentParser.parse_args

    def capture(parser, args=None, namespace=None):
        for action in parser._actions:
            action.required = False
        for group in parser._mutually_exclusive_groups:
            group.required = False
        raise _Parsed(original(parser, args, namespace))

    module = importlib.import_module(module_name)
    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture), \
            mock.patch.object(sys, "argv", [module_name, *argv]):
        try:
            module.main()
        except _Parsed as parsed:
            return parsed.namespace
    raise AssertionError(f"{module_name}.main() did not parse its arguments")


class ConfigTest(unittest.TestCase):
    def test_every_table_parses(self):
        paths = sorted(CONFIGS.glob("*.toml"))
        self.assertGreater(len(paths), 20)
        for path in paths:
            with path.open("rb") as handle:
                tables = tomllib.load(handle)
            for table in tables:
                if table in DOCUMENTATION_TABLES:
                    continue
                with self.subTest(config=path.name, table=table):
                    self.assertIn(table, COMMANDS, f"unknown table [{table}]")
                    try:
                        parse_only(COMMANDS[table], config_arguments(path, table))
                    except ModuleNotFoundError as error:  # optional dependency (pyproject extras)
                        self.skipTest(f"{COMMANDS[table]} needs {error.name}")
                    except SystemExit as error:
                        self.fail(f"{path.name} [{table}] does not parse: exit {error.code}")

    def test_command_line_overrides_config(self):
        argv = config_arguments(CONFIGS / "div2k_x2.toml", "train") + ["--maximum_scale_fraction", "0.003"]
        namespace = parse_only("facet.training.fit", argv)
        self.assertEqual(namespace.maximum_scale_fraction, 0.003)
        self.assertEqual(namespace.minimum_scale, 0.0010)

    def test_paper_defaults(self):
        namespace = parse_only("facet.training.fit", [])
        self.assertEqual(namespace.primitive_count, 500000)
        self.assertEqual(namespace.iterations, 30000)
        self.assertEqual(namespace.active_conditioning_dimensions, 4)
        self.assertEqual(namespace.maximum_scale_fraction, 0.0025)
        self.assertEqual(namespace.minimum_scale, 0.00125)
        self.assertEqual(namespace.scale_regularization, 0.0005)
        self.assertTrue(namespace.restore_clamped_gradient)
        lift = parse_only("facet.lifting.lift", config_arguments(CONFIGS / "lift_hotdog.toml", "lift"))
        self.assertEqual(lift.edge_max_slant, 85.0)


if __name__ == "__main__":
    unittest.main()
