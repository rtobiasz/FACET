"""Per-iteration training history, stored beside the manifest.

A run logs one history entry per iteration. To keep ``manifest.json`` small, the
history is stored in ``history.json.gz`` next to it, as columns, and the manifest
keeps a pointer, the entry count and the last entry.

The format is lossless: Python's JSON writer emits the shortest repr that
round-trips each float, so ``load_history`` returns exactly what training
recorded. It also reads manifests that carry the history inline, so callers
never need to know which kind they have.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

HISTORY_FILE = "history.json.gz"
SCHEMA = "facet2d-training-history-columns-v1"


def history_columns(history: list[dict]) -> dict:
    """Turn a list of per-iteration records into one column per field."""
    fields = list(history[0]) if history else []
    for record in history:
        if list(record) != fields:
            raise ValueError("history records do not share one field order")
    return {"schema": SCHEMA, "fields": fields,
            "columns": {name: [record[name] for record in history] for name in fields}}


def history_records(columns: dict) -> list[dict]:
    if columns.get("schema") != SCHEMA:
        raise ValueError(f"unknown history schema {columns.get('schema')!r}")
    fields = columns["fields"]
    data = columns["columns"]
    length = len(data[fields[0]]) if fields else 0
    return [{name: data[name][index] for name in fields} for index in range(length)]


def write_history(directory: Path, history: list[dict]) -> dict:
    """Write the sidecar and return the manifest fields that point at it."""
    path = Path(directory) / HISTORY_FILE
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(history_columns(history), handle, separators=(",", ":"))
    temporary.replace(path)
    return {"history_file": HISTORY_FILE, "history_length": len(history),
            "history_last": history[-1] if history else None}


def load_history(manifest_path: Path, manifest: dict | None = None) -> list[dict]:
    """Return the per-iteration history for a manifest of either kind."""
    manifest_path = Path(manifest_path)
    if manifest is None:
        manifest = json.loads(manifest_path.read_text())
    training = manifest["training"]
    if "history" in training:
        return training["history"]
    with gzip.open(manifest_path.parent / training["history_file"], "rt",
                   encoding="utf-8") as handle:
        records = history_records(json.load(handle))
    if len(records) != training["history_length"]:
        raise ValueError(f"{manifest_path}: history sidecar has {len(records)} "
                         f"entries, manifest expects {training['history_length']}")
    return records


def last_history_entry(manifest: dict) -> dict | None:
    training = manifest["training"]
    if "history" in training:
        return training["history"][-1] if training["history"] else None
    return training.get("history_last")
