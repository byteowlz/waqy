"""Explicit optional TOML head-export profiles. Never discovers credentials or applies exports."""
from __future__ import annotations

import tomllib
from pathlib import Path

from publication import identifier, selected_heads

FIELDS = {"source", "name", "heads", "repo", "metadata", "consent", "catalog"}
PATHS = {"source", "metadata", "consent", "catalog"}


def _setting(key: str, value, directory: Path):
    if key == "heads":
        if not isinstance(value, list) or not all(isinstance(head, str) for head in value):
            raise TypeError("config heads must be an array of head IDs")
        return selected_heads(",".join(identifier(head) for head in value))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config {key} must be a nonempty string")
    if key not in PATHS:
        return value
    expanded = Path(value).expanduser()
    return expanded if expanded.is_absolute() else directory / expanded


def read_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    with path.open("rb") as file:
        raw = file.read(65_537)
    if len(raw) > 65_536:
        raise ValueError("head-export config exceeds 64 KiB")
    config = tomllib.loads(raw.decode("utf-8"))
    if set(config) != {"export"} or not isinstance(config["export"], dict):
        raise ValueError("config must contain only an [export] table")
    values = config["export"]
    unknown = set(values) - FIELDS
    if unknown:
        raise ValueError(f"unknown head-export settings: {', '.join(sorted(unknown))}; apply is CLI-only")
    return {key: _setting(key, value, path.parent) for key, value in values.items()}


def resolve_options(args) -> dict:
    values = read_config(args.config) if args.config else {}
    for key in FIELDS:
        value = getattr(args, key)
        if value is not None:
            values[key] = selected_heads(value) if key == "heads" else value
    missing = {"source", "name", "heads"} - set(values)
    if missing:
        raise ValueError(f"provide {', '.join(sorted(missing))} via CLI or --config")
    return {"repo": "byteowlz/waqy", "metadata": None, "consent": None, "catalog": None, **values}
