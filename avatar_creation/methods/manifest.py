"""Dataset manifest writing for external method processes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def dataset_manifest(dataset, *, path: Path, required_keys: tuple[str, ...]) -> Path:
    """Write a stable JSON manifest consumed by method environments."""
    manifest = {
        "dataset": {
            "kind": dataset.kind,
            "root": str(dataset.root),
            "required_keys": list(required_keys),
        },
        "subjects": [_json_value(subject) for subject in dataset.subjects],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
