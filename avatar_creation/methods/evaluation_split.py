"""Write the common per-subject evaluation split sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_KEYS = (
    "subject_path",
    "images",
    "masks",
    "labels",
    "calibration",
    "camera_id_mapping",
    "smplx",
    "mhr",
)


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


def evaluation_split_payload(
    subject: dict[str, Any],
    *,
    train_cameras: Iterable[int | str],
    resolution: int | str | list[int],
) -> dict[str, Any]:
    """Build the stable evaluation metadata shared by all method adapters."""
    camera_ids = [_json_value(camera) for camera in train_cameras]
    original_data = {
        key: _json_value(subject[key])
        for key in SOURCE_KEYS
        if subject.get(key) is not None
    }
    payload = {
        "subject_id": str(subject.get("subject_id", "subject")),
        "dataset_kind": str(subject.get("dataset", "unknown")),
        "original_dataset_path": original_data.get("subject_path"),
        "original_data": original_data,
        "train_cameras": camera_ids,
        "train_views": len(camera_ids),
        "resolution": _json_value(resolution),
    }
    train = subject.get("training_config", {})
    if train:
        payload["training_config"] = _json_value(train)
        r = train.get("frame_range")
        payload["train_frame_ids"] = list(range(r["start"], r["stop"], r.get("step", 1))) if r else list(range(train["time_steps"]))
        if "camera_ids" in train:
            payload["train_source_camera_ids"] = train["camera_ids"]
    if subject.get("evaluation_config"):
        payload["evaluation"] = subject["evaluation_config"]
    return payload


def write_evaluation_split(
    subject_dir: Path,
    subject: dict[str, Any],
    *,
    train_cameras: Iterable[int | str],
    resolution: int | str | list[int] | None,
) -> Path:
    """Write ``evaluation_split.json`` into a method's subject directory."""
    subject_dir.mkdir(parents=True, exist_ok=True)
    path = subject_dir / "evaluation_split.json"
    payload = evaluation_split_payload(
        subject,
        train_cameras=train_cameras,
        resolution="source" if resolution in (None, 0) else resolution,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
