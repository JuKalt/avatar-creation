"""Portable, versioned training input definitions (separate from local paths)."""

from __future__ import annotations

import json
from pathlib import Path


TRAIN_FIELDS = ("views", "time_steps", "resolution")


def load_experiment_config(path: Path) -> dict:
    try:
        config = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read dataset config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("Dataset config must be a JSON object")
    allowed = {"schema_version", "name", "dataset_kind", "dataset_path", "subjects", "train", "evaluation"}
    unknown = set(config) - allowed
    if unknown:
        raise SystemExit(f"Unknown dataset config fields: {', '.join(sorted(unknown))}")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise SystemExit("Dataset config schema_version must be 1")
    if config.get("dataset_kind") not in ("thuman", "dna", "dancenet3d", "actorshq"):
        raise SystemExit(
            "Dataset configs currently support dataset_kind thuman, dna, "
            "dancenet3d, or actorshq"
        )
    if "name" in config and not isinstance(config["name"], str):
        raise SystemExit("Dataset config name must be a string")
    subjects = config.get("subjects")
    if not isinstance(subjects, list) or not subjects or any(
        not isinstance(value, str) or not value.strip() for value in subjects
    ):
        raise SystemExit("Dataset config subjects must be a non-empty list of string IDs")
    if len(set(subjects)) != len(subjects):
        raise SystemExit("Dataset config subjects must not contain duplicates")
    train = config.get("train")
    validate_selection(train, "train", training=True)
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict) or set(evaluation) - {"novel_view", "novel_pose"}:
        raise SystemExit("Dataset config evaluation supports novel_view and novel_pose")
    for name, selection in evaluation.items():
        validate_selection(selection, f"evaluation.{name}")
        if "camera_ids" not in train or "frame_range" not in train:
            raise SystemExit("Explicit evaluation requires train.camera_ids and train.frame_range")
        training_frames = set(frame_ids(train))
        evaluation_frames = set(frame_ids(selection))
        if name == "novel_view":
            if set(train["camera_ids"]) & set(selection["camera_ids"]):
                raise SystemExit("Novel-view cameras must be disjoint from training cameras")
            if not evaluation_frames <= training_frames:
                raise SystemExit("Novel-view frames must belong to the training frame range")
        elif training_frames & evaluation_frames:
            raise SystemExit("Novel-pose frames must be disjoint from training frames")
    if "dataset_path" in config:
        value = config["dataset_path"]
        if not isinstance(value, str) or not value.strip():
            raise SystemExit("Dataset config dataset_path must be a non-empty string")
        root = Path(value).expanduser()
        if not root.is_absolute():
            root = path.expanduser().resolve().parent / root
        config["dataset_path"] = str(root.resolve())
    return config


def training_arguments(config: dict, extra_args: list[str]) -> list[str]:
    # Reject aliases too: upstream PiG accepts underscore spellings.
    controlled = {"--views", "--time-steps", "--time_steps", "--resolution",
                  "--dataset-manifest", "--dataset_manifest"}
    # argparse in method adapters also accepts unambiguous flag prefixes.
    flags = [arg.split("=", 1)[0] for arg in extra_args if arg.startswith("--")]
    if any(len(flag) > 2 and any(option.startswith(flag) for option in controlled)
           for flag in flags):
        raise SystemExit(
            "With --run-config/--dataset-config, set views, time_steps and resolution in the config; "
            "remove conflicting method arguments after --."
        )
    arguments = []
    for key in TRAIN_FIELDS:
        arguments.extend(["--" + key.replace("_", "-"), str(training_counts(config["train"])[key])])
    return arguments + extra_args


def frame_ids(selection: dict) -> list[int]:
    if "frame_range" in selection:
        value = selection["frame_range"]
        return list(range(value["start"], value["stop"], value.get("step", 1)))
    return list(range(selection["time_steps"]))


def training_counts(train: dict) -> dict:
    return {"views": len(train["camera_ids"]) if "camera_ids" in train else train["views"],
            "time_steps": len(frame_ids(train)), "resolution": train["resolution"]}


def validate_selection(value: dict, label: str, *, training: bool = False) -> None:
    if not isinstance(value, dict):
        raise SystemExit(f"Dataset config {label} must be an object")
    allowed = {"camera_ids", "frame_range"}
    if training:
        allowed |= {"views", "time_steps", "resolution"}
    if set(value) - allowed:
        raise SystemExit(f"Unknown fields in {label}: {sorted(set(value) - allowed)}")
    if training:
        for explicit, count in (("camera_ids", "views"), ("frame_range", "time_steps")):
            if (explicit in value) == (count in value):
                raise SystemExit(f"{label} must contain exactly one of {explicit} or {count}")
        if type(value.get("resolution")) is not int or value["resolution"] < 0:
            raise SystemExit(f"{label}.resolution must be an integer >= 0")
    elif set(value) != allowed:
        raise SystemExit(f"{label} must contain camera_ids and frame_range")
    for key in ("views", "time_steps"):
        if key in value and (type(value[key]) is not int or value[key] < 1):
            raise SystemExit(f"{label}.{key} must be an integer >= 1")
    if "camera_ids" in value:
        ids = value["camera_ids"]
        if (not isinstance(ids, list) or not ids
                or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise SystemExit(f"{label}.camera_ids must be a non-empty list of unique non-negative integers")
    if "frame_range" in value:
        r = value["frame_range"]
        if (not isinstance(r, dict) or not {"start", "stop"} <= set(r)
                or set(r) - {"start", "stop", "step"}
                or any(type(i) is not int for i in r.values())
                or r["start"] < 0 or r["stop"] <= r["start"] or r.get("step", 1) < 1):
            raise SystemExit(f"{label}.frame_range requires 0 <= start < stop and step >= 1 (stop exclusive)")
