"""Checks for input selections shared by otherwise independent method adapters."""

from __future__ import annotations

import json
from pathlib import Path


def selection_frames(selection):
    r = selection.get("frame_range")
    return list(range(r["start"], r["stop"], r.get("step", 1))) if r else list(range(selection["time_steps"]))


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_thuman_inputs(subject: dict, train: dict) -> dict:
    """Validate cameras and filenames without rejecting missing observations."""
    prefix = f"Run config subject {subject['subject_id']}"
    try:
        images = Path(subject["images"])
        masks = Path(subject["masks"])
        calibration = json.loads(Path(subject["calibration"]).read_text())
        camera_names = sorted(
            path.name for path in images.iterdir()
            if path.is_dir() and (masks / path.name).is_dir()
        )
        if any(not name.startswith("cam") or not name[3:].isdigit() for name in camera_names):
            raise ValueError("camera directories must use cam followed by a numeric ID")
        if camera_names != sorted(camera_names, key=lambda name: int(name[3:])):
            raise ValueError("camera names must sort in numeric order; use zero padding")
        by_id = {int(name[3:]): name for name in camera_names}
        if "camera_ids" in train:
            missing = set(train["camera_ids"]) - set(by_id)
            if missing:
                raise ValueError(f"unavailable camera IDs: {sorted(missing)}")
            selected = [by_id[i] for i in train["camera_ids"]]
        else:
            selected = camera_names[:train["views"]]
        if "views" in train and len(selected) != train["views"]:
            raise ValueError(f"requested {train['views']} views, found {len(selected)}")
        frames = selection_frames(train)
        for camera in selected:
            if camera not in calibration:
                raise ValueError(f"missing calibration for {camera}")
            for root in (images, masks):
                files = [path for path in (root / camera).iterdir()
                         if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS]
                if any(not path.stem.isdigit() for path in files):
                    raise ValueError(f"{camera} requires numeric frame filenames")
                ids = [int(path.stem) for path in files]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate frame IDs in {root / camera}")

        return {"camera_ids": [int(name[3:]) for name in selected], "frame_ids": frames}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{prefix}: {exc}") from exc


def validate_dancenet3d_inputs(subject: dict, selection: dict) -> dict:
    """Require dense canonical media and its native DanceNet frame sidecar."""
    resolved = validate_thuman_inputs(subject, selection)
    subject_id = subject.get("subject_id")
    prefix = f"Run config subject {subject_id}"
    try:
        frame_mapping_path = Path(subject["subject_path"]) / "frame_id_mapping.json"
        frame_mapping = json.loads(frame_mapping_path.read_text(encoding="utf-8"))
        canonical_ids = sorted(int(value) for value in frame_mapping)
        canonical_id_set = set(canonical_ids)
        if canonical_ids != list(range(len(canonical_ids))):
            raise ValueError("frame_id_mapping.json must contain dense IDs starting at zero")
        requested_frames = resolved["frame_ids"]
        outside = [frame for frame in requested_frames if frame not in canonical_id_set]
        if outside:
            preview = ", ".join(str(value) for value in outside[:8])
            suffix = "..." if len(outside) > 8 else ""
            raise ValueError(f"unavailable canonical frame IDs: {preview}{suffix}")
        for camera_id in resolved["camera_ids"]:
            camera = f"cam{camera_id:02d}"
            for key in ("images", "masks"):
                available = {
                    int(path.stem)
                    for path in (Path(subject[key]) / camera).iterdir()
                    if path.is_file() and path.stem.isdigit()
                }
                missing = [frame for frame in requested_frames if frame not in available]
                if missing:
                    preview = ", ".join(str(value) for value in missing[:8])
                    suffix = "..." if len(missing) > 8 else ""
                    raise ValueError(f"{camera} is missing {key} frames: {preview}{suffix}")
        return resolved
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if not (Path(subject.get("subject_path", "")) / "frame_id_mapping.json").is_file():
            exc = ValueError(
                "missing frame_id_mapping.json; refresh this processed subject "
                "with process_dancenet3d.py --skip-extraction"
            )
        raise SystemExit(f"{prefix}: {exc}") from exc


def validate_dancenet3d_camera_identity(subjects: list[dict], config: dict) -> None:
    """Ensure a configured canonical camera ID means one physical camera."""
    selections = [config["train"], *config.get("evaluation", {}).values()]
    selected_ids = sorted({value for item in selections for value in item["camera_ids"]})
    expected: dict[int, str] = {}
    expected_subject: dict[int, str] = {}
    for subject in subjects:
        subject_id = subject.get("subject_id")
        try:
            mapping = json.loads(Path(subject["camera_id_mapping"]).read_text())
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"Run config subject {subject_id}: cannot read camera_id_mapping.json"
            ) from exc
        for camera_id in selected_ids:
            name = f"cam{camera_id:02d}"
            source = mapping.get(name)
            if source is None:
                raise SystemExit(
                    f"Run config subject {subject_id}: unavailable camera ID {camera_id}"
                )
            if camera_id in expected and expected[camera_id] != source:
                raise SystemExit(
                    f"DanceNet camera {name} is not physically consistent: "
                    f"{expected_subject[camera_id]} maps it to {expected[camera_id]}, "
                    f"but {subject_id} maps it to {source}"
                )
            expected[camera_id] = source
            expected_subject[camera_id] = str(subject_id)


def validate_dna_inputs(subject: dict, rgb_reader, annot_reader) -> None:
    """Check native DNA capacity in the method environment, where SMC readers exist."""
    train = subject.get("training_config")
    if train is None:
        return
    prefix = f"Run config subject {subject['subject_id']}"
    calibration = (annot_reader.get_Calibration_all() if hasattr(annot_reader, "get_Calibration_all")
                   else annot_reader.smc["Camera_Parameter"])
    all_cameras = {int(key) for key in calibration if int(key) < 60}
    cameras = sorted(camera for camera in all_cameras if camera < 48)
    expected = train.get("camera_ids", list(range(train.get("views", 0))))
    if "camera_ids" not in train and (train["views"] > 48 or cameras[:train["views"]] != expected):
        raise SystemExit(f"{prefix}: requires consecutive 5MP cameras 0 through {train['views'] - 1}")
    selections = [train, *subject.get("evaluation_config", {}).values()]
    for index, selection in enumerate(selections):
        allowed = all_cameras if index or subject.get("evaluation_selection") else set(cameras)
        missing = set(selection.get("camera_ids", expected)) - allowed
        if missing:
            raise SystemExit(f"{prefix}: unavailable camera IDs: {sorted(missing)}")
    if hasattr(rgb_reader, "get_Camera_5mp_info"):
        rgb_frames = int((rgb_reader.get_Camera_5mp_info() or {}).get("num_frame", 0))
    else:
        capacities = []
        for selection in selections:
            for camera in selection.get("camera_ids", expected):
                group = "Camera_5mp" if camera < 48 else "Camera_12mp"
                entries = rgb_reader.smc.get(f"{group}/{camera}/color", {})
                capacities.append(max((int(key) for key in entries), default=-1) + 1)
        rgb_frames = min(capacities)
    pose_frames = len(annot_reader.get_SMPLx()["transl"])
    requested = max(max(selection_frames(selection)) for selection in selections) + 1
    if min(rgb_frames, pose_frames) < requested:
        raise SystemExit(f"{prefix}: requested {requested} timesteps, "
                         f"only {min(rgb_frames, pose_frames)} synchronized frames available")
