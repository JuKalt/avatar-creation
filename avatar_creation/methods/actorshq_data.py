"""Run-local conversion of native ActorsHQ sequences to the canonical layout."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

from avatar_creation.methods.external_data import ensure_link
from avatar_creation.methods.training_inputs import selection_frames


_MEDIA_RE = re.compile(r"Cam(\d+)_(rgb|mask)(\d+)$")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _media_index(root: Path, kind: str) -> dict[int, dict[int, Path]]:
    result: dict[int, dict[int, Path]] = {}
    for camera in root.iterdir():
        if not camera.is_dir():
            continue
        for path in camera.iterdir():
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            match = _MEDIA_RE.fullmatch(path.stem)
            if match is None or match.group(2) != kind:
                continue
            camera_id = int(match.group(1)) - 1
            result.setdefault(camera_id, {})[int(match.group(3))] = path
    return result


def _rotation_matrix(rotvec: list[float]):
    import numpy as np

    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _read_calibration(path: Path) -> dict[int, dict]:
    import numpy as np

    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["name"]
            if not name.startswith("Cam"):
                continue
            camera_id = int(name[3:]) - 1
            width, height = int(row["w"]), int(row["h"])
            cam_to_world = _rotation_matrix(
                [float(row[key]) for key in ("rx", "ry", "rz")]
            )
            world_to_cam = cam_to_world.T
            center = np.asarray([float(row[key]) for key in ("tx", "ty", "tz")])
            translation = -(world_to_cam @ center)
            result[camera_id] = {
                "K": [
                    float(row["fx"]) * width, 0.0, float(row["px"]) * width,
                    0.0, float(row["fy"]) * height, float(row["py"]) * height,
                    0.0, 0.0, 1.0,
                ],
                "R": world_to_cam.reshape(-1).tolist(),
                "T": translation.tolist(),
                "distCoeff": [0.0, 0.0, 0.0, 0.0, 0.0],
                "imgSize": [width, height],
            }
    return result


def _clear_stage(destination: Path) -> None:
    for name in (
        "images", "masks", "calibration.json", "smpl_params.npz",
        "camera_id_mapping.json", "frame_validity.json", ".vci_actorshq_stage.json",
    ):
        target = destination / name
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target)


def stage_actorshq_subject(
    subject: dict,
    destination: Path,
    *,
    views: int | None = None,
    time_steps: int | None = None,
    resolution: int | None = None,
    renumber_cameras: bool = True,
) -> Path:
    """Stage one sequence without modifying or duplicating source-resolution media."""
    missing = [key for key in ("images", "masks", "calibration", "smplx") if not subject.get(key)]
    if missing:
        raise SystemExit(
            f"ActorsHQ sequence {subject.get('subject_id')} is missing: {', '.join(missing)}. "
            "ActorsHQ does not ship SMPL-X; add a synchronized smpl_params.npz to the sequence."
        )
    if views is not None and views < 1:
        raise SystemExit("--views must be positive")
    if time_steps is not None and time_steps < 1:
        raise SystemExit("--time-steps must be positive")
    if resolution is not None and resolution < 0:
        raise SystemExit("--resolution must be non-negative")

    sources = {key: str(Path(subject[key]).expanduser().resolve()) for key in ("images", "masks", "calibration", "smplx")}
    signature = {
        "staging_version": 1, "sources": sources,
        "training_config": subject.get("training_config"),
        "evaluation_config": subject.get("evaluation_config"),
        "views": views, "time_steps": time_steps, "resolution": resolution, "renumber_cameras": renumber_cameras,
    }
    marker = destination / ".vci_actorshq_stage.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == signature:
                return destination
        except (OSError, json.JSONDecodeError):
            pass

    images = _media_index(Path(subject["images"]), "rgb")
    masks = _media_index(Path(subject["masks"]), "mask")
    calibration = _read_calibration(Path(subject["calibration"]))
    cameras = sorted(set(images) & set(masks) & set(calibration))
    train = subject.get("training_config", {})
    if "camera_ids" in train:
        unavailable = set(train["camera_ids"]) - set(cameras)
        if unavailable:
            raise SystemExit(f"Unavailable ActorsHQ camera IDs: {sorted(unavailable)}")
        cameras = list(train["camera_ids"])
    elif views is not None:
        cameras = cameras[:views]
    if not cameras:
        raise SystemExit("ActorsHQ sequence has no complete RGB/mask/calibration cameras")

    common_frames = set.intersection(
        *(set(images[camera]) & set(masks[camera]) for camera in cameras)
    )
    frames = selection_frames(train) if train else sorted(common_frames)
    if time_steps is not None and not train:
        frames = frames[:time_steps]
    unavailable_frames = set(frames) - common_frames
    if unavailable_frames:
        raise SystemExit(
            "ActorsHQ selected frames are not available in every selected camera: "
            f"{sorted(unavailable_frames)[:10]}"
        )
    if not frames:
        raise SystemExit("ActorsHQ sequence has no synchronized RGB/mask frames")

    destination.mkdir(parents=True, exist_ok=True)
    _clear_stage(destination)
    output_names = {
        camera: f"cam{index:02d}" if renumber_cameras else f"cam{camera:02d}"
        for index, camera in enumerate(cameras)
    }
    calibration_out = {}
    for camera in cameras:
        output_name = output_names[camera]
        camera_data = dict(calibration[camera])
        scale = 1.0
        if resolution not in (None, 0):
            width, height = camera_data["imgSize"]
            scale = min(float(resolution) / max(width, height), 1.0)
            camera_data["K"] = [
                value * scale if index in (0, 2, 4, 5) else value
                for index, value in enumerate(camera_data["K"])
            ]
            camera_data["imgSize"] = [max(1, int(width * scale)), max(1, int(height * scale))]
        calibration_out[output_name] = camera_data
        for frame in frames:
            image_source, mask_source = images[camera][frame], masks[camera][frame]
            image_target = destination / "images" / output_name / f"{frame:08d}{image_source.suffix.lower()}"
            mask_target = destination / "masks" / output_name / f"{frame:08d}{mask_source.suffix.lower()}"
            if scale == 1.0:
                ensure_link(image_source, image_target)
                ensure_link(mask_source, mask_target)
                continue
            import cv2
            for source, target, interpolation in (
                (image_source, image_target, cv2.INTER_LINEAR),
                (mask_source, mask_target, cv2.INTER_NEAREST),
            ):
                value = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                if value is None:
                    raise SystemExit(f"Could not read ActorsHQ media: {source}")
                size = (max(1, int(value.shape[1] * scale)), max(1, int(value.shape[0] * scale)))
                value = cv2.resize(value, size, interpolation=interpolation)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), value):
                    raise SystemExit(f"Could not write staged ActorsHQ media: {target}")

    (destination / "calibration.json").write_text(json.dumps(calibration_out, indent=2) + "\n")
    (destination / "camera_id_mapping.json").write_text(json.dumps({output_names[camera]: camera for camera in cameras}, indent=2) + "\n")
    (destination / "frame_validity.json").write_text(json.dumps({"frame_ids": {"ids": frames, "start": frames[0], "end": frames[-1], "count": len(frames)}, "invalid_frames": {}}, indent=2) + "\n")
    ensure_link(Path(subject["smplx"]), destination / "smpl_params.npz")
    marker.write_text(json.dumps(signature, indent=2) + "\n")
    return destination


def staged_actorshq_subject(subject: dict, data_dir: Path) -> dict:
    """Return a manifest subject pointing at an already-created stage."""
    staged = dict(subject)
    staged.update(
        images=str(data_dir / "images"), masks=str(data_dir / "masks"),
        calibration=str(data_dir / "calibration.json"),
        camera_id_mapping=str(data_dir / "camera_id_mapping.json"),
        smplx=str(data_dir / "smpl_params.npz"), missing_keys=[],
    )
    if not staged.get("training_config"):
        validity = json.loads((data_dir / "frame_validity.json").read_text())
        frames = validity["frame_ids"]["ids"]
        cameras = sorted(path for path in (data_dir / "images").iterdir() if path.is_dir())
        staged["training_config"] = {
            "camera_ids": list(range(len(cameras))),
            "frame_range": {"start": frames[0], "stop": frames[-1] + 1},
            "resolution": 0,
        }
    return staged


def stage_actorshq_evaluation_source(subject: dict, destination: Path) -> dict:
    """Stage every downloaded observation for held-out evaluation metadata."""
    source_subject = dict(subject)
    source_subject.pop("training_config", None)
    source_subject.pop("evaluation_config", None)
    data_dir = stage_actorshq_subject(
        source_subject,
        destination,
        views=None,
        time_steps=None,
        renumber_cameras=False,
        resolution=0,
    )
    staged = staged_actorshq_subject(source_subject, data_dir)
    staged["subject_path"] = subject.get("subject_path")
    if subject.get("training_config") is not None:
        staged["training_config"] = subject["training_config"]
    else:
        staged.pop("training_config", None)
    if subject.get("evaluation_config") is not None:
        staged["evaluation_config"] = subject["evaluation_config"]
    return staged
