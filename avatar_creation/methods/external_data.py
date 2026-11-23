"""Small, dependency-free helpers for staging canonical data for upstream methods."""

from __future__ import annotations

import json
import shutil

from avatar_creation.methods.training_inputs import selection_frames
from pathlib import Path

from multiview_data import Sequence
from multiview_data.reader import index_image_files


CANONICAL_ASSETS = ("images", "masks", "calibration", "smplx")


def _ignore_missing_rmtree_entry(_function, _path, exc_info) -> None:
    """Let cleanup continue when a concurrently removed staged entry vanished."""
    error = exc_info[1]
    if isinstance(error, FileNotFoundError):
        return
    raise error


def load_subjects(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise SystemExit(f"Dataset manifest has no subjects: {manifest_path}")
    return subjects


def safe_subject_id(value: object) -> str:
    raw = str(value)
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)
    return safe or "subject"


def ensure_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Required dataset asset does not exist: {source}")
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise SystemExit(
            f"Refusing to replace existing staged asset: {destination}. "
            "Remove the method output directory to restage it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _clear_staged_assets(destination: Path) -> None:
    for name in (
        "images",
        "masks",
        "calibration.json",
        "smpl_params.npz",
        "frame_validity.json",
        "camera_id_mapping.json",
        ".vci_thuman_stage.json",
    ):
        target = destination / name
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target, onerror=_ignore_missing_rmtree_entry)


def stage_thuman_subject(
    subject: dict,
    destination: Path,
    *,
    views: int | None = None,
    time_steps: int | None = None,
    resolution: int | None = None,
    renumber_cameras: bool = False,
) -> Path:
    """Stage native THuman inputs in method output, never writing to the source."""
    missing = [key for key in CANONICAL_ASSETS if not subject.get(key)]
    if missing:
        raise SystemExit(
            f"Subject {subject.get('subject_id', '<unknown>')} is missing: "
            + ", ".join(missing)
        )
    if views is not None and views < 1:
        raise SystemExit("--views must be positive")
    if time_steps is not None and time_steps < 1:
        raise SystemExit("--time-steps must be positive")
    if resolution is not None and resolution < 0:
        raise SystemExit("--resolution must be non-negative")
    destination.mkdir(parents=True, exist_ok=True)
    if subject.get("dataset") == "processed" or renumber_cameras or subject.get("training_config") or views is not None or time_steps is not None or resolution not in (None, 0):
        sources = {
            key: str(Path(subject[key]).expanduser().resolve())
            for key in CANONICAL_ASSETS
        }
        signature = {
            "staging_version": 6,
            "renumber_cameras": renumber_cameras,
            "training_config": subject.get("training_config"),
            "sources": sources,
            "views": views,
            "time_steps": time_steps,
            "resolution": resolution,
        }
        marker = destination / ".vci_thuman_stage.json"
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == signature:
                    return destination
            except (OSError, json.JSONDecodeError):
                pass

        images_source = Path(subject["images"]).expanduser().resolve()
        masks_source = Path(subject["masks"]).expanduser().resolve()
        image_cameras = {
            path.name: path for path in images_source.iterdir() if path.is_dir()
        }
        mask_cameras = {
            path.name: path for path in masks_source.iterdir() if path.is_dir()
        }
        camera_names = sorted(set(image_cameras) & set(mask_cameras))
        train = subject.get("training_config", {})
        if "camera_ids" in train:
            by_id = {int(name[3:]): name for name in camera_names}
            missing = set(train["camera_ids"]) - set(by_id)
            if missing:
                raise SystemExit(f"Unavailable camera IDs: {sorted(missing)}")
            camera_names = sorted(by_id[i] for i in train["camera_ids"])
        elif views is not None:
            camera_names = camera_names[:views]
        if not camera_names:
            raise SystemExit("No common image/mask cameras found for limited staging")

        sequence = Sequence({"dataset": "thuman", "subject_id": destination.name,
                             "subject_path": images_source.parent, **subject})
        path_indexes = {camera: sequence.observation_paths(camera) for camera in camera_names}
        image_files = {camera: indexes[0] for camera, indexes in path_indexes.items()}
        mask_files = {camera: indexes[1] for camera, indexes in path_indexes.items()}
        valid_frames = {
            camera: set(image_files[camera]) & set(mask_files[camera])
            for camera in camera_names
        }
        observed_frames = set().union(*valid_frames.values())
        if not observed_frames:
            raise SystemExit("No valid image/mask pairs found for limited staging")
        frame_ids = (
            list(range(time_steps))
            if time_steps is not None
            else list(range(min(observed_frames), max(observed_frames) + 1))
        )
        if train:
            frame_ids = selection_frames(train)
        calibration = json.loads(
            Path(subject["calibration"]).read_text(encoding="utf-8")
        )
        missing_calibration = [name for name in camera_names if name not in calibration]
        if missing_calibration:
            raise SystemExit(
                "Calibration is missing selected cameras: "
                + ", ".join(missing_calibration)
            )

        _clear_staged_assets(destination)
        output_names = {name: f"cam{index:02d}" if renumber_cameras else name
                        for index, name in enumerate(camera_names)}
        calibration_out = {output_names[name]: dict(calibration[name]) for name in camera_names}
        invalid_frames: dict[str, list[int]] = {}
        selected_frame_ids = set(frame_ids)
        for camera in camera_names:
            output_camera = output_names[camera]
            scale = 1.0
            camera_valid_frames = valid_frames[camera] & selected_frame_ids
            if not camera_valid_frames:
                raise SystemExit(
                    f"Selected camera {camera} has no valid frames in the requested timeline"
                )
            if resolution not in (None, 0):
                try:
                    import cv2
                except ModuleNotFoundError as exc:
                    raise SystemExit("Resolution-limited staging requires OpenCV in the method environment.") from exc
                sample_frame = min(camera_valid_frames)
                sample = cv2.imread(str(image_files[camera][sample_frame]), cv2.IMREAD_UNCHANGED)
                if sample is None:
                    raise SystemExit(f"Could not read source image: {image_files[camera][sample_frame]}")
                height, width = sample.shape[:2]
                scale = min(float(resolution) / max(width, height), 1.0)
                intrinsics = calibration_out[output_camera].get("K")
                if not isinstance(intrinsics, list) or len(intrinsics) != 9:
                    raise SystemExit(f"Calibration K for {camera} must contain nine values")
                intrinsics = list(intrinsics)
                dst_width, dst_height = max(1, int(width * scale)), max(1, int(height * scale))
                intrinsics[:3] = [value * dst_width / width for value in intrinsics[:3]]
                intrinsics[3:6] = [value * dst_height / height for value in intrinsics[3:6]]
                calibration_out[output_camera]["K"] = intrinsics
                calibration_out[output_camera]["imgSize"] = [dst_width, dst_height]
            invalid_frames[output_camera] = []
            for frame_id in frame_ids:
                if frame_id not in camera_valid_frames:
                    invalid_frames[output_camera].append(frame_id)
                    continue
                image_source = image_files[camera][frame_id]
                mask_source = mask_files[camera][frame_id]
                image_destination = destination / "images" / output_camera / (image_source.name if image_source.stem.isdigit() else f"{frame_id:08d}{image_source.suffix.lower()}")
                mask_destination = destination / "masks" / output_camera / (mask_source.name if mask_source.stem.isdigit() else f"{frame_id:08d}{mask_source.suffix.lower()}")
                if resolution in (None, 0):
                    ensure_link(image_source, image_destination)
                    ensure_link(mask_source, mask_destination)
                    continue
                image = cv2.imread(str(image_source), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(mask_source), cv2.IMREAD_UNCHANGED)
                if image is None or mask is None:
                    raise SystemExit(f"Could not read image or mask for {camera}, frame {frame_id}")
                new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
                if scale < 1.0:
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                mask_destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_destination), image):
                    raise SystemExit(f"Could not write staged image: {image_destination}")
                if not cv2.imwrite(str(mask_destination), mask):
                    raise SystemExit(f"Could not write staged mask: {mask_destination}")
        (destination / "calibration.json").write_text(
            json.dumps(calibration_out, indent=2) + "\n",
            encoding="utf-8",
        )
        if renumber_cameras:
            (destination / "camera_id_mapping.json").write_text(json.dumps({
                output_names[name]: int(name[3:]) for name in camera_names
            }, indent=2) + "\n")
        validity = {
            "frame_ids": {
                **({"ids": frame_ids} if "frame_range" in train else {}),
                "start": frame_ids[0],
                "end": frame_ids[-1],
                "count": len(frame_ids),
            },
            "invalid_frames": invalid_frames,
        }
        (destination / "frame_validity.json").write_text(
            json.dumps(validity, indent=2) + "\n", encoding="utf-8"
        )
        ensure_link(Path(subject["smplx"]), destination / "smpl_params.npz")
        marker.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
        invalid_count = sum(len(ids) for ids in invalid_frames.values())
        print(
            f"Staged {len(camera_names)} camera(s) and fixed timesteps "
            f"{frame_ids[0]}–{frame_ids[-1]} for {subject.get('subject_id', '<unknown>')} "
            f"({invalid_count} invalid camera/timestep pair(s) omitted)"
            + (f" at a maximum resolution of {resolution}px." if resolution not in (None, 0) else ".")
        )
        return destination

    names = {
        "images": "images",
        "masks": "masks",
        "calibration": "calibration.json",
        "smplx": "smpl_params.npz",
    }
    for key, name in names.items():
        ensure_link(Path(subject[key]), destination / name)
    return destination


def numeric_frame_ids(images: Path) -> list[int]:
    camera_dirs = sorted(path for path in images.iterdir() if path.is_dir())
    if not camera_dirs:
        raise SystemExit(f"Expected camera directories below {images}")
    per_camera = []
    for camera in camera_dirs:
        ids = set(index_image_files(camera))
        per_camera.append(ids)
    common = set.intersection(*per_camera)
    if not common:
        raise SystemExit(f"No common numeric frames found below {images}")
    return sorted(common)


def numeric_frame_ids_union(images: Path) -> list[int]:
    camera_dirs = sorted(path for path in images.iterdir() if path.is_dir())
    if not camera_dirs:
        raise SystemExit(f"Expected camera directories below {images}")
    frame_ids = {
        int(path.stem)
        for camera in camera_dirs
        for path in camera.iterdir()
        if path.is_file() and path.stem.isdigit()
    }
    if not frame_ids:
        raise SystemExit(f"No numeric frames found below {images}")
    return sorted(frame_ids)


def camera_ids(images: Path) -> list[int]:
    cameras = sorted(path for path in images.iterdir() if path.is_dir())
    if not cameras:
        raise SystemExit(f"Expected camera directories below {images}")
    return list(range(len(cameras)))


def source_extra(subject: dict, *parts: str) -> Path | None:
    subject_path = subject.get("subject_path")
    if not subject_path:
        return None
    candidate = Path(subject_path).expanduser().resolve().joinpath(*parts)
    return candidate if candidate.exists() else None


def staged_training_frames(data_dir: Path, time_steps: int | None) -> list[int]:
    """Keep the requested timeline, including timesteps with no observations."""
    validity = data_dir / "frame_validity.json"
    if validity.is_file():
        ids = json.loads(validity.read_text()).get("frame_ids", {}).get("ids")
        if ids:
            return ids
    if time_steps is not None:
        return list(range(time_steps))
    observed = numeric_frame_ids_union(data_dir / "images")
    return list(range(observed[0], observed[-1] + 1))


def staged_frame_pairs(data_dir: Path, frames: list[int]) -> tuple[list[tuple[int, int]], list[str]]:
    """Return usable evaluation pairs and the upstream THuman missing-file list."""
    valid, missing = [], []
    cameras = sorted(path for path in (data_dir / "images").iterdir() if path.is_dir())
    for camera_index, camera in enumerate(cameras):
        image_ids = set(index_image_files(camera))
        mask_ids = set(index_image_files(data_dir / "masks" / camera.name))
        for frame in frames:
            if frame in image_ids and frame in mask_ids:
                valid.append((frame, camera_index))
            else:
                missing.append(f"images/{camera.name}/{frame:08d}.jpg")
    if not valid:
        raise SystemExit("Selected timeline has no valid image/mask pairs")
    return sorted(valid), missing
