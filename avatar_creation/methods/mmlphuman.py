"""Manifest adapter for the unmodified MMLP Human checkout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from multiview_data.dna_reader import close_reader as _close_reader, load_reader as _load_dna_reader

import yaml

from avatar_creation.methods.actorshq_data import (
    stage_actorshq_evaluation_source,
    stage_actorshq_subject,
)
from avatar_creation.methods.external_data import (
    camera_ids,
    ensure_link,
    load_subjects,
    staged_training_frames,
    staged_frame_pairs,
    safe_subject_id,
    source_extra,
    stage_thuman_subject,
)
from avatar_creation.methods.evaluation_split import write_evaluation_split
from avatar_creation.methods.training_inputs import validate_dna_inputs, selection_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = PROJECT_ROOT / "external" / "mmlphuman"
MMLPHUMAN_NUM_BETAS = 10


def _normalize_smpl_betas(data_dir: Path) -> None:
    """Materialize the staged SMPL-X parameters with MMLP Human's beta count."""
    import numpy as np

    params_path = data_dir / "smpl_params.npz"
    try:
        with np.load(params_path, allow_pickle=True) as archive:
            params = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not read staged SMPL-X parameters: {params_path}") from exc

    if "betas" not in params:
        raise SystemExit(f"Staged SMPL-X parameters have no 'betas' array: {params_path}")
    betas = np.asarray(params["betas"])
    if betas.ndim == 0:
        raise SystemExit(f"Staged SMPL-X betas must be an array: {params_path}")
    source_count = betas.shape[-1]
    if source_count == MMLPHUMAN_NUM_BETAS:
        return

    if source_count > MMLPHUMAN_NUM_BETAS:
        normalized = betas[..., :MMLPHUMAN_NUM_BETAS]
    else:
        normalized = np.pad(
            betas,
            [(0, 0)] * (betas.ndim - 1)
            + [(0, MMLPHUMAN_NUM_BETAS - source_count)],
        )
    params["betas"] = normalized

    temporary = params_path.with_name(f".{params_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **params)
        os.replace(temporary, params_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"[preprocess] Normalized SMPL-X betas from {source_count} to "
        f"{MMLPHUMAN_NUM_BETAS} in the run-local staging directory.",
        flush=True,
    )


def _ensure_camera_distortion(data_dir: Path) -> None:
    """Add MMLP Human's zero-distortion default to run-local calibration."""
    calibration_path = data_dir / "calibration.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read staged calibration: {calibration_path}") from exc

    missing = []
    for camera_name, camera in calibration.items():
        if not isinstance(camera, dict):
            raise SystemExit(f"Invalid calibration entry for {camera_name}: {calibration_path}")
        if "distCoeff" not in camera:
            calibration[camera_name] = {**camera, "distCoeff": [0.0] * 5}
            missing.append(camera_name)
    if not missing:
        return

    temporary = calibration_path.with_name(
        f".{calibration_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, calibration_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"[preprocess] Added zero distortion coefficients for {len(missing)} camera(s) "
        "in the run-local calibration.",
        flush=True,
    )


def _ensure_jpg_media_aliases(data_dir: Path) -> None:
    """Expose staged media through the .jpg names hardcoded by MMLP Human."""
    alias_count = 0
    for media_name in ("images", "masks"):
        media_dir = data_dir / media_name
        for camera_dir in sorted(path for path in media_dir.iterdir() if path.is_dir()):
            for source in sorted(camera_dir.iterdir()):
                if (
                    not source.is_file()
                    or not source.stem.isdigit()
                    or source.suffix.lower() == ".jpg"
                ):
                    continue
                alias = source.with_suffix(".jpg")
                if alias.exists():
                    continue
                ensure_link(source, alias)
                alias_count += 1
    if alias_count:
        print(
            f"[preprocess] Added {alias_count} run-local .jpg media aliases "
            "required by MMLP Human.",
            flush=True,
        )


def _dna_stage_signature(
    subject: dict, views: int | None, time_steps: int | None, resolution: int | None
) -> dict:
    sources = {}
    for key in ("images", "masks", "calibration", "smplx"):
        source = Path(subject[key]).expanduser().resolve()
        stat = source.stat()
        sources[key] = {
            "path": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {"staging_version": 2, "sources": sources, "views": views, "time_steps": time_steps, "resolution": resolution, "training_config": subject.get("training_config"), "evaluation_config": subject.get("evaluation_config")}


def _clear_dna_stage(destination: Path) -> None:
    for name in (
        "images",
        "masks",
        "calibration.json",
        "smpl_params.npz",
        ".vci_dna_stage.json",
        "frame_validity.json",
        "camera_id_mapping.json",
    ):
        target = destination / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)


def stage_dna_subject(
    subject: dict,
    destination: Path,
    *,
    views: int | None = None,
    time_steps: int | None = None,
    resolution: int | None = None,
    reader_class=None,
    lossless: bool = False,
) -> Path:
    import cv2
    import numpy as np

    missing = [
        key
        for key in ("images", "masks", "calibration", "smplx")
        if not subject.get(key)
    ]
    if missing:
        raise SystemExit("DNA subject is missing SMC assets: " + ", ".join(missing))
    if views is not None and views < 1:
        raise SystemExit("--views must be positive")
    if time_steps is not None and time_steps < 1:
        raise SystemExit("--time-steps must be positive")

    if resolution is not None and resolution < 0:
        raise SystemExit("--resolution must be non-negative")

    signature = _dna_stage_signature(subject, views, time_steps, resolution)
    signature["lossless"] = lossless
    marker = destination / ".vci_dna_stage.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == signature:
                return destination
        except (OSError, json.JSONDecodeError):
            pass

    destination.mkdir(parents=True, exist_ok=True)
    _clear_dna_stage(destination)
    images_dir = destination / "images"
    masks_dir = destination / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    reader_type = reader_class or _load_dna_reader(subject)
    rgb_reader = reader_type(str(Path(subject["images"]).expanduser().resolve()))
    annot_reader = reader_type(str(Path(subject["smplx"]).expanduser().resolve()))
    try:
        validate_dna_inputs(subject, rgb_reader, annot_reader)
        camera_data = annot_reader.get_Calibration_all()
        camera_ids_source = sorted(
            int(camera_id) for camera_id in camera_data if int(camera_id) < 48
        )
        train = subject.get("training_config", {})
        if "camera_ids" in train:
            camera_ids_source = train["camera_ids"]
        elif views is not None:
            camera_ids_source = camera_ids_source[:views]
        if not camera_ids_source:
            raise SystemExit("DNA subject has no 5MP RGB cameras")

        rgb_info = rgb_reader.get_Camera_5mp_info() or {}
        available_frames = int(rgb_info.get("num_frame", 0))
        smplx = annot_reader.get_SMPLx()
        available_frames = min(
            available_frames, int(np.asarray(smplx["transl"]).shape[0])
        )
        frame_count = (
            min(available_frames, time_steps)
            if time_steps is not None
            else available_frames
        )
        if frame_count < 1:
            raise SystemExit("DNA subject has no synchronized RGB and SMPL-X frames")
        frame_ids = selection_frames(train) if train else list(range(frame_count))
        frame_count = len(frame_ids)

        calibration_out = {}
        for output_camera_id, source_camera_id in enumerate(camera_ids_source):
            camera_name = f"cam{output_camera_id:02d}"
            image_camera = images_dir / camera_name
            mask_camera = masks_dir / camera_name
            image_camera.mkdir()
            mask_camera.mkdir()
            print(
                f"Decoding DNA camera {source_camera_id:02d} as {camera_name} "
                f"({frame_count} frames)",
                flush=True,
            )
            camera_scale = None
            for output_frame_id, source_frame_id in enumerate(frame_ids):
                image = rgb_reader.get_img(
                    "Camera_5mp" if source_camera_id < 48 else "Camera_12mp",
                    source_camera_id,
                    "color",
                    Frame_id=source_frame_id,
                    disable_tqdm=True,
                )
                mask = annot_reader.get_mask(
                    source_camera_id,
                    Frame_id=source_frame_id,
                    disable_tqdm=True,
                )
                if image is None or mask is None:
                    raise SystemExit(
                        f"DNA decoding failed for camera {source_camera_id}, frame {source_frame_id}"
                    )
                if camera_scale is None:
                    height, width = image.shape[:2]
                    camera_scale = min(float(resolution) / max(width, height), 1.0) if resolution not in (None, 0) else 1.0
                if camera_scale < 1.0:
                    new_size = (max(1, int(image.shape[1] * camera_scale)), max(1, int(image.shape[0] * camera_scale)))
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
                filename = f"{source_frame_id:08d}.{'png' if lossless else 'jpg'}"
                if not cv2.imwrite(str(image_camera / filename), image, ([] if lossless else [cv2.IMWRITE_JPEG_QUALITY, 95])):
                    raise SystemExit(f"Could not write staged image {image_camera / filename}")
                if not cv2.imwrite(str(mask_camera / filename), mask, ([] if lossless else [cv2.IMWRITE_JPEG_QUALITY, 100])):
                    raise SystemExit(f"Could not write staged mask {mask_camera / filename}")

            source_camera = camera_data[f"{source_camera_id:02d}"]
            camera_to_world = np.asarray(source_camera["RT"], dtype=np.float32)
            world_to_camera = np.linalg.inv(camera_to_world)
            intrinsics = np.asarray(source_camera["K"], dtype=np.float32).copy()
            intrinsics[0, 0] *= camera_scale
            intrinsics[1, 1] *= camera_scale
            intrinsics[0, 2] *= camera_scale
            intrinsics[1, 2] *= camera_scale
            calibration_out[camera_name] = {
                "K": intrinsics.reshape(-1).tolist(),
                "R": world_to_camera[:3, :3].reshape(-1).tolist(),
                "T": world_to_camera[:3, 3].tolist(),
                "distCoeff": np.asarray(source_camera["D"], dtype=np.float32).reshape(-1).tolist(),
            }

        (destination / "calibration.json").write_text(
            json.dumps(calibration_out, indent=2), encoding="utf-8"
        )
        (destination / "frame_validity.json").write_text(json.dumps({"frame_ids": {"ids": frame_ids}}))
        (destination / "camera_id_mapping.json").write_text(json.dumps({
            f"cam{i:02d}": camera for i, camera in enumerate(camera_ids_source)
        }))
        # Preserve source frame indexing and held-out poses for evaluation.
        pose_count = available_frames if train else frame_count
        fullpose = np.asarray(smplx["fullpose"], dtype=np.float32)[:pose_count]
        betas = np.asarray(smplx["betas"], dtype=np.float32)[:pose_count].mean(axis=0, keepdims=True)
        np.savez(
            destination / "smpl_params.npz",
            betas=betas,
            global_orient=fullpose[:, 0, :3].reshape(pose_count, -1),
            body_pose=fullpose[:, 1:22, :].reshape(pose_count, -1),
            jaw_pose=fullpose[:, 22, :].reshape(pose_count, -1),
            leye_pose=fullpose[:, 23, :].reshape(pose_count, -1),
            reye_pose=fullpose[:, 24, :].reshape(pose_count, -1),
            left_hand_pose=fullpose[:, 25:40, :].reshape(pose_count, -1),
            right_hand_pose=fullpose[:, 40:55, :].reshape(pose_count, -1),
            expression=np.asarray(smplx["expression"], dtype=np.float32)[:pose_count],
            transl=np.asarray(smplx["transl"], dtype=np.float32)[:pose_count],
        )
        marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    finally:
        _close_reader(rgb_reader)
        _close_reader(annot_reader)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=UPSTREAM / "config" / "subject01.yaml")
    parser.add_argument("--ip", default="127.0.0.1")
    # Parallel run_methods workers share the host network namespace, so give each
    # active GPU slot its own visualization port. Sequential runs retain 23456.
    slot_index = int(os.environ.get("VCI_DEVICE_SLOT_INDEX", "0"))
    parser.add_argument("--port", type=int, default=23456 + slot_index)
    parser.add_argument("--views", type=int, default=None, help="cameras to stage; default: all")
    parser.add_argument("--time-steps", type=int, default=None, help="synchronized frames to stage; default: all")
    parser.add_argument("--resolution", type=int, default=1024, help="maximum image dimension; default: 1024, use 0 to preserve source resolution")
    return parser.parse_args()


def _generate_lbs_weights(data_dir: Path, subject_out: Path) -> None:
    weights = data_dir / "gaussian" / "lbs_weights_grid.npz"
    if weights.is_file():
        return
    model = UPSTREAM / "smpl_model" / "smplx" / "SMPLX_NEUTRAL.npz"
    point_interpolant = Path(sys.prefix) / "bin" / "PointInterpolant"
    if not model.is_file():
        raise SystemExit(
            f"MMLP Human needs the licensed neutral SMPL-X model at {model}."
        )
    if not point_interpolant.is_file():
        raise SystemExit(
            f"MMLP Human preprocessing needs {point_interpolant}. "
            "Run scripts/setup/setup_mmlphuman_env.sh to install PointInterpolant; "
            "the adapter will run it from a run-local copy."
        )
    print(
        f"[preprocess] Generating MMLP Human LBS weight volume for {subject_out.name}. "
        "This CPU-intensive step can take several minutes.",
        flush=True,
    )
    preprocess_dir = subject_out / "lbs_preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(UPSTREAM / "script" / "gen_weight_volume.py", preprocess_dir)
    shutil.copy2(point_interpolant, preprocess_dir)
    (data_dir / "gaussian").mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "gen_weight_volume.py",
            "--data_dir",
            str(data_dir),
            "--smpl_path",
            str(model),
        ],
        cwd=preprocess_dir,
        check=True,
    )
    if not weights.is_file():
        raise SystemExit(f"LBS preprocessing did not create {weights}")
    print(f"[preprocess] Generated {weights}", flush=True)


def prepare_subject(
    subject: dict,
    output_dir: Path,
    template: dict,
    *,
    views: int | None = None,
    time_steps: int | None = None,
    resolution: int | None = None,
) -> tuple[Path, Path]:
    subject_id = safe_subject_id(subject.get("subject_id", "subject"))
    subject_out = output_dir / "subjects" / subject_id
    if subject.get("dataset") == "dna":
        data_dir = stage_dna_subject(
            subject, subject_out / "data", views=views, time_steps=time_steps, resolution=resolution
        )
    elif subject.get("dataset") == "actorshq":
        data_dir = stage_actorshq_subject(
            subject, subject_out / "data", views=views, time_steps=time_steps, resolution=resolution
        )
    else:
        data_dir = stage_thuman_subject(
            subject, subject_out / "data", views=views, time_steps=time_steps, resolution=resolution
        )
    _normalize_smpl_betas(data_dir)
    _ensure_camera_distortion(data_dir)
    _ensure_jpg_media_aliases(data_dir)
    gaussian_dir = data_dir / "gaussian"
    gaussian_dir.mkdir(exist_ok=True)
    weights_source = source_extra(subject, "gaussian", "lbs_weights_grid.npz")
    if weights_source:
        ensure_link(weights_source, gaussian_dir / "lbs_weights_grid.npz")
    # A dataset-provided template is optional upstream. Keep it out of staged
    # runs so MMLPHuman initializes the template from the SMPL-X mesh instead.
    template_path = gaussian_dir / "template.ply"
    if template_path.is_symlink():
        template_path.unlink()
    weights = gaussian_dir / "lbs_weights_grid.npz"
    if not weights.exists():
        _generate_lbs_weights(data_dir, subject_out)
    if not weights.exists():
        raise SystemExit(
            f"MMLP Human preprocessing is missing for {subject_id}: {weights}. "
            "Generate gaussian/lbs_weights_grid.npz with the upstream "
            "script/gen_weight_volume.py instructions, then rerun."
        )

    frames = staged_training_frames(data_dir, time_steps)
    valid_pairs, _ = staged_frame_pairs(data_dir, frames)
    cams = camera_ids(data_dir / "images")
    evaluation_subject = subject
    if subject.get("dataset") == "actorshq":
        evaluation_subject = stage_actorshq_evaluation_source(
            subject, subject_out / "evaluation_source"
        )
    write_evaluation_split(
        subject_out,
        evaluation_subject,
        train_cameras=cams,
        resolution=resolution,
    )
    config = dict(template)
    config["train_cam_ids"] = cams
    config["begin_ith_frame"] = frames[0]
    config["num_train_frame"] = len(frames)
    config["frame_interval"] = frames[1] - frames[0] if len(frames) > 1 else 1
    config["test"] = dict(config.get("test", {}))
    config["test"].update(
        {"cam_ids": [valid_pairs[-1][1]], "begin_ith_frame": valid_pairs[-1][0], "num_frame": 1, "frame_interval": 1}
    )
    config_path = subject_out / "mmlphuman.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_dir, config_path


def main() -> int:
    args = parse_args()
    if not (UPSTREAM / "train.py").is_file():
        raise SystemExit(
            "MMLP Human checkout is missing. Run scripts/setup/setup_mmlphuman_env.sh."
        )
    template = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for subject in load_subjects(args.dataset_manifest):
        subject_id = safe_subject_id(subject.get("subject_id", "subject"))
        data_dir, config_path = prepare_subject(
            subject, args.output_dir, template, views=args.views, time_steps=args.time_steps, resolution=args.resolution
        )
        model_dir = args.output_dir / "subjects" / subject_id / "model"
        command = [
            sys.executable, "train.py", "--config", str(config_path),
            "--data_dir", str(data_dir), "--out_dir", str(model_dir),
            "--ip", args.ip, "--port", str(args.port),
        ]
        subprocess.run(command, cwd=UPSTREAM, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
