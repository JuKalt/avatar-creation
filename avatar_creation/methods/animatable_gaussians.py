"""Manifest adapter for the unmodified AnimatableGaussians checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = PROJECT_ROOT / "external" / "AnimatableGaussians"
ANIMATABLE_GAUSSIANS_NUM_BETAS = 10


def _upstream_environment() -> dict[str, str]:
    """Keep VCI's top-level modules from shadowing upstream packages."""
    env = os.environ.copy()
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        return env
    excluded = {PROJECT_ROOT.resolve()}
    configured_root = env.get("VCI_DATASET_REPO")
    if configured_root:
        excluded.add(Path(configured_root).expanduser().resolve())
    entries = [
        entry
        for entry in pythonpath.split(os.pathsep)
        if not entry or Path(entry).expanduser().resolve() not in excluded
    ]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        env.pop("PYTHONPATH", None)
    return env


def _normalize_smpl_betas(data_dir: Path) -> None:
    """Materialize SMPL-X parameters with Animatable Gaussians' beta count."""
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
    if source_count == ANIMATABLE_GAUSSIANS_NUM_BETAS:
        return

    if source_count > ANIMATABLE_GAUSSIANS_NUM_BETAS:
        normalized = betas[..., :ANIMATABLE_GAUSSIANS_NUM_BETAS]
    else:
        normalized = np.pad(
            betas,
            [(0, 0)] * (betas.ndim - 1)
            + [(0, ANIMATABLE_GAUSSIANS_NUM_BETAS - source_count)],
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
        f"{ANIMATABLE_GAUSSIANS_NUM_BETAS} in the run-local staging directory.",
        flush=True,
    )


def _ensure_camera_image_sizes(data_dir: Path) -> None:
    """Add image dimensions required by the upstream THuman camera loader."""
    calibration_path = data_dir / "calibration.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read staged calibration: {calibration_path}") from exc

    missing = [
        camera_name
        for camera_name, camera in calibration.items()
        if isinstance(camera, dict) and "imgSize" not in camera
    ]
    if not missing:
        return

    import cv2

    for camera_name in missing:
        camera_dir = data_dir / "images" / camera_name
        sample = next(
            (
                path
                for path in sorted(camera_dir.iterdir())
                if path.is_file() and path.stem.isdigit()
            ),
            None,
        )
        if sample is None:
            raise SystemExit(f"No staged images found for camera {camera_name}")
        image = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"Could not read staged image dimensions: {sample}")
        height, width = image.shape[:2]
        calibration[camera_name] = {
            **calibration[camera_name],
            "imgSize": [width, height],
        }

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
        f"[preprocess] Added image sizes for {len(missing)} camera(s) "
        "in the run-local calibration.",
        flush=True,
    )


def _ensure_jpg_media_aliases(data_dir: Path) -> None:
    """Expose staged media through the .jpg names hardcoded upstream."""
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
                if alias.exists() or alias.is_symlink():
                    continue
                alias.symlink_to(source.name)
                alias_count += 1
    if alias_count:
        print(
            f"[preprocess] Added {alias_count} run-local .jpg media aliases "
            "required by Animatable Gaussians.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=UPSTREAM / "configs" / "subject01" / "avatar.yaml")
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="require existing smpl_pos_map assets instead of generating them",
    )
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=0)
    return parser.parse_args()


def prepare_subject(
    subject: dict,
    output_dir: Path,
    template: dict,
    *,
    views: int | None = None,
    time_steps: int | None = None,
    resolution: int = 0,
) -> tuple[Path, Path, bool]:
    subject_id = safe_subject_id(subject.get("subject_id", "subject"))
    subject_out = output_dir / "subjects" / subject_id
    if subject.get("dataset") == "actorshq":
        data_dir = stage_actorshq_subject(
            subject, subject_out / "data", views=views, time_steps=time_steps,
            resolution=resolution,
        )
    else:
        data_dir = stage_thuman_subject(
            subject, subject_out / "data", views=views, time_steps=time_steps,
            resolution=resolution, renumber_cameras=True,
        )
    _normalize_smpl_betas(data_dir)
    _ensure_camera_image_sizes(data_dir)
    _ensure_jpg_media_aliases(data_dir)
    pos_maps = source_extra(subject, "smpl_pos_map")
    if pos_maps:
        staged_maps = data_dir / "smpl_pos_map"
        staged_maps.mkdir(exist_ok=True)
        for item in pos_maps.iterdir():
            if item.is_file():
                ensure_link(item, staged_maps / item.name)
    weight_volume = source_extra(subject, "cano_weight_volume.npz")
    if weight_volume:
        ensure_link(weight_volume, data_dir / "cano_weight_volume.npz")
    # Let upstream's position-map preprocessing use its SMPL-X template fallback.
    template_path = data_dir / "template.ply"
    if template_path.is_symlink():
        template_path.unlink()

    frames = staged_training_frames(data_dir, time_steps)
    valid_pairs, missing = staged_frame_pairs(data_dir, frames)
    (data_dir / "missing_img_files.txt").write_text(
        "".join(line + "\n" for line in missing), encoding="utf-8"
    )
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
    config["train"] = dict(config["train"])
    config["train"]["data"] = dict(config["train"]["data"])
    config["train"]["data"].update(
        {
            "subject_name": subject_id,
            "data_dir": str(data_dir),
            "frame_range": [frames[0], frames[-1] + 1, frames[1] - frames[0] if len(frames) > 1 else 1],
            "used_cam_ids": cams,
        }
    )
    config["train"].update(
        {
            "net_ckpt_dir": str(subject_out / "model"),
            "pretrained_dir": None,
            "prev_ckpt": None,
            "eval_training_ids": list(valid_pairs[0]),
            "eval_testing_ids": list(valid_pairs[-1]),
        }
    )
    config["test"] = dict(config["test"])
    config["test"]["data"] = dict(config["test"]["data"])
    config["test"]["data"].update(
        {"subject_name": subject_id, "data_dir": str(data_dir), "frame_range": [frames[0], frames[-1] + 1, frames[1] - frames[0] if len(frames) > 1 else 1]}
    )
    config_path = subject_out / "animatable_gaussians.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_dir, config_path, (data_dir / "smpl_pos_map" / "cano_smpl_pos_map.exr").is_file()


def main() -> int:
    args = parse_args()
    if not (UPSTREAM / "main_avatar.py").is_file():
        raise SystemExit(
            "AnimatableGaussians checkout is missing. "
            "Run scripts/setup/setup_animatable_gaussians_env.sh."
        )
    template = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for subject in load_subjects(args.dataset_manifest):
        _, config_path, prepared = prepare_subject(
            subject, args.output_dir, template, views=args.views,
            time_steps=args.time_steps, resolution=args.resolution
        )
        upstream_env = _upstream_environment()
        if not prepared:
            if args.no_preprocess:
                raise SystemExit(
                    "Animatable Gaussians position maps are missing. Remove --no-preprocess "
                    "or provide smpl_pos_map in the native subject directory."
                )
            subprocess.run(
                [sys.executable, "-m", "gen_data.gen_pos_maps", "-c", str(config_path)],
                cwd=UPSTREAM,
                env=upstream_env,
                check=True,
            )
        subprocess.run(
            [sys.executable, "main_avatar.py", "-c", str(config_path), "--mode=train"],
            cwd=UPSTREAM,
            env=upstream_env,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
