#!/usr/bin/env python3
"""Render Animatable Gaussians on the exact VCI novel-view or novel-pose split."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avatar_creation.methods.external_data import stage_thuman_subject


UPSTREAM = PROJECT_ROOT / "external" / "AnimatableGaussians"
SPLITS = {"novel-view": "novel_view", "novel-pose": "novel_pose"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def selection_frames(selection: dict) -> list[int]:
    frame_range = selection["frame_range"]
    return list(
        range(
            frame_range["start"],
            frame_range["stop"],
            frame_range.get("step", 1),
        )
    )


def owner_for(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    start = resolved.parent if resolved.is_file() else resolved
    for candidate in (start, *start.parents):
        if (candidate / "evaluation_split.json").is_file() and (
            candidate / "animatable_gaussians.yaml"
        ).is_file():
            return candidate
    raise ValueError(
        f"Could not find evaluation_split.json and animatable_gaussians.yaml above {resolved}"
    )


def checkpoint_for(path: Path, owner: Path | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file() and resolved.name == "net.pt":
        return resolved.parent
    if resolved.is_dir() and (resolved / "net.pt").is_file():
        return resolved

    owner = owner or owner_for(resolved)
    candidates = [
        net.parent
        for net in (owner / "model").glob("**/net.pt")
        if net.is_file()
    ]
    if not candidates:
        raise ValueError(
            f"No Animatable Gaussians checkpoint directory containing net.pt found under {owner / 'model'}"
        )
    return max(candidates, key=lambda candidate: (candidate / "net.pt").stat().st_mtime_ns)


def _media_ids(directory: Path) -> set[int]:
    if not directory.is_dir():
        return set()
    return {
        int(path.stem)
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in _IMAGE_EXTENSIONS
        and path.stem.isdigit()
    }


def _range_from_list(values) -> list[int]:
    if not isinstance(values, list) or len(values) not in (2, 3):
        raise ValueError("Generated Animatable frame_range must contain two or three integers")
    return list(range(*values))


def evaluation_plan(path: Path, split: str) -> dict:
    owner = owner_for(path)
    checkpoint = checkpoint_for(path, owner)
    metadata = json.loads((owner / "evaluation_split.json").read_text())
    config = yaml.safe_load((owner / "animatable_gaussians.yaml").read_text())

    if metadata.get("dataset_kind") not in {"thuman", "actorshq"}:
        raise ValueError("Animatable Gaussians split evaluation supports THuman and ActorsHQ")
    split_name = SPLITS[split]
    evaluation = metadata.get("evaluation", {})
    if split_name not in evaluation:
        raise ValueError(f"No {split_name} split configured for this run")

    train = metadata.get("training_config")
    if not train or "camera_ids" not in train or "frame_range" not in train:
        raise ValueError("Evaluation requires explicit recorded training camera and frame IDs")
    selection = evaluation[split_name]
    train_frames = selection_frames(train)
    frames = selection_frames(selection)
    train_cameras = list(train["camera_ids"])
    source_cameras = list(selection["camera_ids"])

    if split_name == "novel_view":
        if set(source_cameras) & set(train_cameras):
            raise ValueError("Novel-view cameras overlap training cameras")
        if set(frames) - set(train_frames):
            raise ValueError("Novel-view frames must belong to training frames")
    elif set(frames) & set(train_frames):
        raise ValueError("Novel-pose frames overlap training frames")

    mapping_path = owner / "data" / "camera_id_mapping.json"
    if not mapping_path.is_file():
        raise ValueError(f"Missing recorded training camera mapping: {mapping_path}")
    train_mapping = json.loads(mapping_path.read_text())
    if set(train_mapping.values()) != set(train_cameras):
        raise ValueError("Recorded training cameras disagree with staged camera mapping")

    configured_frames = _range_from_list(config["train"]["data"]["frame_range"])
    if configured_frames != train_frames:
        raise ValueError("Recorded training frames disagree with animatable_gaussians.yaml")
    if config["train"]["data"].get("used_cam_ids") != list(range(len(train_mapping))):
        raise ValueError("Generated training camera IDs disagree with staged cameras")

    source = metadata["original_data"]
    calibration = json.loads(Path(source["calibration"]).read_text())
    names_by_id = {
        int(name[3:]): name
        for name in calibration
        if str(name).startswith("cam") and str(name)[3:].isdigit()
    }
    missing_cameras = set(source_cameras) - set(names_by_id)
    if missing_cameras:
        raise ValueError(f"Configured evaluation cameras unavailable: {sorted(missing_cameras)}")

    pairs, missing_pairs = [], []
    for camera_id in source_cameras:
        name = names_by_id[camera_id]
        image_ids = _media_ids(Path(source["images"]) / name)
        mask_ids = _media_ids(Path(source["masks"]) / name)
        for frame_id in frames:
            if frame_id in image_ids and frame_id in mask_ids:
                pairs.append([camera_id, frame_id])
            else:
                missing_pairs.append([camera_id, frame_id])
    if not pairs:
        raise ValueError("No valid image/mask pairs in the configured evaluation split")

    return {
        "subject_id": metadata["subject_id"],
        "split": split_name,
        "owner": str(owner),
        "checkpoint": str(checkpoint),
        "source_camera_ids": source_cameras,
        "source_frame_ids": frames,
        "training_config": train,
        "evaluation_config": selection,
        "original_data": source,
        "valid_pairs": pairs,
        "missing_pairs": missing_pairs,
        "resolution": train["resolution"],
    }


def prepare_evaluation(plan: dict, output_dir: Path | None = None) -> tuple[list[Path], dict]:
    owner = Path(plan["owner"])
    split = plan["split"]
    evaluation_root = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else owner / "evaluation" / split
    )
    data_dir = owner / "evaluation_inputs" / split
    selection = plan["evaluation_config"]
    subject = {
        **plan["original_data"],
        "subject_id": plan["subject_id"],
        "dataset": "thuman",
        "training_config": {
            **selection,
            "resolution": plan["resolution"],
        },
        "evaluation_config": {},
        "evaluation_selection": True,
    }
    stage_thuman_subject(
        subject,
        data_dir,
        views=len(selection["camera_ids"]),
        time_steps=len(plan["source_frame_ids"]),
        resolution=plan["resolution"],
        renumber_cameras=True,
    )

    mapping = json.loads((data_dir / "camera_id_mapping.json").read_text())
    if set(mapping.values()) != set(plan["source_camera_ids"]):
        raise ValueError("Evaluation staging changed the configured source camera IDs")

    base = yaml.safe_load((owner / "animatable_gaussians.yaml").read_text())
    config_dir = evaluation_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths = []
    rendered_outputs = {}
    frame_range = selection["frame_range"]
    rendered_range = [
        frame_range["start"],
        frame_range["stop"],
        frame_range.get("step", 1),
    ]

    for local_name, source_camera in mapping.items():
        local_camera = int(local_name[3:])
        config = copy.deepcopy(base)
        config["mode"] = "test"
        config["test"] = dict(config.get("test", {}))
        config["test"].pop("pose_data", None)
        config["test"].update(
            {
                "dataset": config["train"].get("dataset", "MvRgbDatasetTHuman4"),
                "data": {
                    "subject_name": plan["subject_id"],
                    "data_dir": str(data_dir),
                    "frame_range": rendered_range,
                },
                "view_setting": "camera",
                "render_view_idx": local_camera,
                "img_scale": 1.0,
                "n_pca": -1,
                "prev_ckpt": plan["checkpoint"],
                "output_dir": str(evaluation_root / f"cam{source_camera:02d}"),
            }
        )
        config_path = config_dir / f"cam{source_camera:02d}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        config_paths.append(config_path)
        rendered_outputs[str(source_camera)] = str(
            evaluation_root / f"cam{source_camera:02d}" / "vanilla" / "rgb_map"
        )

    reference = {
        **plan,
        "local_to_source_camera_ids": mapping,
        "generated_configs": [str(path) for path in config_paths],
        "render_outputs": rendered_outputs,
    }
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "input_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    return config_paths, reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Checkpoint directory, net.pt, or Animatable subject output",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the exact source selection without staging or rendering",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stage evaluation inputs and write configs without rendering",
    )
    args = parser.parse_args(argv)

    plan = evaluation_plan(args.path, args.split)
    if args.validate_only:
        print(json.dumps(plan, indent=2))
        return 0

    if not (UPSTREAM / "main_avatar.py").is_file():
        raise SystemExit(
            "Animatable Gaussians checkout is missing. "
            "Run scripts/setup/setup_animatable_gaussians_env.sh."
        )
    configs, reference = prepare_evaluation(plan, args.output_dir)
    print(
        f"Evaluation: {len(reference['valid_pairs'])} valid pairs, "
        f"{len(reference['missing_pairs'])} missing pairs, "
        f"{len(reference['source_camera_ids'])} cameras"
    )
    if args.prepare_only:
        print(json.dumps(reference, indent=2))
        return 0

    for config_path in configs:
        subprocess.run(
            [sys.executable, "main_avatar.py", "-c", str(config_path), "--mode=test"],
            cwd=UPSTREAM,
            check=True,
        )
    print(f"Renders: {Path(reference['generated_configs'][0]).parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
