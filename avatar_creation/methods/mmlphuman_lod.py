"""Manifest adapter for MMLPHuman with PiG-Avatar-compatible progressive LOD."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from avatar_creation.methods.mmlphuman import UPSTREAM, prepare_subject
from avatar_creation.methods.external_data import load_subjects, safe_subject_id
from avatar_creation.methods.mmlphuman_lod_impl.lod import add_training_arguments, resolve_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=UPSTREAM / "config" / "subject01.yaml")
    parser.add_argument("--ip", default="127.0.0.1")
    slot = int(os.environ.get("VCI_DEVICE_SLOT_INDEX", "0"))
    parser.add_argument("--port", type=int, default=23456 + slot)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1024)
    add_training_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not (UPSTREAM / "train.py").is_file():
        raise SystemExit("MMLP Human checkout is missing. Run scripts/setup/setup_mmlphuman_env.sh.")
    template = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        template["lod"] = resolve_config(template, args)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for subject in load_subjects(args.dataset_manifest):
        subject_id = safe_subject_id(subject.get("subject_id", "subject"))
        data_dir, config_path = prepare_subject(
            subject, args.output_dir, template,
            views=args.views, time_steps=args.time_steps, resolution=args.resolution,
        )
        model_dir = args.output_dir / "subjects" / subject_id / "model"
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / "avatar_creation/methods/mmlphuman_lod_impl/train.py"),
            "--config", str(config_path), "--data_dir", str(data_dir),
            "--out_dir", str(model_dir), "--ip", args.ip, "--port", str(args.port),
        ], cwd=UPSTREAM, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
