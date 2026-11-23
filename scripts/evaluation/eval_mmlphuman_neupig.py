#!/usr/bin/env python3
"""Evaluate an MMLP-Human checkpoint with PiG-Avatar's novel-view evaluator.

The PiG-Avatar checkpoint is used only as the source of the evaluation dataset
configuration.  Rendering is performed by MMLP-Human, while dataset loading,
target masking, metric computation, aggregation, and output serialization are
delegated to PiG-Avatar's unchanged ``eval_novel_view`` implementation.

Run this in the ``neupig`` environment, for example::

    conda run --no-capture-output -n neupig python \
      scripts/evaluation/eval_mmlphuman_neupig.py \
      --mmlp-path /path/to/mmlphuman/subjects/subject00/model \
      --reference-checkpoint /path/to/neupig/subject00/model/avatar_model.pth \
      --device cuda:2
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEUPIG_ROOT = PROJECT_ROOT / "external" / "neu-pig-smplx"
MMLP_ROOT = PROJECT_ROOT / "external" / "mmlphuman"


def _configure_import_paths() -> None:
    # PiG-Avatar owns the top-level ``src`` package. MMLP-Human uses the separate
    # top-level ``scene`` and ``utils`` packages.
    sys.path.insert(0, str(NEUPIG_ROOT))
    sys.path.insert(0, str(MMLP_ROOT))


def _install_numba_fallback() -> None:
    """Allow the PiG-Avatar env to run MMLP's tiny CPU transform loop.

    Numba is part of the MMLP environment but not PiG-Avatar's.  MMLP uses it only
    as a JIT decorator for a 55-joint rigid-transform routine.  Keeping the
    Python implementation is numerically identical and insignificant compared
    with rendering time.
    """

    if importlib.util.find_spec("numba") is not None:
        return

    fallback = types.ModuleType("numba")

    def njit(*args, **_kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    fallback.njit = njit
    fallback.float32 = np.float32
    sys.modules["numba"] = fallback


_configure_import_paths()
_install_numba_fallback()

from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.smpl_utils import init_smpl_pose  # noqa: E402
from src.metrics.evaluation import create_lpips_model  # noqa: E402
from src.opt_pipeline.checkpoint_eval import _checkpoint_for_path  # noqa: E402
from src.opt_pipeline.evaluation import _novel_dataset, eval_novel_view  # noqa: E402
from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: E402


_CHECKPOINT_RE = re.compile(r"^chkpnt(\d+)\.pth$")


def _mmlp_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path

    candidate_directories = [path]
    if (path / "model").is_dir():
        candidate_directories.insert(0, path / "model")

    candidates: list[tuple[int, Path]] = []
    for directory in candidate_directories:
        if not directory.is_dir():
            continue
        for candidate in directory.iterdir():
            match = _CHECKPOINT_RE.match(candidate.name)
            if match and candidate.is_file():
                candidates.append((int(match.group(1)), candidate))

    if not candidates:
        raise FileNotFoundError(f"No chkpnt*.pth MMLP checkpoint found at {path}")
    return max(candidates, key=lambda item: item[0])[1]


def _neupig_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    checkpoint = _checkpoint_for_path(path)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No PiG-Avatar model/avatar_model.pth checkpoint found at {path}"
        )
    return checkpoint.resolve()


def _subject_root(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent


def _load_reference_args(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "args" not in payload:
        raise KeyError(f"PiG-Avatar checkpoint has no saved args: {checkpoint}")
    args = copy.deepcopy(payload["args"])
    if getattr(args, "dataset", None) != "thuman4d":
        raise ValueError(
            "This adapter currently supports THuman4D checkpoints only; "
            f"the reference checkpoint uses {getattr(args, 'dataset', None)!r}."
        )

    args.device = str(device)
    if not hasattr(args, "tmp_args"):
        args.tmp_args = {}
    return args


def _pose_sequence(smplx_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required = (
        "global_orient",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "transl",
    )
    missing = [key for key in required if key not in smplx_params]
    if missing:
        raise KeyError("SMPL-X input is missing: " + ", ".join(missing))

    frame_count = smplx_params["body_pose"].shape[0]
    zeros = torch.zeros((frame_count, 9), dtype=torch.float32)
    poses = torch.cat(
        (
            smplx_params["global_orient"].reshape(frame_count, -1),
            smplx_params["body_pose"].reshape(frame_count, -1),
            zeros,
            smplx_params["left_hand_pose"].reshape(frame_count, -1),
            smplx_params["right_hand_pose"].reshape(frame_count, -1),
        ),
        dim=-1,
    ).to(torch.float32)
    if poses.shape[1] != 165:
        raise ValueError(f"Expected 165 MMLP pose values, got {poses.shape[1]}")

    return {
        "pose": poses.cpu(),
        "Th": smplx_params["transl"].reshape(frame_count, 3).to(torch.float32).cpu(),
        "Rh": torch.eye(3, dtype=torch.float32)[None].expand(frame_count, -1, -1),
    }


def _same_source_pose_data(mmlp_checkpoint: Path, dataset) -> None:
    """Fail if MMLP and PiG-Avatar resolve to different SMPL-X source files."""

    mmlp_pose_path = _subject_root(mmlp_checkpoint) / "data" / "smpl_params.npz"
    neupig_pose_path = Path(dataset.data["smplx_params"])
    if not mmlp_pose_path.exists():
        raise FileNotFoundError(f"MMLP staged pose data is missing: {mmlp_pose_path}")
    if not neupig_pose_path.exists():
        raise FileNotFoundError(f"PiG-Avatar source pose data is missing: {neupig_pose_path}")
    if not mmlp_pose_path.samefile(neupig_pose_path):
        raise ValueError(
            "MMLP and PiG-Avatar do not reference the same SMPL-X source data:\n"
            f"  MMLP:   {mmlp_pose_path.resolve()}\n"
            f"  PiG-Avatar: {neupig_pose_path.resolve()}"
        )


def _load_mmlp_model(checkpoint: Path, device: torch.device) -> GaussianModel:
    init_smpl_pose()

    def preserve_cpu_remap_cuda(storage, location: str):
        # MMLP deliberately stores poses/joints on CPU and Gaussian state on
        # CUDA. A blanket map_location=device breaks its later NumPy calls.
        if location.startswith("cuda"):
            return storage.cuda(device.index)
        return storage

    payload = torch.load(
        checkpoint,
        map_location=preserve_cpu_remap_cuda,
        weights_only=False,
    )
    gaussians = GaussianModel()
    gaussians.restore(payload)
    # Novel-view evaluation uses known training poses. MMLP's PCA-clamped test
    # mode is intended for novel poses and would change this comparison.
    gaussians.is_test = False
    return gaussians


class _SingleLevelLODs:
    def __init__(self, splat_count: int, device: torch.device):
        self.splat_count = splat_count
        self.device = device

    def lod_indices(self, _level: int, total_lods: int) -> torch.Tensor:
        if total_lods != 1:
            raise ValueError(f"MMLP adapter expected one LOD, got {total_lods}")
        return torch.arange(self.splat_count, device=self.device, dtype=torch.long)


class _MMLPSplats:
    def __init__(self, gaussians: GaussianModel, device: torch.device):
        self.gaussians = gaussians
        self.device = device

    def render_rgb(self, data: dict, resolution: tuple[int, int], background: str):
        intrinsics = data["K_CV"].reshape(-1, 3, 3)
        extrinsics = data["V_CV"].reshape(-1, 4, 4)
        if intrinsics.shape[0] != 1 or extrinsics.shape[0] != 1:
            raise ValueError("The MMLP novel-view adapter expects one camera per batch.")

        width, height = (int(resolution[0]), int(resolution[1]))
        value = {"black": 0.0, "white": 1.0}.get(background)
        if value is None:
            raise ValueError(f"Unsupported background: {background}")

        camera = {
            "K": intrinsics[0],
            "w2c": extrinsics[0],
            "width": width,
            "height": height,
        }
        background_tensor = torch.full(
            (3,), value, device=self.device, dtype=torch.float32
        )
        image, alpha, info = self.gaussians.render(
            camera, background=background_tensor
        )
        return image[None, None], alpha[None, None], info


class MMLPNeuPiGAdapter(torch.nn.Module):
    """Minimal AvatarModel-compatible facade consumed by eval_novel_view."""

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        poses: dict[str, torch.Tensor],
        device: torch.device,
    ):
        super().__init__()
        self.args = args
        self.gaussians = gaussians
        self.poses = poses
        self.device = device
        self.lods = _SingleLevelLODs(len(gaussians._xyz), device)

    def forward(self, data: dict, **_kwargs):
        time_grid = data["t_grid"].reshape(-1)
        if time_grid.numel() != 1:
            raise ValueError("The MMLP novel-view adapter expects one timestep per batch.")
        frame = int(time_grid.item())
        if not 0 <= frame < self.poses["pose"].shape[0]:
            raise IndexError(f"Pose frame {frame} is outside the available sequence.")

        self.gaussians.smpl_poses = self.poses["pose"][frame]
        self.gaussians.Th = self.poses["Th"][frame]
        self.gaussians.Rh = self.poses["Rh"][frame]
        splats = _MMLPSplats(self.gaussians, self.device)
        return {"splat_model": splats}, torch.zeros((), device=self.device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmlp-path",
        type=Path,
        required=True,
        help="MMLP checkpoint, model directory, or subject directory.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        required=True,
        help=(
            "Matching PiG-Avatar checkpoint (or its model/run directory). Its saved "
            "dataset arguments define the exact novel-view evaluation data."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation owner directory. PiG-Avatar writes its evaluation/ subtree "
            "below it. Default: the MMLP subject directory."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify checkpoint/data identity and exit without loading or rendering MMLP.",
    )
    return parser.parse_args()


def main() -> int:
    cli = _parse_args()
    device = torch.device(cli.device)
    if device.type != "cuda":
        raise ValueError("MMLP-Human is CUDA-only; pass --device cuda:N.")
    if device.index is not None:
        torch.cuda.set_device(device)

    mmlp_checkpoint = _mmlp_checkpoint(cli.mmlp_path)
    reference_checkpoint = _neupig_checkpoint(cli.reference_checkpoint)
    args = _load_reference_args(reference_checkpoint, device)

    output_owner = (
        cli.output_dir.expanduser().resolve()
        if cli.output_dir is not None
        else _subject_root(mmlp_checkpoint)
    )
    args.io_args["out_path"] = str(output_owner)
    # MMLP has no LOD hierarchy. This makes PiG-Avatar select and report LOD 0.
    args.LODs = 1
    args.lod_condition_denominator = 1
    args.lod_condition_offset = 0

    # Instantiate PiG-Avatar's exact novel-view dataset once for pose provenance and
    # conversion. eval_novel_view creates the same class again for RGB/mask/camera
    # loading, preserving the original evaluator's behavior.
    reference_dataset = _novel_dataset(args, "view")
    _same_source_pose_data(mmlp_checkpoint, reference_dataset)
    poses = _pose_sequence(reference_dataset.smplx_params)
    expected_frames = int(args.method_args["target"]["time_steps"])
    if poses["pose"].shape[0] != expected_frames:
        raise ValueError(
            f"PiG-Avatar evaluates {expected_frames} frames but the pose source has "
            f"{poses['pose'].shape[0]}."
        )

    print(f"MMLP checkpoint: {mmlp_checkpoint}")
    print(f"PiG-Avatar evaluation reference: {reference_checkpoint}")
    print(f"Evaluation data: {args.io_args['input_directory']}")
    print(f"Output: {output_owner / 'evaluation'}")
    print(
        "Validated identical SMPL-X source data; PiG-Avatar will load camera 23 "
        f"at source resolution for {expected_frames} frames."
    )
    if cli.validate_only:
        return 0

    output_owner.mkdir(parents=True, exist_ok=True)
    gaussians = _load_mmlp_model(mmlp_checkpoint, device)
    adapter = MMLPNeuPiGAdapter(args, gaussians, poses, device).to(device)

    audit = {
        "renderer": "mmlphuman",
        "mmlp_checkpoint": str(mmlp_checkpoint),
        "evaluation_implementation": str(
            NEUPIG_ROOT / "src" / "opt_pipeline" / "evaluation.py"
        ),
        "metrics_implementation": str(
            NEUPIG_ROOT / "src" / "metrics" / "evaluation.py"
        ),
        "reference_checkpoint": str(reference_checkpoint),
        "input_directory": str(args.io_args["input_directory"]),
        "pose_source": str(Path(reference_dataset.data["smplx_params"]).resolve()),
        "frames": expected_frames,
        "novel_camera": 23,
        "resolution": "source",
        "metric_background": "black",
    }
    audit_path = output_owner / "evaluation" / "evaluation_reference.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    fid = FrechetInceptionDistance(feature=2048).to(device)
    lpips_model = create_lpips_model(device)
    eval_novel_view(
        LOD_level=-1,
        model=adapter,
        fid=fid,
        lpips_model=lpips_model,
    )
    return 0


if __name__ == "__main__":
    exit_code = main()
    # This PiG-Avatar environment currently segfaults during interpreter teardown
    # even for the stock ``checkpoint_eval --help`` command. All evaluation
    # writes above are synchronous, so bypass the faulty extension teardown
    # after a successful run and return the correct process status.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
