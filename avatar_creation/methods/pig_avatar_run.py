"""Run PiG-Avatar with the VCI unified dataset manifest.

This file runs inside the PiG environment and deliberately does not import the
VCI package. It adapts the stable JSON manifest to PiG's Dataset contract while
leaving the external PiG checkout unchanged.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


if __name__ == "__main__":
    # Main.py temporarily owns __main__. Keep dataset classes in an importable
    # module so the upstream DataLoader can pickle them for spawn workers.
    import pig_avatar_run as runtime

    runtime.main()
    # Native extensions can crash during teardown after a successful run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


PIG_REPO = Path.cwd().resolve()
if not (PIG_REPO / "Main.py").is_file() or not (PIG_REPO / "src").is_dir():
    raise SystemExit(
        f"PiG unified runner must start in the PiG checkout; got {PIG_REPO}."
    )
sys.path.insert(0, str(PIG_REPO))

import numpy as np
import cv2
import torch
from torchvision.io import read_image

_CPU_THREADS = int(os.environ.get("VCI_PIG_CPU_THREADS", "16"))
if _CPU_THREADS <= 0:
    raise SystemExit("VCI_PIG_CPU_THREADS must be greater than zero.")
torch.set_num_threads(_CPU_THREADS)
torch.set_num_interop_threads(_CPU_THREADS)

from src.io.datasets.base import BaseDataset


_FRAME_RE = re.compile(r"(\d+)$")
_FOURDDRESS_SMPLX_RE = re.compile(r"mesh-f(\d+)_smplx\.pkl$")
_FOURDDRESS_LABEL_RE = re.compile(r"label-f(\d+)\.pkl$")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_SMPLX_MODEL_PATH: Path | None = None
_HAND_PCA_DATA: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def _write_evaluation_split(
    subject_dir: Path,
    subject: dict[str, Any],
    *,
    train_cameras: list[int],
    resolution: Any,
) -> None:
    """Write the common sidecar without importing the outer VCI package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vci_evaluation_split", Path(__file__).with_name("evaluation_split.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.write_evaluation_split(subject_dir, subject, train_cameras=train_cameras, resolution=resolution)


def _selection_frames(selection):
    r = selection.get("frame_range")
    return list(range(r["start"], r["stop"], r.get("step", 1))) if r else list(range(selection["time_steps"]))


def _manifest_argument(argv: list[str]) -> tuple[Path, list[str]]:
    try:
        index = argv.index("--dataset-manifest")
        value = argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit("PiG unified runner requires --dataset-manifest PATH") from exc
    remaining = [*argv[:index], *argv[index + 2 :]]
    return Path(value).expanduser().resolve(), remaining


def _pop_flag(argv: list[str], flag: str) -> tuple[bool, list[str]]:
    enabled = flag in argv
    return enabled, [value for value in argv if value != flag]


def _configured_smplx_model(argv: list[str]) -> Path:
    try:
        index = argv.index("--methodConfig")
        config_path = Path(argv[index + 1]).expanduser().resolve()
        config = json.loads(config_path.read_text())
        model_path = Path(
            config["model_args"]["model_args_smplx"]["model_path"]
        ).expanduser().resolve()
    except (ValueError, IndexError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(
            "Unified PiG runner needs a method config with model_args_smplx.model_path."
        ) from exc
    if not model_path.is_file():
        raise SystemExit(f"Configured SMPL-X model not found: {model_path}")
    return model_path


def _redirect_hardcoded_smplx(model_path: Path) -> None:
    import smplx

    global _SMPLX_MODEL_PATH
    _SMPLX_MODEL_PATH = model_path

    original = smplx.SMPLX

    def configured_smplx(requested_path, *args, **kwargs):
        requested = Path(str(requested_path))
        if (
            str(requested).replace("\\", "/") == "models/smplx/SMPLX_NEUTRAL.npz"
            and not requested.is_file()
        ):
            requested_path = str(model_path)
        return original(requested_path, *args, **kwargs)

    smplx.SMPLX = configured_smplx


def _load_subjects(path: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Dataset manifest not found: {path}") from exc
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise SystemExit(f"Dataset manifest contains no subjects: {path}")
    return subjects


def _required_path(subject: dict[str, Any], key: str) -> Path:
    value = subject.get(key)
    path = Path(value).expanduser().resolve() if value else None
    if path is None or not path.exists():
        raise ValueError(
            f"Subject {subject.get('subject_id', '<unknown>')} has no valid {key!r} path."
        )
    return path


def _frame_number(path: Path) -> int | None:
    match = _FRAME_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _media_index(root: Path) -> dict[int, dict[int, Path]]:
    index: dict[int, dict[int, Path]] = defaultdict(dict)
    for camera_dir in sorted(root.iterdir()):
        if not camera_dir.is_dir() or not camera_dir.name.startswith("cam"):
            continue
        try:
            camera_id = int(camera_dir.name[3:])
        except ValueError:
            continue
        for path in sorted(camera_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            frame_id = _frame_number(path)
            if frame_id is not None:
                index[camera_id][frame_id] = path
    return dict(index)


class UnifiedPigDataset(BaseDataset):
    """PiG Dataset implementation backed by one canonical manifest subject."""

    def __init__(self, args, subject: dict[str, Any]):
        super().__init__(args)
        self.subject = subject
        self.image_index = _media_index(_required_path(subject, "images"))
        self.mask_index = _media_index(_required_path(subject, "masks"))
        self.cameras = self._load_cameras(_required_path(subject, "calibration"))
        self.smplx_path = _required_path(subject, "smplx")

        available_cameras = sorted(
            set(self.image_index) & set(self.mask_index) & set(self.cameras)
        )
        if len(available_cameras) < self.C:
            raise ValueError(
                f"Subject {subject.get('subject_id')} needs {self.C} PiG views but only "
                f"{len(available_cameras)} cameras have images, masks, and calibration."
            )
        self.camera_ids = subject.get("training_config", {}).get("camera_ids", available_cameras[: self.C])
        if set(self.camera_ids) - set(available_cameras):
            raise ValueError("Configured cameras are unavailable")

        # T defines the timeline, not the number of observed image files.
        self.frame_ids = _selection_frames(subject["training_config"]) if subject.get("training_config") else list(range(self.T))
        self.args.T = self.T
        self.args.C = self.C

    @staticmethod
    def _load_cameras(path: Path) -> dict[int, dict[str, torch.Tensor]]:
        raw = json.loads(path.read_text())
        cameras = {}
        for name, camera in raw.items():
            if not str(name).startswith("cam"):
                continue
            camera_id = int(str(name)[3:])
            view = torch.eye(4, dtype=torch.float32)
            view[:3, :3] = torch.tensor(camera["R"], dtype=torch.float32).reshape(3, 3)
            view[:3, 3] = torch.tensor(camera["T"], dtype=torch.float32).reshape(3)
            cameras[camera_id] = {
                "K": torch.tensor(camera["K"], dtype=torch.float32).reshape(3, 3),
                "V": view,
            }
        return cameras

    def __len__(self) -> int:
        return len(self.camera_ids)

    @property
    def smplx_params(self) -> dict[str, torch.Tensor]:
        smplx = np.load(self.smplx_path)
        last_frame = max(self.frame_ids)
        params = {}
        for key in smplx.files:
            values = smplx[key]
            if key == "betas":
                params[key] = torch.from_numpy(values).to(torch.float32)
            else:
                if len(values) <= last_frame:
                    raise ValueError(
                        f"SMPL-X parameter {key!r} has {len(values)} frames but frame "
                        f"{last_frame} is required for {self.subject.get('subject_id')}."
                    )
                params[key] = torch.from_numpy(values[self.frame_ids]).to(torch.float32)
        return params

    def load_smplx_params(self, source="dataset"):
        if source != "dataset":
            raise ValueError("Manifest THuman evaluation supports dataset SMPL-X poses")
        return self.smplx_params

    def get_timestep(self, data: dict) -> dict:
        times = torch.arange(self.T, dtype=torch.float32) / self.T
        return data | {"target_time": times}

    def _load_frame(
        self,
        image_path: Path | None,
        mask_path: Path | None,
        *,
        compute_boundary: bool = True,
    ):
        if image_path is None or mask_path is None:
            return None
        try:
            image = self._get_resized_image(read_image(str(image_path)))
            mask = self._get_resized_image(read_image(str(mask_path)))
        except (FileNotFoundError, RuntimeError):
            return None
        if mask.shape[0] > 1:
            mask = mask[:1]
        return (
            image.permute(1, 2, 0),
            mask.permute(1, 2, 0),
            self._get_boundary_mask(mask, kernel_size=3)
            if compute_boundary
            else None,
        )

    def valid_matrix(self) -> torch.Tensor:
        """Map observed source frames onto the selected local time/camera grid."""
        return torch.tensor([
            [frame in self.image_index[camera] and frame in self.mask_index[camera]
             for camera in self.camera_ids]
            for frame in self.frame_ids
        ], dtype=torch.bool)

    def get_frame(
        self, time_idx: int, camera_idx: int, compute_boundary: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load one manifest-backed frame for unified-grid training."""
        if not 0 <= time_idx < self.T:
            raise IndexError(f"Invalid time index {time_idx}; expected [0, {self.T}).")
        if not 0 <= camera_idx < self.C:
            raise IndexError(
                f"Invalid camera index {camera_idx}; expected [0, {self.C})."
            )

        frame_id = self.frame_ids[int(time_idx)]
        camera_id = self.camera_ids[int(camera_idx)]
        frame = self._load_frame(
            self.image_index[camera_id].get(frame_id),
            self.mask_index[camera_id].get(frame_id),
            compute_boundary=compute_boundary,
        )
        if frame is None:
            raise FileNotFoundError(
                f"Missing frame t={frame_id}, c={camera_id} for "
                f"{self.subject.get('subject_id')}."
            )

        image, mask, boundary = frame
        intrinsics = self._get_resized_intrinsics(self.cameras[camera_id]["K"])
        extrinsics = self.cameras[camera_id]["V"]
        camera_center = torch.linalg.inv(extrinsics)[:3, 3]
        output = {
            "rgb": image,
            "mask": mask > (127 if mask.dtype == torch.uint8 else 0.5),
            "K_CV": intrinsics,
            "V_CV": extrinsics,
            "camera_center": camera_center,
            "camera_id": torch.tensor(camera_id, dtype=torch.int32),
            "valid": torch.ones(1, dtype=torch.bool),
            "target_time": torch.tensor(time_idx / self.T, dtype=torch.float32),
        }
        if boundary is not None:
            output["boundary_mask"] = boundary.to(torch.bool)
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        camera_id = self.camera_ids[index]
        images = self.image_index[camera_id]
        masks = self.mask_index[camera_id]
        rgb, mask_values, boundaries, valid = [], [], [], []

        for frame_id in self.frame_ids:
            frame = self._load_frame(images.get(frame_id), masks.get(frame_id))
            if frame is None:
                rgb.append(None)
                mask_values.append(None)
                boundaries.append(None)
                valid.append(False)
            else:
                image, mask, boundary = frame
                rgb.append(image)
                mask_values.append(mask)
                boundaries.append(boundary)
                valid.append(True)

        valid_tensor = torch.tensor(valid, dtype=torch.bool)[:, None]
        if not valid_tensor.any():
            raise ValueError(
                f"No valid frames for subject {self.subject.get('subject_id')} "
                f"camera cam{camera_id:02d}."
            )
        template = int(torch.where(valid_tensor[:, 0])[0][0])
        for frame_index, is_valid in enumerate(valid):
            if is_valid:
                continue
            rgb[frame_index] = torch.zeros_like(rgb[template])
            mask_values[frame_index] = torch.zeros_like(mask_values[template])
            boundaries[frame_index] = torch.zeros_like(boundaries[template])

        intrinsics = self._get_resized_intrinsics(self.cameras[camera_id]["K"])
        extrinsics = self.cameras[camera_id]["V"]
        camera_center = torch.linalg.inv(extrinsics)[:3, 3]
        return {
            "rgb": torch.stack(rgb),
            "mask": torch.stack(mask_values),
            "boundary_mask": torch.stack(boundaries),
            "K_CV": intrinsics,
            "V_CV": extrinsics,
            "camera_id": torch.tensor(camera_id, dtype=torch.int32),
            "valid": valid_tensor,
            "camera_center": camera_center,
        }


def _fourddress_hand_pca() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    global _HAND_PCA_DATA
    if _HAND_PCA_DATA is not None:
        return _HAND_PCA_DATA
    if _SMPLX_MODEL_PATH is None:
        raise ValueError(
            "4D-DRESS hand poses use SMPL-X PCA coefficients, but the configured "
            "SMPL-X model path is unavailable. Launch through pig_avatar.py."
        )
    model = np.load(_SMPLX_MODEL_PATH, allow_pickle=True)
    try:
        _HAND_PCA_DATA = {
            "left": (model["hands_componentsl"], model["hands_meanl"]),
            "right": (model["hands_componentsr"], model["hands_meanr"]),
        }
    except KeyError as exc:
        raise ValueError(
            f"Configured SMPL-X model lacks hand PCA data: {_SMPLX_MODEL_PATH}"
        ) from exc
    return _HAND_PCA_DATA


def _expand_fourddress_hand_pose(values: np.ndarray, side: str) -> np.ndarray:
    if values.shape[-1] == 45:
        return values
    components, mean = _fourddress_hand_pca()[side]
    coefficient_count = values.shape[-1]
    if coefficient_count > components.shape[0]:
        raise ValueError(
            f"Unsupported 4D-DRESS {side} hand pose width {coefficient_count}."
        )
    return values @ components[:coefficient_count] + mean


class FourDDressPigDataset(UnifiedPigDataset):
    """PiG reader for one native 4D-DRESS person/outfit/take sequence."""

    def __init__(
        self, args, subject: dict[str, Any], *, forward_labels: bool = False
    ):
        BaseDataset.__init__(self, args)
        self.subject = subject
        capture = _required_path(subject, "images")
        self.image_index = self._capture_index(capture, "images")
        self.mask_index = self._capture_index(
            _required_path(subject, "masks"), "masks"
        )
        self.cameras = self._load_fourddress_cameras(
            _required_path(subject, "calibration")
        )
        self.smplx_index = self._smplx_index(_required_path(subject, "smplx"))
        self.label_index = (
            self._label_index(_required_path(subject, "labels"))
            if forward_labels
            else None
        )
        self._semantic_labels = None

        available_cameras = sorted(
            set(self.image_index) & set(self.mask_index) & set(self.cameras)
        )
        if not available_cameras:
            raise ValueError(
                f"Subject {subject.get('subject_id')} has no complete 4D-DRESS cameras."
            )
        requested_views = self.C
        view_count = min(requested_views, len(available_cameras))
        self.args.method_args["target"]["views"] = view_count
        self.camera_ids = available_cameras[:view_count]

        common_frames = set(self.smplx_index)
        if self.label_index is not None:
            common_frames &= set(self.label_index)
        for camera_id in self.camera_ids:
            common_frames &= set(self.image_index[camera_id])
            common_frames &= set(self.mask_index[camera_id])
        frame_ids = sorted(common_frames)
        if not frame_ids:
            raise ValueError(
                f"Subject {subject.get('subject_id')} has no synchronized images, "
                "masks, and SMPL-X frames."
            )
        requested_time_steps = self.T
        time_steps = min(requested_time_steps, len(frame_ids))
        self.args.method_args["target"]["time_steps"] = time_steps
        self.frame_ids = frame_ids[:time_steps]
        self.args.T = self.T
        self.args.C = self.C

        if requested_views > view_count:
            print(
                f"4D-DRESS subject {subject.get('subject_id')}: limiting views "
                f"from {requested_views} to {view_count}."
            )
        if requested_time_steps > time_steps:
            print(
                f"4D-DRESS subject {subject.get('subject_id')}: limiting time steps "
                f"from {requested_time_steps} to {time_steps}."
            )

    @staticmethod
    def _capture_index(root: Path, leaf: str) -> dict[int, dict[int, Path]]:
        index: dict[int, dict[int, Path]] = defaultdict(dict)
        for camera_dir in sorted(root.iterdir()):
            if not camera_dir.is_dir() or not camera_dir.name.isdigit():
                continue
            camera_id = int(camera_dir.name)
            media_dir = camera_dir / leaf
            if not media_dir.is_dir():
                continue
            for path in sorted(media_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                frame_id = _frame_number(path)
                if frame_id is not None:
                    index[camera_id][frame_id] = path
        return dict(index)

    @staticmethod
    def _load_fourddress_cameras(path: Path) -> dict[int, dict[str, torch.Tensor]]:
        with path.open("rb") as stream:
            raw = pickle.load(stream)
        cameras = {}
        for name, camera in raw.items():
            camera_id = int(name)
            extrinsics = torch.eye(4, dtype=torch.float32)
            extrinsics[:3, :] = torch.as_tensor(
                camera["extrinsics"], dtype=torch.float32
            ).reshape(3, 4)
            cameras[camera_id] = {
                "K": torch.as_tensor(camera["intrinsics"], dtype=torch.float32).reshape(3, 3),
                "V": extrinsics,
            }
        return cameras

    @staticmethod
    def _smplx_index(root: Path) -> dict[int, Path]:
        index = {}
        for path in sorted(root.glob("*_smplx.pkl")):
            match = _FOURDDRESS_SMPLX_RE.fullmatch(path.name)
            if match is not None:
                index[int(match.group(1))] = path
        return index

    @staticmethod
    def _label_index(root: Path) -> dict[int, Path]:
        index = {}
        for path in sorted(root.glob("label-f*.pkl")):
            match = _FOURDDRESS_LABEL_RE.fullmatch(path.name)
            if match is not None:
                index[int(match.group(1))] = path
        return index

    @property
    def semantic_labels(self) -> torch.Tensor | None:
        if self.label_index is None:
            return None
        if self._semantic_labels is not None:
            return self._semantic_labels

        labels = []
        for frame_id in self.frame_ids:
            with self.label_index[frame_id].open("rb") as stream:
                frame = pickle.load(stream)
            if not isinstance(frame, dict) or "scan_labels" not in frame:
                raise ValueError(
                    f"4D-DRESS label frame {frame_id} has no scan_labels array."
                )
            values = np.asarray(frame["scan_labels"])
            if values.ndim != 1 or not np.allclose(values, np.rint(values)):
                raise ValueError(
                    f"4D-DRESS label frame {frame_id} is not a 1D categorical array."
                )
            labels.append(values.astype(np.int64, copy=False))
        try:
            stacked = np.stack(labels)
        except ValueError as exc:
            raise ValueError(
                "4D-DRESS scan_labels arrays have inconsistent vertex counts."
            ) from exc
        self._semantic_labels = torch.from_numpy(stacked)
        return self._semantic_labels

    @property
    def smplx_params(self) -> dict[str, torch.Tensor]:
        frames = []
        for frame_id in self.frame_ids:
            with self.smplx_index[frame_id].open("rb") as stream:
                frames.append(pickle.load(stream))
        shared_keys = set.intersection(*(set(frame) for frame in frames))
        raw = {
            key: np.stack([np.asarray(frame[key]) for frame in frames])
            for key in shared_keys
        }
        raw["left_hand_pose"] = _expand_fourddress_hand_pose(
            raw["left_hand_pose"], "left"
        )
        raw["right_hand_pose"] = _expand_fourddress_hand_pose(
            raw["right_hand_pose"], "right"
        )
        raw["betas"] = raw["betas"].mean(axis=0, keepdims=True)
        return {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in raw.items()
        }


class TalkBody4DPigDataset(BaseDataset):
    """Zero-copy PiG reader for native TalkBody4D videos and annotations."""

    def __init__(self, args, subject: dict[str, Any]):
        super().__init__(args)
        self.subject = subject
        self.video_index = self._video_index(_required_path(subject, "images"))
        self.mask_index = self._mask_index(_required_path(subject, "masks"))
        self.smplx_index = self._smplx_index(_required_path(subject, "smplx"))
        self.cameras = self._load_cameras(_required_path(subject, "calibration"))

        available_cameras = sorted(
            set(self.video_index) & set(self.mask_index) & set(self.cameras)
        )
        if not available_cameras:
            raise ValueError(
                f"Subject {subject.get('subject_id')} has no complete TalkBody4D cameras."
            )
        requested_views = self.C
        view_count = min(requested_views, len(available_cameras))
        self.args.method_args["target"]["views"] = view_count
        self.camera_ids = available_cameras[:view_count]

        common_frames = set(self.smplx_index)
        for camera_id in self.camera_ids:
            common_frames &= set(self.mask_index[camera_id])
        frame_ids = sorted(common_frames)
        if not frame_ids:
            raise ValueError(
                f"Subject {subject.get('subject_id')} has no synchronized video, "
                "matting, and SMPL-X frames."
            )
        requested_time_steps = self.T
        time_steps = min(requested_time_steps, len(frame_ids))
        self.args.method_args["target"]["time_steps"] = time_steps
        self.frame_ids = frame_ids[:time_steps]
        self.args.T = self.T
        self.args.C = self.C

        if requested_views > view_count:
            print(
                f"TalkBody4D subject {subject.get('subject_id')}: limiting views "
                f"from {requested_views} to {view_count}."
            )
        if requested_time_steps > time_steps:
            print(
                f"TalkBody4D subject {subject.get('subject_id')}: limiting time steps "
                f"from {requested_time_steps} to {time_steps}."
            )

    @staticmethod
    def _video_index(root: Path) -> dict[int, Path]:
        return {
            int(path.stem): path
            for path in sorted(root.glob("*.mp4"))
            if path.stem.isdigit()
        }

    @staticmethod
    def _mask_index(root: Path) -> dict[int, dict[int, Path]]:
        result: dict[int, dict[int, Path]] = defaultdict(dict)
        for camera_dir in sorted(root.iterdir()):
            if not camera_dir.is_dir() or not camera_dir.name.isdigit():
                continue
            camera_id = int(camera_dir.name)
            for path in sorted(camera_dir.glob("*.png")):
                frame_id = _frame_number(path)
                if frame_id is not None:
                    result[camera_id][frame_id] = path
        return dict(result)

    @staticmethod
    def _smplx_index(root: Path) -> dict[int, Path]:
        return {
            int(path.stem): path
            for path in sorted(root.glob("*.json"))
            if path.stem.isdigit()
        }

    @staticmethod
    def _load_cameras(path: Path) -> dict[int, dict[str, Any]]:
        raw = json.loads(path.read_text())
        intrinsics = raw.get("cameras", {})
        poses = raw.get("camera_poses", {})
        cameras = {}
        for name in sorted(set(intrinsics) & set(poses)):
            if not str(name).isdigit():
                continue
            camera = intrinsics[name]
            pose = poses[name]
            view = torch.eye(4, dtype=torch.float32)
            view[:3, :3] = torch.as_tensor(pose["R"], dtype=torch.float32).reshape(3, 3)
            view[:3, 3] = torch.as_tensor(pose["T"], dtype=torch.float32).reshape(3)
            cameras[int(name)] = {
                "K": torch.as_tensor(camera["K"], dtype=torch.float32).reshape(3, 3),
                "dist": np.asarray(camera.get("dist", []), dtype=np.float32),
                "V": view,
            }
        return cameras

    def __len__(self) -> int:
        return len(self.camera_ids)

    def get_timestep(self, data: dict) -> dict:
        times = torch.arange(self.T, dtype=torch.float32) / self.T
        return data | {"target_time": times}

    def _load_video(self, camera_id: int) -> torch.Tensor:
        capture = cv2.VideoCapture(str(self.video_index[camera_id]))
        if not capture.isOpened():
            raise ValueError(f"Could not open TalkBody4D video {self.video_index[camera_id]}")
        images = []
        try:
            for frame_id in self.frame_ids:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, image = capture.read()
                if not ok:
                    raise ValueError(
                        f"Could not decode frame {frame_id} from "
                        f"{self.video_index[camera_id]}"
                    )
                camera = self.cameras[camera_id]
                image = cv2.undistort(
                    image, camera["K"].numpy(), camera["dist"]
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = self._get_resized_image(
                    torch.from_numpy(image).permute(2, 0, 1)
                )
                images.append(resized.permute(1, 2, 0))
        finally:
            capture.release()
        return torch.stack(images)

    def _load_masks(self, camera_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        masks = []
        boundaries = []
        camera = self.cameras[camera_id]
        for frame_id in self.frame_ids:
            path = self.mask_index[camera_id][frame_id]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read TalkBody4D matting {path}")
            mask = cv2.undistort(mask, camera["K"].numpy(), camera["dist"])
            resized = self._get_resized_image(torch.from_numpy(mask)[None])[:1]
            masks.append(resized.permute(1, 2, 0))
            boundaries.append(self._get_boundary_mask(resized, kernel_size=3))
        return torch.stack(masks), torch.stack(boundaries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        camera_id = self.camera_ids[index]
        rgb = self._load_video(camera_id)
        masks, boundaries = self._load_masks(camera_id)
        intrinsics = self._get_resized_intrinsics(self.cameras[camera_id]["K"])
        extrinsics = self.cameras[camera_id]["V"]
        camera_center = torch.linalg.inv(extrinsics)[:3, 3]
        return {
            "rgb": rgb,
            "mask": masks,
            "boundary_mask": boundaries,
            "K_CV": intrinsics,
            "V_CV": extrinsics,
            "camera_id": torch.tensor(camera_id, dtype=torch.int32),
            "valid": torch.ones((self.T, 1), dtype=torch.bool),
            "camera_center": camera_center,
        }

    @property
    def smplx_params(self) -> dict[str, torch.Tensor]:
        frames = [json.loads(self.smplx_index[frame].read_text()) for frame in self.frame_ids]

        def stacked(source: str, width: int | None = None) -> np.ndarray:
            values = np.stack(
                [np.asarray(frame[source], dtype=np.float32).reshape(-1) for frame in frames]
            )
            return values[:, :width] if width is not None else values

        params = {
            "global_orient": stacked("Rh"),
            "transl": stacked("Th"),
            "body_pose": stacked("body_pose"),
            "jaw_pose": stacked("jaw_pose"),
            "betas": stacked("betas", 10).mean(axis=0, keepdims=True),
            "expression": stacked("expression", 10),
            "left_hand_pose": stacked("left_hand_pose"),
            "right_hand_pose": stacked("right_hand_pose"),
        }
        for key in ("leye_pose", "reye_pose"):
            if key in frames[0]:
                params[key] = stacked(key)
        return {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in params.items()
        }


def unified_dataset_from_subject(
    args,
    subject: dict[str, Any],
    *,
    novel_kind: str | None = None,
    forward_labels: bool = False,
):
    """Select the PiG reader matching the canonical subject's storage format."""
    if subject.get("dataset") == "4ddress":
        return FourDDressPigDataset(args, subject, forward_labels=forward_labels)
    if subject.get("dataset") == "talkbody4d":
        return TalkBody4DPigDataset(args, subject)
    if subject.get("dataset") != "dna":
        return UnifiedPigDataset(args, subject)

    import ext

    subject_root = Path(subject.get("subject_path") or "")
    sample_code_roots = (
        PIG_REPO / "ext",
        subject_root / "dna_rendering_sample_code",
        subject_root.parent / "dna_rendering_sample_code",
    )
    for root in sample_code_roots:
        if (root / "dna_rendering_sample_code" / "SMCReader.py").is_file():
            root_string = str(root.resolve())
            if root_string not in ext.__path__:
                ext.__path__.append(root_string)

    from src.io.datasets.dna import (
        DNADataset,
        DNADataset_NovelPose,
        DNADataset_NovelView,
    )

    dataset_class = DNADataset if subject.get("training_config") else {
        None: DNADataset,
        "pose": DNADataset_NovelPose,
        "view": DNADataset_NovelView,
    }[novel_kind]

    class ManifestDNADataset(dataset_class):
        def __init__(self, dataset_args, manifest_subject):
            self.manifest_subject = manifest_subject
            dataset_args.io_args["directory"] = str(manifest_subject["subject_id"])
            super().__init__(dataset_args)
            train = manifest_subject.get("training_config")
            if train:
                self.camera_ids = train["camera_ids"] if "camera_ids" in train else list(range(train["views"]))
            if novel_kind is None and manifest_subject.get("training_config") is not None:
                # Import by file path: upstream owns the src package in this process.
                import importlib.util
                helper_path = Path(__file__).with_name("training_inputs.py")
                helper_spec = importlib.util.spec_from_file_location("vci_training_inputs", helper_path)
                helper = importlib.util.module_from_spec(helper_spec)
                helper_spec.loader.exec_module(helper)
                self._read_smc("rgb", lambda rgb: self._read_smc(
                    "smplx", lambda annot: helper.validate_dna_inputs(manifest_subject, rgb, annot)
                ))

        @property
        def frame_ids(self):
            train = self.manifest_subject.get("training_config")
            return _selection_frames(train) if train else super().frame_ids

        def load_smplx_params(self, source="dataset"):
            if not self.manifest_subject.get("training_config") or source != "dataset":
                return super().load_smplx_params(source)
            return self._smplx_params_from_raw(self._load_smplx_raw(), self.frame_ids)

        def __getitem__(self, index):
            if hasattr(self, "camera_ids"):
                return DNADataset.__getitem__(self, self.camera_ids[index])
            return super().__getitem__(index)

        def get_frame(self, time_idx, camera_idx, compute_boundary=False):
            if not 0 <= camera_idx < self.C:
                raise IndexError(f"Invalid camera index {camera_idx}")
            previous = getattr(self, "start_camera", 0)
            self._single_frame = True
            if hasattr(self, "camera_ids"):
                # Keep upstream's bounds check on the local index while mapping the source ID.
                self.start_camera = self.camera_ids[camera_idx] - camera_idx
            try:
                return DNADataset.get_frame(self, time_idx, camera_idx, compute_boundary)
            finally:
                self.start_camera = previous
                self._single_frame = False

        def valid_matrix(self):
            if not hasattr(self, "camera_ids"):
                return super().valid_matrix()
            from src.io.datasets.dna_smc import DiskSMCReader
            valid = torch.zeros((self.T, self.C), dtype=torch.bool)
            rgb, mask = DiskSMCReader(self.smc_file["rgb"]), DiskSMCReader(self.smc_file["mask"])
            try:
                for c, camera in enumerate(self.camera_ids):
                    for t, frame in enumerate(self.frame_ids):
                        valid[t, c] = rgb.frame_exists(f"{self._camera_group(camera)}/{camera}/color/{frame}") and mask.frame_exists(f"Mask/{camera}/mask/{frame}")
            finally:
                rgb.close()
                mask.close()
            return valid

        def _find_smc_files(self, _file_identifier: str) -> dict[str, Path]:
            return {
                "rgb": _required_path(self.manifest_subject, "images"),
                "mask": _required_path(self.manifest_subject, "masks"),
                "smplx": _required_path(self.manifest_subject, "smplx"),
            }

        def _get_resized_image(self, image: torch.Tensor) -> torch.Tensor:
            had_batch_dimension = image.ndim == 4
            resized = super()._get_resized_image(image)
            if had_batch_dimension and resized.ndim == 3 and not getattr(self, "_single_frame", False):
                resized = resized.unsqueeze(0)
            return resized

    return ManifestDNADataset(args, subject)


class _DiskFrameSource:
    """Load only the sampled time/camera pairs from their source files."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.T, self.C = int(dataset.T), len(dataset)
        self._videos = {}
        self._probe = self._first_valid()

    def __del__(self):
        for capture in self._videos.values():
            capture.release()

    def _camera_id(self, camera_index):
        ids = getattr(self.dataset, "camera_ids", None)
        return ids[camera_index] if ids is not None else camera_index

    def _first_valid(self):
        for camera_index in range(self.C):
            for time_index in range(self.T):
                frame = self._read(time_index, camera_index)
                if frame is not None:
                    return frame
        raise ValueError("The disk-backed PiG dataset contains no readable frames.")

    def metadata(self):
        values = {"K_CV": [], "V_CV": [], "camera_center": [], "camera_id": []}
        for camera_index in range(self.C):
            camera_id = self._camera_id(camera_index)
            if hasattr(self.dataset, "cameras"):
                camera = self.dataset.cameras[camera_id]
                K = self.dataset._get_resized_intrinsics(camera["K"])
                V = camera["V"]
                # V maps world to camera; its inverse translation is the
                # world-space camera center (-R.T @ t for a rigid pose).
                center = torch.linalg.inv(V)[:3, 3]
            else:
                K, V, center = self.dataset._load_camera(camera_id)
            values["K_CV"].append(K)
            values["V_CV"].append(V)
            values["camera_center"].append(center)
            values["camera_id"].append(camera_id)
        return {
            key: (
                torch.tensor(value, dtype=torch.int32)
                if key == "camera_id"
                else torch.stack(value)
            )
            for key, value in values.items()
        }

    def _talkbody(self, time_index, camera_index):
        camera_id = self._camera_id(camera_index)
        frame_id = self.dataset.frame_ids[time_index]
        capture = self._videos.get(camera_id)
        if capture is None:
            capture = cv2.VideoCapture(str(self.dataset.video_index[camera_id]))
            if not capture.isOpened():
                raise ValueError(f"Could not open {self.dataset.video_index[camera_id]}")
            self._videos[camera_id] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
        mask_path = self.dataset.mask_index[camera_id].get(frame_id)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        if not ok or mask is None:
            return None
        camera = self.dataset.cameras[camera_id]
        image = cv2.undistort(image, camera["K"].numpy(), camera["dist"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = self.dataset._get_resized_image(
            torch.from_numpy(image).permute(2, 0, 1)
        ).permute(1, 2, 0)
        mask = cv2.undistort(mask, camera["K"].numpy(), camera["dist"])
        mask = self.dataset._get_resized_image(torch.from_numpy(mask)[None])[:1]
        return rgb, mask.permute(1, 2, 0), self.dataset._get_boundary_mask(mask, 3)

    def _dna(self, time_index, camera_index):
        camera_id = self._camera_id(camera_index)
        frame_id = self.dataset.frame_ids[time_index]
        rgb = self.dataset._read_smc(
            "rgb",
            lambda reader: reader.get_img(
                self.dataset._camera_group(camera_id), camera_id, "color",
                Frame_id=[frame_id], disable_tqdm=True,
            ),
        )
        mask = self.dataset._read_smc(
            "mask",
            lambda reader: reader.get_mask(
                camera_id, Frame_id=[frame_id], disable_tqdm=True
            ),
        )
        if rgb is None or mask is None:
            return None
        rgb = self.dataset._get_resized_image(
            torch.as_tensor(rgb).flip(dims=[-1])
        )[0].permute(1, 2, 0)
        mask = self.dataset._get_resized_image(torch.as_tensor(mask)[..., None])[0, :1]
        return rgb, mask.permute(1, 2, 0), self.dataset._get_boundary_mask(mask, 3)

    def _read(self, time_index, camera_index):
        if isinstance(self.dataset, TalkBody4DPigDataset):
            return self._talkbody(time_index, camera_index)
        if hasattr(self.dataset, "smc_file"):
            return self._dna(time_index, camera_index)
        camera_id = self._camera_id(camera_index)
        frame_id = self.dataset.frame_ids[time_index]
        return self.dataset._load_frame(
            self.dataset.image_index[camera_id].get(frame_id),
            self.dataset.mask_index[camera_id].get(frame_id),
        )

    def __getitem__(self, index):
        time_index, camera_index = divmod(int(index), self.C)
        frame = self._read(time_index, camera_index)
        valid = frame is not None
        if frame is None:
            rgb, mask, boundary = (torch.zeros_like(item) for item in self._probe)
        else:
            rgb, mask, boundary = frame
        if mask.dtype == torch.uint8:
            mask = mask > (127 if mask.max() > 1 else 0)
        elif mask.dtype != torch.bool:
            mask = mask > 0.5
        boundary = boundary.to(torch.bool)
        rgb = rgb.clone()
        rgb.masked_fill_(~mask, 0)
        return {
            "index": (time_index, camera_index),
            "rgb": rgb,
            "mask": mask,
            "boundary_mask": boundary,
            "valid": torch.tensor([valid], dtype=torch.bool),
        }


class _DiskBackedBatchDataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        del device
        self.source = data.pop("_vci_disk_source")
        self.T, self.C = self.source.T, self.source.C
        self.target_time = data.pop("target_time")
        self.semantic_labels = data.pop("_vci_semantic_labels", None)

    def __len__(self):
        return self.T * self.C

    def __getitem__(self, index):
        output = self.source[index]
        time_index = output["index"][0]
        output["target_time"] = self.target_time[time_index]
        if self.semantic_labels is not None:
            output["semantic_labels"] = self.semantic_labels[time_index]
        return output


def _enable_disk_backed_batches(data_module, runner_module):
    import shutil
    from src.io.datasets import opt as dataset_opt_module
    from src.opt import base as opt_base_module
    from src.opt import model as opt_model_module

    def target_shape(data):
        if "_vci_shape" in data:
            return data["_vci_shape"]
        for key in ("rgb", "mask"):
            if key in data:
                return data[key].shape
        raise ValueError("No valid dataset shape was provided.")

    def prepare_data(args, num_workers=0):
        del num_workers
        dataset = data_module._dataset_from_args(args)
        source = _DiskFrameSource(dataset)
        height, width = source._probe[0].shape[:2]
        print(f"Training inputs: views={source.C}, timesteps={source.T}, "
              f"decoded image resolution={width}x{height} (width x height)", flush=True)
        data = source.metadata()
        data["_vci_disk_source"] = source
        data["_vci_shape"] = (source.T, source.C, height, width, 3)
        data["smplx_params"] = {
            key: value.to(torch.float32) for key, value in dataset.smplx_params.items()
        }
        if args.noise > 0:
            data["smplx_params"] = data_module.add_so3_noise_to_body_pose(
                data["smplx_params"], sigma_rad=args.noise, seed=42
            )
        torch.save(
            data["smplx_params"],
            Path(args.io_args["out_path"]) / "smplx_params_noised.pt",
        )
        data = dataset.get_timestep(data)
        labels = getattr(dataset, "semantic_labels", None)
        if labels is not None:
            data["_vci_semantic_labels"] = labels
        args.method_args["target"]["resolution"] = (height, width)
        print(
            "Using disk-backed training data; only sampled time/camera frames "
            "are decoded into system memory."
        )
        with (Path(args.io_args["out_path"]) / "method_args.json").open("w") as stream:
            json.dump(args.method_args, stream)
        shutil.copytree("src", Path(args.io_args["out_path"]) / "src", dirs_exist_ok=True)
        return args, data

    data_module.prepare_data = prepare_data
    runner_module.prepare_data = prepare_data
    dataset_opt_module.target_shape = target_shape
    opt_base_module.target_shape = target_shape
    opt_model_module.BatchDataset = _DiskBackedBatchDataset
    opt_base_module.BatchDataset = _DiskBackedBatchDataset

def _enable_semantic_label_batches(data_module) -> None:
    original_load = data_module.load_dataset_streaming

    def load_dataset_streaming(dataset):
        data = original_load(dataset)
        labels = getattr(dataset, "semantic_labels", None)
        if labels is not None:
            data["semantic_labels"] = labels
        return data

    from src.io.datasets.opt import BatchDataset, TCCollate

    original_batch_init = BatchDataset.__init__
    original_batch_getitem = BatchDataset.__getitem__
    original_collate = TCCollate.__call__

    def batch_init(self, data, device):
        labels = data.pop("semantic_labels", None)
        original_batch_init(self, data, device)
        if labels is not None:
            self.semantic_labels = labels

    def batch_getitem(self, index):
        output = original_batch_getitem(self, index)
        if hasattr(self, "semantic_labels"):
            time_index = int(index // self.C)
            output["semantic_labels"] = self.semantic_labels[time_index]
        return output

    def collate(self, batch):
        output = original_collate(self, batch)
        if "semantic_labels" in batch[0]:
            camera_count = len({item["index"][1] for item in batch})
            time_count = len(batch) // camera_count
            labels = torch.stack([item["semantic_labels"] for item in batch])
            output["semantic_labels"] = labels.view(
                time_count, camera_count, *labels.shape[1:]
            ).to(self.device)
        return output

    data_module.load_dataset_streaming = load_dataset_streaming
    BatchDataset.__init__ = batch_init
    BatchDataset.__getitem__ = batch_getitem
    TCCollate.__call__ = collate


def configured_novel_dataset(args, subject: dict, kind: str):
    """Use the same source-ID selection for in-process and checkpoint evaluation."""
    name = {"view": "novel_view", "pose": "novel_pose"}[kind]
    if name not in subject.get("evaluation_config", {}):
        raise ValueError(f"No {name} split configured")
    selection = subject["evaluation_config"][name]
    training_frames = _selection_frames(subject["training_config"])
    frames = _selection_frames(selection)
    training_cameras = subject["training_config"]["camera_ids"]
    if kind == "view":
        if set(selection["camera_ids"]) & set(training_cameras):
            raise ValueError("Novel-view cameras overlap training cameras")
        if set(frames) - set(training_frames):
            raise ValueError("Novel-view frames must belong to training frames")
    elif set(frames) & set(training_frames):
        raise ValueError("Novel-pose frames overlap training frames")
    eval_args = copy.deepcopy(args)
    eval_args.method_args["target"].update(
        views=len(selection["camera_ids"]), time_steps=len(frames),
        resolution=subject["training_config"]["resolution"],
    )
    eval_subject = {
        **subject, "evaluation_selection": True, "evaluation_config": {},
        "training_config": {**selection, "resolution": subject["training_config"]["resolution"]},
    }
    if subject.get("evaluation_source"):
        eval_subject.update(subject["evaluation_source"])
        eval_subject["training_config"] = {**selection, "resolution": subject["training_config"]["resolution"]}
    dataset = unified_dataset_from_subject(eval_args, eval_subject)
    if kind == "view":
        # A subset of training frames keeps its original model-time coordinates.
        indices = {frame: index for index, frame in enumerate(training_frames)}
        times = torch.tensor([indices[frame] / len(training_frames) for frame in frames])
        dataset.get_timestep = lambda data: data | {"target_time": times}
    return dataset


def _enable_manifest_multi_subject_datasets(
    multi_subject_module, dataset_factory
) -> None:
    """Route unified-grid subject construction through the VCI manifest."""
    multi_subject_module._make_dataset = dataset_factory


def main() -> None:
    manifest_path, upstream_argv = _manifest_argument(sys.argv)
    forward_labels, upstream_argv = _pop_flag(
        upstream_argv, "--vci-forward-labels"
    )
    data_on_disk, upstream_argv = _pop_flag(
        upstream_argv, "--vci-data-on-disk"
    )
    subjects = _load_subjects(manifest_path)
    if forward_labels:
        unsupported = [
            str(subject.get("subject_id"))
            for subject in subjects
            if subject.get("dataset") != "4ddress"
        ]
        if unsupported:
            raise SystemExit(
                "--forward-labels currently supports only 4D-DRESS subjects; "
                f"unsupported: {', '.join(unsupported)}"
            )
    _redirect_hardcoded_smplx(_configured_smplx_model(upstream_argv))
    subjects_by_id = {
        str(subject.get("subject_id") or index): subject
        for index, subject in enumerate(subjects)
    }

    from src.opt_pipeline import data as data_module
    from src.opt_pipeline import evaluation as evaluation_module
    from src.opt_pipeline import runner as runner_module
    Opt_Run = runner_module.Opt_Run

    original_novel_dataset = evaluation_module._novel_dataset
    if forward_labels:
        _enable_semantic_label_batches(data_module)
    if data_on_disk:
        _enable_disk_backed_batches(data_module, runner_module)

    def dataset_from_args(args):
        subject_id = str(args.io_args["directory"])
        dataset = unified_dataset_from_subject(
            args,
            subjects_by_id[subject_id],
            forward_labels=forward_labels,
        )
        args.vci_subject = copy.deepcopy(subjects_by_id[subject_id])
        train_cameras = list(getattr(dataset, "camera_ids", range(len(dataset))))
        resolution = args.method_args.get("target", {}).get("resolution")
        split_subject = dict(subjects_by_id[subject_id])
        if split_subject.get("evaluation_source"):
            split_subject.update(split_subject["evaluation_source"])
        _write_evaluation_split(
            Path(args.io_args["out_path"]),
            split_subject,
            train_cameras=train_cameras,
            resolution=resolution,
        )
        return dataset

    def novel_dataset(args, kind: str):
        subject_id = str(args.io_args["directory"])
        subject = subjects_by_id[subject_id]
        if subject.get("evaluation_config"):
            return configured_novel_dataset(args, subject, kind)
        if subject.get("dataset") != "dna":
            return original_novel_dataset(args, kind)
        return unified_dataset_from_subject(args, subject, novel_kind=kind)

    def directories(_self):
        return list(subjects_by_id)

    original_args_for_directory = Opt_Run._args_for_directory

    def args_for_directory(self, directory: str):
        args = original_args_for_directory(self, directory)
        subject = subjects_by_id[directory]
        subject_path = Path(subject.get("subject_path") or manifest_path.parent)
        args.io_args["input_directory"] = subject_path
        if subject.get("dataset") == "dna":
            args.dataset = "dna"
            args.io_args["directory_path"] = str(subject_path)
        elif subject.get("dataset") == "4ddress":
            args.dataset = "4ddress"
        elif subject.get("dataset") == "talkbody4d":
            args.dataset = "talkbody4d"
        return args

    data_module._dataset_from_args = dataset_from_args
    evaluation_module._novel_dataset = novel_dataset
    from src.opt_pipeline import novel_pose_evaluation
    novel_pose_evaluation._novel_pose_dataset = lambda args: novel_dataset(args, "pose")
    Opt_Run._directories = directories
    Opt_Run._args_for_directory = args_for_directory

    # Unified-grid training constructs datasets through the upstream
    # MultiSubjectDiskDataset instead of data_module._dataset_from_args. Route
    # that loader through the same manifest-backed factory; otherwise its
    # default ``thuman4d`` argument makes it look for data.json beneath the
    # configured THuman root even when the manifest contains 4D-DRESS takes.
    from src.io.datasets import multi_subject as multi_subject_module

    _enable_manifest_multi_subject_datasets(
        multi_subject_module, dataset_from_args
    )

    sys.argv = upstream_argv
    runpy.run_path("Main.py", run_name="__main__")
