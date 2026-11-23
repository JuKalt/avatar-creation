from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from multiview_data.capacity import dataset_capacity
from multiview_data.loader import AVAILABLE_DATASETS
from avatar_creation.methods.actorshq_data import stage_actorshq_subject
from avatar_creation.methods.manifest import dataset_manifest
from avatar_creation.methods.pig_avatar import stage_actorshq_manifest


REQUIRED_KEYS = ("images", "masks", "calibration", "smplx")


def create_actorshq_root(root: Path, *, with_smplx: bool = True) -> Path:
    sequence = root / "Actor01" / "Sequence1"
    scale = sequence / "4x"
    rgb = scale / "rgbs" / "Cam001"
    mask = scale / "masks" / "Cam001"
    rgb.mkdir(parents=True)
    mask.mkdir(parents=True)
    for frame in (15, 16):
        (rgb / f"Cam001_rgb{frame:06d}.jpg").write_bytes(b"rgb")
        (mask / f"Cam001_mask{frame:06d}.png").write_bytes(b"mask")
    with (scale / "calibration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("name", "w", "h", "rx", "ry", "rz", "tx", "ty", "tz", "fx", "fy", "px", "py"),
        )
        writer.writeheader()
        writer.writerow(
            dict(name="Cam001", w=8, h=6, rx=0, ry=0, rz=0, tx=1, ty=2, tz=3, fx=1.5, fy=2, px=.5, py=.5)
        )
    (sequence / "scene.json").write_text(json.dumps({"num_frames": 20}))
    if with_smplx:
        zeros = np.zeros((20, 3), dtype=np.float32)
        np.savez(
            sequence / "smpl_params.npz",
            betas=np.zeros((1, 10), dtype=np.float32),
            global_orient=zeros,
            body_pose=np.zeros((20, 63), dtype=np.float32),
            left_hand_pose=np.zeros((20, 45), dtype=np.float32),
            right_hand_pose=np.zeros((20, 45), dtype=np.float32),
            transl=zeros,
        )
    return sequence


class ActorsHQTests(unittest.TestCase):
    def test_native_adapter_supports_root_sequence_scale_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = create_actorshq_root(root)
            for kind in ("actorshq", "actors-hq", "humanrf"):
                for path in (root, sequence, sequence / "4x"):
                    with self.subTest(kind=kind, path=path):
                        dataset = AVAILABLE_DATASETS[kind](path, required_keys=REQUIRED_KEYS)
                        subject = dataset.subjects[0]
                        self.assertEqual(dataset.kind, "actorshq")
                        self.assertEqual(subject["subject_id"], "Actor01_Sequence1")
                        self.assertEqual(subject["images"], sequence / "4x" / "rgbs")
                        self.assertEqual(subject["smplx"], sequence / "smpl_params.npz")
                        self.assertEqual(subject["missing_keys"], ())

    def test_missing_smplx_sidecar_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_actorshq_root(root, with_smplx=False)
            dataset = AVAILABLE_DATASETS["actorshq"](root, required_keys=REQUIRED_KEYS)
            self.assertEqual(dataset.subjects[0]["missing_keys"], ("smplx",))

    def test_run_local_stage_converts_names_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_actorshq_root(root)
            subject = AVAILABLE_DATASETS["actorshq"](root, required_keys=REQUIRED_KEYS).subjects[0]
            destination = root / "method-output" / "data"
            stage_actorshq_subject(subject, destination, views=1, time_steps=2, resolution=0)

            self.assertTrue((destination / "images" / "cam00" / "00000015.jpg").is_symlink())
            self.assertTrue((destination / "masks" / "cam00" / "00000016.png").is_symlink())
            calibration = json.loads((destination / "calibration.json").read_text())
            self.assertEqual(calibration["cam00"]["K"], [12.0, 0.0, 4.0, 0.0, 12.0, 3.0, 0.0, 0.0, 1.0])
            self.assertEqual(calibration["cam00"]["T"], [-1.0, -2.0, -3.0])
            validity = json.loads((destination / "frame_validity.json").read_text())
            self.assertEqual(validity["frame_ids"]["ids"], [15, 16])

    def test_rotated_camera_stage_preserves_world_space_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = create_actorshq_root(root)
            calibration_path = sequence / "4x" / "calibration.csv"
            with calibration_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["rz"] = str(np.pi / 2)
            with calibration_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            subject = AVAILABLE_DATASETS["actorshq"](root, required_keys=REQUIRED_KEYS).subjects[0]
            destination = stage_actorshq_subject(
                subject, root / "stage", views=1, time_steps=2, resolution=0,
            )
            camera = json.loads((destination / "calibration.json").read_text())["cam00"]
            rotation = np.asarray(camera["R"]).reshape(3, 3)
            translation = np.asarray(camera["T"])
            np.testing.assert_allclose(rotation, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-7)
            np.testing.assert_allclose(-rotation.T @ translation, [1, 2, 3], atol=1e-7)

    def test_pig_manifest_is_rewritten_to_the_run_local_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_actorshq_root(root)
            dataset = AVAILABLE_DATASETS["actorshq"](root, required_keys=REQUIRED_KEYS)
            manifest = dataset_manifest(dataset, path=root / "manifest.json", required_keys=REQUIRED_KEYS)
            staged = stage_actorshq_manifest(
                manifest,
                root / "pig-output",
                Namespace(views=1, time_steps=2, resolution=0),
            )
            subject = json.loads(staged.read_text())["subjects"][0]
            self.assertEqual(subject["dataset"], "actorshq")
            self.assertEqual(subject["training_config"]["frame_range"], {"start": 15, "stop": 17})
            self.assertTrue(Path(subject["calibration"]).is_file())

    def test_capacity_uses_native_media_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_actorshq_root(root)
            dataset = AVAILABLE_DATASETS["actorshq"](root, required_keys=REQUIRED_KEYS)
            result = dataset_capacity(
                "actorshq", dataset.subjects, dna_commands=[],
                dna_helper=root / "unused.py", verbose=True,
            )
            self.assertEqual(result["views"], {"min": 1, "max": 1})
            self.assertEqual(result["timesteps"], {"min": 2, "max": 2})


if __name__ == "__main__":
    unittest.main()
