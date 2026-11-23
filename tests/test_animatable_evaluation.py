"""Animatable evaluation must preserve exact VCI source camera and frame IDs."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from avatar_creation.methods.evaluation_split import write_evaluation_split


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "animatable_eval",
    ROOT / "scripts/evaluation/eval_animatable_gaussians.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class AnimatableEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / "run" / "subjects" / "subject00"
        self.source = self.root / "source"
        self.checkpoint = self.owner / "model" / "batch_10"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "net.pt").touch()

        calibration = {}
        for camera in range(5):
            name = f"cam{camera:02d}"
            calibration[name] = {
                "K": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "T": [0, 0, 0],
                "distCoeff": [0, 0, 0, 0, 0],
                "imgSize": [8, 8],
            }
            for kind in ("images", "masks"):
                directory = self.source / kind / name
                directory.mkdir(parents=True)
                for frame in range(5):
                    (directory / f"{frame:08d}.jpg").touch()
        (self.source / "calibration.json").write_text(json.dumps(calibration))
        (self.source / "smpl_params.npz").touch()

        self.subject = {
            "subject_id": "subject00",
            "subject_path": str(self.source),
            "dataset": "thuman",
            "images": str(self.source / "images"),
            "masks": str(self.source / "masks"),
            "calibration": str(self.source / "calibration.json"),
            "smplx": str(self.source / "smpl_params.npz"),
            "training_config": {
                "camera_ids": [0, 2, 4],
                "frame_range": {"start": 0, "stop": 3},
                "resolution": 0,
            },
            "evaluation_config": {
                "novel_view": {
                    "camera_ids": [1, 3],
                    "frame_range": {"start": 0, "stop": 3},
                },
                "novel_pose": {
                    "camera_ids": [0, 2, 4],
                    "frame_range": {"start": 3, "stop": 5},
                },
            },
        }
        self.owner.mkdir(parents=True, exist_ok=True)
        write_evaluation_split(
            self.owner,
            self.subject,
            train_cameras=[0, 1, 2],
            resolution=0,
        )
        (self.owner / "data").mkdir()
        (self.owner / "data" / "camera_id_mapping.json").write_text(
            json.dumps({"cam00": 0, "cam01": 2, "cam02": 4})
        )
        (self.owner / "animatable_gaussians.yaml").write_text(
            yaml.safe_dump(
                {
                    "train": {
                        "dataset": "MvRgbDatasetTHuman4",
                        "data": {
                            "data_dir": str(self.owner / "data"),
                            "frame_range": [0, 3, 1],
                            "used_cam_ids": [0, 1, 2],
                        },
                    },
                    "test": {"pose_data": {"data_path": "wrong.npz"}},
                    "model": {},
                }
            )
        )

    def test_plans_exact_source_splits(self):
        view = evaluation.evaluation_plan(self.owner, "novel-view")
        pose = evaluation.evaluation_plan(self.checkpoint, "novel-pose")
        self.assertEqual(view["source_camera_ids"], [1, 3])
        self.assertEqual(view["source_frame_ids"], [0, 1, 2])
        self.assertEqual(view["valid_pairs"], [
            [camera, frame] for camera in (1, 3) for frame in range(3)
        ])
        self.assertEqual(pose["source_camera_ids"], [0, 2, 4])
        self.assertEqual(pose["source_frame_ids"], [3, 4])

    def test_generated_configs_map_each_local_camera_to_source(self):
        plan = evaluation.evaluation_plan(self.owner, "novel-view")
        paths, reference = evaluation.prepare_evaluation(plan)
        self.assertEqual(
            reference["local_to_source_camera_ids"],
            {"cam00": 1, "cam01": 3},
        )
        self.assertEqual(len(paths), 2)
        for local, path in enumerate(paths):
            config = yaml.safe_load(path.read_text())
            source = (1, 3)[local]
            self.assertNotIn("pose_data", config["test"])
            self.assertEqual(config["test"]["render_view_idx"], local)
            self.assertEqual(config["test"]["data"]["frame_range"], [0, 3, 1])
            self.assertEqual(config["test"]["prev_ckpt"], str(self.checkpoint))
            self.assertTrue(config["test"]["output_dir"].endswith(f"cam{source:02d}"))

        staged = self.owner / "evaluation_inputs" / "novel_view"
        self.assertEqual(
            sorted(path.name for path in (staged / "images").iterdir()),
            ["cam00", "cam01"],
        )
        self.assertEqual(
            json.loads((staged / "camera_id_mapping.json").read_text()),
            {"cam00": 1, "cam01": 3},
        )

    def test_validate_only_does_not_stage_or_render(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = evaluation.main(
                ["--split", "novel-pose", "--path", str(self.owner), "--validate-only"]
            )
        self.assertEqual(status, 0)
        self.assertIn('"source_camera_ids": [', output.getvalue())
        self.assertFalse((self.owner / "evaluation_inputs").exists())
        self.assertFalse((self.owner / "evaluation").exists())

    def test_rejects_checkpoint_training_mismatch(self):
        mapping = self.owner / "data" / "camera_id_mapping.json"
        mapping.write_text(json.dumps({"cam00": 0, "cam01": 2}))
        with self.assertRaisesRegex(ValueError, "training cameras disagree"):
            evaluation.evaluation_plan(self.owner, "novel-view")


if __name__ == "__main__":
    unittest.main()
