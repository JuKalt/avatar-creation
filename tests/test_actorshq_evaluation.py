from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.eval_animatable_gaussians import evaluation_plan as animatable_plan
from scripts.evaluation.eval_mmlphuman import evaluation_plan as mmlp_plan
from multiview_data.loader import AVAILABLE_DATASETS
from avatar_creation.methods import animatable_gaussians, mmlphuman
from tests.test_actorshq import REQUIRED_KEYS, create_actorshq_root


class ActorsHQEvaluationTests(unittest.TestCase):
    def _subject_with_heldout_camera(self, root: Path) -> dict:
        sequence = create_actorshq_root(root)
        scale = sequence / "4x"
        source_rgb = scale / "rgbs" / "Cam001" / "Cam001_rgb000015.jpg"
        source_mask = scale / "masks" / "Cam001" / "Cam001_mask000015.png"
        rgb = scale / "rgbs" / "Cam003"
        mask = scale / "masks" / "Cam003"
        rgb.mkdir()
        mask.mkdir()
        for frame in (15, 16):
            (rgb / f"Cam003_rgb{frame:06d}.jpg").write_bytes(source_rgb.read_bytes())
            (mask / f"Cam003_mask{frame:06d}.png").write_bytes(source_mask.read_bytes())
        with (scale / "calibration.csv").open("a", encoding="utf-8") as stream:
            stream.write("Cam003,8,6,0,0,0,3,2,1,1.5,2,0.5,0.5\n")
        subject = AVAILABLE_DATASETS["actorshq"](
            root, required_keys=REQUIRED_KEYS
        ).subjects[0]
        subject["training_config"] = {
            "camera_ids": [2],
            "frame_range": {"start": 15, "stop": 17},
            "resolution": 0,
        }
        subject["evaluation_config"] = {
            "novel_view": {
                "camera_ids": [0],
                "frame_range": {"start": 15, "stop": 17},
            }
        }
        return subject

    def test_mmlp_plan_resolves_source_ids_after_training_renumbering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._subject_with_heldout_camera(root)

            def create_weights(data_dir: Path, _subject_out: Path) -> None:
                (data_dir / "gaussian" / "lbs_weights_grid.npz").write_bytes(b"fixture")

            with patch.object(mmlphuman, "_generate_lbs_weights", side_effect=create_weights):
                _, config = mmlphuman.prepare_subject(
                    subject, root / "mmlp", {}, views=1, time_steps=2, resolution=0
                )
            checkpoint = config.parent / "model" / "chkpnt1.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"fixture")
            _, _, pairs, audit = mmlp_plan(checkpoint, "novel-view")
            self.assertEqual(audit["train_cameras"], ["cam02"])
            self.assertEqual(audit["evaluation_cameras"], ["cam00"])
            self.assertEqual(len(pairs), 2)

    def test_animatable_plan_reads_canonical_actorshq_evaluation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._subject_with_heldout_camera(root)
            template = {"train": {"data": {}}, "test": {"data": {}}}
            _, config, _ = animatable_gaussians.prepare_subject(
                subject, root / "animatable", template,
                views=1, time_steps=2, resolution=0,
            )
            checkpoint = config.parent / "model" / "iteration_1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "net.pt").write_bytes(b"fixture")
            plan = animatable_plan(checkpoint, "novel-view")
            self.assertEqual(plan["source_camera_ids"], [0])
            self.assertEqual(plan["valid_pairs"], [[0, 15], [0, 16]])


if __name__ == "__main__":
    unittest.main()
