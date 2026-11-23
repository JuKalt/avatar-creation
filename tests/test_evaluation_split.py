from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from avatar_creation.methods.evaluation_split import write_evaluation_split


class EvaluationSplitTests(unittest.TestCase):
    def test_sidecar_records_training_split_and_original_subject_data(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            images = source / "images"
            masks = source / "masks"
            calibration = source / "calibration.json"
            smplx = source / "smpl_params.npz"
            subject = {
                "subject_id": "subject_01",
                "dataset": "processed",
                "subject_path": source,
                "images": images,
                "masks": masks,
                "calibration": calibration,
                "smplx": smplx,
            }

            path = write_evaluation_split(
                Path(output_tmp),
                subject,
                train_cameras=[0, 2, 4],
                resolution=1024,
            )

            split = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(split["subject_id"], "subject_01")
            self.assertEqual(split["dataset_kind"], "processed")
            self.assertEqual(split["original_dataset_path"], str(source))
            self.assertEqual(split["original_data"]["images"], str(images))
            self.assertEqual(split["original_data"]["masks"], str(masks))
            self.assertEqual(split["original_data"]["calibration"], str(calibration))
            self.assertEqual(split["original_data"]["smplx"], str(smplx))
            self.assertEqual(split["train_cameras"], [0, 2, 4])
            self.assertEqual(split["train_views"], 3)
            self.assertEqual(split["resolution"], 1024)

    def test_unscaled_data_uses_source_resolution_marker(self) -> None:
        with tempfile.TemporaryDirectory() as output_tmp:
            path = write_evaluation_split(
                Path(output_tmp),
                {"subject_id": "subject_01", "subject_path": "/data/subject_01"},
                train_cameras=["cam00"],
                resolution=0,
            )

            split = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(split["resolution"], "source")
            self.assertEqual(split["train_cameras"], ["cam00"])


if __name__ == "__main__":
    unittest.main()
