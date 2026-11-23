from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from avatar_creation.methods.training_inputs import validate_dna_inputs, validate_thuman_inputs
from tests.support import create_subject


class TrainingInputTests(unittest.TestCase):
    def test_thuman_selection_preserves_timeline_with_missing_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_subject(Path(directory), "subject01")
            subject = {"subject_id": "subject01", "images": root / "images",
                       "masks": root / "masks", "calibration": root / "calibration.json"}
            train = {"views": 1, "time_steps": 1, "resolution": 512}
            self.assertEqual(validate_thuman_inputs(subject, train),
                             {"camera_ids": [0], "frame_ids": [0]})
            with self.assertRaises(SystemExit):
                validate_thuman_inputs(subject, train | {"views": 2})
            # Missing pairs, including globally absent timesteps, do not compact T.
            (root / "masks/cam00/00000000.png").unlink()
            selection = validate_thuman_inputs(subject, train | {"time_steps": 2000})
            self.assertEqual(selection["frame_ids"], list(range(2000)))

    def test_dna_rejects_truncation_and_different_camera_ids(self):
        rgb, annot = Mock(), Mock()
        rgb.get_Camera_5mp_info.return_value = {"num_frame": 3}
        annot.get_SMPLx.return_value = {"transl": [0, 0, 0]}
        annot.get_Calibration_all.return_value = {"00": {}, "01": {}}
        subject = {"subject_id": "0166_04", "training_config":
                   {"views": 2, "time_steps": 3, "resolution": 0}}
        validate_dna_inputs(subject, rgb, annot)
        annot.get_Calibration_all.return_value = {"00": {}, "02": {}}
        with self.assertRaisesRegex(SystemExit, "consecutive 5MP"):
            validate_dna_inputs(subject, rgb, annot)
        annot.get_Calibration_all.return_value = {"00": {}, "01": {}}
        annot.get_SMPLx.return_value = {"transl": [0]}
        with self.assertRaisesRegex(SystemExit, "synchronized frames"):
            validate_dna_inputs(subject, rgb, annot)


if __name__ == "__main__":
    unittest.main()
