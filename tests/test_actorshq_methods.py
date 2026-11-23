from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiview_data.loader import AVAILABLE_DATASETS
from avatar_creation.methods import animatable_gaussians, mmlphuman
from avatar_creation.methods.actorshq_data import stage_actorshq_subject
from tests.test_actorshq import REQUIRED_KEYS, create_actorshq_root


class ActorsHQMethodDispatchTests(unittest.TestCase):
    def _subject(self, root: Path) -> dict:
        create_actorshq_root(root)
        return AVAILABLE_DATASETS["actorshq"](
            root, required_keys=REQUIRED_KEYS
        ).subjects[0]

    def test_mmlphuman_uses_actorshq_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._subject(root)

            def create_weights(data_dir: Path, _subject_out: Path) -> None:
                weights = data_dir / "gaussian" / "lbs_weights_grid.npz"
                weights.write_bytes(b"fixture")

            with (
                patch.object(
                    mmlphuman,
                    "stage_actorshq_subject",
                    wraps=stage_actorshq_subject,
                ) as stage,
                patch.object(mmlphuman, "_generate_lbs_weights", side_effect=create_weights),
            ):
                data_dir, config = mmlphuman.prepare_subject(
                    subject, root / "output", {}, views=1, time_steps=2, resolution=0
                )
            stage.assert_called_once()
            self.assertTrue((data_dir / "calibration.json").is_file())
            self.assertTrue(config.is_file())

    def test_animatable_gaussians_uses_actorshq_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._subject(root)
            template = {"train": {"data": {}}, "test": {"data": {}}}
            with patch.object(
                animatable_gaussians,
                "stage_actorshq_subject",
                wraps=stage_actorshq_subject,
            ) as stage:
                data_dir, config, prepared = animatable_gaussians.prepare_subject(
                    subject, root / "output", template, views=1, time_steps=2, resolution=0
                )
            stage.assert_called_once()
            self.assertTrue((data_dir / "camera_id_mapping.json").is_file())
            self.assertTrue(config.is_file())
            self.assertFalse(prepared)


if __name__ == "__main__":
    unittest.main()
