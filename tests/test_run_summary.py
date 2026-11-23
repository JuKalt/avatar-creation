import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from avatar_creation.methods.run_summary import build_run_summary


class RunSummaryTests(unittest.TestCase):
    def test_explicit_splits_and_measured_non_square_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for camera, size in ((0, (80, 120)), (1, (160, 90))):
                folder = root / f"cam{camera:02}"
                folder.mkdir()
                Image.new("RGB", size).save(folder / "0003.png")
            subject = dict(subject_id="subject00", dataset="thuman", images=str(root),
                           training_config=dict(camera_ids=[0], frame_range=dict(start=3, stop=7, step=2), resolution=0),
                           evaluation_config=dict(novel_view=dict(camera_ids=[1], frame_range=dict(start=3, stop=7, step=2))))
            dataset = SimpleNamespace(kind="thuman", root=root, subjects=[subject])
            summary = build_run_summary("pig-avatar", dataset, [], ("0", "1"))
            self.assertIn("train: views=1, timesteps=2, cameras=[0], frames=3..5 (step=2)", summary)
            self.assertIn("eval/novel_view: views=1, timesteps=2, cameras=[1]", summary)
            self.assertIn("80x120", summary)
            self.assertIn("160x90", summary)
            self.assertIn("0 (source image size)", summary)
            self.assertIn("GPU pool: 0,1", summary)
            subject["training_config"]["resolution"] = 60
            summary = build_run_summary("pig-avatar", dataset, [], ())
            self.assertIn("from samples): 40x60", summary)
            self.assertIn("from samples): 60x33", summary)

    def test_missing_config_does_not_invent_split(self):
        dataset = SimpleNamespace(kind="dna", root=Path('/data'), subjects=[dict(subject_id="s")])
        self.assertIn("no shared run config", build_run_summary("mmlphuman", dataset, [], ()))
