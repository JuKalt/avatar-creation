"""The LOD method stays separate from the baseline MMLPHuman adapter."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_creation.methods import mmlphuman_lod
from avatar_creation.methods.mmlphuman_lod_impl.lod import LODSelection, lod_config
from avatar_creation.methods.registry import discover_methods


class MMLPHumanLODTests(unittest.TestCase):
    def test_registered_method_uses_separate_trainer_and_existing_evaluator(self):
        methods = discover_methods()
        baseline = methods["mmlphuman"]
        lod = methods["mmlphuman-lod"]
        self.assertEqual(lod.conda_env, baseline.conda_env)
        self.assertIn("/mmlphuman_lod.py", " ".join(lod.command))
        self.assertIn("/mmlphuman.py", " ".join(baseline.command))
        for command in lod.evaluation_scripts.values():
            script = command.split()[0]
            self.assertTrue((Path(__file__).resolve().parents[1] / script).is_file())

    def test_adapter_passes_lod_config_to_shared_staging_and_private_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "subject.yaml"
            config.write_text("seed: 11\nlod:\n  min_fraction: 0.2\n")
            args = ["--dataset-manifest", str(root / "manifest.json"),
                    "--output-dir", str(root / "runs"), "--config", str(config),
                    "--lod-max-fraction", "0.75", "--lod-seed", "7"]
            (root / "train.py").touch()
            with patch.object(mmlphuman_lod, "UPSTREAM", root), \
                 patch.object(mmlphuman_lod, "load_subjects", return_value=[{"subject_id": "test"}]), \
                 patch.object(mmlphuman_lod, "prepare_subject", return_value=(root / "data", root / "run.yaml")) as prepare, \
                 patch.object(mmlphuman_lod.subprocess, "run") as run:
                self.assertEqual(mmlphuman_lod.main(args), 0)
            template = prepare.call_args.args[2]
            self.assertEqual(template["seed"], 11)
            self.assertEqual(template["lod"], lod_config(0.2, 0.75, 7))
            command = run.call_args.args[0]
            self.assertEqual(Path(command[1]).name, "train.py")
            self.assertEqual(Path(command[1]).parent.name, "mmlphuman_lod_impl")
            self.assertEqual(run.call_args.kwargs["cwd"], root)

    def test_subset_is_nested_and_permutation_is_checkpointed(self):
        import torch

        lod = LODSelection(seed=13)
        lod.initialize(20, torch.device("cpu"))
        small = lod.indices(0.25)
        large = lod.indices(0.75)
        self.assertEqual(len(small), 5)
        self.assertEqual(len(large), 15)
        self.assertEqual(small.tolist(), large[:5].tolist())
        restored = LODSelection(seed=999)
        restored.restore(lod.permutation.clone(), 20, torch.device("cpu"))
        self.assertEqual(restored.indices(0.25).tolist(), small.tolist())
        with self.assertRaisesRegex(ValueError, "every Gaussian exactly once"):
            restored.restore(torch.zeros(20, dtype=torch.long), 20, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
