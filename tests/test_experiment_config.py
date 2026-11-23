from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_creation.methods.experiment_config import load_experiment_config
from avatar_creation.methods.cli import main, parse_args
from avatar_creation.methods.registry import discover_methods
from avatar_creation.methods.runner import prepare_method
from avatar_creation.methods.training_inputs import validate_dancenet3d_camera_identity
from tests.support import create_dna_root, create_subject


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "input.json"
        self.config = {
            "schema_version": 1, "dataset_kind": "thuman", "subjects": ["subject01"],
            "train": {"views": 1, "time_steps": 1, "resolution": 0},
        }
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.config))

    def test_shipped_configs(self):
        root = Path(__file__).resolve().parents[1] / "configs/datasets"
        for kind in ("thuman", "dna", "dancenet3d"):
            self.assertEqual(load_experiment_config(root / f"{kind}.json")["dataset_kind"], kind)

    def test_cli_defaults_paths_overrides_and_method_options(self):
        self.config["dataset_path"] = "data"
        self.write()
        args, extra = parse_args([
            "mmlphuman", "--dataset-config", str(self.path), "--subjects", "subject01",
            "--", "--device", "0",
        ])
        self.assertEqual(args.dataset_kind, "thuman")
        self.assertEqual(args.dataset_path, self.root / "data")
        self.assertEqual(args.subjects, ["subject01"])
        self.assertEqual(extra, ["--views", "1", "--time-steps", "1", "--resolution", "0", "--device", "0"])
        args, _ = parse_args(["mmlphuman", "--dataset-config", str(self.path),
                              "--dataset-path", str(self.root / "override")])
        self.assertEqual(args.dataset_path, self.root / "override")
        self.assertEqual(parse_args(["mmlphuman"])[0].dataset_kind, "processed")

    def test_run_config_alias_and_subject_conflict(self):
        args, extra = parse_args(["mmlphuman", "--run-config", str(self.path)])
        self.assertEqual(args.training_config, self.config)
        self.assertIn("--resolution", extra)
        with self.assertRaises(SystemExit):
            parse_args(["mmlphuman", "--run-config", str(self.path), "--subjects", "other"])

    def test_invalid_configs(self):
        for update in (
            {"schema_version": 2}, {"schema_version": True}, {"dataset_kind": "other"},
            {"subjects": []}, {"subjects": [1]}, {"subjects": ["a", "a"]},
            {"vews": 3}, {"train": {"views": 1}}, {"dataset_path": False},
        ):
            with self.subTest(update=update), self.assertRaises(SystemExit):
                self.path.write_text(json.dumps({**self.config, **update}))
                load_experiment_config(self.path)
        for field, value in (("views", 0), ("time_steps", -1), ("resolution", -1), ("views", True), ("views", 1.5)):
            with self.subTest(field=field, value=value), self.assertRaises(SystemExit):
                self.path.write_text(json.dumps({**self.config, "train": {**self.config["train"], field: value}}))
                load_experiment_config(self.path)
        self.path.write_text("{")
        with self.assertRaisesRegex(SystemExit, "Cannot read"):
            load_experiment_config(self.path)

    def test_conflicting_method_flags_fail(self):
        for flag in ("--views=2", "--time_steps=2", "--time-steps=2", "--resolution=512", "--res=512", "--vie=2", "--dataset-manifest=other.json"):
            with self.subTest(flag=flag), self.assertRaisesRegex(SystemExit, "conflicting method arguments"):
                parse_args(["mmlphuman", "--dataset-config", str(self.path), "--", flag])

    def test_both_methods_receive_same_config_in_plain_and_tui_modes(self):
        for tui in (False, True):
            target = "avatar_creation.methods.textual_ui.run_textual_methods" if tui else "avatar_creation.methods.cli.run_method"
            if tui:
                try:
                    import textual  # noqa: F401  # availability probe
                except ModuleNotFoundError:
                    continue
            with patch("avatar_creation.methods.cli.validate_method_environments"), patch(target, return_value=0) as run:
                main(["pig-avatar", "mmlphuman", "--dataset-config", str(self.path),
                      *([] if tui else ["--no-tui"])])
            self.assertEqual(run.call_count, 1 if tui else 2)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["training_config"], self.config)
                self.assertEqual(call.kwargs["subject_ids"], ("subject01",))
                self.assertEqual(call.kwargs["extra_args"], ["--views", "1", "--time-steps", "1", "--resolution", "0"])

    def test_dna_animatable_rejected_before_environment_checks(self):
        self.config["dataset_kind"] = "dna"
        self.write()
        with patch("avatar_creation.methods.cli.validate_method_environments") as validate:
            with self.assertRaisesRegex(SystemExit, "does not support native DNA"):
                main(["animatable-gaussians", "--dataset-config", str(self.path), "--no-tui"])
            validate.assert_not_called()

    def test_saved_config_can_be_loaded_for_rerun(self):
        for kind in ("thuman", "dna"):
            root = self.root / kind
            root.mkdir()
            if kind == "dna":
                create_dna_root(root, "0166_04")
                subject = "0166_04"
            else:
                create_subject(root, "subject01")
                subject = "subject01"
            self.config.update(dataset_kind=kind, subjects=[subject])
            with patch.dict(os.environ, {
                "VCI_DATASET_CONFIG": str(self.root / "paths.json"),
                "VCI_DATASET_OUTPUT_CONFIG": str(self.root / "outputs.json"),
            }):
                prepared = prepare_method(
                    discover_methods()["mmlphuman"], dataset_kind=kind, dataset_path=root,
                    output_dir=root / "runs", required_keys_value=None,
                    extra_args=["--views", "1", "--time-steps", "1", "--resolution", "0"],
                    subject_ids=(subject,), training_config=self.config, warn_missing=False,
                )
            resolved = load_experiment_config(prepared.output_dir / "run_config.json")
            self.assertEqual(resolved["dataset_path"], str(root))
            self.assertEqual(resolved["subjects"], [subject])
            self.assertEqual(resolved["train"], self.config["train"])

    def test_dancenet_camera_identity_rejects_shifted_indices(self):
        subjects = []
        for name, source in (("first", "1516"), ("second", "1634")):
            path = self.root / name / "camera_id_mapping.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"cam19": source}))
            subjects.append({"subject_id": name, "camera_id_mapping": path})
        config = {"train": {"camera_ids": [19]}, "evaluation": {}}
        with self.assertRaisesRegex(SystemExit, "not physically consistent"):
            validate_dancenet3d_camera_identity(subjects, config)


if __name__ == "__main__":
    unittest.main()
