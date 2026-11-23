from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_creation.methods.cli import main, parse_args as parse_method_args
from avatar_creation.methods.registry import discover_methods, load_method_spec
from avatar_creation.methods.runner import prepare_method

from tests.support import (
    create_dna_root,
    create_fourddress_root,
    create_subject,
    create_talkbody4d_root,
    tree_snapshot,
)


DATASET_KINDS = (
    "processed",
    "thuman",
    "dna",
    "dancenet3d",
    "talkbody4d",
    "4ddress",
)


class MethodDatasetContractTests(unittest.TestCase):
    def test_pig_avatar_lists_only_the_new_name_and_reuses_legacy_output(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--list-methods"]), 0)
        self.assertIn("pig-avatar - Train PiG-Avatar", output.getvalue())
        self.assertNotIn("neu-pig", output.getvalue())
        self.assertEqual(load_method_spec("neu-pig-smplx"), load_method_spec("pig-avatar"))
        self.assertEqual(load_method_spec("pig-avatar").output_subdir, "neu-pig-smplx")

    def test_every_registered_method_accepts_every_dataset_manifest(self) -> None:
        methods = discover_methods()
        self.assertTrue(methods)

        for method_name, spec in methods.items():
            for dataset_kind in DATASET_KINDS:
                with (
                    self.subTest(method=method_name, dataset=dataset_kind),
                    tempfile.TemporaryDirectory() as dataset_tmp,
                    tempfile.TemporaryDirectory() as output_tmp,
                ):
                    dataset_root = Path(dataset_tmp)
                    if dataset_kind == "dna":
                        dna_paths = create_dna_root(dataset_root)
                        expected_subject_path = dataset_root
                        expected_paths = {
                            "images": dna_paths["images"],
                            "masks": dna_paths["masks"],
                            "calibration": dna_paths["smplx"],
                            "smplx": dna_paths["smplx"],
                        }
                    elif dataset_kind == "4ddress":
                        paths = create_fourddress_root(dataset_root)
                        expected_subject_path = paths["sequence"]
                        expected_paths = {
                            "images": paths["capture"],
                            "masks": paths["capture"],
                            "calibration": paths["calibration"],
                            "smplx": paths["smplx"],
                        }
                    elif dataset_kind == "talkbody4d":
                        paths = create_talkbody4d_root(dataset_root)
                        expected_subject_path = paths["actor"]
                        expected_paths = {
                            "images": paths["images"],
                            "masks": paths["masks"],
                            "calibration": paths["calibration"],
                            "smplx": paths["smplx"],
                        }
                    else:
                        source_subject = create_subject(dataset_root)
                        expected_subject_path = source_subject
                        expected_paths = {
                            "images": source_subject / "images",
                            "masks": source_subject / "masks",
                            "calibration": source_subject / "calibration.json",
                            "smplx": source_subject / "smpl_params.npz",
                        }
                    before = tree_snapshot(dataset_root)

                    with patch.dict(
                        os.environ,
                        {
                            "VCI_DATASET_CONFIG": str(Path(output_tmp) / "datasets.json"),
                            "VCI_DATASET_OUTPUT_CONFIG": str(
                                Path(output_tmp) / "outputs.json"
                            ),
                        },
                    ):
                        prepared = prepare_method(
                            spec,
                            dataset_kind=dataset_kind,
                            dataset_path=dataset_root,
                            output_dir=Path(output_tmp),
                            required_keys_value=None,
                            extra_args=[],
                            warn_missing=False,
                        )

                    self.assertEqual(tree_snapshot(dataset_root), before)
                    manifest_path = prepared.output_dir / "dataset_manifest.json"
                    self.assertTrue(manifest_path.is_file())
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["dataset"]["kind"], dataset_kind)
                    self.assertEqual(len(manifest["subjects"]), 1)
                    subject = manifest["subjects"][0]
                    self.assertEqual(
                        subject["subject_path"], str(expected_subject_path)
                    )
                    for key, path in expected_paths.items():
                        self.assertEqual(subject[key], str(path))
                    self.assertIn(str(manifest_path), prepared.command)
                    self.assertIn(str(prepared.output_dir), prepared.command)


    def test_subject_filter_is_consumed_and_limits_dna_manifest(self) -> None:
        args, forwarded = parse_method_args(
            [
                "pig-avatar",
                "--dataset-kind",
                "dna",
                "--subjects",
                "0166_04",
                "--cpu-threads",
                "8",
                "--",
                "--device",
                "0",
                "--time-steps",
                "180",
            ]
        )
        self.assertEqual(args.subjects, ["0166_04"])
        self.assertEqual(args.cpu_threads, 8)
        self.assertEqual(forwarded, ["--", "--device", "0", "--time-steps", "180"])

        with (
            tempfile.TemporaryDirectory() as dataset_tmp,
            tempfile.TemporaryDirectory() as output_tmp,
        ):
            dataset_root = Path(dataset_tmp)
            create_dna_root(dataset_root, "0166_04")
            create_dna_root(dataset_root, "0170_01")
            spec = discover_methods()["pig-avatar"]
            with patch.dict(
                os.environ,
                {
                    "VCI_DATASET_CONFIG": str(Path(output_tmp) / "datasets.json"),
                    "VCI_DATASET_OUTPUT_CONFIG": str(Path(output_tmp) / "outputs.json"),
                },
            ):
                prepared = prepare_method(
                    spec,
                    dataset_kind="dna",
                    dataset_path=dataset_root,
                    output_dir=Path(output_tmp),
                    required_keys_value="images,masks,calibration,smplx",
                    extra_args=["--device", "0", "--time-steps", "180"],
                    subject_ids=("0166_04",),
                    warn_missing=False,
                )

            manifest = json.loads(
                (prepared.output_dir / "dataset_manifest.json").read_text()
            )
            self.assertEqual(
                [subject["subject_id"] for subject in manifest["subjects"]],
                ["0166_04"],
            )
            self.assertNotIn("--subjects", prepared.command)
            self.assertIn("--device", prepared.command)

    def test_unknown_subject_filter_fails_before_method_launch(self) -> None:
        with (
            tempfile.TemporaryDirectory() as dataset_tmp,
            tempfile.TemporaryDirectory() as output_tmp,
        ):
            dataset_root = Path(dataset_tmp)
            create_dna_root(dataset_root, "0166_04")
            spec = discover_methods()["pig-avatar"]
            with (
                patch.dict(
                    os.environ,
                    {
                        "VCI_DATASET_CONFIG": str(Path(output_tmp) / "datasets.json"),
                        "VCI_DATASET_OUTPUT_CONFIG": str(
                            Path(output_tmp) / "outputs.json"
                        ),
                    },
                ),
                self.assertRaisesRegex(SystemExit, "Unknown dataset subject ID"),
            ):
                prepare_method(
                    spec,
                    dataset_kind="dna",
                    dataset_path=dataset_root,
                    output_dir=Path(output_tmp),
                    required_keys_value=None,
                    extra_args=[],
                    subject_ids=("missing",),
                    warn_missing=False,
                )

    def test_registered_method_cwd_is_independent_of_shell_directory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                spec = discover_methods()["pig-avatar"]
            finally:
                os.chdir(original_cwd)

        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(spec.cwd, (project_root / "external" / "neu-pig-smplx").resolve())


    def test_devices_wrap_method_in_subject_scheduler(self) -> None:
        with (
            tempfile.TemporaryDirectory() as dataset_tmp,
            tempfile.TemporaryDirectory() as output_tmp,
        ):
            dataset_root = Path(dataset_tmp)
            create_subject(dataset_root, "subject00")
            create_subject(dataset_root, "subject01")
            spec = discover_methods()["pig-avatar"]
            with patch.dict(
                os.environ,
                {
                    "VCI_DATASET_CONFIG": str(Path(output_tmp) / "datasets.json"),
                    "VCI_DATASET_OUTPUT_CONFIG": str(Path(output_tmp) / "outputs.json"),
                },
            ):
                prepared = prepare_method(
                    spec,
                    dataset_kind="processed",
                    dataset_path=dataset_root,
                    output_dir=Path(output_tmp),
                    required_keys_value=None,
                    extra_args=[],
                    devices=("0", "cuda:2"),
                    warn_missing=False,
                )

            self.assertEqual(prepared.output_dir, Path(output_tmp) / "neu-pig-smplx")
            self.assertEqual(
                prepared.command[:3],
                [sys.executable, "-m", "avatar_creation.methods.subject_scheduler"],
            )
            self.assertIn("0,cuda:2", prepared.command)
            self.assertIn("conda", prepared.command)


if __name__ == "__main__":
    unittest.main()
