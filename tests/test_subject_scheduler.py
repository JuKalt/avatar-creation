from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_creation.methods.subject_scheduler import run_subjects


class SubjectSchedulerTests(unittest.TestCase):
    def test_quiet_mode_marks_subject_running_without_known_progress_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            manifest.write_text(
                json.dumps({"dataset": {}, "subjects": [{"subject_id": "one"}]}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return_code = run_subjects(
                    manifest_path=manifest,
                    devices=["0"],
                    command=[
                        sys.executable,
                        "-c",
                        "print('TP: 1%|#| 10/1000')",
                        str(manifest),
                    ],
                    log_dir=root / "logs",
                )

            rendered = output.getvalue()
            self.assertEqual(return_code, 0)
            self.assertIn("one                      running", rendered)
            self.assertIn("TP: 1%", (root / "logs/0000_one.log").read_text())

    def test_quiet_mode_continuously_redraws_optimization_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            manifest.write_text(
                json.dumps({"dataset": {}, "subjects": [{"subject_id": "one"}]}),
                encoding="utf-8",
            )
            script = (
                "print('hidden setup'); "
                "print('Subject 1/1 • Optimization 0% • Loss: 1.0 0:00:01 0:01:39'); "
                "print('Subject 1/1 • Optimization 0% • Loss: 0.9 0:00:02 0:01:38'); "
                "print('Subject 1/1 • Optimization 1% • Loss: 0.8 0:00:03 0:01:37')"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return_code = run_subjects(
                    manifest_path=manifest,
                    devices=["0"],
                    command=[sys.executable, "-c", script, str(manifest)],
                    log_dir=root / "logs",
                )

            rendered = output.getvalue()
            self.assertEqual(return_code, 0)
            self.assertNotIn("hidden setup", rendered)
            self.assertEqual(rendered.count("L="), 3)
            self.assertIn("0%", rendered)
            self.assertIn("1%", rendered)
            self.assertIn("ETA=0:01:39", rendered)
            self.assertIn("ETA=0:01:37", rendered)
            self.assertIn("hidden setup", (root / "logs/0000_one.log").read_text())

    def test_each_subject_gets_a_single_subject_manifest_and_one_visible_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": {"kind": "fixture", "root": str(root)},
                        "subjects": [
                            {"subject_id": "one"},
                            {"subject_id": "two"},
                            {"subject_id": "three"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result_dir = root / "results"
            result_dir.mkdir()
            script = (
                "import json, os, pathlib, sys; "
                "m=json.load(open(sys.argv[1])); s=m['subjects']; "
                "p=pathlib.Path(sys.argv[2]) / s[0]['subject_id']; "
                "p.write_text(json.dumps({'count': len(s), "
                "'gpu': os.environ['CUDA_VISIBLE_DEVICES'], "
                "'run_id': os.environ['VCI_SUBJECT_RUN_ID']}))"
            )

            return_code = run_subjects(
                manifest_path=manifest,
                devices=["cuda:2", "5"],
                command=[sys.executable, "-c", script, str(manifest), str(result_dir)],
                log_dir=root / "logs",
            )

            self.assertEqual(return_code, 0)
            for subject_id in ("one", "two", "three"):
                result = json.loads((result_dir / subject_id).read_text())
                self.assertEqual(result["count"], 1)
                self.assertIn(result["gpu"], {"2", "5"})
                self.assertEqual(result["run_id"], subject_id)

    def test_a_subject_failure_does_not_prevent_queued_subjects_from_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": {"kind": "fixture", "root": str(root)},
                        "subjects": [
                            {"subject_id": "bad"},
                            {"subject_id": "good"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            marker = root / "good-ran"
            script = (
                "import json, pathlib, sys; "
                "s=json.load(open(sys.argv[1]))['subjects'][0]['subject_id']; "
                "pathlib.Path(sys.argv[2]).touch() if s == 'good' else sys.exit(9)"
            )
            return_code = run_subjects(
                manifest_path=manifest,
                devices=["0"],
                command=[sys.executable, "-c", script, str(manifest), str(marker)],
                log_dir=root / "logs",
            )

            self.assertEqual(return_code, 1)
            self.assertTrue(marker.is_file())

    def test_subjects_use_launch_time_code_and_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method = root / "external" / "method"
            method.mkdir(parents=True)
            manifest = root / "dataset_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": {},
                        "subjects": [
                            {"subject_id": "one"},
                            {"subject_id": "two"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = method / "method.json"
            config.write_text('{"value": "original"}', encoding="utf-8")
            script = method / "run.py"
            script.write_text(
                "import json, os, pathlib, sys\n"
                "subject = json.load(open(sys.argv[1]))['subjects'][0]['subject_id']\n"
                "value = json.load(open(pathlib.Path(__file__).with_name('method.json')))['value']\n"
                "(pathlib.Path(sys.argv[2]) / subject).write_text('original-code:' + value)\n"
                "if subject == 'one':\n"
                "    pathlib.Path(os.environ['ORIGINAL_SCRIPT']).write_text(\"import sys; sys.exit(8)\\n\")\n"
                "    pathlib.Path(os.environ['ORIGINAL_CONFIG']).write_text('{\"value\": \"changed\"}')\n",
                encoding="utf-8",
            )
            # An output path can also live below the repository, but must not
            # be redirected into the temporary source snapshot.
            results = root / "results"
            results.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(method)
                with patch.dict(
                    os.environ,
                    {
                        "VCI_DATASET_REPO": str(root),
                        "PYTHONPATH": str(root),
                        "ORIGINAL_SCRIPT": str(script),
                        "ORIGINAL_CONFIG": str(config),
                    },
                ):
                    return_code = run_subjects(
                        manifest_path=manifest,
                        devices=["0"],
                        command=[sys.executable, str(script), str(manifest), str(results)],
                        log_dir=root / "logs",
                    )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(return_code, 0)
            self.assertEqual((results / "one").read_text(), "original-code:original")
            self.assertEqual((results / "two").read_text(), "original-code:original")


if __name__ == "__main__":
    unittest.main()
