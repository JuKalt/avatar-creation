from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_creation.methods.registry import MethodSpec
from avatar_creation.methods.runner import PreparedMethod, run_method


class RunnerFailureLoggingTests(unittest.TestCase):
    def test_failed_plain_run_prints_and_retains_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "failed-method"
            spec = MethodSpec(name="failing", command=[])
            prepared = PreparedMethod(
                spec=spec,
                command=[
                    sys.executable,
                    "-c",
                    "print('failure detail', flush=True); raise SystemExit(7)",
                ],
                output_dir=output_dir,
                cwd=None,
                env=os.environ.copy(),
                summary="Run inputs: train=20 views, eval=4 views\n",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("avatar_creation.methods.runner.prepare_method", return_value=prepared),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(subprocess.CalledProcessError) as raised,
            ):
                run_method(
                    spec,
                    dataset_kind="processed",
                    dataset_path=None,
                    output_dir=None,
                    registry_path=None,
                    required_keys_value=None,
                    extra_args=[],
                )

            log_path = output_dir / "runner.log"
            self.assertEqual(raised.exception.returncode, 7)
            self.assertEqual(log_path.read_text(), prepared.summary + "failure detail\n")
            self.assertLess(stdout.getvalue().index("Run inputs:"), stdout.getvalue().index("failure detail"))
            self.assertIn("failure detail", stdout.getvalue())
            self.assertIn(f"Method failed. Full log: {log_path}", stderr.getvalue())

    def test_plain_run_preserves_in_place_progress_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "progress-method"
            spec = MethodSpec(name="progress", command=[])
            prepared = PreparedMethod(
                spec=spec,
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import sys; sys.stdout.write(\x27loss=1\\r\x27); "
                        "sys.stdout.flush(); sys.stdout.write(\x27loss=2\\r\x27); "
                        "sys.stdout.flush(); sys.stdout.write(\x27done\\n\x27)"
                    ),
                ],
                output_dir=output_dir,
                cwd=None,
                env=os.environ.copy(),
            )
            stdout = io.StringIO()

            with (
                patch("avatar_creation.methods.runner.prepare_method", return_value=prepared),
                contextlib.redirect_stdout(stdout),
            ):
                run_method(
                    spec,
                    dataset_kind="processed",
                    dataset_path=None,
                    output_dir=None,
                    registry_path=None,
                    required_keys_value=None,
                    extra_args=[],
                )

            progress = "loss=1\rloss=2\rdone\n"
            self.assertIn(progress, stdout.getvalue())
            self.assertEqual(
                (output_dir / "runner.log").read_bytes().decode(), progress
            )


if __name__ == "__main__":
    unittest.main()
