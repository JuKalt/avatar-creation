from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_creation.methods.pig_avatar import (
    DEFAULT_CPU_THREADS,
    _configure_cpu_environment,
    build_command,
    parse_args,
)


class PigAvatarArgumentsTests(unittest.TestCase):
    def _command(self, *extra: str) -> list[str]:
        args = parse_args(
            [
                "--dataset-manifest",
                "manifest.json",
                "--output-dir",
                "outputs",
                "--method-config",
                "method.json",
                *extra,
            ]
        )
        return build_command(args, Path("manifest.json"), Path("outputs"))

    def test_data_is_loaded_in_memory_by_default(self) -> None:
        args = parse_args(
            [
                "--dataset-manifest",
                "manifest.json",
                "--output-dir",
                "outputs",
            ]
        )
        self.assertTrue(args.data_in_memory)
        self.assertNotIn("--vci-data-on-disk", self._command())

    def test_cpu_threads_default_and_override(self) -> None:
        args = parse_args(
            ["--dataset-manifest", "manifest.json", "--output-dir", "outputs"]
        )
        overridden = parse_args(
            [
                "--dataset-manifest",
                "manifest.json",
                "--output-dir",
                "outputs",
                "--cpu-threads",
                "8",
            ]
        )
        self.assertEqual(args.cpu_threads, DEFAULT_CPU_THREADS)
        self.assertEqual(overridden.cpu_threads, 8)

    def test_cpu_environment_caps_threads_and_uses_worker_affinity(self) -> None:
        env: dict[str, str] = {}
        with (
            patch.dict("os.environ", {"VCI_DEVICE_SLOT_INDEX": "1"}),
            patch("os.sched_getaffinity", return_value=set(range(32))),
            patch("os.sched_setaffinity") as set_affinity,
        ):
            selected = _configure_cpu_environment(env, 8)

        self.assertEqual(selected, tuple(range(8, 16)))
        self.assertEqual(env["VCI_PIG_CPU_THREADS"], "8")
        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        set_affinity.assert_called_once_with(0, set(range(8, 16)))

    def test_disk_backed_loading_accepts_hyphenated_flag(self) -> None:
        args = parse_args(
            [
                "--dataset-manifest",
                "manifest.json",
                "--output-dir",
                "outputs",
                "--no-data-in-memory",
            ]
        )
        self.assertFalse(args.data_in_memory)
        self.assertIn("--vci-data-on-disk", self._command("--no-data-in-memory"))

    def test_data_in_memory_accepts_an_explicit_false_value(self) -> None:
        args = parse_args(
            [
                "--dataset-manifest",
                "manifest.json",
                "--output-dir",
                "outputs",
                "--data_in_memory",
                "false",
            ]
        )
        self.assertFalse(args.data_in_memory)

    def test_disk_backed_loading_accepts_underscore_flag(self) -> None:
        args = parse_args(
            [
                "--dataset-manifest",
                "manifest.json",
                "--output-dir",
                "outputs",
                "--no_data_in_memory",
            ]
        )
        self.assertFalse(args.data_in_memory)

    def test_unified_grid_uses_in_process_manifest_loader(self) -> None:
        command = self._command(
            "--training_mode",
            "unified_grid",
            "--num_workers",
            "8",
        )
        self.assertEqual(command[-2:], ["--num_workers", "0"])

    def test_single_subject_does_not_override_data_workers(self) -> None:
        command = self._command("--training_mode=single_subject")
        self.assertNotEqual(command[-2:], ["--num_workers", "0"])


if __name__ == "__main__":
    unittest.main()
