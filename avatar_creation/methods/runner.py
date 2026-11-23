"""Prepare and launch registered methods against unified dataset manifests."""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys

import multiview_data
from dataclasses import dataclass
from pathlib import Path

from multiview_data.config import resolve_output_path
from multiview_data.actorshq import validate_actorshq_selection
from multiview_data.loader import load_dataset, parse_required_keys

from .manifest import dataset_manifest
from .run_summary import build_run_summary
from .registry import MethodSpec
from .training_inputs import (
    validate_dancenet3d_camera_identity,
    validate_dancenet3d_inputs,
    validate_thuman_inputs,
)


@dataclass(frozen=True)
class PreparedMethod:
    spec: MethodSpec
    command: list[str]
    output_dir: Path
    cwd: Path | None
    env: dict[str, str]
    summary: str = ""


def render_command(
    spec: MethodSpec,
    *,
    manifest_path: Path,
    output_dir: Path,
    extra_args: list[str],
) -> list[str]:
    values = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "method": spec.name,
        "project_root": str(Path(__file__).resolve().parents[2]),
    }
    command = [part.format(**values) for part in spec.command]
    return [*spec.runner_prefix, *command, *extra_args]


def method_environment(spec: MethodSpec) -> dict[str, str]:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    pythonpath = env.get("PYTHONPATH")
    env["VCI_DATASET_REPO"] = str(project_root)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("FORCE_COLOR", "1")
    env.setdefault("TERM", "xterm-256color")
    # The adapter runs in a separate method environment, which may have no
    # installed entry for the dataset package yet.
    data_package_root = Path(multiview_data.__file__).resolve().parents[1]
    entries = [str(project_root), str(data_package_root)]
    if pythonpath:
        entries.extend(pythonpath.split(os.pathsep))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    if spec.env:
        env.update(spec.env)
    return env


def _select_subjects(dataset, subject_ids: tuple[str, ...]) -> None:
    if not subject_ids:
        return

    subjects_by_id = {str(item["subject_id"]): item for item in dataset.subjects}
    missing = [subject_id for subject_id in subject_ids if subject_id not in subjects_by_id]
    if missing:
        available = sorted(subjects_by_id)
        preview = ", ".join(available[:20]) or "<none>"
        if len(available) > 20:
            preview += f", ... ({len(available)} total)"
        raise SystemExit(
            "Unknown dataset subject ID(s): "
            + ", ".join(missing)
            + f". Available: {preview}"
        )

    selected = []
    seen = set()
    for subject_id in subject_ids:
        if subject_id not in seen:
            selected.append(subjects_by_id[subject_id])
            seen.add(subject_id)
    dataset.subjects = selected


def prepare_method(
    spec: MethodSpec,
    *,
    dataset_kind: str,
    dataset_path: Path | None,
    output_dir: Path | None,
    required_keys_value: str | None,
    extra_args: list[str],
    training_config: dict | None = None,
    subject_ids: tuple[str, ...] = (),
    devices: tuple[str, ...] = (),
    verbose_subject_output: bool = False,
    warn_missing: bool = True,
    persist_paths: bool = True,
) -> PreparedMethod:
    required_keys = (
        parse_required_keys(required_keys_value)
        if required_keys_value is not None
        else spec.required_keys
    )
    dataset = load_dataset(
        dataset_kind,
        path=dataset_path,
        interactive=False,
        persist_path=persist_paths,
        required_keys=required_keys,
        warn_missing=False,
    )
    _select_subjects(dataset, subject_ids)
    if training_config is not None:
        if dataset.kind != training_config["dataset_kind"]:
            raise SystemExit("Dataset kind conflicts with the shared run config")
        actual_subjects = [str(subject["subject_id"]) for subject in dataset.subjects]
        if actual_subjects != training_config["subjects"]:
            raise SystemExit("Selected subjects conflict with the shared run config")
        for subject in dataset.subjects:
            subject["training_config"] = dict(training_config["train"])
            subject["evaluation_config"] = training_config.get("evaluation", {})
            if dataset.kind == "thuman":
                subject["training_selection"] = validate_thuman_inputs(
                    subject, training_config["train"]
                )
                for selection in subject["evaluation_config"].values():
                    validate_thuman_inputs(subject, selection)
            elif dataset.kind == "actorshq":
                validate_actorshq_selection(subject, training_config["train"])
                for selection in subject["evaluation_config"].values():
                    validate_actorshq_selection(subject, selection)
            elif dataset.kind == "dancenet3d":
                validate_dancenet3d_inputs(subject, training_config["train"])
                for selection in subject["evaluation_config"].values():
                    validate_dancenet3d_inputs(subject, selection)
        if dataset.kind == "dancenet3d":
            validate_dancenet3d_camera_identity(dataset.subjects, training_config)
    if warn_missing:
        dataset.warn_missing()
    method_out = resolve_output_path(dataset_kind, output_dir, persist=persist_paths) / (spec.output_subdir or spec.name)
    manifest_path = dataset_manifest(
        dataset,
        path=method_out / "dataset_manifest.json",
        required_keys=required_keys,
    )
    if training_config is not None:
        resolved = {
            **training_config,
            "dataset_kind": dataset.kind,
            "dataset_path": str(dataset.root),
            "subjects": [str(subject["subject_id"]) for subject in dataset.subjects],
        }
        for filename in ("run_config.json", "dataset_config.json"):
            (method_out / filename).write_text(
                json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
            )
    command = render_command(
        spec,
        manifest_path=manifest_path,
        output_dir=method_out,
        extra_args=extra_args,
    )
    if devices:
        command = [
            sys.executable, "-m", "avatar_creation.methods.subject_scheduler",
            "--manifest", str(manifest_path),
            "--devices", ",".join(devices),
            "--log-dir", str(method_out / "subject_logs"),
            *(["--verbose"] if verbose_subject_output else []),
            "--", *command,
        ]
    summary = build_run_summary(spec.name, dataset, extra_args, devices)
    (method_out / "run_summary.log").write_text(summary, encoding="utf-8")
    return PreparedMethod(
        spec=spec,
        command=command,
        output_dir=method_out,
        cwd=spec.cwd,
        env=method_environment(spec),
        summary=summary,
    )


def run_method(
    spec: MethodSpec,
    *,
    dataset_kind: str,
    dataset_path: Path | None,
    output_dir: Path | None,
    registry_path: Path | None,
    required_keys_value: str | None,
    extra_args: list[str],
    training_config: dict | None = None,
    subject_ids: tuple[str, ...] = (),
    devices: tuple[str, ...] = (),
    verbose_subject_output: bool = False,
    dry_run: bool = False,
) -> Path:
    del registry_path
    prepared = prepare_method(
        spec,
        dataset_kind=dataset_kind,
        dataset_path=dataset_path,
        output_dir=output_dir,
        required_keys_value=required_keys_value,
        extra_args=extra_args,
        training_config=training_config,
        subject_ids=subject_ids,
        devices=devices,
        verbose_subject_output=verbose_subject_output,
        persist_paths=not dry_run,
    )
    print(prepared.summary, end="", flush=True)
    print("Method command:")
    print("  " + " ".join(prepared.command))
    if dry_run:
        return prepared.output_dir
    prepared.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = prepared.output_dir / "runner.log"
    with log_path.open(
        "w", encoding="utf-8", errors="replace", newline=""
    ) as log_file:
        log_file.write(prepared.summary)
        log_file.flush()
        process = subprocess.Popen(
            prepared.command,
            cwd=prepared.cwd,
            env=prepared.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := process.stdout.read(4096):
            output = decoder.decode(chunk)
            sys.stdout.write(output)
            sys.stdout.flush()
            log_file.write(output)
            log_file.flush()
        output = decoder.decode(b"", final=True)
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()
            log_file.write(output)
            log_file.flush()
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        print(f"Method failed. Full log: {log_path}", file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(return_code, prepared.command)
    return prepared.output_dir
