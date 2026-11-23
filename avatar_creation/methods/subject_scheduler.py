"""Run one manifest subject per GPU without modifying external methods."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue


_SNAPSHOT_SUFFIXES = {
    "", ".c", ".cc", ".cfg", ".cmake", ".conf", ".cpp", ".cu", ".cuh",
    ".h", ".hpp", ".ini", ".json", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
_SNAPSHOT_IGNORED_DIRS = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}


@dataclass(frozen=True)
class _RunSnapshot:
    snapshot_root: Path
    cwd: Path
    command: list[str]
    env: dict[str, str]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _copy_snapshot_tree(source: Path, target: Path, active_cwd: Path) -> None:
    """Copy mutable source/config files and link bulky runtime assets."""
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.name in _SNAPSHOT_IGNORED_DIRS:
            continue
        if item.is_symlink():
            destination.symlink_to(item.readlink(), target_is_directory=item.is_dir())
        elif item.is_dir():
            # Only the active external checkout needs a deep snapshot. Other
            # checkouts can contain many GB of unrelated models and assets.
            if item.parent.name == "external" and not (
                _is_relative_to(active_cwd, item) or _is_relative_to(item, active_cwd)
            ):
                destination.symlink_to(item, target_is_directory=True)
            else:
                _copy_snapshot_tree(item, destination, active_cwd)
        elif item.suffix.lower() in _SNAPSHOT_SUFFIXES:
            shutil.copy2(item, destination)
        else:
            destination.symlink_to(item)


def _rewrite_snapshot_path(value: str, source_root: Path, snapshot_root: Path) -> str:
    prefix, separator, raw_path = value.partition("=")
    candidate = Path(raw_path if separator else value).expanduser()
    if (
        not candidate.is_absolute()
        or not _is_relative_to(candidate, source_root)
        or not candidate.is_file()
        or candidate.suffix.lower() not in _SNAPSHOT_SUFFIXES
    ):
        return value
    rewritten = snapshot_root / candidate.relative_to(source_root)
    return f"{prefix}={rewritten}" if separator else str(rewritten)


def _prepare_run_snapshot(
    *,
    command: list[str],
    manifest_path: Path,
    temporary_root: Path,
    base_env: dict[str, str],
) -> _RunSnapshot | None:
    configured_root = base_env.get("VCI_DATASET_REPO")
    if not configured_root:
        return None
    source_root = Path(configured_root).expanduser().resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"VCI_DATASET_REPO is not a directory: {source_root}")
    active_cwd = Path.cwd().resolve()
    snapshot_root = temporary_root / "repo"
    _copy_snapshot_tree(source_root, snapshot_root, active_cwd)
    snapshot_cwd = (
        snapshot_root / active_cwd.relative_to(source_root)
        if _is_relative_to(active_cwd, source_root)
        else active_cwd
    )
    snapshot_command = [
        token
        if token == str(manifest_path)
        else _rewrite_snapshot_path(token, source_root, snapshot_root)
        for token in command
    ]
    snapshot_env = dict(base_env)
    snapshot_env["VCI_DATASET_REPO"] = str(snapshot_root)
    snapshot_env["VCI_RUN_SNAPSHOT"] = str(snapshot_root)
    pythonpath = snapshot_env.get("PYTHONPATH")
    if pythonpath:
        rewritten_entries = []
        for entry in pythonpath.split(os.pathsep):
            if not entry:
                continue
            resolved = Path(entry).expanduser().resolve()
            rewritten_entries.append(
                str(snapshot_root / resolved.relative_to(source_root))
                if _is_relative_to(resolved, source_root)
                else entry
            )
        snapshot_env["PYTHONPATH"] = os.pathsep.join(rewritten_entries)
    return _RunSnapshot(
        snapshot_root=snapshot_root,
        cwd=snapshot_cwd,
        command=snapshot_command,
        env=snapshot_env,
    )


def _safe_name(value: object) -> str:
    rendered = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("._")
    return rendered or "subject"


def _visible_device(value: str) -> str:
    normalized = value.strip().removeprefix("cuda:")
    if not normalized:
        raise SystemExit("Device entries must not be empty")
    return normalized


def _progress_summary(device: str, subject_id: str, line: str) -> str:
    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line)
    percent_match = re.search(r"(\d+)%", plain)
    metrics_match = re.search(
        r"Loss:\s*(\S+)(?:\s+(\S+)\s+(\S+))?", plain
    )
    percent = min(100, int(percent_match.group(1))) if percent_match else 0
    loss = metrics_match.group(1) if metrics_match else "-"
    eta = metrics_match.group(3) if metrics_match and metrics_match.group(3) else "-"
    bar_width = 8
    filled = round(percent * bar_width / 100)
    bar = "━" * filled + "─" * (bar_width - filled)
    return (
        f"G{device[:4]:<4} {subject_id[:20]:<20} "
        f"{bar} {percent:3d}% L={loss} ETA={eta}"
    )


def _write_subject_manifests(manifest_path: Path, log_dir: Path) -> list[tuple[str, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise SystemExit(f"Dataset manifest contains no subjects: {manifest_path}")

    manifest_dir = log_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shards: list[tuple[str, Path]] = []
    for index, subject in enumerate(subjects):
        if not isinstance(subject, dict):
            raise SystemExit(f"Invalid subject entry {index} in {manifest_path}")
        subject_id = str(subject.get("subject_id") or index)
        shard_path = manifest_dir / f"{index:04d}_{_safe_name(subject_id)}.json"
        shard = dict(manifest)
        shard["subjects"] = [subject]
        shard_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
        shards.append((subject_id, shard_path))
    return shards


def run_subjects(
    *,
    manifest_path: Path,
    devices: list[str],
    command: list[str],
    log_dir: Path,
    verbose: bool = False,
) -> int:
    """Run command once per subject, exposing exactly one requested GPU each time."""
    if not devices:
        raise SystemExit("--devices must contain at least one device")
    if not command:
        raise SystemExit("No method command was provided after --")

    manifest_path = manifest_path.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    snapshot_temporary = tempfile.TemporaryDirectory(prefix="vci-method-run-")
    snapshot = _prepare_run_snapshot(
        command=command,
        manifest_path=manifest_path,
        temporary_root=Path(snapshot_temporary.name),
        base_env=base_env,
    )
    if snapshot is not None:
        command = snapshot.command
        base_env = snapshot.env
    shards = _write_subject_manifests(manifest_path, log_dir)
    slots: Queue[tuple[int, str]] = Queue()
    for index, device in enumerate(devices):
        slots.put((index, _visible_device(device)))
    output_lock = threading.Lock()
    slot_lines = {
        index: f"[GPU {_visible_device(device)}] waiting"
        for index, device in enumerate(devices)
    }
    dashboard_initialized = False

    def render_dashboard_locked() -> None:
        nonlocal dashboard_initialized
        if verbose:
            return
        if dashboard_initialized:
            sys.stdout.write(f"\x1b[{len(devices)}F")
        for index in range(len(devices)):
            sys.stdout.write(f"\x1b[2K{slot_lines[index]}\n")
        sys.stdout.flush()
        dashboard_initialized = True

    def update_slot(slot_index: int, message: str) -> None:
        with output_lock:
            slot_lines[slot_index] = message
            render_dashboard_locked()

    def announce(message: str, *, error: bool = False) -> None:
        with output_lock:
            print(message, file=sys.stderr if error else sys.stdout, flush=True)

    def run_one(item: tuple[str, Path]) -> tuple[str, int, Path]:
        subject_id, shard_path = item
        slot_index, device = slots.get()
        log_path = log_dir / f"{shard_path.stem}.log"
        try:
            subject_command = [
                str(shard_path) if token == str(manifest_path) else token
                for token in command
            ]
            if subject_command == command:
                raise RuntimeError(
                    f"Method command does not contain manifest path {manifest_path}"
                )
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["VCI_DEVICE_SLOT_INDEX"] = str(slot_index)
            env["VCI_DEVICE_SLOT_COUNT"] = str(len(devices))
            env["VCI_SUBJECT_RUN_ID"] = _safe_name(subject_id)
            if verbose:
                announce(f"[start] {subject_id} on GPU {device} (log: {log_path})")
            else:
                update_slot(
                    slot_index,
                    f"G{device[:4]:<4} {subject_id[:24]:<24} preparing",
                )
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                process = subprocess.Popen(
                    subject_command,
                    cwd=snapshot.cwd if snapshot is not None else None,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                if not verbose:
                    update_slot(
                        slot_index,
                        f"G{device[:4]:<4} {subject_id[:24]:<24} running",
                    )
                assert process.stdout is not None
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    if verbose:
                        with output_lock:
                            sys.stdout.write(f"[{subject_id}] {line}")
                            sys.stdout.flush()
                    elif "Optimization" in line:
                        update_slot(
                            slot_index,
                            _progress_summary(device, subject_id, line),
                        )
                process.stdout.close()
                return_code = process.wait()
            if return_code == 0:
                update_slot(slot_index, f"G{device[:4]:<4} {subject_id[:24]:<24} done")
            else:
                update_slot(
                    slot_index,
                    f"G{device[:4]:<4} {subject_id[:24]:<24} "
                    f"failed {return_code} (see log)",
                )
            return subject_id, return_code, log_path
        finally:
            slots.put((slot_index, device))

    announce(
        f"Parallel subject processing enabled for {len(shards)} subject(s) "
        f"on {len(devices)} GPU(s): {', '.join(_visible_device(item) for item in devices)}"
    )
    if not verbose:
        with output_lock:
            render_dashboard_locked()
    failures: list[tuple[str, int, Path]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {pool.submit(run_one, item): item[0] for item in shards}
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                completed_id, return_code, log_path = future.result()
            except BaseException as exc:
                announce(f"[failed] {subject_id}: {exc}", error=True)
                failures.append((subject_id, 1, log_dir))
                continue
            if verbose:
                announce(
                    f"[done]  {completed_id}"
                    if return_code == 0
                    else f"[failed] {completed_id}: exit {return_code} (log: {log_path})",
                    error=return_code != 0,
                )
            if return_code != 0:
                failures.append((completed_id, return_code, log_path))

    if failures:
        announce("One or more subject runs failed:", error=True)
        for subject_id, return_code, log_path in failures:
            announce(
                f"  - {subject_id}: exit {return_code} (log: {log_path})",
                error=True,
            )
        snapshot_temporary.cleanup()
        return 1
    announce("All subject runs complete.")
    snapshot_temporary.cleanup()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream complete prefixed subject output in addition to writing logs.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    return run_subjects(
        manifest_path=args.manifest,
        devices=devices,
        command=args.command,
        log_dir=args.log_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
