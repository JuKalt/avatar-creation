"""Adapter for running the external PiG-Avatar method."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from avatar_creation.methods.actorshq_data import (
    stage_actorshq_evaluation_source,
    stage_actorshq_subject,
    staged_actorshq_subject,
)
from avatar_creation.methods.external_data import safe_subject_id

DEFAULT_CPU_THREADS = 16


def _repo_root() -> Path:
    env_root = os.environ.get("VCI_DATASET_REPO")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _device_index(value: str) -> str:
    normalized = value.removeprefix("cuda:")
    if not normalized.isdigit():
        raise SystemExit(
            f"Invalid PiG device {value!r}. Use a CUDA index such as 7 or cuda:7."
        )
    return normalized


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _configure_cuda_build_environment(env: dict[str, str]) -> None:
    """Expose Conda CUDA target paths for gsplat and other JIT extensions."""
    prefix = Path(sys.prefix)
    target = prefix / "targets" / "x86_64-linux"
    include = target / "include"
    library = target / "lib"
    if (include / "cuda_runtime_api.h").is_file():
        env.setdefault("CUDA_HOME", str(prefix))
        for name, path in (
            ("CPATH", include),
            ("CPLUS_INCLUDE_PATH", include),
            ("LIBRARY_PATH", library),
            ("LD_LIBRARY_PATH", library),
        ):
            current = env.get(name)
            env[name] = str(path) + (os.pathsep + current if current else "")
    if "TORCH_CUDA_ARCH_LIST" not in env:
        try:
            import torch

            major, minor = torch.cuda.get_device_capability()
            env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass


def _configure_cpu_environment(
    env: dict[str, str], cpu_threads: int
) -> tuple[int, ...] | None:
    """Cap PiG CPU pools and isolate concurrent GPU workers."""
    if cpu_threads <= 0:
        raise SystemExit("--cpu-threads must be greater than zero")

    rendered = str(cpu_threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = rendered
    env["VCI_PIG_CPU_THREADS"] = rendered

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        return None

    try:
        slot_index = int(os.environ.get("VCI_DEVICE_SLOT_INDEX", "0"))
    except ValueError:
        slot_index = 0
    start = (max(0, slot_index) * cpu_threads) % len(allowed)
    selected = tuple(
        allowed[(start + offset) % len(allowed)]
        for offset in range(min(cpu_threads, len(allowed)))
    )
    os.sched_setaffinity(0, set(selected))
    return selected


def prepare_method_config(args: argparse.Namespace, repo: Path, output_dir: Path) -> Path:
    source = (
        args.method_config.expanduser().resolve()
        if args.method_config is not None
        else repo / "configs" / "method" / "pig_avatar_smplx.json"
    )
    if not source.is_file():
        raise SystemExit(f"PiG method config not found: {source}")
    config = json.loads(source.read_text())
    model_args = config.get("model_args", {}).get("model_args_smplx", {})
    configured = Path(str(model_args.get("model_path", "")))
    configured = configured if configured.is_absolute() else repo / configured
    if configured.is_file():
        model_path = configured.resolve()
    else:
        override = os.environ.get("VCI_SMPLX_MODEL")
        candidates = [
            Path(override).expanduser() if override else None,
            _repo_root() / "external" / "mamma" / "data" / "body_models"
            / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz",
        ]
        model_path = next(
            (path.resolve() for path in candidates if path is not None and path.is_file()),
            None,
        )
    if model_path is None:
        raise SystemExit(
            "PiG requires the licensed SMPLX_NEUTRAL.npz model. Set "
            "VCI_SMPLX_MODEL to its path or install the MAMMA body-model assets."
        )
    model_args["model_path"] = str(model_path)
    subject_run_id = os.environ.get("VCI_SUBJECT_RUN_ID")
    target = (
        output_dir / "subject_configs" / f"{subject_run_id}.json"
        if subject_run_id
        else output_dir / "pig_method_config.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


def build_command(args: argparse.Namespace, manifest_path: Path, out_path: Path) -> list[str]:
    runner = _repo_root() / "avatar_creation" / "methods" / "pig_avatar_run.py"
    cmd = [
        "python",
        str(runner),
        "--dataset-manifest",
        str(manifest_path),
        "--out_path",
        str(out_path),
        "--devices",
        _device_index(args.device),
        "--time_steps",
        str(args.time_steps),
        "--views",
        str(args.views),
        "--resolution",
        str(args.resolution),
    ]
    cmd.extend(["--methodConfig", str(args.method_config)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size_time is not None:
        cmd.extend(["--batch_size_time", str(args.batch_size_time)])
    if args.batch_size_cameras is not None:
        cmd.extend(["--batch_size_cameras", str(args.batch_size_cameras)])
    if args.checkpoint_log_every is not None:
        cmd.extend(["--checkpoint_log_every", str(args.checkpoint_log_every)])
    if args.noise is not None:
        cmd.extend(["--noise", str(args.noise)])
    if args.mhr:
        cmd.append("--mhr")
    if args.forward_labels:
        cmd.append("--vci-forward-labels")
    if not getattr(args, "data_in_memory", True):
        cmd.append("--vci-data-on-disk")
    if args.extra:
        cmd.extend(args.extra)
    if _uses_unified_grid(args.extra):
        # The manifest dataset factory is installed at runtime by
        # pig_avatar_run.py. PyTorch spawn workers import the upstream dataset
        # module afresh and otherwise fall back to its THuman4D factory.
        cmd.extend(["--num_workers", "0"])
    return cmd


def _uses_unified_grid(extra: list[str]) -> bool:
    for index, value in enumerate(extra):
        if value == "--training_mode" and index + 1 < len(extra):
            return extra[index + 1] == "unified_grid"
        if value.startswith("--training_mode="):
            return value.partition("=")[2] == "unified_grid"
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PiG-Avatar with the unified VCI dataset manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, default=None)
    parser.add_argument("--device", default="0", help="CUDA device index, optionally prefixed with cuda:")
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--method-config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-time", type=int, default=None)
    parser.add_argument("--batch-size-cameras", type=int, default=None)
    parser.add_argument("--checkpoint-log-every", type=int, default=None)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="Maximum CPU threads used by this PiG subject worker.",
    )
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--mhr", action="store_true")
    parser.add_argument(
        "--forward-labels",
        action="store_true",
        help="Load 4D-DRESS scan labels and forward them as semantic_labels.",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-in-memory",
        "--data_in_memory",
        dest="data_in_memory",
        type=_boolean,
        nargs="?",
        const=True,
        default=True,
        help="Preload all training images and masks into system memory.",
    )
    data_group.add_argument(
        "--no-data-in-memory",
        "--no_data_in_memory",
        dest="data_in_memory",
        action="store_false",
        help="Read only the sampled training frames from disk.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.extra = extra
    return args


def stage_actorshq_manifest(
    manifest_path: Path, output_dir: Path, args: argparse.Namespace
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = manifest.get("subjects", [])
    if not any(subject.get("dataset") == "actorshq" for subject in subjects):
        return manifest_path
    staged_subjects = []
    for subject in subjects:
        if subject.get("dataset") != "actorshq":
            staged_subjects.append(subject)
            continue
        data_dir = output_dir / "actorshq_staging" / safe_subject_id(subject["subject_id"])
        stage_actorshq_subject(
            subject,
            data_dir,
            views=args.views,
            time_steps=args.time_steps,
            resolution=args.resolution,
        )
        staged_subject = staged_actorshq_subject(subject, data_dir)
        staged_subject["evaluation_source"] = stage_actorshq_evaluation_source(
            subject, output_dir / "actorshq_evaluation_source" / safe_subject_id(subject["subject_id"])
        )
        staged_subjects.append(staged_subject)
    staged_manifest = dict(manifest)
    staged_manifest["subjects"] = staged_subjects
    run_id = os.environ.get("VCI_SUBJECT_RUN_ID")
    target = output_dir / (
        f"actorshq_dataset_manifest_{safe_subject_id(run_id)}.json"
        if run_id
        else "actorshq_dataset_manifest.json"
    )
    target.write_text(json.dumps(staged_manifest, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_path.expanduser().resolve() if args.repo_path else _repo_root() / "external" / "neu-pig-smplx"
    if not (repo / "Main.py").is_file():
        raise SystemExit(
            f"PiG-Avatar checkout not found at {repo}. "
            "Run scripts/setup/setup_pig_avatar_env.sh to install a compatible checkout."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_manifest.expanduser().resolve()
    manifest_path = stage_actorshq_manifest(manifest_path, output_dir, args)
    args.method_config = prepare_method_config(args, repo, output_dir)
    out_path = output_dir / "pig_outputs"
    cmd = build_command(args, manifest_path, out_path)
    print("PiG command:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    project_root = str(_repo_root())
    pythonpath = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and str(Path(entry).expanduser().resolve()) != project_root
    ]
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    else:
        env.pop("PYTHONPATH", None)
    _configure_cuda_build_environment(env)
    assigned_cpus = _configure_cpu_environment(env, args.cpu_threads)
    if assigned_cpus:
        print(
            f"PiG CPU worker cap: {args.cpu_threads} thread(s), "
            f"affinity CPUs {assigned_cpus[0]}-{assigned_cpus[-1]}."
        )
    subprocess.run(cmd, cwd=repo, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
