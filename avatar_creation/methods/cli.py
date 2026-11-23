"""CLI for running registered methods in their own environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiview_data.loader import AVAILABLE_DATASETS
from avatar_creation.methods.experiment_config import load_experiment_config, training_arguments

from .registry import discover_methods, load_method_spec, validate_method_environments
from .runner import run_method


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a registered method against a processed dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("methods", nargs="*", help="Registered method name(s).")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run every registered method.")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--run-config", "--dataset-config", dest="dataset_config", type=Path,
                        help="Shared JSON subjects and training inputs (--dataset-config is an alias).")
    parser.add_argument("--dataset-kind", default=None, choices=sorted(AVAILABLE_DATASETS))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--required-keys", default=None)
    parser.add_argument(
        "--subjects",
        nargs="+",
        metavar="ID",
        help="Run only canonical dataset subject IDs, e.g. 0166_04.",
    )
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated GPU pool. Runs one selected subject per GPU at a time "
            "without modifying the external method checkout, e.g. 0,1,2,3."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="CPU thread cap per PiG subject worker (default: 16).",
    )
    parser.add_argument(
        "--verbose-subject-output",
        action="store_true",
        help="Stream full parallel subject output; by default it is log-only.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output root; stored per dataset kind when provided.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-tui", action="store_true", help="Use plain streaming terminal output.")
    args, extra = parser.parse_known_args(argv)
    args.training_config = None
    if args.dataset_config is not None:
        config = load_experiment_config(args.dataset_config)
        if args.dataset_kind is not None and args.dataset_kind != config["dataset_kind"]:
            parser.error("--dataset-kind conflicts with --dataset-config")
        args.dataset_kind = config["dataset_kind"]
        args.dataset_path = args.dataset_path or (
            Path(config["dataset_path"]) if "dataset_path" in config else None
        )
        if args.subjects is not None and args.subjects != config["subjects"]:
            parser.error("--subjects conflicts with --run-config; edit subjects in the config")
        args.subjects = config["subjects"]
        args.training_config = {**config, "subjects": args.subjects}
        extra = training_arguments(config, extra[1:] if extra[:1] == ["--"] else extra)
    args.dataset_kind = args.dataset_kind or "processed"
    return args, extra


def main(argv: list[str] | None = None) -> int:
    args, unknown_args = parse_args(argv)
    if args.list_methods:
        for name, spec in sorted(discover_methods(args.registry).items()):
            suffix = f" - {spec.description}" if spec.description else ""
            print(f"{name}{suffix}")
            for key, label in (("novel_view", "Novel view"), ("novel_pose", "Novel pose")):
                script = (spec.evaluation_scripts or {}).get(key, "Not registered")
                print(f"  {label}: {script}")
        return 0
    if args.all:
        specs = list(discover_methods(args.registry).values())
    else:
        if not args.methods:
            raise SystemExit("Provide at least one method name, --all, or --list-methods.")
        # A legacy alias and its canonical name must not schedule the same method twice.
        selected = {}
        for name in args.methods:
            spec = load_method_spec(name, args.registry)
            selected.setdefault(spec.name, spec)
        specs = list(selected.values())
    if args.training_config is not None:
        for spec in specs:
            if spec.name not in ("pig-avatar", "neu-pig-smplx", "mmlphuman", "mmlphuman-lod", "animatable-gaussians", "humangs"):
                raise SystemExit(f"Dataset configs are not supported by method {spec.name!r}")
            if spec.name == "animatable-gaussians" and args.dataset_kind == "dna":
                raise SystemExit("Animatable Gaussians does not support native DNA inputs")
    validate_method_environments(specs)
    extra_args = list(unknown_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            raise SystemExit("--cpu-threads must be greater than zero")
        if any(spec.name not in ("pig-avatar", "neu-pig-smplx") for spec in specs):
            raise SystemExit("--cpu-threads is only supported for pig-avatar")
        extra_args = ["--cpu-threads", str(args.cpu_threads), *extra_args]
    devices = tuple(
        item.strip() for item in (args.devices or "").split(",") if item.strip()
    )
    if args.devices is not None and not devices:
        raise SystemExit("--devices must contain at least one GPU, e.g. --devices 0,1")
    normalized_devices = tuple(item.removeprefix("cuda:") for item in devices)
    if any(not item for item in normalized_devices):
        raise SystemExit("Invalid empty CUDA device in --devices")
    if len(set(normalized_devices)) != len(normalized_devices):
        raise SystemExit("--devices must list each GPU only once")
    if devices and any(
        item == "--device" or item.startswith("--device=") for item in extra_args
    ):
        raise SystemExit(
            "Do not combine runner --devices with a method --device argument. "
            "Each worker sees its assigned GPU as device 0 automatically."
        )
    if devices and any(spec.name in ("mmlphuman", "mmlphuman-lod") for spec in specs) and any(
        item == "--port" or item.startswith("--port=") for item in extra_args
    ):
        raise SystemExit(
            "Do not pass --port during a parallel MMLP Human run; the runner "
            "assigns a distinct visualization port to every active GPU slot."
        )
    if not args.no_tui:
        try:
            from .textual_ui import run_textual_methods
        except ModuleNotFoundError as exc:
            if exc.name == "textual":
                raise SystemExit(
                    "Textual is required for the full-screen runner. "
                    "Run ./install.sh --cli-only from the Avatar Creation checkout "
                    "(using the same VCI_CLI_ENV), or pass --no-tui."
                ) from exc
            raise
        return run_textual_methods(
            specs,
            dataset_kind=args.dataset_kind,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            required_keys_value=args.required_keys,
            extra_args=extra_args,
            training_config=args.training_config,
            subject_ids=tuple(args.subjects or ()),
            devices=devices,
            verbose_subject_output=args.verbose_subject_output,
            dry_run=args.dry_run,
        )
    for spec in specs:
        out = run_method(
            spec,
            dataset_kind=args.dataset_kind,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            registry_path=args.registry,
            required_keys_value=args.required_keys,
            extra_args=extra_args,
            training_config=args.training_config,
            subject_ids=tuple(args.subjects or ()),
            devices=devices,
            verbose_subject_output=args.verbose_subject_output,
            dry_run=args.dry_run,
        )
        print(f"Method output: {out}")
    return 0
