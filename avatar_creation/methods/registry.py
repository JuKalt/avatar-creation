"""Method registry loading.

Method specs are intentionally plain JSON so method code can live outside this
repository and in its own environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("methods.json")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    command: list[str]
    conda_env: str | None = None
    python: str | None = None
    cwd: Path | None = None
    env: dict[str, str] | None = None
    description: str = ""
    evaluation_scripts: dict[str, str] | None = None
    required_keys: tuple[str, ...] = ()
    setup_command: str | None = None
    environment_check: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()
    required_any_files: tuple[tuple[str, ...], ...] = ()
    output_subdir: str | None = None

    @property
    def runner_prefix(self) -> list[str]:
        if self.conda_env and self.python:
            raise SystemExit(
                f"Method {self.name!r} sets both conda_env and python; choose one."
            )
        if self.conda_env:
            return ["conda", "run", "--no-capture-output", "-n", self.conda_env]
        if self.python:
            return [self.python]
        return []


def _required_path(spec: MethodSpec, raw: str) -> Path | None:
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if "$" in expanded:
        return None
    path = Path(expanded)
    if not path.is_absolute():
        path = (spec.cwd or PROJECT_ROOT) / path
    return path.resolve()


def required_file_errors(spec: MethodSpec) -> list[str]:
    reasons: list[str] = []
    if spec.cwd is not None and not spec.cwd.is_dir():
        reasons.append(f"method checkout is missing: {spec.cwd}")
    for raw in spec.required_files:
        path = _required_path(spec, raw)
        if path is None:
            reasons.append(f"required file uses an unset environment variable: {raw}")
        elif not path.is_file():
            reasons.append(f"required file is missing: {path}")
    for group in spec.required_any_files:
        candidates = [path for raw in group if (path := _required_path(spec, raw))]
        if not any(path.is_file() for path in candidates):
            rendered = ", ".join(str(path) for path in candidates) or ", ".join(group)
            reasons.append(f"none of the alternative required files exists: {rendered}")
    return reasons

def environment_errors(specs: list[MethodSpec]) -> list[str]:
    conda_specs = [spec for spec in specs if spec.conda_env]
    conda_envs: set[str] = set()
    conda_error: str | None = None
    if conda_specs:
        if shutil.which("conda") is None:
            conda_error = "conda executable was not found on PATH"
        else:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                conda_error = result.stderr.strip() or "conda env list failed"
            else:
                try:
                    env_paths = json.loads(result.stdout).get("envs", [])
                    conda_envs = {Path(path).name for path in env_paths}
                except (json.JSONDecodeError, AttributeError):
                    conda_error = "conda env list returned invalid JSON"

    errors: list[str] = []
    for spec in specs:
        reasons = required_file_errors(spec)
        cwd_ready = spec.cwd is None or spec.cwd.is_dir()

        if spec.conda_env and conda_error:
            reasons.append(conda_error)
        elif spec.conda_env and spec.conda_env not in conda_envs:
            reasons.append(f"Conda environment {spec.conda_env!r} is not installed")
        elif spec.python and shutil.which(spec.python) is None and not Path(spec.python).is_file():
            reasons.append(f"Python executable {spec.python!r} was not found")
        elif spec.environment_check and cwd_ready:
            result = subprocess.run(
                [*spec.runner_prefix, *spec.environment_check],
                cwd=spec.cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stdout.strip() or result.stderr.strip()
                reason = "environment validation failed"
                if detail:
                    reason += f"\n  {detail.replace(chr(10), chr(10) + chr(32) * 2)}"
                reasons.append(reason)
        if reasons:
            setup = spec.setup_command or "No setup command is registered for this method."
            detail = "\n  ".join(reasons)
            errors.append(f"{spec.name}: {detail}\n  Setup: {setup}")
    return errors

def validate_method_environments(specs: list[MethodSpec]) -> None:
    errors = environment_errors(specs)
    if errors:
        raise SystemExit("Method environment preflight failed:\n\n" + "\n\n".join(errors))


def registry_path(path: Path | None = None) -> Path:
    raw = path or os.environ.get("VCI_METHOD_REGISTRY")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_REGISTRY_PATH


def _load_registry(path: Path | None = None) -> dict[str, Any]:
    resolved = registry_path(path)
    if not resolved.is_file():
        raise SystemExit(f"Method registry not found: {resolved}")
    data = json.loads(resolved.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Method registry root must be an object: {resolved}")
    return data


def discover_methods(path: Path | None = None) -> dict[str, MethodSpec]:
    data = _load_registry(path)
    methods = data.get("methods", {})
    if not isinstance(methods, dict):
        raise SystemExit("Method registry key 'methods' must be an object.")
    return {name: _parse_method(name, spec) for name, spec in methods.items()}


def load_method_spec(name: str, path: Path | None = None) -> MethodSpec:
    methods = discover_methods(path)
    # Keep explicit entries in custom registries authoritative.
    if name == "neu-pig-smplx" and name not in methods and "pig-avatar" in methods:
        return methods["pig-avatar"]
    if name not in methods:
        known = ", ".join(sorted(methods)) or "<none>"
        raise SystemExit(f"Unknown method {name!r}. Available methods: {known}")
    return methods[name]


def _parse_method(name: str, spec: Any) -> MethodSpec:
    if not isinstance(spec, dict):
        raise SystemExit(f"Method {name!r} spec must be an object.")
    command = spec.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise SystemExit(f"Method {name!r} must define command as a string list.")
    cwd_raw = spec.get("cwd")
    cwd_path = Path(cwd_raw).expanduser() if cwd_raw else None
    cwd = (
        (cwd_path if cwd_path.is_absolute() else PROJECT_ROOT / cwd_path).resolve()
        if cwd_path is not None
        else None
    )
    env = spec.get("env") or None
    if env is not None and (
        not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
    ):
        raise SystemExit(f"Method {name!r} env must be an object of strings.")
    setup_command = spec.get("setup_command")
    if setup_command is not None and not isinstance(setup_command, str):
        raise SystemExit(f"Method {name!r} setup_command must be a string.")
    evaluation_scripts = spec.get("evaluation_scripts", {})
    if (not isinstance(evaluation_scripts, dict)
            or set(evaluation_scripts) - {"novel_view", "novel_pose"}
            or not all(isinstance(value, str) and value.strip() for value in evaluation_scripts.values())):
        raise SystemExit(
            f"Method {name!r} evaluation_scripts must map novel_view/novel_pose to non-empty strings."
        )
    required_files = spec.get("required_files", [])
    if not isinstance(required_files, list) or not all(isinstance(item, str) for item in required_files):
        raise SystemExit(f"Method {name!r} required_files must be a string list.")
    required_any_files = spec.get("required_any_files", [])
    if not isinstance(required_any_files, list) or not all(
        isinstance(group, list) and group and all(isinstance(item, str) for item in group)
        for group in required_any_files
    ):
        raise SystemExit(f"Method {name!r} required_any_files must be a list of non-empty string lists.")
    environment_check = spec.get("environment_check", [])
    if not isinstance(environment_check, list) or not all(isinstance(item, str) for item in environment_check):
        raise SystemExit(f"Method {name!r} environment_check must be a string list.")
    required_keys = spec.get("required_keys", [])
    if not isinstance(required_keys, list) or not all(
        isinstance(key, str) and key.strip() for key in required_keys
    ):
        raise SystemExit(f"Method {name!r} required_keys must be a string list.")
    output_subdir = spec.get("output_subdir")
    if output_subdir is not None and (
        not isinstance(output_subdir, str) or output_subdir in ("", ".", "..")
        or Path(output_subdir).name != output_subdir
    ):
        raise SystemExit(f"Method {name!r} output_subdir must be one directory name.")
    return MethodSpec(
        name=name,
        output_subdir=output_subdir,
        command=command,
        conda_env=spec.get("conda_env"),
        python=spec.get("python"),
        cwd=cwd,
        env=env,
        description=str(spec.get("description", "")),
        evaluation_scripts=evaluation_scripts,
        required_keys=tuple(required_keys),
        setup_command=setup_command,
        environment_check=tuple(environment_check),
        required_files=tuple(required_files),
        required_any_files=tuple(tuple(group) for group in required_any_files),
    )
