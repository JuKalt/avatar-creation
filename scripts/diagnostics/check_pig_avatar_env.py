#!/usr/bin/env python3
"""Run upstream PiG checks with assets used by the processed-dataset adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PREPROCESSING_ASSETS = {"models/V02_05", "models/smplx_parts_segm.pkl"}


def main() -> int:
    checker_path = (
        Path(__file__).resolve().parents[2]
        / "external/neu-pig-smplx/scripts/check_env.py"
    )
    spec = importlib.util.spec_from_file_location("vci_pig_environment_check", checker_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load PiG environment checker: {checker_path}")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.ASSET_PATHS = {
        label: path for label, path in checker.ASSET_PATHS.items()
        if path not in PREPROCESSING_ASSETS
    }
    return checker.main()


if __name__ == "__main__":
    raise SystemExit(main())
