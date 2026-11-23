#!/usr/bin/env python3
"""Compatibility entry point for the PiG-Avatar checkpoint evaluator."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("eval_pig_avatar.py")), run_name="__main__")
