"""Entrypoint alias for launching one or more registered methods."""

from __future__ import annotations

from avatar_creation.methods.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
