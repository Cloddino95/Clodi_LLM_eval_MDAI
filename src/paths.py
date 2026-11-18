"""Stable project and configuration-relative path handling."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a CLI path independently of the caller's working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_config_path(value: str | Path, config_file: str | Path) -> Path:
    """Resolve a path relative to the YAML file that declared it."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_file).resolve().parent / path
    return path.resolve()

