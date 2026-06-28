"""File helper utilities for experiment I/O."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return the directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    """Create parent directory if needed and return the path object."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


__all__ = ["ensure_dir", "ensure_parent"]
