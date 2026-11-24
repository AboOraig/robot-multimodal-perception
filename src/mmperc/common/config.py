"""YAML configuration loading, with no machine-specific paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root, resolved relative to this file (src/mmperc/common/config.py)."""
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file. Relative paths are resolved against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p} must contain a YAML mapping at top level")
    return cfg