"""YAML config loading with `include:` support and placeholder-path checks."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER_PREFIX = "/path/to/"


def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Files listed under `include:` (relative to this file) are
    loaded first and this file is merged on top of them."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged: dict[str, Any] = {}
    for inc in cfg.pop("include", []) or []:
        merged = deep_merge(merged, load_config(path.parent / inc))
    return deep_merge(merged, cfg)


def is_placeholder(path: str | Path | None) -> bool:
    return path is None or str(path).startswith(PLACEHOLDER_PREFIX)


def require_path(path: str | Path | None, what: str) -> Path:
    """Fail early, with a pointer to configs/paths.yaml, on placeholder or missing paths."""
    if is_placeholder(path):
        raise FileNotFoundError(
            f"{what} is still the placeholder {path!r}; set it in configs/paths.yaml"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}")
    return path


def output_dir(cfg: dict, subdir: str) -> Path:
    out = Path(cfg.get("output_dir", "outputs")) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out
