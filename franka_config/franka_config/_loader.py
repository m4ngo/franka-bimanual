"""YAML discovery, caching, and dotted-path access for config/."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR = "FRANKA_CONFIG_DIR"
_DIR_NAME = "config"
_SECTIONS = (
    "world", "arms", "cameras", "control", "rig", "teleop", "policy", "calibration",
)

_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}
_dir_cache: Path | None = None


def _search_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [p for p in here.parents]
    roots.append(Path.home() / "franka_ws")
    roots.append(Path.cwd())
    roots.extend(Path.cwd().parents)
    return roots


def config_dir() -> Path:
    """Directory holding the YAML config files.

    Resolution order: $FRANKA_CONFIG_DIR, then the nearest `config/` directory
    walking up from this file (works for editable installs), then ~/franka_ws,
    then upward from the CWD.
    """
    global _dir_cache
    if _dir_cache is not None:
        return _dir_cache

    override = os.environ.get(_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / "world.yaml").is_file():
            raise FileNotFoundError(f"{_ENV_VAR}={path} does not contain world.yaml")
        _dir_cache = path
        return path

    seen: set[Path] = set()
    for root in _search_roots():
        candidate = root / _DIR_NAME
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "world.yaml").is_file():
            _dir_cache = candidate.resolve()
            return _dir_cache

    raise FileNotFoundError(
        "Could not locate the franka_ws config/ directory. "
        f"Set {_ENV_VAR} to point at it."
    )


def repo_root() -> Path:
    """Workspace root — the parent of config/."""
    return config_dir().parent


def section(name: str) -> dict[str, Any]:
    """Parsed contents of config/<name>.yaml (cached)."""
    with _lock:
        if name in _cache:
            return _cache[name]
        path = config_dir() / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"missing config file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")
        _cache[name] = data
        return data


def get(path: str, default: Any = ...) -> Any:
    """Dotted lookup, e.g. get("control.gains.joint_pd.kp").

    The first segment names the YAML file. Raises KeyError on a missing key
    unless a default is given — a silent None here would become a silent
    control-gain change on hardware.
    """
    head, _, rest = path.partition(".")
    node: Any = section(head)
    if not rest:
        return node
    for key in rest.split("."):
        if not isinstance(node, dict) or key not in node:
            if default is not ...:
                return default
            raise KeyError(f"config path not found: {path!r} (missing {key!r})")
        node = node[key]
    return node


def reload() -> None:
    """Drop the cache so the next access re-reads from disk."""
    global _dir_cache
    with _lock:
        _cache.clear()
        _dir_cache = None


def all_sections() -> dict[str, dict[str, Any]]:
    """Every config file, parsed — used for run-metadata snapshots."""
    return {name: section(name) for name in _SECTIONS}
