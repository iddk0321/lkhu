"""Config loader — merges package defaults + user config. Design doc §14.1.

Deep-merges the user ``config.yaml`` on top of ``defaults.yaml``.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_defaults", "load_config", "deep_merge"]


def load_defaults() -> dict[str, Any]:
    """Read the default config bundled with the package."""
    text = importlib.resources.files("lkhu.config").joinpath("defaults.yaml").read_text("utf-8")
    return yaml.safe_load(text)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries (override wins). The original is preserved."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(user_config_path: str | Path | None = None) -> dict[str, Any]:
    """Merge user config onto the default config and return it.

    Args:
        user_config_path: Path to the user ``config.yaml`` (defaults only when absent or
            missing).

    Returns:
        The merged config dictionary.
    """
    config = load_defaults()
    if user_config_path:
        path = Path(user_config_path)
        if path.exists():
            user = yaml.safe_load(path.read_text("utf-8")) or {}
            config = deep_merge(config, user)
    return config
