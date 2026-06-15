"""Automatic registration/unregistration of Claude Code settings. Design doc §8.3.

Registers the lkhu MCP server in the Claude settings file (claude_desktop_config.json)
across 3 OSes. OS-specific path differences are handled only in this platform layer
(hard rule 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["claude_config_path", "register", "unregister", "is_registered"]

SERVER_NAME = "lkhu"


def claude_config_path() -> Path:
    """OS-specific path to the Claude desktop settings file.

    Can be overridden via the ``LKHU_CLAUDE_CONFIG`` environment variable
    (for testing/portability).
    """
    import os

    override = os.environ.get("LKHU_CLAUDE_CONFIG")
    if override:
        return Path(override)
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform.startswith("win"):
        import os

        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return base / "Claude" / "claude_desktop_config.json"
    # Linux and others
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def register(
    config_path: str | Path | None = None,
    command: str = "lkhu",
    args: list[str] | None = None,
) -> Path:
    """Register the lkhu MCP server in the Claude settings (preserving existing servers).

    Args:
        config_path: Path to the settings file (default OS path).
        command: Command to run (default ``lkhu``).
        args: Arguments (default ``["serve"]``).

    Returns:
        Path to the modified settings file.
    """
    path = Path(config_path) if config_path else claude_config_path()
    data = _load(path)
    servers = data.setdefault("mcpServers", {})
    servers[SERVER_NAME] = {"command": command, "args": args if args is not None else ["serve"]}
    _save(path, data)
    return path


def unregister(config_path: str | Path | None = None) -> bool:
    """Remove the lkhu server from the Claude settings (data is preserved).

    Returns:
        True if removed, False if it was not present.
    """
    path = Path(config_path) if config_path else claude_config_path()
    data = _load(path)
    servers = data.get("mcpServers", {})
    if SERVER_NAME in servers:
        del servers[SERVER_NAME]
        _save(path, data)
        return True
    return False


def is_registered(config_path: str | Path | None = None) -> bool:
    """Whether lkhu is registered in the Claude settings."""
    path = Path(config_path) if config_path else claude_config_path()
    return SERVER_NAME in _load(path).get("mcpServers", {})
