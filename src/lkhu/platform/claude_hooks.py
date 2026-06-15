"""Register/unregister Claude Code hooks in settings.json. Design §8.2 (auto-invocation policy).

For automatic recall/save, hooks ``lkhu hook <event>`` onto SessionStart / UserPromptSubmit / Stop.
Existing hooks are preserved; only the entries added by lkhu are updated/removed (idempotent).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["settings_path", "install_hooks", "uninstall_hooks", "hooks_installed", "default_prefix"]

# Marker identifying lkhu hooks (used for idempotent update/removal)
_MARKER = "lkhu hook"

# (Claude Code event, lkhu event, matcher, timeout)
_HOOK_SPECS = [
    ("SessionStart", "session-start", "startup|clear|compact", 60),
    ("UserPromptSubmit", "user-prompt", None, 30),
    ("Stop", "stop", None, 30),
]


def settings_path() -> Path:
    """Path to the Claude Code user settings.json (overridable via ``LKHU_CC_SETTINGS``)."""
    override = os.environ.get("LKHU_CC_SETTINGS")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "settings.json"


def default_prefix() -> str:
    """The lkhu invocation prefix the hooks run (uses the current interpreter, PATH-independent)."""
    return f'"{sys.executable}" -m lkhu'


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_lkhu(hooks: dict[str, Any]) -> None:
    """Remove only the hook groups registered by lkhu (other hooks are preserved)."""
    for event, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        kept = []
        for group in groups:
            cmds = " ".join(
                h.get("command", "") for h in group.get("hooks", []) if isinstance(h, dict)
            )
            if _MARKER not in cmds:
                kept.append(group)
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]


def install_hooks(
    config_path: str | Path | None = None,
    prefix: str | None = None,
) -> Path:
    """Register the lkhu hooks (updating existing lkhu entries, preserving other hooks).

    Hooks all three: SessionStart (auto recall) / UserPromptSubmit (recall+save) / Stop (save).

    Args:
        config_path: Path to settings.json (default user settings).
        prefix: Hook command prefix (default the current interpreter ``-m lkhu``).

    Returns:
        Path to the modified settings.json.
    """
    path = Path(config_path) if config_path else settings_path()
    prefix = prefix or default_prefix()
    data = _load(path)
    hooks = data.setdefault("hooks", {})
    _strip_lkhu(hooks)  # Idempotent: remove existing lkhu entries then add anew

    for event, lkhu_event, matcher, timeout in _HOOK_SPECS:
        hook_entry = {
            "type": "command",
            "command": f"{prefix} hook {lkhu_event}",
            "timeout": timeout,
        }
        group: dict[str, Any] = {"hooks": [hook_entry]}
        if matcher:
            group["matcher"] = matcher
        hooks.setdefault(event, []).append(group)

    _save(path, data)
    return path


def uninstall_hooks(config_path: str | Path | None = None) -> bool:
    """Remove the lkhu hooks (other hooks/settings are preserved).

    Returns:
        True if it saved after attempting removal.
    """
    path = Path(config_path) if config_path else settings_path()
    data = _load(path)
    if "hooks" not in data:
        return False
    _strip_lkhu(data["hooks"])
    if not data["hooks"]:
        del data["hooks"]
    _save(path, data)
    return True


def hooks_installed(config_path: str | Path | None = None) -> bool:
    """Whether the lkhu hooks are installed."""
    path = Path(config_path) if config_path else settings_path()
    text = json.dumps(_load(path))
    return _MARKER in text
