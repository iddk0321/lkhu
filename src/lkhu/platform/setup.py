"""Install/uninstall orchestration.

Claude Code integration is done via a **plugin** (repo-root marketplace) — directly editing
``~/.claude.json``/``settings.json`` is unreliable because Claude Code overwrites and discards it
(clobbering). Plugins are managed by a separate registry, so they are stable.

Therefore ``lkhu install`` only does (1) codebook creation and (2) **Claude Desktop** MCP
registration (Claude Desktop is a separate app that does not support plugins, and writing its
config file directly does persist).
Portability principle: build the command from the current interpreter (``sys.executable``)
without hardcoding paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from lkhu.platform import claude_code, claude_hooks, mcp_config

__all__ = ["portable_command", "install", "uninstall"]


def portable_command() -> tuple[str, list[str]]:
    """The portable (command, args) that launches the MCP server.

    Returns:
        ``(sys.executable, ["-m", "lkhu", "serve"])``.
    """
    return sys.executable, ["-m", "lkhu", "serve"]


def install(base: str | Path | None = None, register_desktop: bool = True) -> dict[str, Any]:
    """Create the codebook and register the Claude Desktop MCP (idempotent).

    Claude Code integration (auto-memory hooks + MCP) is installed as a plugin:
    ``claude plugin marketplace add <repo> && claude plugin install lkhu@lkhu``.

    Args:
        base: Data root override (for testing).
        register_desktop: Whether to register the Claude Desktop MCP.

    Returns:
        Summary of what was performed.
    """
    from lkhu.core.engine import initialize

    info = initialize(base=base, register_mcp=False)  # codebook/directories only
    command, args = portable_command()

    desktop_path = None
    if register_desktop:
        desktop_path = str(mcp_config.register(command=command, args=args))

    return {
        "data_dir": info["data_dir"],
        "codebook_created": info["codebook_created"],
        "desktop_mcp": desktop_path,
        "command": f"{command} {' '.join(args)}",
    }


def uninstall() -> dict[str, bool]:
    """Unregister the lkhu integration (data/codebook are preserved).

    Removes the Claude Desktop MCP, and also cleans up any legacy leftovers (directly registered
    Claude Code MCP/hooks) that might remain. The plugin itself is removed via
    ``claude plugin uninstall lkhu@lkhu``.

    Returns:
        Whether each item was removed.
    """
    return {
        "desktop_mcp": mcp_config.unregister(),
        "legacy_claude_code_mcp": claude_code.unregister(),
        "legacy_hooks": claude_hooks.uninstall_hooks(),
    }
