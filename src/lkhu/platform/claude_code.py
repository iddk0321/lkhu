"""Claude Code (CLI) MCP registration/unregistration — uses the supported ``claude mcp`` CLI.

⚠️ ``~/.claude.json`` is a file that Claude Code manages directly, so writing to it externally
gets clobbered when the session overwrites it with its own state. Therefore you must register via
the ``claude mcp add/remove`` CLI for it to persist (Claude Desktop is a separate app, so writing
its config file directly is OK — see ``mcp_config.py``).

If the ``claude`` CLI is not present (i.e. a Claude Desktop-only user), it is silently skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

__all__ = ["claude_cli", "is_available", "register", "unregister", "is_registered"]

SERVER_NAME = "lkhu"

# Runner injectable in tests (default subprocess.run)
Runner = Callable[..., subprocess.CompletedProcess]


def claude_cli() -> str | None:
    """Path to the ``claude`` executable (None if not found)."""
    return shutil.which("claude")


def is_available() -> bool:
    """Whether the Claude Code CLI is installed."""
    return claude_cli() is not None


def _run(runner: Runner, *args: str) -> subprocess.CompletedProcess:
    cli = claude_cli()
    return runner([cli, *args], capture_output=True, text=True)


def register(command: str, args: list[str], runner: Runner = subprocess.run) -> bool:
    """Register the lkhu MCP server in Claude Code (user scope, idempotent).

    Args:
        command: Command to run (an absolute interpreter path is recommended for portability).
        args: Arguments (e.g. ``["-m", "lkhu", "serve"]``).
        runner: Command runner (for test injection).

    Returns:
        Whether registration succeeded (False if the claude CLI is absent).
    """
    if not is_available():
        return False
    # Idempotent: remove any existing entry then add (remove is ignored if already absent)
    _run(runner, "mcp", "remove", SERVER_NAME, "--scope", "user")
    result = _run(runner, "mcp", "add", SERVER_NAME, "--scope", "user", "--", command, *args)
    return result.returncode == 0


def unregister(runner: Runner = subprocess.run) -> bool:
    """Remove the lkhu server from Claude Code."""
    if not is_available():
        return False
    return _run(runner, "mcp", "remove", SERVER_NAME, "--scope", "user").returncode == 0


def is_registered(runner: Runner = subprocess.run) -> bool:
    """Whether lkhu is registered in Claude Code."""
    if not is_available():
        return False
    return _run(runner, "mcp", "get", SERVER_NAME).returncode == 0
