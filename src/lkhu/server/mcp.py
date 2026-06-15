"""FastMCP stdio server entry point. Design doc §4, §8.

Invoked by ``lkhu serve``. After confirming the daemon is up (launching it if not), it serves the
tools wired to the daemon client over stdio. Since the data (SQLite/FAISS) is owned exclusively by
the daemon, consistency is guaranteed.
"""

from __future__ import annotations

from lkhu.server.client import LkhuClient, ensure_daemon_running
from lkhu.server.tools import build_server

__all__ = ["run"]


def run() -> None:
    """Confirm/launch the daemon, then bring up the MCP stdio server."""
    ensure_daemon_running()
    client = LkhuClient()
    mcp = build_server(client)
    mcp.run()  # stdio (default)
