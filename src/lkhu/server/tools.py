"""MCP tool definitions. Design doc §8.1.

The tools operate through the daemon client, so hooks, MCP, and CLI all see the same memory
(owned exclusively by a single daemon).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from lkhu.server.client import LkhuClient

__all__ = ["build_server", "TOOL_NAMES"]

TOOL_NAMES = ["recall", "remember", "forget", "recall_session", "status", "export"]


def build_server(client: LkhuClient, name: str = "lkhu") -> FastMCP:
    """Build a FastMCP server wired to the daemon client.

    Args:
        client: Daemon HTTP client.
        name: Server name.

    Returns:
        A ``FastMCP`` instance with the tools registered.
    """
    from fastmcp import FastMCP

    mcp: FastMCP = FastMCP(name)

    @mcp.tool
    def recall(query: str, k: int = 5) -> dict[str, Any]:
        """Search the top-K relevant memories and decode the synthesized scent into language."""
        return client.recall(query, k=k)

    @mcp.tool
    def remember(content: str, kind: str = "fact") -> dict[str, Any]:
        """Store an explicit memory strongly."""
        res = client.remember(content, kind=kind)
        return {"id": res["id"], "stored": True}

    @mcp.tool
    def forget(query: str, confirm: bool = False) -> dict[str, Any]:
        """Archive memories matching the query (audit is preserved)."""
        return client.forget(query, confirm=confirm)

    @mcp.tool
    def recall_session(session_id: str) -> str:
        """Restore the full audit of a specific session."""
        return client.recall_session(session_id)

    @mcp.tool
    def status() -> dict[str, Any]:
        """System statistics (scent count, strength distribution, decoder tier ratios)."""
        return client.status()

    @mcp.tool
    def export(out_path: str) -> dict[str, Any]:
        """Export the audit data as JSONL."""
        return client.export(out_path)

    return mcp
