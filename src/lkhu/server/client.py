"""lkhu daemon HTTP client + auto-launch.

The daemon (``lkhu daemon``) exclusively owns the LkhuEngine, and the hooks, MCP server, and
CLI all connect to it through this client. This avoids the problem of multiple processes opening
SQLite/FAISS directly and having their indexes drift apart.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

__all__ = ["LkhuClient", "daemon_host", "daemon_port", "daemon_url", "ensure_daemon_running"]

DEFAULT_BASE_PORT = 37700


def daemon_host() -> str:
    """Daemon host (local only)."""
    return os.environ.get("LKHU_DAEMON_HOST", "127.0.0.1")


def daemon_port() -> int:
    """Daemon port. ``LKHU_DAEMON_PORT`` takes priority; otherwise spread out per user."""
    env = os.environ.get("LKHU_DAEMON_PORT")
    if env:
        return int(env)
    try:
        uid = os.getuid()  # POSIX
    except AttributeError:
        uid = 0  # Windows
    return DEFAULT_BASE_PORT + (uid % 100)


def daemon_url() -> str:
    """Daemon base URL."""
    return f"http://{daemon_host()}:{daemon_port()}"


class LkhuClient:
    """Thin HTTP client that connects to the daemon.

    Args:
        base_url: Daemon URL (defaults to automatic environment-based resolution).
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or daemon_url()).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (localhost)
            payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else None

    # ----- diagnostics -----

    def health(self) -> bool:
        """Check whether the daemon is responsive."""
        try:
            return self._request("GET", "/health").get("ok", False)
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # ----- operations -----

    def recall(self, query: str, k: int = 5) -> dict[str, Any]:
        return self._request("POST", "/recall", {"query": query, "k": k})

    def remember(self, content: str, kind: str = "fact", session_id: str = "") -> dict[str, Any]:
        return self._request(
            "POST", "/remember", {"content": content, "kind": kind, "session_id": session_id}
        )

    def observe(
        self, content: str, session_id: str = "", strength: float | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/observe",
            {"content": content, "session_id": session_id, "strength": strength},
        )

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self._request("POST", "/recent", {"n": n})["memories"]

    def forget(self, query: str, confirm: bool = False) -> dict[str, Any]:
        return self._request("POST", "/forget", {"query": query, "confirm": confirm})

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def dashboard_stats(self) -> dict[str, Any]:
        return self._request("GET", "/api/stats")

    def memories(self, include_archived: bool = False) -> list[dict[str, Any]]:
        flag = 1 if include_archived else 0
        return self._request("GET", f"/api/memories?archived={flag}")["memories"]

    def recall_session(self, session_id: str) -> str:
        return self._request("POST", "/recall_session", {"session_id": session_id})["text"]

    def export(self, out_path: str) -> dict[str, Any]:
        return self._request("POST", "/export", {"out_path": out_path})


def ensure_daemon_running(timeout: float = 20.0) -> bool:
    """Check whether the daemon is up, and launch it in the background if not.

    Args:
        timeout: Maximum time in seconds to wait for a response after launch.

    Returns:
        True if the daemon is responsive.
    """
    client = LkhuClient()
    if client.health():
        return True

    # Run `python -m lkhu daemon` detached using the current interpreter
    log_path = os.environ.get("LKHU_DAEMON_LOG")
    stdout = open(log_path, "ab") if log_path else subprocess.DEVNULL  # noqa: SIM115
    kwargs: dict[str, Any] = {"stdout": stdout, "stderr": stdout}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # detach from parent
    else:  # Windows
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, "-m", "lkhu", "daemon"], **kwargs)  # noqa: S603

    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.health():
            return True
        time.sleep(0.3)
    return False
