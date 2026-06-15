"""lkhu daemon — a resident HTTP service that exclusively owns the LkhuEngine.

The hooks, MCP server, and CLI all become clients of this daemon and share the same memory
(a single SQLite/FAISS). The APScheduler lifecycle (daily/weekly) also runs inside this process.
Uses only the standard library.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from lkhu.server.client import daemon_host, daemon_port

if TYPE_CHECKING:
    from lkhu.core.engine import LkhuEngine

__all__ = ["DaemonServer", "run_daemon"]


class _Handler(BaseHTTPRequestHandler):
    """Daemon request handler. Maps each path to an engine operation."""

    server_version = "lkhu-daemon/0.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102
        pass  # suppress access logs (avoid polluting stdio)

    @property
    def _engine(self) -> LkhuEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def _send(self, code: int, obj: Any) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        try:
            parsed = urlparse(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            eng = self._engine
            if path == "/health":
                self._send(200, {"ok": True, "memories": eng.vault.count()})
            elif path == "/status":
                self._send(200, eng.status())
            elif path in ("/", "/dashboard"):
                from lkhu.server.dashboard import render_dashboard

                self._send_html(render_dashboard())
            elif path == "/api/stats":
                stats = eng.dashboard_stats()
                stats["data_dir"] = str(eng.paths.data_dir)
                self._send(200, stats)
            elif path == "/api/memories":
                include_archived = query.get("archived", ["0"])[0] == "1"
                self._send(200, {"memories": eng.dump(include_archived=include_archived)})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            eng = self._engine
            if self.path == "/recall":
                self._send(200, eng.recall(body["query"], k=body.get("k", 5)))
            elif self.path == "/remember":
                mem = eng.remember(
                    body["content"],
                    kind=body.get("kind", "fact"),
                    session_id=body.get("session_id", ""),
                )
                self._send(200, {"id": mem.id})
            elif self.path == "/observe":
                mem = eng.observe(
                    body["content"],
                    session_id=body.get("session_id", ""),
                    strength=body.get("strength"),
                )
                self._send(200, {"id": mem.id})
            elif self.path == "/recent":
                self._send(200, {"memories": eng.recent(n=body.get("n", 10))})
            elif self.path == "/forget":
                self._send(200, eng.forget(body["query"], confirm=body.get("confirm", False)))
            elif self.path == "/recall_session":
                self._send(200, {"text": eng.recall_session(body["session_id"])})
            elif self.path == "/export":
                self._send(
                    200, {"exported": eng.export(body["out_path"]), "path": body["out_path"]}
                )
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})


class DaemonServer:
    """A daemon that owns the engine and exposes it over HTTP.

    Args:
        engine: The LkhuEngine to own exclusively.
        host: Bind host.
        port: Bind port.
        start_scheduler: Whether to also run the lifecycle scheduler (daily/weekly).
    """

    def __init__(
        self,
        engine: LkhuEngine,
        host: str | None = None,
        port: int | None = None,
        start_scheduler: bool = False,
    ):
        self.engine = engine
        host = host or daemon_host()
        port = port if port is not None else daemon_port()
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.engine = engine  # type: ignore[attr-defined]
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]
        self._scheduler = None
        if start_scheduler:
            from lkhu.platform.scheduler import LkhuScheduler

            self._scheduler = LkhuScheduler()
            self._scheduler.register_lifecycle(
                daily_job=engine.run_daily,
                weekly_job=engine.run_weekly,
                daily_cron=engine.config["consolidation"].get("schedule_cron", "0 3 * * *"),
                weekly_cron=engine.config["cleansing"].get("schedule_cron", "30 3 * * 0"),
            )
            self._scheduler.start()
            # Catch up on any lifecycle missed while the daemon was down (e.g. the machine slept
            # through 03:00). Runs off-thread so startup is not blocked; fail-open.
            threading.Thread(target=self._run_due_lifecycle_safely, daemon=True).start()

    def _run_due_lifecycle_safely(self) -> None:
        try:
            self.engine.run_due_lifecycle()
        except Exception:  # noqa: BLE001  (lifecycle must never crash the daemon)
            pass

    def serve_forever(self) -> None:
        """Blocking service loop."""
        self._httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        """Start the service in a background thread (for tests/embedded use)."""
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        return t

    def shutdown(self) -> None:
        """Shut down the service and close the engine."""
        if self._scheduler is not None:
            self._scheduler.shutdown()
        self._httpd.shutdown()
        self._httpd.server_close()
        self.engine.close()


def run_daemon() -> None:
    """Production daemon entry point (``lkhu daemon``). Exits quietly if already up."""
    from lkhu.core.engine import LkhuEngine
    from lkhu.platform.ollama import OllamaEmbedder
    from lkhu.server.client import LkhuClient

    if LkhuClient().health():
        return  # already running

    engine = LkhuEngine.open(embedder=OllamaEmbedder())
    server = DaemonServer(engine, start_scheduler=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
