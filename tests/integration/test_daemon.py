"""Phase 8 — daemon + client integration tests (HTTP round-trip)."""

from __future__ import annotations

import pytest

from lkhu.core.encoder import HashingEmbedder
from lkhu.core.engine import LkhuEngine, initialize
from lkhu.server.client import LkhuClient
from lkhu.server.daemon import DaemonServer


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "data"))
    initialize(base=None, mcp_config_path=tmp_path / "claude.json")
    engine = LkhuEngine.open(embedder=HashingEmbedder(dim=1024))
    server = DaemonServer(engine, host="127.0.0.1", port=0)  # port 0 → OS assigns a free port
    server.start_background()
    c = LkhuClient(base_url=f"http://127.0.0.1:{server.port}")
    yield c
    server.shutdown()


def test_health(client) -> None:
    assert client.health() is True


def test_remember_then_recall(client) -> None:
    client.remember("The main language is Python and development happens on macOS", kind="fact")
    client.remember("The project name is lkhu", kind="fact")
    res = client.recall("development language", k=3)
    assert "Python" in res["text"]
    assert res["tier"] in (1, 2, 3)


def test_observe_and_recent(client) -> None:
    client.observe("first conversation turn", session_id="s1")
    client.observe("second conversation turn", session_id="s1")
    recent = client.recent(n=5)
    assert len(recent) == 2
    assert all("audit_text" in m for m in recent)


def test_status(client) -> None:
    client.remember("a single memory", kind="fact")
    s = client.status()
    assert s["total_memories"] == 1
    assert s["dim"] == 1024


def test_forget(client) -> None:
    client.remember("secret quux to delete", kind="fact")
    assert client.status()["total_memories"] == 1
    res = client.forget("secret quux", confirm=True)
    assert res["archived"] >= 1
    assert client.status()["total_memories"] == 0


def test_recall_session(client) -> None:
    client.remember("session note one", kind="fact", session_id="sx")
    client.remember("session note two", kind="fact", session_id="sx")
    text = client.recall_session("sx")
    assert "one" in text and "two" in text


def test_export(client, tmp_path) -> None:
    client.remember("memory to export", kind="fact")
    out = tmp_path / "exp.jsonl"
    res = client.export(str(out))
    assert res["exported"] == 1
    assert out.exists()


# ── dashboard ───────────────────────────────────────────────────────────────


def test_dashboard_stats(client) -> None:
    client.remember("a strong explicit memory", kind="fact")
    client.observe("an ordinary turn memory", session_id="s")
    stats = client.dashboard_stats()
    assert stats["total_active"] == 2
    assert "explicit" in stats["kinds"] and "turn" in stats["kinds"]
    assert sum(stats["strength_buckets"].values()) == 2
    assert sum(stats["age_buckets"].values()) == 2
    assert stats["lifecycle"]["daily_decay"] == 0.99


def test_dashboard_memories_endpoint(client) -> None:
    client.remember("dashboard memory alpha", kind="fact")
    mems = client.memories()
    assert len(mems) == 1
    m = mems[0]
    # all metadata must be expanded
    for key in ("id", "audit_text", "strength", "kind", "created_at", "access_count", "archived"):
        assert key in m
    assert m["audit_text"] == "dashboard memory alpha"


def test_dashboard_html_served(client) -> None:
    import urllib.request

    with urllib.request.urlopen(client.base_url + "/") as resp:
        html = resp.read().decode("utf-8")
        assert resp.headers["Content-Type"].startswith("text/html")
    assert "lkhu Memory Dashboard" in html
    assert "/api/memories" in html  # JS polls the endpoint
