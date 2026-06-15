"""Phase 6 verification — MCP tools E2E (init → engine → tool round-trip).

Verifies all layers with HashingEmbedder, without a real Claude/Ollama.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lkhu.core.encoder import HashingEmbedder
from lkhu.core.engine import LkhuEngine, initialize
from lkhu.server.client import LkhuClient
from lkhu.server.daemon import DaemonServer
from lkhu.server.tools import TOOL_NAMES, build_server


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "data"))
    cfg = tmp_path / "claude.json"
    info = initialize(base=None, mcp_config_path=cfg)
    assert info["codebook_created"]
    assert json.loads(cfg.read_text())["mcpServers"]["lkhu"]["command"] == "lkhu"
    eng = LkhuEngine.open(embedder=HashingEmbedder(dim=1024))
    yield eng
    eng.close()


def test_engine_open_requires_codebook(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        LkhuEngine.open(embedder=HashingEmbedder(dim=1024))


@pytest.fixture
def tool_server(tmp_path, monkeypatch):
    """MCP server wired to a daemon + client (exclusive data ownership)."""
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "data2"))
    initialize(base=None, mcp_config_path=tmp_path / "claude2.json")
    engine = LkhuEngine.open(embedder=HashingEmbedder(dim=1024))
    server = DaemonServer(engine, host="127.0.0.1", port=0)
    server.start_background()
    client = LkhuClient(base_url=f"http://127.0.0.1:{server.port}")
    yield build_server(client)
    server.shutdown()


def test_mcp_server_exposes_six_tools(tool_server) -> None:
    names = sorted(t.name for t in asyncio.run(tool_server.list_tools()))
    assert names == sorted(TOOL_NAMES)


def test_remember_recall_roundtrip_via_tools(tool_server) -> None:
    mcp = tool_server

    async def scenario():
        await mcp.call_tool("remember", {"content": "The main language is Python", "kind": "fact"})
        await mcp.call_tool("remember", {"content": "The project name is lkhu", "kind": "fact"})
        recall = await mcp.call_tool("recall", {"query": "Python language", "k": 3})
        status = await mcp.call_tool("status", {})
        return recall.structured_content, status.structured_content

    recall, status = asyncio.run(scenario())
    assert "Python" in recall["text"]
    assert recall["tier"] in (1, 2, 3)
    assert status["total_memories"] == 2


def test_forget_archives(engine) -> None:
    engine.remember("secret memory xyzzy to be deleted", kind="fact", session_id="s")
    assert engine.status()["total_memories"] == 1
    res = engine.forget("secret xyzzy", confirm=True)
    assert res["archived"] >= 1
    assert engine.status()["total_memories"] == 0


def test_forget_requires_confirm(engine) -> None:
    engine.remember("memory to be preserved", kind="fact")
    res = engine.forget("preserved", confirm=False)
    assert res["archived"] == 0
    assert engine.status()["total_memories"] == 1


def test_recall_session_restores_audit(engine) -> None:
    engine.remember("session memory one", kind="fact", session_id="sx")
    engine.remember("session memory two", kind="fact", session_id="sx")
    restored = engine.recall_session("sx")
    assert "one" in restored and "two" in restored


def test_export_writes_jsonl(engine, tmp_path) -> None:
    engine.remember("memory to export", kind="fact", session_id="s")
    out = tmp_path / "export.jsonl"
    count = engine.export(out)
    assert count == 1
    assert "memory to export" in out.read_text()


def test_daily_and_weekly_cycle(engine) -> None:
    for i in range(4):
        engine.observe(f"session turn {i} content", session_id="day")
    daily = engine.run_daily()
    assert daily["consolidated"] == 1  # 4 turns → 1 summary
    weekly = engine.run_weekly()
    assert "merged" in weekly and "archived_weak" in weekly
