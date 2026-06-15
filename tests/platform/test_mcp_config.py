"""Phase 6 — tests for automatic Claude settings registration/unregistration."""

from __future__ import annotations

import json

from lkhu.platform import mcp_config


def test_register_creates_entry(tmp_path) -> None:
    cfg = tmp_path / "claude.json"
    mcp_config.register(cfg)
    data = json.loads(cfg.read_text())
    assert data["mcpServers"]["lkhu"] == {"command": "lkhu", "args": ["serve"]}
    assert mcp_config.is_registered(cfg)


def test_register_preserves_other_servers(tmp_path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    mcp_config.register(cfg)
    data = json.loads(cfg.read_text())
    assert "other" in data["mcpServers"]
    assert "lkhu" in data["mcpServers"]


def test_unregister_removes_only_lkhu(tmp_path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}, "lkhu": {}}}))
    assert mcp_config.unregister(cfg) is True
    data = json.loads(cfg.read_text())
    assert "lkhu" not in data["mcpServers"]
    assert "other" in data["mcpServers"]


def test_unregister_when_absent(tmp_path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    assert mcp_config.unregister(cfg) is False


def test_env_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "custom" / "claude.json"
    monkeypatch.setenv("LKHU_CLAUDE_CONFIG", str(target))
    assert mcp_config.claude_config_path() == target
