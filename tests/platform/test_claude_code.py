"""Phase 9 — Claude Code (CLI) MCP registration tests (claude mcp CLI, runner injection)."""

from __future__ import annotations

import subprocess

import pytest

from lkhu.platform import claude_code


class FakeRunner:
    """A fake runner that records ``claude`` invocations."""

    def __init__(self, rc: int = 0):
        self.calls: list[list[str]] = []
        self.rc = rc

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, self.rc, "", "")


@pytest.fixture
def has_claude(monkeypatch):
    """Make it look as if the claude CLI is present."""
    monkeypatch.setattr(claude_code, "claude_cli", lambda: "/usr/bin/claude")


def test_register_builds_claude_mcp_add(has_claude) -> None:
    runner = FakeRunner(rc=0)
    ok = claude_code.register("/py", ["-m", "lkhu", "serve"], runner=runner)
    assert ok is True
    # add called after remove (idempotent)
    assert runner.calls[0][:4] == ["/usr/bin/claude", "mcp", "remove", "lkhu"]
    add = runner.calls[1]
    assert add[:4] == ["/usr/bin/claude", "mcp", "add", "lkhu"]
    assert add[-4:] == ["--", "/py", "-m", "lkhu"] or "serve" in add  # portable command included
    assert "/py" in add and "serve" in add


def test_register_returns_false_without_cli(monkeypatch) -> None:
    monkeypatch.setattr(claude_code, "claude_cli", lambda: None)
    assert claude_code.register("/py", ["-m", "lkhu", "serve"]) is False


def test_unregister_calls_remove(has_claude) -> None:
    runner = FakeRunner(rc=0)
    assert claude_code.unregister(runner=runner) is True
    assert runner.calls[0][:4] == ["/usr/bin/claude", "mcp", "remove", "lkhu"]


def test_is_registered_uses_get(has_claude) -> None:
    present = FakeRunner(rc=0)
    assert claude_code.is_registered(runner=present) is True
    assert present.calls[0][:4] == ["/usr/bin/claude", "mcp", "get", "lkhu"]

    absent = FakeRunner(rc=1)
    assert claude_code.is_registered(runner=absent) is False


def test_is_available_reflects_cli(monkeypatch) -> None:
    monkeypatch.setattr(claude_code, "claude_cli", lambda: "/usr/bin/claude")
    assert claude_code.is_available() is True
    monkeypatch.setattr(claude_code, "claude_cli", lambda: None)
    assert claude_code.is_available() is False
