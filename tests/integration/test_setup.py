"""Phase 9 — install/uninstall orchestration tests.

Claude Code integration is handled by the plugin, so install() only does codebook + Desktop MCP.
uninstall() cleans up Desktop + legacy (directly registered Code MCP/hooks).
"""

from __future__ import annotations

import sys

import pytest

from lkhu.platform import claude_code, claude_hooks, mcp_config, setup


@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    """Point all integration paths at temporary locations and stub out the Claude Code CLI calls."""
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("LKHU_CLAUDE_CONFIG", str(tmp_path / "claude_desktop.json"))
    monkeypatch.setenv("LKHU_CC_SETTINGS", str(tmp_path / "settings.json"))

    # On uninstall (legacy cleanup), stub claude_code so the real claude CLI isn't called
    state = {"registered": True}
    monkeypatch.setattr(claude_code, "is_available", lambda: True)
    monkeypatch.setattr(
        claude_code, "unregister", lambda: state.__setitem__("registered", False) or True
    )
    monkeypatch.setattr(claude_code, "is_registered", lambda: state["registered"])
    return tmp_path


def test_portable_command_uses_current_interpreter() -> None:
    cmd, args = setup.portable_command()
    assert cmd == sys.executable  # no hardcoded path → portable
    assert args == ["-m", "lkhu", "serve"]


def test_install_creates_codebook_and_desktop_mcp(temp_env) -> None:
    result = setup.install()
    assert result["codebook_created"] is True
    # only the Claude Desktop MCP is registered (Claude Code is a plugin)
    assert mcp_config.is_registered()
    assert result["desktop_mcp"] is not None
    # install() no longer touches settings.json hooks (handled by the plugin)
    assert not claude_hooks.hooks_installed()
    # whether the registration command is portable (current interpreter)
    assert sys.executable in result["command"]


def test_install_is_idempotent(temp_env) -> None:
    setup.install()
    setup.install()  # the second run also passes without error (codebook preserved)
    assert mcp_config.is_registered()


def test_uninstall_removes_desktop_and_legacy(temp_env) -> None:
    setup.install()
    result = setup.uninstall()
    assert result["desktop_mcp"] is True
    assert "legacy_claude_code_mcp" in result
    assert "legacy_hooks" in result
    assert not mcp_config.is_registered()
