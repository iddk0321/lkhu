"""Phase 1 — CLI skeleton tests."""

from __future__ import annotations

from typer.testing import CliRunner

from lkhu import __version__
from lkhu.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_is_0_1_0() -> None:
    assert __version__ == "0.1.0"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help → show help and exit (exit code 0 or 2 allowed)
    assert "lkhu" in result.stdout.lower()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    for cmd in ["init", "serve", "status", "recall", "doctor", "export"]:
        assert cmd in result.stdout


def test_help_lists_hook_commands() -> None:
    result = runner.invoke(app, ["--help"])
    for cmd in ["daemon", "hook", "install-hooks", "uninstall-hooks"]:
        assert cmd in result.stdout


def test_install_and_uninstall_hooks(monkeypatch, tmp_path) -> None:
    from lkhu.platform import claude_hooks

    cfg = tmp_path / "settings.json"
    monkeypatch.setenv("LKHU_CC_SETTINGS", str(cfg))

    result = runner.invoke(app, ["install-hooks"])
    assert result.exit_code == 0
    assert claude_hooks.hooks_installed(cfg)

    result = runner.invoke(app, ["uninstall-hooks"])
    assert result.exit_code == 0
    assert not claude_hooks.hooks_installed(cfg)


def _temp_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("LKHU_CLAUDE_CONFIG", str(tmp_path / "claude.json"))


def test_reset_requires_confirm(monkeypatch, tmp_path) -> None:
    _temp_env(monkeypatch, tmp_path)
    result = runner.invoke(app, ["reset"])
    assert result.exit_code == 1


def test_init_then_uninstall(monkeypatch, tmp_path) -> None:
    _temp_env(monkeypatch, tmp_path)
    from lkhu.platform import claude_code, mcp_config

    # Stub out uninstall's legacy cleanup so it doesn't call the real claude CLI
    monkeypatch.setattr(claude_code, "unregister", lambda: False)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "data" / "codebook.npy").exists()
    assert mcp_config.is_registered(tmp_path / "claude.json")

    # uninstall only removes the registration and preserves data
    result = runner.invoke(app, ["uninstall"])
    assert result.exit_code == 0, result.stdout
    assert not mcp_config.is_registered(tmp_path / "claude.json")
    assert (tmp_path / "data" / "codebook.npy").exists()


def test_doctor_runs(monkeypatch, tmp_path) -> None:
    _temp_env(monkeypatch, tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout.lower()
