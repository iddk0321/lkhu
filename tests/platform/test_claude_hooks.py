"""Phase 8 — Claude Code hook registration/unregistration tests."""

from __future__ import annotations

import json

from lkhu.platform import claude_hooks


def _events(path) -> dict:
    return json.loads(path.read_text()).get("hooks", {})


def test_install_registers_three_events(tmp_path) -> None:
    cfg = tmp_path / "settings.json"
    claude_hooks.install_hooks(config_path=cfg, prefix="lkhu")
    hooks = _events(cfg)
    assert set(hooks) == {"SessionStart", "UserPromptSubmit", "Stop"}
    # the command contains 'lkhu hook'
    cmd = hooks["SessionStart"][0]["hooks"][0]["command"]
    assert "lkhu hook session-start" in cmd
    assert hooks["SessionStart"][0]["matcher"] == "startup|clear|compact"
    assert claude_hooks.hooks_installed(cfg)


def test_install_preserves_other_hooks(tmp_path) -> None:
    cfg = tmp_path / "settings.json"
    cfg.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other-tool"}]}]}})
    )
    claude_hooks.install_hooks(config_path=cfg, prefix="lkhu")
    stop_cmds = [h["command"] for g in _events(cfg)["Stop"] for h in g["hooks"]]
    assert "other-tool" in stop_cmds
    assert any("lkhu hook stop" in c for c in stop_cmds)


def test_install_is_idempotent(tmp_path) -> None:
    cfg = tmp_path / "settings.json"
    claude_hooks.install_hooks(config_path=cfg, prefix="lkhu")
    claude_hooks.install_hooks(config_path=cfg, prefix="lkhu")
    # exactly one lkhu group per event, no duplicates
    assert len(_events(cfg)["SessionStart"]) == 1
    assert len(_events(cfg)["UserPromptSubmit"]) == 1


def test_uninstall_removes_only_lkhu(tmp_path) -> None:
    cfg = tmp_path / "settings.json"
    cfg.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other-tool"}]}]}})
    )
    claude_hooks.install_hooks(config_path=cfg, prefix="lkhu")
    claude_hooks.uninstall_hooks(config_path=cfg)
    hooks = _events(cfg)
    assert not claude_hooks.hooks_installed(cfg)
    # other hooks are preserved
    assert "other-tool" in [h["command"] for g in hooks["Stop"] for h in g["hooks"]]
    assert "SessionStart" not in hooks
