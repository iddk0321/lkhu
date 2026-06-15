"""Phase 6 — config loader tests."""

from __future__ import annotations

from lkhu.config.loader import deep_merge, load_config, load_defaults


def test_load_defaults_has_core_sections() -> None:
    cfg = load_defaults()
    for section in ["encoder", "codebook", "recall", "decoder", "consolidation", "cleansing"]:
        assert section in cfg
    assert cfg["encoder"]["dim"] == 1024
    assert len(cfg["codebook"]["initial_keys"]) == 15


def test_deep_merge_overrides_nested() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 9}, "c": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    # original preserved
    assert base["a"]["y"] == 2


def test_load_config_applies_user_override(tmp_path) -> None:
    user = tmp_path / "config.yaml"
    user.write_text("recall:\n  default_k: 12\n")
    cfg = load_config(user)
    assert cfg["recall"]["default_k"] == 12
    # unmerged values keep their defaults
    assert cfg["encoder"]["dim"] == 1024


def test_load_config_without_user_file(tmp_path) -> None:
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg["decoder"]["tier3_llm_max_tokens"] == 80
