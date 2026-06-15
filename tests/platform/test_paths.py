"""Phase 2 — platform path layer tests.

A single gateway that guarantees no hardcoded paths (hard rule 1). All artifacts
must live under data_dir.
"""

from __future__ import annotations

from pathlib import Path

from lkhu.platform.paths import LkhuPaths


def test_explicit_base_roots_data_dir(tmp_path) -> None:
    paths = LkhuPaths(base=tmp_path)
    assert paths.data_dir == tmp_path


def test_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LKHU_DATA", str(tmp_path / "custom"))
    paths = LkhuPaths()
    assert paths.data_dir == tmp_path / "custom"


def test_default_uses_platformdirs(monkeypatch) -> None:
    monkeypatch.delenv("LKHU_DATA", raising=False)
    paths = LkhuPaths()
    assert "lkhu" in str(paths.data_dir).lower()


def test_runtime_artifacts_under_data_dir(tmp_path) -> None:
    paths = LkhuPaths(base=tmp_path)
    for p in [
        paths.codebook_path,
        paths.codebook_backup_path,
        paths.db_path,
        paths.faiss_path,
        paths.short_term_path,
        paths.audit_dir,
        paths.backups_dir,
        paths.logs_dir,
        paths.stats_path,
    ]:
        assert Path(p).is_relative_to(paths.data_dir)


def test_codebook_backup_targets_includes_documents(tmp_path) -> None:
    paths = LkhuPaths(base=tmp_path)
    targets = paths.codebook_backup_targets()
    # Local backup + Documents backup (secondary/tertiary of the triple backup)
    assert paths.codebook_backup_path in targets
    assert any("Documents" in str(t) for t in targets)


def test_ensure_creates_directories(tmp_path) -> None:
    paths = LkhuPaths(base=tmp_path / "deep" / "nested")
    paths.ensure()
    assert paths.data_dir.is_dir()
    assert paths.audit_dir.is_dir()
    assert paths.backups_dir.is_dir()
    assert paths.logs_dir.is_dir()
