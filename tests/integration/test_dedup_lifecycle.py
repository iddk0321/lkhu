"""Tests for auto-save dedup and lifecycle catch-up (the consolidation-never-ran fix)."""

from __future__ import annotations

import pytest

from lkhu.core.encoder import HashingEmbedder
from lkhu.eval.harness import build_engine


@pytest.fixture
def engine(tmp_path):
    eng = build_engine(tmp_path, HashingEmbedder(dim=1024))
    yield eng
    eng.close()


def test_observe_dedups_identical_content(engine) -> None:
    first = engine.observe("we use PostgreSQL as the primary database", session_id="s1")
    second = engine.observe("we use PostgreSQL as the primary database", session_id="s1")
    assert first.id == second.id  # deduped, not a new row
    assert engine.vault.count() == 1


def test_observe_keeps_distinct_content(engine) -> None:
    engine.observe("we use PostgreSQL as the primary database", session_id="s1")
    engine.observe("the frontend is built with React and Vite", session_id="s1")
    assert engine.vault.count() == 2


def test_run_due_lifecycle_runs_then_is_idempotent(engine) -> None:
    for text in ("decision one about api", "decision two about auth", "decision three about db"):
        engine.observe(text, session_id="sess")

    ran = engine.run_due_lifecycle()
    assert "daily" in ran  # nothing had run before → daily fires
    assert engine.paths.lifecycle_state_path.exists()

    again = engine.run_due_lifecycle()
    assert again == {}  # fresh within the window → nothing re-runs


def test_reembed_rebuilds_vectors_preserving_metadata(engine) -> None:
    import numpy as np

    m = engine.remember("the backend is written in Python", kind="fact")
    before = engine.vault.get(m.id)
    # Simulate a model change by overwriting the stored vector with noise.
    engine.vault.update_vectors([(m.id, np.zeros_like(before.vector))])
    assert np.allclose(engine.vault.get(m.id).vector, 0.0)

    count = engine.reembed()
    assert count == engine.vault.count(include_archived=True)
    after = engine.vault.get(m.id)
    # Vector rebuilt from audit_text (no longer zero); metadata preserved.
    assert not np.allclose(after.vector, 0.0)
    assert after.audit_text == before.audit_text
    assert after.strength == before.strength
    assert after.kind == before.kind


def test_consolidation_actually_produces_summary(engine) -> None:
    for text in ("api uses rest", "auth uses jwt tokens", "db is postgres for records"):
        engine.observe(text, session_id="sess")
    engine.run_due_lifecycle()
    summaries = [m for m in engine.vault.all() if m.kind == "summary"]
    assert len(summaries) == 1
