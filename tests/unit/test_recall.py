"""Phase 4 — RecallEngine tests (rerank, Hebbian update, composite scent)."""

from __future__ import annotations

import numpy as np
import pytest

from lkhu.core.codebook import Codebook
from lkhu.core.encoder import Encoder, HashingEmbedder
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory
from lkhu.core.recall import RecallEngine, recency_score

DIM = 64
KEYS = ["K_TOPIC", "K_DECISION", "K_LANGUAGE", "K_FILE", "K_PREFERENCE", "K_PROJECT"]


@pytest.fixture
def engine(tmp_path):
    emb = HashingEmbedder(dim=DIM)
    cb = Codebook.generate(KEYS, dim=DIM, seed=1)
    enc = Encoder(embedder=emb, codebook=cb)
    vault = LongTermVault(tmp_path / "m.db", dim=DIM)
    eng = RecallEngine(vault=vault, encoder=enc)
    yield eng
    vault.close()


def _store(engine: RecallEngine, text: str, **kw) -> Memory:
    vec = engine.encoder.encode(text)
    return engine.vault.insert(Memory.make(vec, audit_text=text, **kw))


def test_recall_returns_composite_and_memories(engine) -> None:
    _store(engine, "the user works with Python")
    _store(engine, "the project name is lkhu")
    _store(engine, "lunch today is kimchi stew")
    result = engine.recall("Python work", k=2)
    assert result.composite.shape == (DIM,)
    assert np.isclose(np.linalg.norm(result.composite), 1.0)
    assert 1 <= len(result.memories) <= 2


def test_recall_finds_relevant(engine) -> None:
    _store(engine, "the main language is Python")
    _store(engine, "totally unrelated small talk the weather is nice")
    result = engine.recall("Python language", k=1)
    assert "Python" in result.memories[0].audit_text


def test_hebbian_update_persisted(engine) -> None:
    m = _store(engine, "the user works with Python", strength=1.0)
    engine.recall("Python", k=1)
    refreshed = engine.vault.get(m.id)
    assert refreshed.access_count == 1
    assert refreshed.strength > 1.0  # ×1.05 reinforcement
    assert refreshed.strength <= 1.5


def test_strength_capped(engine) -> None:
    m = _store(engine, "Python work log", strength=1.49)
    engine.recall("Python", k=1)
    assert engine.vault.get(m.id).strength <= 1.5


def test_recency_score_monotonic() -> None:
    now = 1_000_000_000.0
    fresh = recency_score(now_iso_at(now), now=now)
    old = recency_score(now_iso_at(now - 30 * 86400), now=now)
    assert 0.0 < old < fresh <= 1.0


def now_iso_at(epoch: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, UTC).isoformat()


def test_rerank_prefers_stronger_when_similar(engine) -> None:
    """When similarity is comparable, a stronger memory gets a higher score."""
    base = "Python data pipeline work"
    weak = _store(engine, base, strength=0.2)
    strong_text = base + " extra"
    strong = _store(engine, strong_text, strength=1.5)
    result = engine.recall(base, k=2)
    ids = [m.id for m in result.memories]
    # Both are recalled, and the stronger memory is likely to be first
    assert strong.id in ids and weak.id in ids
    assert result.memories[0].id == strong.id
