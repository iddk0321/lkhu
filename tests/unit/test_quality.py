"""Tests for the recall-quality improvements: noise filter + gated Hebbian reinforcement."""

from __future__ import annotations

import pytest

from lkhu.core.codebook import Codebook
from lkhu.core.encoder import Encoder, HashingEmbedder
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory
from lkhu.core.recall import RecallEngine
from lkhu.server import hooks

DIM = 64
KEYS = ["K_TOPIC", "K_DECISION", "K_LANGUAGE", "K_FILE", "K_PREFERENCE", "K_PROJECT"]


# ── save filter ──────────────────────────────────────────────────────────────


def test_is_noise_drops_url_and_symbols() -> None:
    assert hooks._is_noise("https://github.com/DDDangkong/lkhu")
    assert hooks._is_noise("👍👍🔥")
    assert hooks._is_noise("...")
    assert hooks._is_noise("   ")


def test_is_noise_keeps_real_content_in_any_language() -> None:
    assert not hooks._is_noise("The backend uses FastAPI and Python 3.11.")
    assert not hooks._is_noise("프론트엔드는 React와 TypeScript를 사용한다.")


def test_skip_save_combines_trivial_and_noise() -> None:
    assert hooks._skip_save("ㅇㅇ")  # trivial (too short)
    assert hooks._skip_save("🔥🔥🔥")  # noise (no alnum)
    assert not hooks._skip_save("We decided to use PostgreSQL as the primary database.")


# ── gated Hebbian reinforcement ──────────────────────────────────────────────


@pytest.fixture
def recall_factory(tmp_path):
    def _make(threshold: float) -> RecallEngine:
        emb = HashingEmbedder(dim=DIM)
        cb = Codebook.generate(KEYS, dim=DIM, seed=1)
        enc = Encoder(embedder=emb, codebook=cb)
        vault = LongTermVault(tmp_path / f"m{threshold}.db", dim=DIM)
        return RecallEngine(vault=vault, encoder=enc, reinforce_sim_threshold=threshold)

    return _make


def _store(engine: RecallEngine, text: str, **kw) -> Memory:
    vec = engine.encoder.encode(text)
    return engine.vault.insert(Memory.make(vec, audit_text=text, **kw))


def test_ungated_reinforces_even_unrelated(recall_factory) -> None:
    eng = recall_factory(0.0)
    m = _store(eng, "the user works with Python", strength=1.0)
    eng.recall("a completely unrelated banana query about nothing", k=1)
    assert eng.vault.get(m.id).strength > 1.0  # boosted regardless of relevance


def test_gated_skips_low_similarity_filler(recall_factory) -> None:
    eng = recall_factory(0.99)  # only near-identical hits reinforce
    m = _store(eng, "the user works with Python", strength=1.0)
    eng.recall("a completely unrelated banana query about nothing", k=1)
    # Returned as the only candidate, but similarity is far below 0.99 → strength untouched.
    refreshed = eng.vault.get(m.id)
    assert refreshed.strength == pytest.approx(1.0)
    assert refreshed.access_count == 1  # still counted as accessed


def test_gated_still_reinforces_true_hit(recall_factory) -> None:
    eng = recall_factory(0.5)
    m = _store(eng, "the user works with Python", strength=1.0)
    eng.recall("the user works with Python", k=1)  # identical → high similarity
    assert eng.vault.get(m.id).strength > 1.0
