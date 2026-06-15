"""Phase 4 — 3-Tier Decoder tests (audit excerpt / key probe / LLM fallback)."""

from __future__ import annotations

import numpy as np

from lkhu.core import vsa
from lkhu.core.codebook import Codebook
from lkhu.core.decoder import AuditVocab, Decoder, format_kv
from lkhu.core.encoder import HashingEmbedder
from lkhu.core.memory import Memory
from lkhu.core.recall import RecallResult

DIM = 64
KEYS = ["K_TOPIC", "K_DECISION", "K_LANGUAGE", "K_FILE", "K_PREFERENCE", "K_PROJECT"]


def _setup():
    emb = HashingEmbedder(dim=DIM)
    cb = Codebook.generate(KEYS, dim=DIM, seed=1)
    return emb, cb


def _mem(audit: str, vec: np.ndarray | None = None) -> Memory:
    if vec is None:
        vec = np.zeros(DIM, dtype=np.float32)
    return Memory.make(vec, audit_text=audit)


def test_format_kv() -> None:
    assert format_kv({"OS": "macOS", "priority": "accuracy"}) == "OS=macOS, priority=accuracy"


def test_tier1_short_audits() -> None:
    emb, cb = _setup()
    dec = Decoder(codebook=cb, embedder=emb)
    result = RecallResult(
        composite=np.zeros(DIM, dtype=np.float32),
        memories=[_mem("the user is on macOS"), _mem("main language Python")],
    )
    out = dec.decode(result)
    assert out.tier == 1
    assert "macOS" in out.text and "Python" in out.text
    assert not out.llm_used


def test_tier1_skipped_when_audit_too_long_then_tier3() -> None:
    emb, cb = _setup()
    dec = Decoder(codebook=cb, embedder=emb, tier2_threshold=0.99)
    long_text = "a" * 200
    result = RecallResult(
        composite=vsa.normalize(np.random.default_rng(0).standard_normal(DIM).astype(np.float32)),
        memories=[_mem(long_text)],
    )
    out = dec.decode(result)
    # When the audit is too long and no key matches, it falls back to Tier3
    assert out.tier == 3


def test_tier2_key_probe() -> None:
    emb, cb = _setup()
    # Pure structural composite scent: bind(K_LANGUAGE, Python) + bind(K_PROJECT, lkhu)
    v_py = vsa.normalize(emb.embed("Python"))
    v_proj = vsa.normalize(emb.embed("lkhu"))
    composite = vsa.bundle([vsa.bind(cb["K_LANGUAGE"], v_py), vsa.bind(cb["K_PROJECT"], v_proj)])

    dec = Decoder(codebook=cb, embedder=emb, tier2_threshold=0.5)
    vocab = AuditVocab(emb)
    for term in ["Python", "lkhu", "car", "kimchi"]:
        vocab.add(term)

    # audit is empty so Tier1 is skipped → Tier2 key probe
    result = RecallResult(composite=composite, memories=[_mem(""), _mem("")])
    out = dec.decode(result, vocab=vocab)
    assert out.tier == 2
    assert "Python" in out.text and "lkhu" in out.text


def test_tier3_llm_called_when_provided() -> None:
    emb, cb = _setup()
    calls = {}

    def fake_llm(memories, max_tokens):
        calls["n"] = len(memories)
        calls["budget"] = max_tokens
        return "a summarized one-liner"

    dec = Decoder(codebook=cb, embedder=emb, tier2_threshold=0.99, llm=fake_llm)
    long_text = "b" * 200
    result = RecallResult(
        composite=vsa.normalize(np.random.default_rng(1).standard_normal(DIM).astype(np.float32)),
        memories=[_mem(long_text)],
    )
    out = dec.decode(result)
    assert out.tier == 3
    assert out.llm_used
    assert out.text == "a summarized one-liner"
    assert calls["budget"] == 80


def test_tier3_extractive_fallback_without_llm() -> None:
    emb, cb = _setup()
    dec = Decoder(codebook=cb, embedder=emb, tier2_threshold=0.99)
    long_text = "c" * 200
    result = RecallResult(
        composite=vsa.normalize(np.random.default_rng(2).standard_normal(DIM).astype(np.float32)),
        memories=[_mem(long_text)],
    )
    out = dec.decode(result)
    assert out.tier == 3
    assert not out.llm_used
    assert len(out.text) > 0


def test_tier_counts_and_ratio() -> None:
    emb, cb = _setup()
    dec = Decoder(codebook=cb, embedder=emb)
    for _ in range(9):
        dec.decode(RecallResult(np.zeros(DIM, dtype=np.float32), memories=[_mem("a short memory")]))
    # Force tier3 once
    dec2 = Decoder(codebook=cb, embedder=emb, tier2_threshold=0.99)
    dec2.decode(RecallResult(np.zeros(DIM, dtype=np.float32), memories=[_mem("d" * 200)]))

    assert dec.counts[1] == 9
    assert dec.tier3_ratio == 0.0
    assert dec2.counts[3] == 1
    assert dec2.tier3_ratio == 1.0


def test_empty_result_returns_empty() -> None:
    emb, cb = _setup()
    dec = Decoder(codebook=cb, embedder=emb)
    out = dec.decode(RecallResult(np.zeros(DIM, dtype=np.float32), memories=[]))
    assert out.text == ""
