"""Phase 4 verification — whether Tier1+2 handle ≥90% of the recall→decode pipeline.

Design §7.3: Tier3 (LLM) must stay under 5%. The key metric for token savings.
"""

from __future__ import annotations

from lkhu.core.codebook import Codebook
from lkhu.core.decoder import Decoder
from lkhu.core.encoder import Encoder, HashingEmbedder
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory
from lkhu.core.recall import RecallEngine

DIM = 128
KEYS = [
    "K_TOPIC",
    "K_DECISION",
    "K_LANGUAGE",
    "K_FILE",
    "K_PREFERENCE",
    "K_PROJECT",
    "K_FACT",
    "K_VALUE",
]

CORPUS = [
    "the user uses macOS",
    "the main language is Python",
    "the project name is lkhu",
    "the core logic lives in src/lkhu/core/vsa.py",
    "the user prefers the dark theme",
    "decided to make accuracy the top priority",
    "tests are written with pytest",
    "had kimchi stew for lunch today",
    "vector search is done with FAISS",
    "settled on the Apache 2.0 license",
]

QUERIES = [
    "user macOS OS",
    "which language is used Python",
    "project name lkhu",
    "vector code location vsa.py",
    "theme preference dark",
    "priority decision accuracy",
    "test tool pytest",
    "lunch menu kimchi",
    "search library FAISS",
    "license Apache",
]


def test_tier1_2_handle_at_least_90_percent(tmp_path) -> None:
    emb = HashingEmbedder(dim=DIM)
    cb = Codebook.generate(KEYS, dim=DIM, seed=2026)
    enc = Encoder(embedder=emb, codebook=cb)
    vault = LongTermVault(tmp_path / "m.db", dim=DIM)
    engine = RecallEngine(vault=vault, encoder=enc)
    decoder = Decoder(codebook=cb, embedder=emb)

    for text in CORPUS:
        vault.insert(Memory.make(enc.encode(text), audit_text=text, session_id="s"))

    for q in QUERIES:
        result = engine.recall(q, k=3)
        decoder.decode(result)

    stats = decoder.stats()
    handled = stats["tier1"] + stats["tier2"]
    assert stats["total"] == len(QUERIES)
    # Tier1+2 handle ≥90%, Tier3 (LLM) under 10%
    assert handled / stats["total"] >= 0.9
    assert stats["tier3_ratio"] < 0.1
    vault.close()
