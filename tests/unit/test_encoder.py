"""Phase 2 — Encoder (natural language → scent) tests.

Verifies determinism, normalization, semantic+structure synthesis, and key-value
structure round-tripping.
"""

from __future__ import annotations

import numpy as np

from lkhu.core import vsa
from lkhu.core.codebook import Codebook
from lkhu.core.encoder import Encoder, HashingEmbedder, extract_kv

DIM = 256
KEYS = [
    "K_USER",
    "K_TOPIC",
    "K_VALUE",
    "K_DECISION",
    "K_PREFERENCE",
    "K_FILE",
    "K_LANGUAGE",
    "K_PROJECT",
]


def _make_encoder() -> Encoder:
    emb = HashingEmbedder(dim=DIM)
    cb = Codebook.generate(KEYS, dim=DIM, seed=2024)
    return Encoder(embedder=emb, codebook=cb)


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=DIM)
    a = emb.embed("hello world")
    b = emb.embed("hello world")
    assert a.shape == (DIM,)
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_hashing_embedder_shared_tokens_more_similar() -> None:
    emb = HashingEmbedder(dim=512)
    base = emb.embed("python data pipeline")
    near = emb.embed("python data tool")
    far = emb.embed("a completely different sentence kimchi stew")
    assert vsa.cosine(base, near) > vsa.cosine(base, far)


def test_encode_deterministic_dim_normalized() -> None:
    enc = _make_encoder()
    v1 = enc.encode("the user is on macOS and their main language is Python")
    v2 = enc.encode("the user is on macOS and their main language is Python")
    assert v1.shape == (DIM,)
    assert v1.dtype == np.float32
    assert np.isclose(np.linalg.norm(v1), 1.0)
    assert np.allclose(v1, v2)


def test_encode_semantic_only_when_no_kv() -> None:
    enc = _make_encoder()
    v = enc.encode("hmm just some ordinary small talk")
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_extract_kv_finds_language() -> None:
    pairs = extract_kv("the main language is Python")
    assert ("K_LANGUAGE", "Python") in pairs


def test_extract_kv_finds_file_path() -> None:
    pairs = extract_kv("the core logic lives in src/lkhu/core/vsa.py")
    assert any(k == "K_FILE" and "vsa.py" in v for k, v in pairs)


def test_extract_kv_caps_at_seven() -> None:
    text = "written in Python Java Rust Go C++ Ruby Kotlin Swift PHP"
    pairs = extract_kv(text)
    assert len(pairs) <= 7


def test_structure_roundtrip_probe() -> None:
    """Unbinding a scent that encoded a key-value by that key lands closer to the value."""
    enc = _make_encoder()
    encoded = enc.encode("the main language in use is Python")
    probe = vsa.unbind(encoded, enc.codebook["K_LANGUAGE"])

    py = vsa.normalize(enc.embedder.embed("Python"))
    java = vsa.normalize(enc.embedder.embed("Java"))
    assert vsa.cosine(probe, py) > vsa.cosine(probe, java)
