"""Phase 2 — OllamaEmbedder tests.

These require an actual Ollama runtime, so they are isolated with
``@pytest.mark.ollama`` (skipped by default in CI). They are skipped
automatically when the runtime is unavailable.
"""

from __future__ import annotations

import pytest

from lkhu.core import vsa
from lkhu.core.encoder import Embedder
from lkhu.platform.ollama import OllamaEmbedder

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def embedder() -> OllamaEmbedder:
    emb = OllamaEmbedder()
    if not emb.is_available():
        pytest.skip("Skipping: Ollama runtime (bge-m3) is not available.")
    return emb


def test_satisfies_embedder_protocol(embedder: OllamaEmbedder) -> None:
    assert isinstance(embedder, Embedder)


def test_embed_dim_and_finite(embedder: OllamaEmbedder) -> None:
    import numpy as np

    v = embedder.embed("The user is on macOS and their main language is Python")
    assert v.shape == (1024,)
    assert np.all(np.isfinite(v))


def test_semantic_similarity(embedder: OllamaEmbedder) -> None:
    a = vsa.normalize(embedder.embed("A dog is a cute animal"))
    b = vsa.normalize(embedder.embed("A cat is a lovely animal"))
    c = vsa.normalize(embedder.embed("I changed the car engine oil"))
    assert vsa.cosine(a, b) > vsa.cosine(a, c)
