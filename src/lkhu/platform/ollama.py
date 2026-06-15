"""Ollama integration — Snowflake Arctic Embed 2 embedding wrapper. Design doc §5.3, §11.2.

Provides ``OllamaEmbedder``, which satisfies the core ``Embedder`` protocol.
Ollama is an external runtime that the user installs themselves, so this module
helps with detection/model checks/automatic pull.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OllamaEmbedder", "ollama_available"]

DEFAULT_MODEL = "snowflake-arctic-embed2"  # 1024-dim multilingual; best retrieval in lkhu eval
DEFAULT_DIM = 1024


def _client(host: str | None = None):
    """Create an ollama client (lazy import)."""
    import ollama

    return ollama.Client(host=host) if host else ollama.Client()


def ollama_available(host: str | None = None) -> bool:
    """Check whether the Ollama server is reachable (does not check whether a model runs)."""
    try:
        _client(host).list()
        return True
    except Exception:
        return False


class OllamaEmbedder:
    """Semantic embedder using Ollama (Snowflake Arctic Embed 2).

    Args:
        model: Embedding model name (default ``snowflake-arctic-embed2``).
        dim: Expected dimension (default 1024). An error is raised if it differs
            from the response dimension.
        host: Ollama host (default uses environment config/local).
    """

    def __init__(self, model: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM, host: str | None = None):
        self.model = model
        self.dim = dim
        self.host = host

    def _embed_raw(self, text: str) -> list[float]:
        resp = _client(self.host).embed(model=self.model, input=text)
        # The ollama response is a dict or an object (.embeddings) depending on version
        embeddings = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
        return list(embeddings[0])

    def detect_dim(self) -> int:
        """Probe the model's actual output dimension (one embedding call)."""
        return len(self._embed_raw("probe"))

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a ``dim``-dimensional ``float32`` vector.

        Args:
            text: Natural-language text to embed.

        Returns:
            ``float32`` embedding vector.

        Raises:
            ValueError: When the response dimension differs from the expected dimension.
        """
        vec = np.asarray(self._embed_raw(text), dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vec.shape}")
        return vec

    # ----- diagnostics / preparation -----

    def is_available(self) -> bool:
        """Check whether embeddings can actually be produced (server + model running)."""
        try:
            self.embed("ping")
            return True
        except Exception:
            return False

    def has_model(self) -> bool:
        """Check whether the target model exists locally."""
        try:
            resp = _client(self.host).list()
            models = resp.get("models", []) if isinstance(resp, dict) else resp.models
            names = [m["model"] if isinstance(m, dict) else m.model for m in models]
            return any(self.model in (n or "") for n in names)
        except Exception:
            return False

    def ensure_model(self) -> None:
        """Pull the model if it is missing (requires network)."""
        if not self.has_model():
            _client(self.host).pull(self.model)
