"""Encoder — encodes natural language into a scent. Design doc §7.1.

scent = semantic scent (0.6) + structure scent (0.4).
    - semantic scent: sentence embedding produced by the embedder (Ollama Snowflake Arctic Embed 2).
    - structure scent: rule-extracted key-values ``bind``-ed and summed (Σ bind(K, V)).

Key-value extraction is rule-based to avoid LLM calls (CLAUDE.md hard rule 3).
The embedder is abstracted behind the ``Embedder`` protocol so core knows nothing
about Ollama or the OS.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from lkhu.core import vsa
from lkhu.core.codebook import Codebook

__all__ = ["Embedder", "HashingEmbedder", "Encoder", "extract_kv"]


@runtime_checkable
class Embedder(Protocol):
    """Interface for an embedder that turns natural language into a fixed-dim vector."""

    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a ``dim``-dimensional ``float32`` vector."""
        ...


class HashingEmbedder:
    """Dependency-free deterministic hashing embedder (for tests and offline use).

    Sums unit vectors built by hashing tokens. The more tokens two texts share, the
    more similar they are. Its semantic quality falls short of Snowflake Arctic Embed 2, but it lets
    every layer be validated without Ollama.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        if not tokens:
            # Empty input: build a single vector from the hash of the whole text
            tokens = [text.strip() or "∅"]
        acc = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            digest = hashlib.blake2b(tok.lower().encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            acc += rng.standard_normal(self.dim).astype(np.float32)
        return vsa.normalize(acc)


def _tokenize(text: str) -> list[str]:
    """Unicode word tokenization (word characters: Latin, digits, Hangul, etc.)."""
    return re.findall(r"\w+", text, flags=re.UNICODE)


# ----- rule-based key-value extraction -----

# Recognized programming languages (canonical spelling preserved)
_LANGUAGES = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "rust": "Rust",
    "go": "Go",
    "golang": "Go",
    "c++": "C++",
    "ruby": "Ruby",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "php": "PHP",
    "c#": "C#",
    "scala": "Scala",
    "sql": "SQL",
}

# File paths/names (common source extensions)
_FILE_RE = re.compile(
    r"\b[\w./\\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|rb|kt|swift|php|cs|cpp|c|h|"
    r"yaml|yml|toml|json|md|txt|cfg|ini|sh)\b"
)

_MAX_PAIRS = 7  # Design §13: cap at 7 key-values per session (avoid bundle overflow)


def extract_kv(text: str) -> list[tuple[str, str]]:
    """Extract (key, value) pairs from text using rules (no LLM).

    Args:
        text: Natural language to analyze.

    Returns:
        ``[(key_name, value_text), ...]`` (at most 7).
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    lowered = text.lower()

    def _add(key: str, value: str) -> None:
        item = (key, value)
        if item not in seen:
            seen.add(item)
            pairs.append(item)

    # File paths
    for m in _FILE_RE.finditer(text):
        _add("K_FILE", m.group(0))

    # Programming languages (longer tokens first so 'c' doesn't mask 'c++')
    for token in sorted(_LANGUAGES, key=len, reverse=True):
        if re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", lowered):
            _add("K_LANGUAGE", _LANGUAGES[token])

    return pairs[:_MAX_PAIRS]


class Encoder:
    """Natural language → composite scent encoder.

    Args:
        embedder: Semantic embedder (the Ollama model, or the hashing embedder for tests).
        codebook: Key scent dictionary.
        semantic_weight: Weight of the semantic scent (default 0.6).
        structure_weight: Weight of the structure scent (default 0.4).
        auto_discovery: Whether to add unregistered keys to the codebook when seen.
    """

    def __init__(
        self,
        embedder: Embedder,
        codebook: Codebook,
        semantic_weight: float = 0.6,
        structure_weight: float = 0.4,
        auto_discovery: bool = True,
    ):
        if embedder.dim != codebook.dim:
            raise ValueError(
                f"Embedder dim ({embedder.dim}) and codebook dim ({codebook.dim}) differ."
            )
        self.embedder = embedder
        self.codebook = codebook
        self.semantic_weight = semantic_weight
        self.structure_weight = structure_weight
        self.auto_discovery = auto_discovery

    @property
    def dim(self) -> int:
        return self.codebook.dim

    def _structure_vector(self, pairs: list[tuple[str, str]]) -> np.ndarray | None:
        """Bundle key-value pairs into a structure scent via ``Σ bind(K, V)``."""
        bound: list[np.ndarray] = []
        for key_name, value_text in pairs:
            if key_name not in self.codebook:
                if not self.auto_discovery:
                    continue
                self.codebook.add_key(key_name)
            key_vec = self.codebook[key_name]
            value_vec = vsa.normalize(self.embedder.embed(value_text))
            bound.append(vsa.bind(key_vec, value_vec))
        if not bound:
            return None
        return vsa.bundle(bound)

    def encode(self, text: str) -> np.ndarray:
        """Encode natural language into a composite scent.

        Args:
            text: Natural language to encode.

        Returns:
            Unit-norm ``float32`` composite scent.
        """
        semantic = vsa.normalize(self.embedder.embed(text))
        structure = self._structure_vector(extract_kv(text))
        if structure is None:
            return semantic
        final = self.semantic_weight * semantic + self.structure_weight * structure
        return vsa.normalize(final)
