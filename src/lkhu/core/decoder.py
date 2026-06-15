"""3-Tier Decoder — scent to natural language. Design doc §7.3. The crux of token savings.

    Tier 1: audit_text excerpt (LLM 0)       — shows the short existing shadow copy as-is
    Tier 2: key unbind probe (LLM 0)         — unbind composite by core keys, match a value dict
    Tier 3: LLM fallback (80 tokens, <5%)    — brief LLM summary if injected, else excerpt

In real patterns ~70% finish at Tier1, ~25% at Tier2, and Tier3 stays under 5%.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lkhu.core import vsa
from lkhu.core.codebook import Codebook
from lkhu.core.encoder import Embedder, extract_kv
from lkhu.core.memory import Memory
from lkhu.core.metrics import estimate_tokens
from lkhu.core.recall import RecallResult

__all__ = ["Decoder", "DecodeOutput", "AuditVocab", "format_kv"]

# Core keys to probe in Tier2 (only those that exist are used)
DEFAULT_TOP_KEYS = [
    "K_TOPIC",
    "K_DECISION",
    "K_LANGUAGE",
    "K_FILE",
    "K_PREFERENCE",
    "K_PROJECT",
    "K_FACT",
    "K_VALUE",
]

# Tier3 LLM signature: (memories, max_tokens) -> summary string
LLMFunc = Callable[[list[Memory], int], str]


def format_kv(extracted: dict[str, str]) -> str:
    """Format a key-value dict as ``key=value, key=value``."""
    return ", ".join(f"{k}={v}" for k, v in extracted.items())


def _key_label(key: str) -> str:
    """Display label for a key (strips the ``K_`` prefix)."""
    return key[2:] if key.startswith("K_") else key


@dataclass
class DecodeOutput:
    """Result of decoding.

    Attributes:
        text: Natural language output.
        tier: Tier used (1/2/3).
        llm_used: Whether an LLM was actually called in Tier3.
    """

    text: str
    tier: int
    llm_used: bool = False


class AuditVocab:
    """Value dictionary (cleanup memory) to match unbind-probe results against.

    Built by embedding value phrases extracted from the audit_text of recalled
    memories. No LLM used.

    Args:
        embedder: Embedder for value phrases.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._entries: list[tuple[str, np.ndarray]] = []
        self._seen: set[str] = set()

    def add(self, text: str) -> None:
        """Add a single value phrase to the dictionary."""
        text = text.strip()
        if not text or text in self._seen:
            return
        self._seen.add(text)
        self._entries.append((text, vsa.normalize(self.embedder.embed(text))))

    def add_from_memories(self, memories: list[Memory]) -> None:
        """Extract key-value values from memories' audit_text and fill the dictionary."""
        for m in memories:
            if not m.audit_text:
                continue
            for _key, value in extract_kv(m.audit_text):
                self.add(value)

    def nearest(self, probe: np.ndarray, threshold: float) -> tuple[str, float] | None:
        """Return the entry nearest to probe (None if below threshold)."""
        best: tuple[str, float] | None = None
        for text, vec in self._entries:
            sim = vsa.cosine(probe, vec)
            if sim >= threshold and (best is None or sim > best[1]):
                best = (text, sim)
        return best


class Decoder:
    """3-tier decoder.

    Args:
        codebook: Key scent dictionary.
        embedder: Embedder for building the value dictionary.
        top_keys: List of keys to probe in Tier2.
        tier1_audit_max_len: Max allowed length for a Tier1 excerpt (default 150).
        tier2_threshold: Tier2 unbind match threshold (default 0.7).
        tier3_max_tokens: Token cap for the Tier3 summary (default 80).
        llm: Tier3 LLM function (falls back to excerpt if absent).
    """

    def __init__(
        self,
        codebook: Codebook,
        embedder: Embedder,
        top_keys: list[str] | None = None,
        tier1_audit_max_len: int = 150,
        tier2_threshold: float = 0.7,
        tier3_max_tokens: int = 80,
        llm: LLMFunc | None = None,
    ):
        self.codebook = codebook
        self.embedder = embedder
        self.top_keys = top_keys if top_keys is not None else DEFAULT_TOP_KEYS
        self.tier1_audit_max_len = tier1_audit_max_len
        self.tier2_threshold = tier2_threshold
        self.tier3_max_tokens = tier3_max_tokens
        self.llm = llm
        self.counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.llm_tokens_used: int = 0

    @property
    def total_decodes(self) -> int:
        return sum(self.counts.values())

    @property
    def tier3_ratio(self) -> float:
        """LLM fallback (Tier3) ratio. 0 when the denominator is 0."""
        total = self.total_decodes
        return self.counts[3] / total if total else 0.0

    def decode(self, result: RecallResult, vocab: AuditVocab | None = None) -> DecodeOutput:
        """Decode a composite scent and its memories into natural language.

        Args:
            result: Recall result.
            vocab: Tier2 value dictionary (built automatically from memories if absent).

        Returns:
            ``DecodeOutput``.
        """
        top = result.memories[:3]
        if not top:
            return DecodeOutput(text="", tier=1)

        # ----- Tier 1: audit excerpt -----
        audits = [m.audit_text for m in top]
        if audits and all(0 < len(a) < self.tier1_audit_max_len for a in audits):
            self.counts[1] += 1
            return DecodeOutput(text="\n".join(audits), tier=1)

        # ----- Tier 2: key unbind probe -----
        if vocab is None:
            vocab = AuditVocab(self.embedder)
            vocab.add_from_memories(result.memories)
        extracted: dict[str, str] = {}
        for key in self.top_keys:
            if key not in self.codebook:
                continue
            probe = vsa.unbind(result.composite, self.codebook[key])
            match = vocab.nearest(probe, self.tier2_threshold)
            if match:
                extracted[_key_label(key)] = match[0]
        if len(extracted) >= 2:
            self.counts[2] += 1
            return DecodeOutput(text=format_kv(extracted), tier=2)

        # ----- Tier 3: LLM fallback (or non-LLM excerpt) -----
        self.counts[3] += 1
        if self.llm is not None:
            text = self.llm(top, self.tier3_max_tokens)
            self.llm_tokens_used += estimate_tokens(text)
            return DecodeOutput(text=text, tier=3, llm_used=True)
        return DecodeOutput(text=self._extractive(top), tier=3, llm_used=False)

    def stats(self) -> dict[str, float | int]:
        """Decoding stats (for status display): per-tier counts, Tier3 ratio, LLM token estimate."""
        return {
            "tier1": self.counts[1],
            "tier2": self.counts[2],
            "tier3": self.counts[3],
            "total": self.total_decodes,
            "tier3_ratio": round(self.tier3_ratio, 4),
            "llm_tokens_used": self.llm_tokens_used,
        }

    def _extractive(self, memories: list[Memory]) -> str:
        """Excerpt summary built by cutting audit_text to the token budget, without an LLM."""
        words: list[str] = []
        for m in memories:
            words.extend(m.audit_text.split())
            if len(words) >= self.tier3_max_tokens:
                break
        return " ".join(words[: self.tier3_max_tokens])
