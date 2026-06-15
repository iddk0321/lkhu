"""Token usage measurement utility. Design doc §7.3 (exposed in lkhu status).

Uses a lightweight estimate instead of an exact tokenizer (roughly 4 chars ≈ 1 token). The goal is
not absolute accuracy but to show the user "how much natural-language / LLM calling was saved by
scent processing".
"""

from __future__ import annotations

__all__ = ["estimate_tokens"]


def estimate_tokens(text: str) -> int:
    """Estimate the approximate token count of a string (≈ 4 chars/token, at least word count)."""
    if not text:
        return 0
    char_estimate = max(1, len(text) // 4)
    word_estimate = len(text.split())
    return max(char_estimate, word_estimate)
