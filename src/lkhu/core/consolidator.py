"""Consolidator — nightly consolidation (short-term → long-term). Design doc §2.1(2), §7.4.

Corresponds to hippocampal-cortical consolidation. Bundles the scents of a single
session into one 'summary' scent via a strength-weighted sum.
★ Key point: just add them up. Zero LLM calls.
"""

from __future__ import annotations

from lkhu.core import vsa
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory

__all__ = ["Consolidator"]

# Source kinds eligible for consolidation
_SOURCE_KINDS = {"turn", "explicit"}


class Consolidator:
    """Per-session scent weighted-sum consolidation.

    Args:
        vault: Long-term store.
        min_session_size: Minimum session size to consolidate (default 3).
        summary_strength: Summary scent strength (default 1.2, slightly stronger).
    """

    def __init__(
        self,
        vault: LongTermVault,
        min_session_size: int = 3,
        summary_strength: float = 1.2,
    ):
        self.vault = vault
        self.min_session_size = min_session_size
        self.summary_strength = summary_strength

    def consolidate(self, days: int = 2) -> list[Memory]:
        """Consolidate recent sessions.

        Args:
            days: Number of recent days to consider for consolidation.

        Returns:
            The list of generated summary Memory objects.
        """
        created: list[Memory] = []
        for session_id in self.vault.recent_sessions(days):
            members = self.vault.by_session(session_id)
            # Skip if a summary already exists (idempotent)
            if any(m.kind == "summary" for m in members):
                continue
            sources = [m for m in members if m.kind in _SOURCE_KINDS]
            if len(sources) < self.min_session_size:
                continue

            summary_vec = vsa.bundle(
                [m.vector * m.strength for m in sources],
                normalized=True,
            )
            summary = Memory.make(
                vector=summary_vec,
                kind="summary",
                strength=self.summary_strength,
                session_id=session_id,
                audit_text=f"[session {session_id}: merged {len(sources)} items]",
                source_ids=[m.id for m in sources],
            )
            self.vault.insert(summary)
            created.append(summary)
        return created
