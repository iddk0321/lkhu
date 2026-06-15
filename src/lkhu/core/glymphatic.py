"""GlymphaticCleaner — weekly cleanse. Design doc §2.1(5), §7.6.

Corresponds to the glymphatic system. (1) Merges nearly identical scents, and
(2) archives weak, old scents. Originals are not deleted but archived to preserve
audit_text (hard rule 4).
"""

from __future__ import annotations

from lkhu.core import vsa
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory

__all__ = ["GlymphaticCleaner"]


class GlymphaticCleaner:
    """Duplicate merging + weak scent cleanup.

    Args:
        vault: Long-term store.
        duplicate_threshold: Merge similarity threshold (default 0.95).
        weak_strength: Upper strength bound for cleanup targets (default 0.1).
        weak_min_age_days: Minimum age for cleanup targets (default 30 days).
    """

    def __init__(
        self,
        vault: LongTermVault,
        duplicate_threshold: float = 0.95,
        weak_strength: float = 0.1,
        weak_min_age_days: int = 30,
    ):
        self.vault = vault
        self.duplicate_threshold = duplicate_threshold
        self.weak_strength = weak_strength
        self.weak_min_age_days = weak_min_age_days

    def cleanse(self) -> dict[str, int]:
        """Run the cleanse and return a statistics report.

        Returns:
            ``{"merged": n, "archived_weak": n}``.
        """
        merged = self._merge_duplicates()
        archived_weak = self._archive_weak()
        return {"merged": merged, "archived_weak": archived_weak}

    def _merge_duplicates(self) -> int:
        """Merge pairs above the similarity threshold (each original used once)."""
        pairs = self.vault.find_duplicate_pairs(self.duplicate_threshold)
        used: set[str] = set()
        merged_count = 0
        for a, b in pairs:
            if a.id in used or b.id in used:
                continue
            merged_vec = vsa.normalize(a.vector + b.vector)
            merged = Memory.make(
                vector=merged_vec,
                kind="merged",
                strength=max(a.strength, b.strength),
                audit_text=f"{a.audit_text} / {b.audit_text}",
                source_ids=[a.id, b.id],
            )
            self.vault.insert(merged)
            self.vault.archive([a.id, b.id])
            used.update({a.id, b.id})
            merged_count += 1
        return merged_count

    def _archive_weak(self) -> int:
        """Archive weak, old scents (preserving audit)."""
        weak = self.vault.select_weak(
            strength_below=self.weak_strength,
            min_age_days=self.weak_min_age_days,
        )
        if weak:
            self.vault.archive([m.id for m in weak])
        return len(weak)
