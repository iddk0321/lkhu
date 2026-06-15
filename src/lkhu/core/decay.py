"""DecayEngine — forgetting curve. Design doc §2.1(4), §7.5.

Each day, multiply long-term scent strength by ×0.99 and short-term accumulated
scent by ×0.7. The ×1.05 reinforcement on recall is handled by RecallEngine. The
result traces an Ebbinghaus-style exponential decay curve.
"""

from __future__ import annotations

from lkhu.core.long_term import LongTermVault
from lkhu.core.short_term import ShortTermBundle

__all__ = ["DecayEngine"]


class DecayEngine:
    """Daily strength decay engine.

    Args:
        vault: Long-term store.
        short_term: Short-term accumulated scent.
        daily_rate: Long-term daily decay rate (default 0.99).
        short_daily: Short-term daily decay rate (default 0.7).
    """

    def __init__(
        self,
        vault: LongTermVault,
        short_term: ShortTermBundle,
        daily_rate: float = 0.99,
        short_daily: float = 0.7,
    ):
        self.vault = vault
        self.short_term = short_term
        self.daily_rate = daily_rate
        self.short_daily = short_daily

    def run_daily(self) -> dict[str, float | int]:
        """Apply one day's worth of decay.

        Returns:
            An application summary (number of affected memories, decay rates used).
        """
        count = self.vault.count()
        self.vault.multiply_strength(self.daily_rate)
        self.short_term.decay(self.short_daily)
        return {
            "decayed_memories": count,
            "daily_rate": self.daily_rate,
            "short_daily": self.short_daily,
        }
