"""WorkingMemory — current-session turn buffer (volatile). Design doc §2.1, §4.

Corresponds to prefrontal working memory. Keeps only the most recent ~50 turns in RAM; older
turns are pushed out. After a period of idle time it becomes a flush candidate (moved to the
short-term store).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable

__all__ = ["WorkingMemory"]


class WorkingMemory:
    """Session RAM turn buffer.

    Args:
        max_turns: Maximum number of turns to keep at once (default 50).
    """

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self._turns: deque[dict] = deque(maxlen=max_turns)
        self._last_activity: float | None = None

    def add(self, turn: dict, at: float | None = None) -> None:
        """Add a turn (overflow automatically pushes out the oldest first).

        Args:
            turn: Turn data (an arbitrary dict with text/vector/ts etc.).
            at: Activity timestamp (default now). Injectable in tests.
        """
        self._turns.append(turn)
        self._last_activity = at if at is not None else time.time()

    @property
    def turns(self) -> list[dict]:
        """List of turns in the current buffer (oldest → newest)."""
        return list(self._turns)

    def recent(self, n: int) -> list[dict]:
        """The most recent n turns (oldest → newest order preserved)."""
        items = list(self._turns)
        return items[-n:] if n > 0 else []

    def extend(self, turns: Iterable[dict], at: float | None = None) -> None:
        """Add several turns at once."""
        for t in turns:
            self.add(t, at=at)

    def should_flush(self, idle_minutes: int, now: float | None = None) -> bool:
        """Whether at least idle_minutes have passed since the last activity."""
        if self._last_activity is None or not self._turns:
            return False
        now = now if now is not None else time.time()
        return (now - self._last_activity) >= idle_minutes * 60

    def clear(self) -> None:
        """Empty the buffer."""
        self._turns.clear()
        self._last_activity = None

    def __len__(self) -> int:
        return len(self._turns)
