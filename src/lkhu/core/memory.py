"""Memory record definition. Design doc §6.3.

One memory = a composite scent + metadata + a natural-language shadow (audit_text).
``id`` uses a chronologically sortable UUIDv7.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

__all__ = ["Memory", "MemoryKind", "uuid7", "now_iso"]

# kind enum: turn | summary | merged | explicit
MemoryKind = str

# Default strength per kind
DEFAULT_STRENGTH: dict[str, float] = {
    "turn": 1.0,
    "explicit": 1.3,
    "summary": 1.2,
    "merged": 1.0,
}


def uuid7() -> str:
    """Generate a chronologically sortable UUIDv7 string (RFC 9562).

    The high 48 bits are the Unix time in milliseconds; the rest is random.
    """
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")  # 80-bit random
    rand_a = (rand >> 64) & 0xFFF  # 12 bits
    rand_b = rand & ((1 << 62) - 1)  # 62 bits
    value = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(uuid.UUID(int=value))


def now_iso() -> str:
    """Return the current time as an ISO 8601 (UTC) string."""
    return datetime.now(UTC).isoformat()


@dataclass
class Memory:
    """A memory unit in long-term storage.

    Attributes:
        vector: Composite scent (float32[dim], L2 normalized).
        strength: Current strength (0~1.5, subject to decay).
        kind: turn | summary | merged | explicit.
        audit_text: Natural-language shadow copy (not a primary search target, must be preserved).
        session_id: Identifier of the same conversation session.
        created_at / last_accessed_at: ISO timestamps.
        access_count: Cumulative recall count.
        source_ids: Original ids when consolidated/merged.
        id: UUIDv7.
        rowid: SQLite integer PK (used as the FAISS id). None when not yet stored.
        archived: Whether it has been archived.
    """

    vector: np.ndarray
    strength: float = 1.0
    kind: MemoryKind = "turn"
    audit_text: str = ""
    session_id: str = ""
    created_at: str = field(default_factory=now_iso)
    last_accessed_at: str = field(default_factory=now_iso)
    access_count: int = 0
    source_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=uuid7)
    rowid: int | None = None
    archived: bool = False

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32)

    @classmethod
    def make(
        cls,
        vector: np.ndarray,
        kind: MemoryKind = "turn",
        audit_text: str = "",
        session_id: str = "",
        strength: float | None = None,
        source_ids: list[str] | None = None,
    ) -> Memory:
        """Create a new Memory, applying the default strength for the kind."""
        return cls(
            vector=vector,
            kind=kind,
            audit_text=audit_text,
            session_id=session_id,
            strength=strength if strength is not None else DEFAULT_STRENGTH.get(kind, 1.0),
            source_ids=source_ids or [],
        )
