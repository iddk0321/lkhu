"""LongTermVault — permanent storage + search for individual scents. Design doc §4, §6.3.

Storage uses SQLite (metadata + scent BLOB); search uses FAISS (IndexFlatIP, cosine = inner
product). SQLite is treated as the single source of truth, and the FAISS index is rebuilt from
SQLite on open (to avoid search errors from a stale index file → "FAISS reproducibility").
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import faiss
import numpy as np

from lkhu.core.memory import Memory

__all__ = ["LongTermVault"]

# Pin faiss to a single thread to avoid OpenMP duplicate-runtime conflicts (together with the
# KMP setting in __init__).
faiss.omp_set_num_threads(1)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    rowid           INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT UNIQUE NOT NULL,
    vector          BLOB NOT NULL,
    strength        REAL NOT NULL,
    created_at      TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'turn',
    audit_text      TEXT NOT NULL DEFAULT '',
    source_ids      TEXT NOT NULL DEFAULT '[]',
    archived        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_archived ON memories(archived);
"""


def _synchronized(method):
    """Decorator that serializes method calls with the instance RLock (thread-safe)."""

    @functools.wraps(method)
    def wrapper(self: LongTermVault, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class LongTermVault:
    """Long-term memory store backed by SQLite + FAISS.

    Args:
        db_path: Path to the SQLite file.
        dim: Scent dimensionality.
    """

    def __init__(self, db_path: str | Path, dim: int):
        self.dim = dim
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # The FastMCP worker thread and the scheduler background thread share the same vault, so
        # we use check_same_thread=False + RLock to serialize (preventing concurrent SQLite/FAISS
        # access).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._rebuild_index()

    # ----- index -----

    @_synchronized
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from the non-archived scents in SQLite."""
        self._index.reset()
        rows = self._conn.execute(
            "SELECT rowid, vector FROM memories WHERE archived = 0"
        ).fetchall()
        if not rows:
            return
        ids = np.array([r["rowid"] for r in rows], dtype=np.int64)
        mat = np.stack([_from_blob(r["vector"]) for r in rows]).astype(np.float32)
        self._index.add_with_ids(mat, ids)

    # ----- write -----

    @_synchronized
    def insert(self, memory: Memory) -> Memory:
        """Store a memory and add it to FAISS. Returns the Memory with rowid filled in."""
        cur = self._conn.execute(
            """INSERT INTO memories
               (id, vector, strength, created_at, last_accessed_at, access_count,
                session_id, kind, audit_text, source_ids, archived)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory.id,
                _to_blob(memory.vector),
                float(memory.strength),
                memory.created_at,
                memory.last_accessed_at,
                int(memory.access_count),
                memory.session_id,
                memory.kind,
                memory.audit_text,
                json.dumps(memory.source_ids),
                int(memory.archived),
            ),
        )
        self._conn.commit()
        memory.rowid = int(cur.lastrowid)
        if not memory.archived:
            vec = np.ascontiguousarray(memory.vector, dtype=np.float32)[None, :]
            self._index.add_with_ids(vec, np.array([memory.rowid], dtype=np.int64))
        return memory

    # ----- read -----

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            vector=_from_blob(row["vector"]),
            strength=row["strength"],
            kind=row["kind"],
            audit_text=row["audit_text"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            source_ids=json.loads(row["source_ids"]),
            id=row["id"],
            rowid=row["rowid"],
            archived=bool(row["archived"]),
        )

    @_synchronized
    def get(self, memory_id: str) -> Memory | None:
        """Look up a memory by id."""
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    @_synchronized
    def _get_by_rowid(self, rowid: int) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE rowid = ?", (int(rowid),)).fetchone()
        return self._row_to_memory(row) if row else None

    @_synchronized
    def faiss_search(self, query: np.ndarray, k: int = 5) -> list[tuple[Memory, float]]:
        """Return the top-k memories closest to the query scent as (Memory, similarity)."""
        if self._index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query, dtype=np.float32)[None, :]
        k = min(k, self._index.ntotal)
        sims, ids = self._index.search(q, k)
        out: list[tuple[Memory, float]] = []
        for sim, rowid in zip(sims[0], ids[0], strict=False):
            if rowid == -1:
                continue
            mem = self._get_by_rowid(int(rowid))
            if mem is not None:
                out.append((mem, float(sim)))
        return out

    @_synchronized
    def by_session(self, session_id: str, include_archived: bool = False) -> list[Memory]:
        """Return the memories belonging to a session id."""
        sql = "SELECT * FROM memories WHERE session_id = ?"
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY created_at"
        rows = self._conn.execute(sql, (session_id,)).fetchall()
        return [self._row_to_memory(r) for r in rows]

    @_synchronized
    def all(self, include_archived: bool = False) -> list[Memory]:
        """Return all memories."""
        sql = "SELECT * FROM memories"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY created_at"
        return [self._row_to_memory(r) for r in self._conn.execute(sql).fetchall()]

    @_synchronized
    def count(self, include_archived: bool = False) -> int:
        """Number of memories (non-archived only by default)."""
        sql = "SELECT COUNT(*) AS c FROM memories"
        if not include_archived:
            sql += " WHERE archived = 0"
        return int(self._conn.execute(sql).fetchone()["c"])

    @_synchronized
    def recent_sessions(self, days: int) -> list[str]:
        """List of session ids for memories created within the last ``days`` (deduplicated)."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """SELECT DISTINCT session_id FROM memories
               WHERE archived = 0 AND session_id != '' AND created_at >= ?""",
            (cutoff,),
        ).fetchall()
        return [r["session_id"] for r in rows]

    # ----- update -----

    @_synchronized
    def batch_update(self, memories: list[Memory]) -> None:
        """Bulk-update strength/last_accessed_at/access_count."""
        self._conn.executemany(
            """UPDATE memories
               SET strength = ?, last_accessed_at = ?, access_count = ?
               WHERE id = ?""",
            [(float(m.strength), m.last_accessed_at, int(m.access_count), m.id) for m in memories],
        )
        self._conn.commit()

    @_synchronized
    def update_vectors(self, pairs: list[tuple[str, np.ndarray]]) -> None:
        """Replace the scent vectors of the given memory ids, then rebuild the index.

        Used by re-embedding (e.g. after switching the embedding model): metadata, strength, and
        audit_text are untouched — only the vectors change.
        """
        self._conn.executemany(
            "UPDATE memories SET vector = ? WHERE id = ?",
            [(_to_blob(vec), mid) for mid, vec in pairs],
        )
        self._conn.commit()
        self._rebuild_index()

    @_synchronized
    def multiply_strength(self, factor: float) -> None:
        """Multiply the strength of every non-archived memory by factor (daily decay)."""
        self._conn.execute(
            "UPDATE memories SET strength = strength * ? WHERE archived = 0", (float(factor),)
        )
        self._conn.commit()

    @_synchronized
    def set_created_at(self, memory_id: str, created_at: str) -> None:
        """Set the creation timestamp (for tests/migrations)."""
        self._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?", (created_at, memory_id)
        )
        self._conn.commit()

    @_synchronized
    def select_weak(self, strength_below: float, min_age_days: int) -> list[Memory]:
        """Select low-strength, old memories (cleanse candidates)."""
        cutoff = (datetime.now(UTC) - timedelta(days=min_age_days)).isoformat()
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE archived = 0 AND strength < ? AND created_at < ?""",
            (float(strength_below), cutoff),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ----- archive / delete -----

    @_synchronized
    def archive(self, memory_ids: list[str]) -> None:
        """Archive memories and remove them from the search index (audit_text is preserved)."""
        for mid in memory_ids:
            self._conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (mid,))
        self._conn.commit()
        self._rebuild_index()

    @_synchronized
    def delete(self, memory_id: str) -> None:
        """Delete a memory completely."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        self._rebuild_index()

    @_synchronized
    def find_duplicate_pairs(self, threshold: float) -> list[tuple[Memory, Memory]]:
        """Find (a, b) memory pairs whose similarity exceeds the threshold (for cleanse)."""
        mems = self.all()
        pairs: list[tuple[Memory, Memory]] = []
        from lkhu.core import vsa

        for i, a in enumerate(mems):
            for b in mems[i + 1 :]:
                if vsa.cosine(a.vector, b.vector) > threshold:
                    pairs.append((a, b))
        return pairs

    # ----- persist / close -----

    def persist_index(self, faiss_path: str | Path) -> None:
        """Write the FAISS index to disk (for external inspection/warm-up)."""
        faiss.write_index(self._index, str(faiss_path))

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
