"""AuditLog — natural-language shadow log (JSONL, split by month). Design doc §4, §10.2.

audit_text is not the primary search target (Hard Rule 4), but it is preserved for debugging,
recovery, and user visibility. File layout: ``audit/YYYY-MM/DD.jsonl`` (month directory, day file).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from lkhu.core.memory import now_iso

__all__ = ["AuditLog"]


class AuditLog:
    """Append-only natural-language shadow log.

    Args:
        audit_dir: Root directory for the log.
    """

    def __init__(self, audit_dir: str | Path):
        self.audit_dir = Path(audit_dir)

    def _file_for(self, created_at: str) -> Path:
        """Build the ``YYYY-MM/DD.jsonl`` path from an ISO timestamp."""
        date_part = created_at[:10]  # YYYY-MM-DD
        year_month = date_part[:7]  # YYYY-MM
        day = date_part[8:10]  # DD
        return self.audit_dir / year_month / f"{day}.jsonl"

    def append(self, record: dict) -> None:
        """Append a single record as JSONL.

        If ``created_at`` is missing, fill it with the current time.
        """
        record = dict(record)
        record.setdefault("created_at", now_iso())
        path = self._file_for(record["created_at"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> Iterator[dict]:
        """Iterate over all records in chronological (file/line) order."""
        if not self.audit_dir.exists():
            return
        for path in sorted(self.audit_dir.glob("*/*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def by_session(self, session_id: str) -> list[dict]:
        """Collect only the records for a given session (supports recall_session in the design)."""
        return [r for r in self.read_all() if r.get("session_id") == session_id]

    def export_jsonl(self, out_path: str | Path) -> int:
        """Export all records to a single JSONL file. Returns the number of records exported."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for record in self.read_all():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count
