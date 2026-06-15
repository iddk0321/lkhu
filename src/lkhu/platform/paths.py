"""Single gateway that provides OS-specific paths in one place. Design doc §5, §10.2.

⚠️ No hardcoded paths (CLAUDE.md hard rule 1). OS-specific paths like ``~/Library/...``,
``C:\\...`` etc. are obtained only through this module's ``platformdirs`` wrapper. Business
logic (core/) never builds paths directly.

The ``LKHU_DATA`` environment variable can force the data root (for testing/portability).
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

__all__ = ["LkhuPaths"]

APP_NAME = "lkhu"


class LkhuPaths:
    """Single source of truth for all paths used by lkhu.

    Args:
        base: Explicit data root (for testing/portability). If ``None``, uses the
            ``LKHU_DATA`` environment variable, and if that is also unset, the
            platformdirs default path.
        app_name: Application name (directory name).
    """

    def __init__(self, base: str | Path | None = None, app_name: str = APP_NAME):
        self.app_name = app_name
        if base is not None:
            self._data_dir = Path(base)
        elif os.environ.get("LKHU_DATA"):
            self._data_dir = Path(os.environ["LKHU_DATA"])
        else:
            self._data_dir = Path(platformdirs.user_data_dir(app_name, appauthor=False))

    # ----- top-level directories -----

    @property
    def data_dir(self) -> Path:
        """Root of all runtime artifacts (codebook/db/faiss/audit/backups/logs)."""
        return self._data_dir

    @property
    def config_dir(self) -> Path:
        """Configuration file directory (OS convention)."""
        return Path(platformdirs.user_config_dir(self.app_name, appauthor=False))

    @property
    def cache_dir(self) -> Path:
        """Cache directory (OS convention)."""
        return Path(platformdirs.user_cache_dir(self.app_name, appauthor=False))

    # ----- sub-paths (relative to data_dir) -----

    @property
    def codebook_path(self) -> Path:
        """Key scent dictionary (permanently fixed)."""
        return self.data_dir / "codebook.npy"

    @property
    def codebook_backup_path(self) -> Path:
        """Secondary codebook backup (inside the data directory)."""
        return self.data_dir / "codebook.backup.npy"

    @property
    def documents_backup_path(self) -> Path:
        """Tertiary codebook backup (user's Documents)."""
        return Path.home() / "Documents" / "lkhu_codebook.backup.npy"

    def codebook_backup_targets(self) -> list[Path]:
        """Secondary/tertiary backup paths passed to ``Codebook.save(backups=...)``."""
        return [self.codebook_backup_path, self.documents_backup_path]

    @property
    def db_path(self) -> Path:
        """SQLite meta/audit database."""
        return self.data_dir / "memories.db"

    @property
    def faiss_path(self) -> Path:
        """FAISS vector index."""
        return self.data_dir / "vectors.faiss"

    @property
    def short_term_path(self) -> Path:
        """Short-term accumulated scent."""
        return self.data_dir / "short_term.npy"

    @property
    def audit_dir(self) -> Path:
        """Natural-language shadow log (split by month)."""
        return self.data_dir / "audit"

    @property
    def backups_dir(self) -> Path:
        """Daily/weekly/monthly backup root."""
        return self.data_dir / "backups"

    @property
    def logs_dir(self) -> Path:
        """Server/consolidation/cleanse logs."""
        return self.data_dir / "logs"

    @property
    def stats_path(self) -> Path:
        """Statistics snapshot."""
        return self.data_dir / "stats.json"

    @property
    def lifecycle_state_path(self) -> Path:
        """Last-run timestamps for the daily/weekly lifecycle (enables catch-up on startup)."""
        return self.data_dir / "lifecycle_state.json"

    @property
    def config_path(self) -> Path:
        """User configuration file."""
        return self.config_dir / "config.yaml"

    # ----- preparation -----

    def ensure(self) -> None:
        """Create the required directories (ignored if they already exist)."""
        for d in (
            self.data_dir,
            self.audit_dir,
            self.backups_dir,
            self.backups_dir / "daily",
            self.backups_dir / "weekly",
            self.backups_dir / "monthly",
            self.logs_dir,
            self.config_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
