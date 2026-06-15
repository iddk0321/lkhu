"""LkhuEngine — facade tying all components together. Shared by CLI and MCP server.

Assembles codebook, encoder, vault, recall, decoder, lifecycle, short_term, and audit in
one place, and provides high-level operations like remember/recall/forget/recall_session/
status/export.

Design principle: the codebook is fixed at init time and never changed at runtime
(auto_discovery=False). Every key emitted by extract_kv is part of the initial key set,
so no new key derivation is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lkhu.config.loader import load_config
from lkhu.core.audit import AuditLog
from lkhu.core.codebook import Codebook
from lkhu.core.consolidator import Consolidator
from lkhu.core.decay import DecayEngine
from lkhu.core.decoder import Decoder
from lkhu.core.encoder import Embedder, Encoder
from lkhu.core.glymphatic import GlymphaticCleaner
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory, now_iso
from lkhu.core.recall import RecallEngine
from lkhu.core.short_term import ShortTermBundle
from lkhu.platform.paths import LkhuPaths

__all__ = ["LkhuEngine", "initialize"]


def initialize(
    base: str | Path | None = None,
    register_mcp: bool = True,
    mcp_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the data directory and codebook, and register Claude config (lkhu init).

    If the codebook already exists it is never regenerated (sacrosanct). Makes triple backups.

    Args:
        base: Data root override (for testing).
        register_mcp: Whether to register into Claude config.
        mcp_config_path: Claude config path override (for testing).

    Returns:
        Initialization summary dictionary.
    """
    from lkhu.platform import mcp_config as mcpc

    paths = LkhuPaths(base=base)
    paths.ensure()
    config = load_config(paths.config_path)

    codebook_created = False
    if not Codebook.is_initialized(paths.codebook_path):
        cb = Codebook.generate(
            config["codebook"]["initial_keys"],
            dim=config["encoder"].get("dim", 1024),
            seed=config["codebook"].get("random_seed"),
        )
        cb.save(paths.codebook_path, backups=paths.codebook_backup_targets())
        codebook_created = True

    registered = False
    if register_mcp:
        mcpc.register(mcp_config_path)
        registered = True

    return {
        "data_dir": str(paths.data_dir),
        "codebook_created": codebook_created,
        "codebook_path": str(paths.codebook_path),
        "mcp_registered": registered,
    }


class LkhuEngine:
    """lkhu memory system facade."""

    def __init__(
        self,
        paths: LkhuPaths,
        config: dict[str, Any],
        codebook: Codebook,
        embedder: Embedder,
    ):
        self.paths = paths
        self.config = config
        self.codebook = codebook
        self.embedder = embedder

        enc_cfg = config["encoder"]
        rec_cfg = config["recall"]
        dec_cfg = config["decoder"]

        self.encoder = Encoder(
            embedder=embedder,
            codebook=codebook,
            semantic_weight=enc_cfg.get("semantic_weight", 0.6),
            structure_weight=enc_cfg.get("structure_weight", 0.4),
            auto_discovery=False,  # guarantees codebook immutability
        )
        self.vault = LongTermVault(paths.db_path, dim=codebook.dim)
        self.short_term = ShortTermBundle.load(paths.short_term_path, dim=codebook.dim)
        self.audit = AuditLog(paths.audit_dir)
        weights = rec_cfg.get("rerank_weights", {})
        self.recall_engine = RecallEngine(
            vault=self.vault,
            encoder=self.encoder,
            sim_weight=weights.get("similarity", 0.6),
            strength_weight=weights.get("strength", 0.2),
            recency_weight=weights.get("recency", 0.2),
            candidate_multiplier=rec_cfg.get("candidate_multiplier", 3),
            recall_boost=config["long_term"].get("recall_boost", 1.05),
            max_strength=config["long_term"].get("max_strength", 1.5),
            reinforce_sim_threshold=rec_cfg.get("reinforce_sim_threshold", 0.0),
            min_similarity=rec_cfg.get("min_similarity", 0.0),
        )
        self.dedup_threshold = config["long_term"].get("dedup_threshold", 0.95)
        self.dedup_reinforce = config["long_term"].get("dedup_reinforce", 1.02)
        self.max_strength = config["long_term"].get("max_strength", 1.5)
        self.decoder = Decoder(
            codebook=codebook,
            embedder=embedder,
            tier1_audit_max_len=dec_cfg.get("tier1_audit_max_len", 150),
            tier2_threshold=dec_cfg.get("tier2_unbind_threshold", 0.7),
            tier3_max_tokens=dec_cfg.get("tier3_llm_max_tokens", 80),
        )

    # ----- Construction -----

    @classmethod
    def open(cls, embedder: Embedder, base: str | Path | None = None) -> LkhuEngine:
        """Open the engine from an existing data directory (codebook must exist).

        Args:
            embedder: Semantic embedder (OllamaEmbedder or a test one).
            base: Data root override (for testing).

        Raises:
            FileNotFoundError: When the codebook is missing (run ``lkhu init`` first).
        """
        paths = LkhuPaths(base=base)
        if not Codebook.is_initialized(paths.codebook_path):
            raise FileNotFoundError(
                f"codebook not found: {paths.codebook_path}. Run 'lkhu init' first."
            )
        config = load_config(paths.config_path)
        codebook = Codebook.load(paths.codebook_path)
        if embedder.dim != codebook.dim:
            raise ValueError(
                f"embedder dim ({embedder.dim}) differs from codebook dim ({codebook.dim})."
            )
        return cls(paths=paths, config=config, codebook=codebook, embedder=embedder)

    # ----- High-level operations -----

    def _store(
        self,
        content: str,
        kind: str,
        session_id: str,
        strength: float | None,
        vec: np.ndarray | None = None,
    ) -> Memory:
        if vec is None:
            vec = self.encoder.encode(content)
        mem = Memory.make(
            vector=vec,
            kind=kind,
            audit_text=content,
            session_id=session_id,
            strength=strength,
        )
        self.vault.insert(mem)
        self.short_term.add(vec)
        self.audit.append(
            {
                "id": mem.id,
                "session_id": session_id,
                "kind": kind,
                "audit_text": content,
                "created_at": mem.created_at,
            }
        )
        return mem

    def remember(self, content: str, kind: str = "fact", session_id: str = "") -> Memory:
        """Store an explicit memory strongly (design §8.1 remember)."""
        return self._store(content, kind="explicit", session_id=session_id, strength=1.3)

    def observe(self, content: str, session_id: str = "", strength: float | None = None) -> Memory:
        """Store a conversation turn, deduplicating against near-identical recent memories.

        Auto-capture hooks fire on every prompt/turn, so the same statement is often observed
        repeatedly. Instead of inserting a near-duplicate (cosine ≥ ``dedup_threshold``), the
        existing memory is lightly reinforced and returned — this keeps the store from filling
        with copies of the same line and is fully language-agnostic (it compares scents).
        """
        vec = self.encoder.encode(content)
        hits = self.vault.faiss_search(vec, k=1)
        if hits:
            existing, sim = hits[0]
            # Dedup only within the SAME session and only against other turns. Cross-session
            # repeats are kept as distinct rows so each session still has the members its
            # consolidation needs; reinforcing a curated explicit/summary from a casual turn is
            # likewise avoided.
            if (
                sim >= self.dedup_threshold
                and existing.kind == "turn"
                and existing.session_id == session_id
            ):
                # Don't add a duplicate vector row, but still record the observation in the
                # fast-decaying short-term bundle and the audit log (hard rule 4: the
                # natural-language shadow is always preserved) and reinforce the existing row.
                existing.access_count += 1
                existing.last_accessed_at = now_iso()
                existing.strength = min(self.max_strength, existing.strength * self.dedup_reinforce)
                self.vault.batch_update([existing])
                self.short_term.add(vec)
                self.audit.append(
                    {
                        "id": existing.id,
                        "session_id": session_id,
                        "kind": "turn",
                        "audit_text": content,
                        "created_at": now_iso(),
                        "deduped": True,
                    }
                )
                return existing
        return self._store(content, kind="turn", session_id=session_id, strength=strength, vec=vec)

    def recall(self, query: str, k: int = 5) -> dict[str, Any]:
        """Search and decode relevant memories, returning natural language and sources
        (design §8.1 recall)."""
        result = self.recall_engine.recall(query, k=k)
        output = self.decoder.decode(result)
        return {
            "text": output.text,
            "tier": output.tier,
            "llm_used": output.llm_used,
            "sources": [
                {"id": m.id, "audit_text": m.audit_text, "strength": round(m.strength, 3)}
                for m in result.memories
            ],
        }

    def forget(self, query: str, confirm: bool, k: int = 5) -> dict[str, Any]:
        """Archive memories matching the query (audit is preserved, design §8.1 forget)."""
        if not confirm:
            return {"archived": 0, "confirmed": False}
        result = self.recall_engine.recall(query, k=k)
        ids = [m.id for m in result.memories]
        if ids:
            self.vault.archive(ids)
        return {"archived": len(ids), "confirmed": True, "ids": ids}

    def recall_session(self, session_id: str) -> str:
        """Restore the full audit for a specific session (design §8.1 recall_session)."""
        records = self.audit.by_session(session_id)
        return "\n".join(r.get("audit_text", "") for r in records)

    def status(self) -> dict[str, Any]:
        """System statistics (design §8.1 status)."""
        mems = self.vault.all()
        strengths = [m.strength for m in mems]
        kinds: dict[str, int] = {}
        for m in mems:
            kinds[m.kind] = kinds.get(m.kind, 0) + 1
        return {
            "total_memories": len(mems),
            "archived": self.vault.count(include_archived=True) - len(mems),
            "kinds": kinds,
            "strength_avg": round(float(np.mean(strengths)), 3) if strengths else 0.0,
            "strength_max": round(float(np.max(strengths)), 3) if strengths else 0.0,
            "codebook_keys": len(self.codebook),
            "dim": self.codebook.dim,
            "decoder": self.decoder.stats(),
            "short_term_norm": round(float(np.linalg.norm(self.short_term.raw)), 4),
        }

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Top N memories by strength and recency (for session-start context injection).

        Args:
            n: Number to fetch.

        Returns:
            ``[{id, audit_text, strength, kind, created_at}, ...]`` (by strength, newest first).
        """
        mems = self.vault.all()
        mems.sort(key=lambda m: (m.strength, m.created_at), reverse=True)
        return [
            {
                "id": m.id,
                "audit_text": m.audit_text,
                "strength": round(m.strength, 3),
                "kind": m.kind,
                "created_at": m.created_at,
            }
            for m in mems[:n]
        ]

    def dump(self, include_archived: bool = True) -> list[dict[str, Any]]:
        """Return all memories expanded with metadata (for the dashboard)."""
        return [
            {
                "id": m.id,
                "audit_text": m.audit_text,
                "strength": round(m.strength, 3),
                "kind": m.kind,
                "created_at": m.created_at,
                "last_accessed_at": m.last_accessed_at,
                "access_count": m.access_count,
                "session_id": m.session_id,
                "archived": m.archived,
                "source_ids": m.source_ids,
            }
            for m in self.vault.all(include_archived=include_archived)
        ]

    def dashboard_stats(self) -> dict[str, Any]:
        """Dashboard aggregation: kind/strength/age distributions, consolidation status,
        lifecycle settings."""
        from datetime import UTC, datetime

        all_mems = self.vault.all(include_archived=True)
        active = [m for m in all_mems if not m.archived]
        archived = [m for m in all_mems if m.archived]

        kinds: dict[str, int] = {}
        for m in active:
            kinds[m.kind] = kinds.get(m.kind, 0) + 1

        # Strength distribution
        strength_buckets = {
            "0–0.2 (near extinction)": 0,
            "0.2–0.5 (weak)": 0,
            "0.5–1.0 (normal)": 0,
            "1.0–1.5 (strong)": 0,
        }
        for m in active:
            s = m.strength
            if s < 0.2:
                strength_buckets["0–0.2 (near extinction)"] += 1
            elif s < 0.5:
                strength_buckets["0.2–0.5 (weak)"] += 1
            elif s < 1.0:
                strength_buckets["0.5–1.0 (normal)"] += 1
            else:
                strength_buckets["1.0–1.5 (strong)"] += 1

        # Age distribution
        now = datetime.now(UTC)
        age_buckets = {"today": 0, "this week": 0, "within a month": 0, "over a month": 0}
        for m in active:
            try:
                age_days = (now - datetime.fromisoformat(m.created_at)).days
            except (ValueError, TypeError):
                age_days = 0
            if age_days < 1:
                age_buckets["today"] += 1
            elif age_days < 7:
                age_buckets["this week"] += 1
            elif age_days < 30:
                age_buckets["within a month"] += 1
            else:
                age_buckets["over a month"] += 1

        summaries = [m for m in active if m.kind == "summary"]
        consolidated_sources = sum(len(m.source_ids) for m in summaries)

        lt = self.config["long_term"]
        cons = self.config["consolidation"]
        clean = self.config["cleansing"]
        return {
            "total_active": len(active),
            "total_archived": len(archived),
            "kinds": kinds,
            "strength_buckets": strength_buckets,
            "age_buckets": age_buckets,
            "summaries": len(summaries),
            "consolidated_sources": consolidated_sources,
            "lifecycle": {
                "daily_decay": lt.get("daily_decay", 0.99),
                "recall_boost": lt.get("recall_boost", 1.05),
                "max_strength": lt.get("max_strength", 1.5),
                "consolidation_cron": cons.get("schedule_cron", "0 3 * * *"),
                "min_session_size": cons.get("min_session_size", 3),
                "cleansing_cron": clean.get("schedule_cron", "30 3 * * 0"),
                "weak_strength": clean.get("weak_strength", 0.1),
                "weak_min_age_days": clean.get("weak_min_age_days", 30),
                "duplicate_threshold": clean.get("duplicate_threshold", 0.95),
            },
        }

    def export(self, out_path: str | Path) -> int:
        """Export audit data as JSONL (design §8.1 export). Returns the record count."""
        return self.audit.export_jsonl(out_path)

    # ----- Lifecycle (for the scheduler) -----

    def _load_lifecycle_state(self) -> dict[str, str]:
        """Read the last-run timestamps ({} if absent/corrupt)."""
        import json

        path = self.paths.lifecycle_state_path
        try:
            return json.loads(path.read_text("utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def _save_lifecycle_state(self, state: dict[str, str]) -> None:
        import json
        import os

        path = self.paths.lifecycle_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state), "utf-8")
        os.replace(tmp, path)  # atomic — a crash mid-write can't corrupt the state

    def run_daily(self) -> dict[str, Any]:
        """Daily job: decay + consolidation. Records its run time for catch-up."""
        decay = DecayEngine(
            vault=self.vault,
            short_term=self.short_term,
            daily_rate=self.config["long_term"].get("daily_decay", 0.99),
            short_daily=self.config["short_term"].get("daily_decay", 0.7),
        )
        decay_report = decay.run_daily()
        self.short_term.save(self.paths.short_term_path)
        cons = Consolidator(
            vault=self.vault,
            min_session_size=self.config["consolidation"].get("min_session_size", 3),
        )
        created = cons.consolidate(days=2)
        state = self._load_lifecycle_state()
        state["last_daily"] = now_iso()
        self._save_lifecycle_state(state)
        return {"decay": decay_report, "consolidated": len(created)}

    def run_weekly(self) -> dict[str, int]:
        """Weekly job: glymphatic cleansing. Records its run time for catch-up."""
        cl_cfg = self.config["cleansing"]
        cleaner = GlymphaticCleaner(
            vault=self.vault,
            duplicate_threshold=cl_cfg.get("duplicate_threshold", 0.95),
            weak_strength=cl_cfg.get("weak_strength", 0.1),
            weak_min_age_days=cl_cfg.get("weak_min_age_days", 30),
        )
        result = cleaner.cleanse()
        state = self._load_lifecycle_state()
        state["last_weekly"] = now_iso()
        self._save_lifecycle_state(state)
        return result

    def run_due_lifecycle(self) -> dict[str, Any]:
        """Run any overdue lifecycle jobs (catch-up).

        The APScheduler cron only fires while the daemon is alive at the scheduled wall-clock
        time, so on a laptop that sleeps through 03:00 the daily decay/consolidation would never
        run. Calling this on daemon startup makes the lifecycle actually happen: if a day (or a
        week) has elapsed since the last run, the job is executed now. Idempotent within the
        window.
        """
        from datetime import UTC, datetime

        state = self._load_lifecycle_state()
        now = datetime.now(UTC)
        ran: dict[str, Any] = {}

        def _stale(key: str, min_hours: float) -> bool:
            stamp = state.get(key)
            if not stamp:
                return True
            try:
                last = datetime.fromisoformat(stamp)
            except (ValueError, TypeError):
                return True
            if last.tzinfo is None:  # tolerate a legacy/naive timestamp
                last = last.replace(tzinfo=UTC)
            return (now - last).total_seconds() >= min_hours * 3600.0

        # On a brand-new store there is nothing to clean yet, so seed the weekly clock instead of
        # firing glymphatic cleanse at install time (the first real cleanse then lands a week out).
        weekly_unseeded = "last_weekly" not in state
        if _stale("last_daily", 20):
            ran["daily"] = self.run_daily()
        if weekly_unseeded:
            seed = self._load_lifecycle_state()
            seed["last_weekly"] = now_iso()
            self._save_lifecycle_state(seed)
        elif _stale("last_weekly", 24 * 7):
            ran["weekly"] = self.run_weekly()
        return ran

    def reembed(self) -> int:
        """Re-encode every memory's audit_text with the current embedder and replace its vector.

        Use this after changing the embedding model: vectors saved in one model's space are
        meaningless to another, so recall breaks until they are rebuilt. Metadata, strength, and
        the audit shadow are preserved; only the scent vectors change. Returns the count rebuilt.
        """
        mems = self.vault.all(include_archived=True)
        pairs = [(m.id, self.encoder.encode(m.audit_text)) for m in mems if m.audit_text]
        if pairs:
            self.vault.update_vectors(pairs)
        return len(pairs)

    def persist(self) -> None:
        """Persist volatile state to disk."""
        self.short_term.save(self.paths.short_term_path)

    def close(self) -> None:
        """Close the engine (save state + close connections)."""
        self.persist()
        self.vault.close()
