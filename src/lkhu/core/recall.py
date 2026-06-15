"""RecallEngine — search + rerank + Hebbian update + composite scent. Design doc §7.2.

The first stage of token savings. FAISS quickly finds K×3 candidates, reranks them by
(similarity, strength, recency), reinforces the strength of recalled memories (Hebbian),
and builds a composite scent via a weighted sum. Zero LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from lkhu.core import vsa
from lkhu.core.encoder import Encoder
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory, now_iso

__all__ = ["RecallEngine", "RecallResult", "recency_score"]

_RECENCY_HALFLIFE_DAYS = 7.0


def recency_score(last_accessed_at: str, now: float | None = None) -> float:
    """Recency score (0~1]. Closer to 1 the more recent, decaying exponentially.

    Args:
        last_accessed_at: ISO timestamp.
        now: Reference time (epoch seconds). Defaults to now.

    Returns:
        A score of the form ``0.5 ** (age_days / halflife)``.
    """
    now = now if now is not None else datetime.now(UTC).timestamp()
    try:
        ts = datetime.fromisoformat(last_accessed_at).timestamp()
    except (ValueError, TypeError):
        return 0.0
    age_days = max(0.0, (now - ts) / 86400.0)
    return float(0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS))


@dataclass
class RecallResult:
    """Recall result.

    Attributes:
        composite: Weighted-sum composite scent of recalled memories (normalized).
        memories: Reranked top memories (descending by score).
        scores: Rerank score of each memory (same ordering as memories).
    """

    composite: np.ndarray
    memories: list[Memory] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


class RecallEngine:
    """Search, rerank, strength-update, and synthesis engine.

    Args:
        vault: Long-term vault.
        encoder: Query encoder.
        sim_weight/strength_weight/recency_weight: Rerank weights (default 0.6/0.2/0.2).
        candidate_multiplier: Candidate multiplier (default 3 → K×3 candidates).
        recall_boost: Strength multiplier on recall (default 1.05).
        max_strength: Strength cap (default 1.5).
        reinforce_sim_threshold: Only memories whose query similarity reaches this value are
            reinforced (default 0.0 = reinforce every returned memory). A positive value
            implements the Hebbian "fire together" rule properly: filler results that merely
            rank top-k by being the least-bad match are returned but NOT strengthened, which
            breaks the rich-get-richer loop that otherwise pins frequently-recalled noise at
            the strength cap.
    """

    def __init__(
        self,
        vault: LongTermVault,
        encoder: Encoder,
        sim_weight: float = 0.6,
        strength_weight: float = 0.2,
        recency_weight: float = 0.2,
        candidate_multiplier: int = 3,
        recall_boost: float = 1.05,
        max_strength: float = 1.5,
        reinforce_sim_threshold: float = 0.0,
        min_similarity: float = 0.0,
    ):
        self.vault = vault
        self.encoder = encoder
        self.sim_weight = sim_weight
        self.strength_weight = strength_weight
        self.recency_weight = recency_weight
        self.candidate_multiplier = candidate_multiplier
        self.recall_boost = recall_boost
        self.max_strength = max_strength
        self.reinforce_sim_threshold = reinforce_sim_threshold
        self.min_similarity = min_similarity

    def recall(self, query: str, k: int = 5, reinforce: bool = True) -> RecallResult:
        """Recall the top-k memories relevant to the query and build a composite scent.

        Args:
            query: Natural language query.
            k: Number of memories to return.
            reinforce: Apply the Hebbian strength/access update to the returned memories. Set
                ``False`` for side-effect-free reads (e.g. evaluation), so measuring recall
                does not itself mutate strengths and make results order-dependent.

        Returns:
            ``RecallResult``.
        """
        q = self.encoder.encode(query)
        candidates = self.vault.faiss_search(q, k=k * self.candidate_multiplier)
        # Similarity floor: drop candidates that aren't actually close to the query. This trims
        # low-relevance noise from results directly (independent of the strength lifecycle), which
        # matters for embedders whose noise sits well below genuine matches in cosine.
        if self.min_similarity > 0.0:
            candidates = [(m, s) for m, s in candidates if s >= self.min_similarity]
        if not candidates:
            return RecallResult(composite=np.zeros(self.encoder.dim, dtype=np.float32))

        now = datetime.now(UTC).timestamp()
        scored: list[tuple[Memory, float, float]] = []
        for mem, sim in candidates:
            # Similarity is the primary, query-conditional signal. Strength and recency act as
            # MULTIPLICATIVE modulators, not additive bonuses — so a globally-strong but
            # off-topic memory cannot outrank a clearly more similar one (an additive strength
            # term, being query-independent, did exactly that and hurt recall). Weak/stale
            # memories are gently demoted; among comparably-similar items, strength breaks ties.
            strength_norm = mem.strength / self.max_strength if self.max_strength else 0.0
            modulator = (
                self.sim_weight
                + self.strength_weight * strength_norm
                + self.recency_weight * recency_score(mem.last_accessed_at, now=now)
            )
            scored.append((mem, sim * modulator, sim))

        scored.sort(key=lambda x: -x[1])
        top = scored[:k]

        # Hebbian strength update — only genuine hits (sim ≥ threshold) are reinforced, so
        # low-relevance filler that merely survived the top-k cut does not accrue strength.
        if reinforce:
            stamp = now_iso()
            reinforced: list[Memory] = []
            for mem, _, sim in top:
                mem.access_count += 1
                mem.last_accessed_at = stamp
                if sim >= self.reinforce_sim_threshold:
                    mem.strength = min(self.max_strength, mem.strength * self.recall_boost)
                reinforced.append(mem)
            self.vault.batch_update(reinforced)

        # Composite scent = Σ (scent × score)
        composite = vsa.bundle(
            [mem.vector * score for mem, score, _ in top],
            normalized=True,
        )
        return RecallResult(
            composite=composite,
            memories=[m for m, _, _ in top],
            scores=[s for _, s, _ in top],
        )
