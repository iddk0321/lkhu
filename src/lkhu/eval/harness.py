"""Evaluation harness — builds an isolated engine over the gold corpus and scores it.

No production data is touched: everything runs in a throwaway data directory with a fresh
codebook. Recall/multilingual/Hebbian metrics need real semantic embeddings (Ollama); the
save-filter metric is deterministic and runs offline too.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lkhu.config.loader import load_config
from lkhu.core.codebook import Codebook
from lkhu.core.encoder import Embedder
from lkhu.core.engine import LkhuEngine
from lkhu.eval import corpus
from lkhu.platform.paths import LkhuPaths
from lkhu.server.hooks import _clean_prompt, _skip_save

__all__ = ["Scorecard", "run_eval", "build_engine"]

# Fixed codebook seed for the eval so the benchmark is reproducible. (In production the codebook
# is generated once with OS entropy and then persisted; here a fresh codebook is built each run,
# so without a fixed seed the structural scent — and thus rankings near ties — would vary.)
_EVAL_SEED = 20240614


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_engine(
    base: str | Path,
    embedder: Embedder,
    overrides: dict[str, Any] | None = None,
) -> LkhuEngine:
    """Construct a fresh isolated engine (own data dir + codebook) with optional config tweaks."""
    paths = LkhuPaths(base=base)
    paths.ensure()
    config = load_config(paths.config_path)  # defaults (no user config in a temp dir)
    config["encoder"]["dim"] = embedder.dim  # codebook + encoder follow the embedder's dim
    if overrides:
        _deep_update(config, overrides)
    if not Codebook.is_initialized(paths.codebook_path):
        cb = Codebook.generate(
            config["codebook"]["initial_keys"],
            dim=embedder.dim,
            seed=config["codebook"].get("random_seed") or _EVAL_SEED,
        )
        cb.save(paths.codebook_path)
    codebook = Codebook.load(paths.codebook_path)
    return LkhuEngine(paths=paths, config=config, codebook=codebook, embedder=embedder)


# Lookup tables built once from the corpus.
_SIGNAL_TOPIC = {it.text: it.topic for it in corpus.SIGNAL}
_SOFT_TEXTS = {it.text for it in corpus.SOFT_NOISE}


def _load_corpus(engine: LkhuEngine) -> None:
    """Insert every gold SIGNAL + SOFT_NOISE item (bypassing dedup so all are present)."""
    for item in [*corpus.SIGNAL, *corpus.SOFT_NOISE]:
        engine._store(item.text, kind="turn", session_id="eval", strength=1.0)


def _warmup(engine: LkhuEngine, rounds: int, decay: float) -> None:
    """Simulate real use over time: repeatedly recall real topics (reinforcing genuine hits via
    the similarity gate) and decay each round. Signal that keeps getting asked about rises;
    never-asked noise falls behind. Uses held-out paraphrases, so the test queries stay unseen.
    """
    for _ in range(rounds):
        for wq in corpus.WARMUP_QUERIES:
            engine.recall_engine.recall(wq.text, k=engine.recall_engine.candidate_multiplier)
        engine.vault.multiply_strength(decay)


def _recall_metrics(engine: LkhuEngine, k: int) -> dict[str, float]:
    """Run every query and average hit@k / precision@k / noise@k (+ cross-lingual hit@k).

    Uses a side-effect-free recall (``reinforce=False``) so measuring does not mutate strengths
    and make the metric order-dependent — the score is deterministic for a given store state.
    """
    hits = precisions = noises = 0.0
    cross_hits = cross_n = 0.0
    for q in corpus.QUERIES:
        memories = engine.recall_engine.recall(q.text, k=k, reinforce=False).memories
        texts = [m.audit_text for m in memories]
        n = len(texts) or 1
        relevant = sum(1 for t in texts if _SIGNAL_TOPIC.get(t) == q.topic)
        noise = sum(1 for t in texts if t in _SOFT_TEXTS)
        hit = 1.0 if relevant >= 1 else 0.0
        hits += hit
        precisions += relevant / n
        noises += noise / n
        if q.cross_lingual:
            cross_hits += hit
            cross_n += 1
    nq = len(corpus.QUERIES)
    return {
        "hit_at_k": round(hits / nq, 3),
        "precision_at_k": round(precisions / nq, 3),
        "noise_rate": round(noises / nq, 3),
        "cross_lingual_hit_at_k": round(cross_hits / cross_n, 3) if cross_n else 0.0,
    }


def _would_save(text: str) -> bool:
    """Mirror the hook's real decision: clean (strip system blocks) then apply the save gate."""
    cleaned = _clean_prompt(text)
    return bool(cleaned) and not _skip_save(cleaned)


def _filter_metrics() -> dict[str, float]:
    """Deterministic save-filter quality (no embedding needed)."""
    kept_signal = sum(1 for it in corpus.SIGNAL if _would_save(it.text))
    dropped_hard = sum(1 for it in corpus.HARD_NOISE if not _would_save(it.text))
    return {
        "signal_kept_rate": round(kept_signal / len(corpus.SIGNAL), 3),
        "hard_noise_dropped_rate": round(dropped_hard / len(corpus.HARD_NOISE), 3),
    }


def _strength_separation(engine: LkhuEngine) -> dict[str, float]:
    """After a warm-up, measure how strength separates signal from noise.

    Faithful to real use: only topics the user keeps asking about are warmed (via held-out
    paraphrases), so well-behaved reinforcement leaves never-asked soft noise weak. Reports the
    count of soft-noise items that nonetheless saturated near the cap, plus mean strengths.
    """
    cap = engine.max_strength
    mems = engine.vault.all()
    sig = [m.strength for m in mems if _SIGNAL_TOPIC.get(m.audit_text)]
    noise = [m.strength for m in mems if m.audit_text in _SOFT_TEXTS]
    noise_saturated = sum(1 for s in noise if s >= 0.97 * cap)
    return {
        "noise_saturated": float(noise_saturated),
        "signal_mean_strength": round(sum(sig) / len(sig), 3) if sig else 0.0,
        "noise_mean_strength": round(sum(noise) / len(noise), 3) if noise else 0.0,
    }


@dataclass
class Scorecard:
    """Aggregated eval results."""

    mode: str  # "ollama" | "offline"
    k: int
    filter: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)  # warm (steady-state, after lifecycle)
    recall_cold: dict[str, float] = field(default_factory=dict)  # cold start (first session)
    hebbian_baseline: dict[str, float] = field(default_factory=dict)
    hebbian_gated: dict[str, float] = field(default_factory=dict)
    corpus_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_eval(embedder: Embedder, k: int = 5, offline: bool = False, rounds: int = 20) -> Scorecard:
    """Run the full evaluation and return a Scorecard.

    Args:
        embedder: Semantic embedder. Ollama for real recall metrics; a hashing embedder is
            only meaningful for the filter metric.
        k: Top-k for recall.
        offline: Skip the embedding-dependent metrics (recall + Hebbian).
        rounds: Repeat count for the Hebbian saturation probe.
    """
    card = Scorecard(
        mode="offline" if offline else "ollama",
        k=k,
        corpus_sizes={
            "signal": len(corpus.SIGNAL),
            "soft_noise": len(corpus.SOFT_NOISE),
            "hard_noise": len(corpus.HARD_NOISE),
            "queries": len(corpus.QUERIES),
        },
    )
    card.filter = _filter_metrics()
    if offline:
        return card

    cfg = load_config()
    threshold = cfg["recall"].get("reinforce_sim_threshold", 0.45)
    decay = cfg["long_term"].get("daily_decay", 0.99)

    # Gated engine: cold recall → warm up (real-use simulation) → warm recall + separation.
    with tempfile.TemporaryDirectory() as tmp:
        engine = build_engine(
            tmp, embedder, overrides={"recall": {"reinforce_sim_threshold": threshold}}
        )
        _load_corpus(engine)
        card.recall_cold = _recall_metrics(engine, k=k)  # first-ever session, no history
        _warmup(engine, rounds=rounds, decay=decay)  # simulate weeks of real use
        card.recall = _recall_metrics(engine, k=k)  # steady-state
        card.hebbian_gated = _strength_separation(engine)
        engine.close()

    # Baseline engine: same warm-up but with the similarity gate OFF (ungated reinforcement).
    with tempfile.TemporaryDirectory() as tmp:
        base = build_engine(tmp, embedder, overrides={"recall": {"reinforce_sim_threshold": 0.0}})
        _load_corpus(base)
        _warmup(base, rounds=rounds, decay=decay)
        card.hebbian_baseline = _strength_separation(base)
        base.close()
    return card
