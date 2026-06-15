"""Tests for the evaluation harness (offline mode — no Ollama required)."""

from __future__ import annotations

from lkhu.core.encoder import HashingEmbedder
from lkhu.eval import run_eval
from lkhu.eval.harness import _would_save, build_engine


def test_offline_eval_produces_filter_scorecard() -> None:
    card = run_eval(HashingEmbedder(dim=64), offline=True)
    assert card.mode == "offline"
    # The gold corpus is designed so the filter keeps all signal and drops all hard noise.
    assert card.filter["signal_kept_rate"] == 1.0
    assert card.filter["hard_noise_dropped_rate"] == 1.0
    assert card.corpus_sizes["signal"] > 0
    assert card.recall == {}  # skipped in offline mode


def test_would_save_mirrors_hook_pipeline() -> None:
    assert _would_save("We decided to standardize API responses on snake_case.")
    assert not _would_save("ㅇㅇ")
    assert not _would_save("<task-notification><tool-use-id>x</tool-use-id></task-notification>")


def test_build_engine_is_isolated(tmp_path) -> None:
    eng = build_engine(tmp_path, HashingEmbedder(dim=64))
    try:
        assert eng.vault.count() == 0
        assert eng.paths.data_dir == tmp_path
    finally:
        eng.close()
