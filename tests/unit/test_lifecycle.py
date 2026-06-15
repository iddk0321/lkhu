"""Phase 5 — lifecycle tests: decay / consolidation / glymphatic cleanse."""

from __future__ import annotations

import numpy as np
import pytest

from lkhu.core import vsa
from lkhu.core.consolidator import Consolidator
from lkhu.core.decay import DecayEngine
from lkhu.core.glymphatic import GlymphaticCleaner
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory
from lkhu.core.short_term import ShortTermBundle

DIM = 64


def _vec(rng) -> np.ndarray:
    return vsa.normalize(rng.standard_normal(DIM).astype(np.float32))


@pytest.fixture
def vault(tmp_path):
    v = LongTermVault(tmp_path / "m.db", dim=DIM)
    yield v
    v.close()


# ----- Decay -----


def test_daily_decay_scales_strength(vault) -> None:
    rng = np.random.default_rng(0)
    ids = [vault.insert(Memory.make(_vec(rng), strength=1.0)).id for _ in range(3)]
    st = ShortTermBundle(dim=DIM)
    st.add(_vec(rng))
    before = float(np.linalg.norm(st.raw))

    engine = DecayEngine(vault=vault, short_term=st)
    engine.run_daily()

    for i in ids:
        assert vault.get(i).strength == pytest.approx(0.99)
    assert float(np.linalg.norm(st.raw)) == pytest.approx(before * 0.7, abs=1e-5)


def test_decay_curve_is_ebbinghaus(vault) -> None:
    """With no recall, daily decay makes strength monotonically decrease (exponential)."""
    rng = np.random.default_rng(1)
    mid = vault.insert(Memory.make(_vec(rng), strength=1.0)).id
    engine = DecayEngine(vault=vault, short_term=ShortTermBundle(dim=DIM))
    series = [vault.get(mid).strength]
    for _ in range(30):
        engine.run_daily()
        series.append(vault.get(mid).strength)
    # Monotonically decreasing
    assert all(series[i] > series[i + 1] for i in range(len(series) - 1))
    # After 30 days ≈ 0.99**30
    assert series[-1] == pytest.approx(0.99**30, rel=1e-4)


# ----- Consolidation -----


def test_consolidation_creates_summary_with_source_ids(vault) -> None:
    rng = np.random.default_rng(2)
    ids = [
        vault.insert(Memory.make(_vec(rng), session_id="S", audit_text=f"turn{i}")).id
        for i in range(4)
    ]
    cons = Consolidator(vault=vault)
    created = cons.consolidate(days=3650)
    assert len(created) == 1
    summary = vault.get(created[0].id)
    assert summary.kind == "summary"
    assert summary.strength == pytest.approx(1.2)
    assert set(summary.source_ids) == set(ids)
    assert np.isclose(np.linalg.norm(summary.vector), 1.0)


def test_consolidation_skips_small_sessions(vault) -> None:
    rng = np.random.default_rng(3)
    for _ in range(2):  # below min_session_size=3
        vault.insert(Memory.make(_vec(rng), session_id="small"))
    cons = Consolidator(vault=vault)
    assert cons.consolidate(days=3650) == []


def test_consolidation_is_idempotent(vault) -> None:
    rng = np.random.default_rng(4)
    for i in range(3):
        vault.insert(Memory.make(_vec(rng), session_id="S", audit_text=f"t{i}"))
    cons = Consolidator(vault=vault)
    first = cons.consolidate(days=3650)
    second = cons.consolidate(days=3650)
    assert len(first) == 1
    assert second == []  # do not regenerate if a summary already exists


# ----- Glymphatic cleanse -----


def test_cleanse_merges_duplicates(vault) -> None:
    rng = np.random.default_rng(5)
    base = _vec(rng)
    near = vsa.normalize(base + 0.001 * _vec(rng))  # nearly identical (cosine>0.95)
    a = vault.insert(Memory.make(base, strength=0.8, audit_text="A"))
    b = vault.insert(Memory.make(near, strength=1.1, audit_text="B"))

    cleaner = GlymphaticCleaner(vault=vault)
    report = cleaner.cleanse()

    assert report["merged"] >= 1
    merged = [m for m in vault.all() if m.kind == "merged"]
    assert merged
    assert set(merged[0].source_ids) == {a.id, b.id}
    assert merged[0].strength == pytest.approx(1.1)  # max(a,b)
    # Originals drop out of search (archived)
    assert vault.count() == 1


def test_cleanse_archives_weak_old(vault) -> None:
    rng = np.random.default_rng(6)
    weak = vault.insert(Memory.make(_vec(rng), strength=0.05))
    strong = vault.insert(Memory.make(_vec(rng), strength=1.0))
    vault.set_created_at(weak.id, "2000-01-01T00:00:00+00:00")
    vault.set_created_at(strong.id, "2000-01-01T00:00:00+00:00")

    cleaner = GlymphaticCleaner(vault=vault)
    report = cleaner.cleanse()

    assert report["archived_weak"] >= 1
    assert vault.get(weak.id).archived is True
    assert vault.get(strong.id).archived is False
