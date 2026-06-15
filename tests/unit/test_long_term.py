"""Phase 3 — LongTermVault (SQLite + FAISS) tests."""

from __future__ import annotations

import numpy as np
import pytest

from lkhu.core import vsa
from lkhu.core.long_term import LongTermVault
from lkhu.core.memory import Memory

DIM = 64


def _vec(rng: np.random.Generator) -> np.ndarray:
    return vsa.normalize(rng.standard_normal(DIM).astype(np.float32))


@pytest.fixture
def vault(tmp_path) -> LongTermVault:
    v = LongTermVault(tmp_path / "memories.db", dim=DIM)
    yield v
    v.close()


def test_insert_and_get_roundtrip(vault) -> None:
    rng = np.random.default_rng(0)
    m = Memory.make(_vec(rng), kind="explicit", audit_text="hello", session_id="s1")
    saved = vault.insert(m)
    assert saved.rowid is not None

    got = vault.get(m.id)
    assert got is not None
    assert got.audit_text == "hello"
    assert got.kind == "explicit"
    assert got.session_id == "s1"
    assert np.allclose(got.vector, m.vector, atol=1e-6)


def test_count(vault) -> None:
    rng = np.random.default_rng(1)
    for _ in range(5):
        vault.insert(Memory.make(_vec(rng)))
    assert vault.count() == 5


def test_faiss_search_finds_nearest(vault) -> None:
    rng = np.random.default_rng(2)
    target = _vec(rng)
    target_mem = vault.insert(Memory.make(target, audit_text="target"))
    for _ in range(20):
        vault.insert(Memory.make(_vec(rng)))

    results = vault.faiss_search(target, k=3)
    assert results
    top_mem, sim = results[0]
    assert top_mem.id == target_mem.id
    assert sim > 0.99


def test_search_reproducible_after_reopen(tmp_path) -> None:
    rng = np.random.default_rng(3)
    path = tmp_path / "m.db"
    v = LongTermVault(path, dim=DIM)
    target = _vec(rng)
    tid = v.insert(Memory.make(target, audit_text="t")).id
    for _ in range(10):
        v.insert(Memory.make(_vec(rng)))
    v.close()

    v2 = LongTermVault(path, dim=DIM)
    results = v2.faiss_search(target, k=1)
    assert results[0][0].id == tid
    v2.close()


def test_batch_update_strength(vault) -> None:
    rng = np.random.default_rng(4)
    m = vault.insert(Memory.make(_vec(rng)))
    m.strength = 1.45
    m.access_count = 7
    vault.batch_update([m])
    got = vault.get(m.id)
    assert got.strength == pytest.approx(1.45)
    assert got.access_count == 7


def test_by_session(vault) -> None:
    rng = np.random.default_rng(5)
    for _ in range(3):
        vault.insert(Memory.make(_vec(rng), session_id="A"))
    for _ in range(2):
        vault.insert(Memory.make(_vec(rng), session_id="B"))
    assert len(vault.by_session("A")) == 3
    assert len(vault.by_session("B")) == 2


def test_archive_removes_from_search(vault) -> None:
    rng = np.random.default_rng(6)
    target = _vec(rng)
    m = vault.insert(Memory.make(target, audit_text="x"))
    vault.archive([m.id])
    results = vault.faiss_search(target, k=5)
    assert all(r[0].id != m.id for r in results)
    assert vault.count() == 0  # default count is non-archived only


def test_decay_all(vault) -> None:
    rng = np.random.default_rng(7)
    ids = [vault.insert(Memory.make(_vec(rng), strength=1.0)).id for _ in range(3)]
    vault.multiply_strength(0.99)
    for i in ids:
        assert vault.get(i).strength == pytest.approx(0.99)


def test_select_weak(vault) -> None:
    rng = np.random.default_rng(8)
    weak = vault.insert(Memory.make(_vec(rng), strength=0.05))
    strong = vault.insert(Memory.make(_vec(rng), strength=1.0))
    # force-update created_at to push it into the past
    vault.set_created_at(weak.id, "2000-01-01T00:00:00+00:00")
    vault.set_created_at(strong.id, "2000-01-01T00:00:00+00:00")
    weak_ids = [m.id for m in vault.select_weak(strength_below=0.1, min_age_days=30)]
    assert weak.id in weak_ids
    assert strong.id not in weak_ids
