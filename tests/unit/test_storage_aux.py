"""Phase 3 — Audit / ShortTerm / WorkingMemory tests."""

from __future__ import annotations

import numpy as np

from lkhu.core import vsa
from lkhu.core.audit import AuditLog
from lkhu.core.short_term import ShortTermBundle
from lkhu.core.working_memory import WorkingMemory

DIM = 32


# ----- AuditLog -----


def test_audit_append_and_read_all(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit")
    log.append({"id": "1", "session_id": "s1", "audit_text": "first memory"})
    log.append({"id": "2", "session_id": "s2", "audit_text": "second"})
    records = list(log.read_all())
    assert len(records) == 2
    assert {r["id"] for r in records} == {"1", "2"}


def test_audit_monthly_file_layout(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit")
    log.append({"id": "1", "audit_text": "x", "created_at": "2026-03-15T10:00:00+00:00"})
    f = tmp_path / "audit" / "2026-03" / "15.jsonl"
    assert f.exists()


def test_audit_by_session(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit")
    log.append({"id": "1", "session_id": "A", "audit_text": "a"})
    log.append({"id": "2", "session_id": "B", "audit_text": "b"})
    log.append({"id": "3", "session_id": "A", "audit_text": "c"})
    got = log.by_session("A")
    assert len(got) == 2
    assert {r["id"] for r in got} == {"1", "3"}


# ----- ShortTermBundle -----


def test_short_term_membership() -> None:
    st = ShortTermBundle(dim=DIM)
    rng = np.random.default_rng(0)
    members = [vsa.normalize(rng.standard_normal(DIM).astype(np.float32)) for _ in range(3)]
    outsider = vsa.normalize(rng.standard_normal(DIM).astype(np.float32))
    for m in members:
        st.add(m)
    b = st.bundle()
    for m in members:
        assert vsa.cosine(b, m) > vsa.cosine(b, outsider)


def test_short_term_decay_reduces_norm() -> None:
    st = ShortTermBundle(dim=DIM)
    rng = np.random.default_rng(1)
    st.add(vsa.normalize(rng.standard_normal(DIM).astype(np.float32)))
    before = float(np.linalg.norm(st.raw))
    st.decay(0.7)
    after = float(np.linalg.norm(st.raw))
    assert after < before
    assert np.isclose(after, before * 0.7, atol=1e-5)


def test_short_term_save_load_roundtrip(tmp_path) -> None:
    st = ShortTermBundle(dim=DIM)
    rng = np.random.default_rng(2)
    st.add(vsa.normalize(rng.standard_normal(DIM).astype(np.float32)))
    path = tmp_path / "short_term.npy"
    st.save(path)

    st2 = ShortTermBundle.load(path, dim=DIM)
    assert np.allclose(st2.raw, st.raw)


def test_short_term_load_missing_is_empty(tmp_path) -> None:
    st = ShortTermBundle.load(tmp_path / "nope.npy", dim=DIM)
    assert np.linalg.norm(st.raw) == 0.0


# ----- WorkingMemory -----


def test_working_memory_evicts_oldest() -> None:
    wm = WorkingMemory(max_turns=3)
    for i in range(5):
        wm.add({"text": f"turn{i}"})
    assert len(wm) == 3
    texts = [t["text"] for t in wm.turns]
    assert texts == ["turn2", "turn3", "turn4"]


def test_working_memory_recent_order() -> None:
    wm = WorkingMemory(max_turns=10)
    for i in range(5):
        wm.add({"text": f"t{i}"})
    recent = wm.recent(2)
    assert [t["text"] for t in recent] == ["t3", "t4"]


def test_working_memory_idle_flush() -> None:
    wm = WorkingMemory(max_turns=10)
    wm.add({"text": "x"}, at=1000.0)
    assert wm.should_flush(idle_minutes=30, now=1000.0 + 31 * 60)
    assert not wm.should_flush(idle_minutes=30, now=1000.0 + 10 * 60)


def test_working_memory_clear() -> None:
    wm = WorkingMemory(max_turns=10)
    wm.add({"text": "x"})
    wm.clear()
    assert len(wm) == 0
