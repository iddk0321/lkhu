"""Phase 5 — scheduler tests (embedded APScheduler)."""

from __future__ import annotations

import time

from lkhu.platform.scheduler import LkhuScheduler


def test_add_cron_registers_job() -> None:
    sched = LkhuScheduler()
    sched.add_cron(lambda: None, "0 3 * * *", job_id="daily")
    assert "daily" in sched.jobs()


def test_register_lifecycle_adds_two_jobs() -> None:
    sched = LkhuScheduler()
    sched.register_lifecycle(daily_job=lambda: None, weekly_job=lambda: None)
    ids = sched.jobs()
    assert "lkhu-daily" in ids
    assert "lkhu-weekly" in ids


def test_invalid_cron_raises() -> None:
    sched = LkhuScheduler()
    try:
        sched.add_cron(lambda: None, "not a cron", job_id="bad")
    except ValueError:
        return
    raise AssertionError("an invalid cron should raise ValueError")


def test_scheduler_actually_fires() -> None:
    """Register a short-delay job to actually verify background execution."""
    sched = LkhuScheduler()
    fired = {"ok": False}
    sched.add_after_seconds(lambda: fired.__setitem__("ok", True), 0.2, job_id="probe")
    sched.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and not fired["ok"]:
            time.sleep(0.05)
    finally:
        sched.shutdown()
    assert fired["ok"]
