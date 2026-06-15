"""Scheduler — APScheduler wrapper (embedded in the server). Design doc §5.4.

Without relying on the OS-native cron/launchd/Task Scheduler, it runs daily
consolidation/decay and weekly cleanse inside the lkhu server process. A single
``lkhu serve`` is all it takes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

__all__ = ["LkhuScheduler"]

# Default schedule, design §14.1
DEFAULT_DAILY_CRON = "0 3 * * *"  # every day at 03:00
DEFAULT_WEEKLY_CRON = "30 3 * * 0"  # Sunday at 03:30


class LkhuScheduler:
    """lkhu background scheduler."""

    def __init__(self) -> None:
        self._sched = BackgroundScheduler(timezone="UTC")

    def add_cron(self, func: Callable[[], None], cron_expr: str, job_id: str) -> None:
        """Register a job with a crontab string.

        Args:
            func: Function to run.
            cron_expr: ``minute hour day month weekday`` format.
            job_id: Job identifier.

        Raises:
            ValueError: When the cron string is invalid.
        """
        trigger = CronTrigger.from_crontab(cron_expr, timezone="UTC")
        self._sched.add_job(func, trigger=trigger, id=job_id, replace_existing=True)

    def add_after_seconds(self, func: Callable[[], None], seconds: float, job_id: str) -> None:
        """Register a one-shot job that runs after the given number of seconds."""
        run_at = datetime.now(UTC) + timedelta(seconds=seconds)
        self._sched.add_job(
            func, trigger=DateTrigger(run_date=run_at), id=job_id, replace_existing=True
        )

    def register_lifecycle(
        self,
        daily_job: Callable[[], None],
        weekly_job: Callable[[], None],
        daily_cron: str = DEFAULT_DAILY_CRON,
        weekly_cron: str = DEFAULT_WEEKLY_CRON,
    ) -> None:
        """Register the daily (decay + consolidation) and weekly (cleanse) jobs."""
        self.add_cron(daily_job, daily_cron, job_id="lkhu-daily")
        self.add_cron(weekly_job, weekly_cron, job_id="lkhu-weekly")

    def jobs(self) -> list[str]:
        """List of registered job ids."""
        return [job.id for job in self._sched.get_jobs()]

    def start(self) -> None:
        """Start the scheduler (background thread)."""
        if not self._sched.running:
            self._sched.start()

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the scheduler."""
        if self._sched.running:
            self._sched.shutdown(wait=wait)
