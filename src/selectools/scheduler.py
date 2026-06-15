"""
Scheduled agents — run an :class:`~selectools.agent.core.Agent` on a cron or
interval schedule.

For periodic, unattended work: monitoring, reporting, digesting, cleanup. A
:class:`ScheduledJob` binds an agent + a prompt + a :class:`Schedule`; an
:class:`AgentScheduler` owns the jobs and fires each one when it comes due. The
scheduler is stdlib-only (``asyncio`` + ``datetime``) — no APScheduler/cron
daemon dependency — mirroring the loop-detection module's philosophy.

Usage::

    from selectools import Agent, AgentScheduler, cron, every

    scheduler = AgentScheduler()
    scheduler.add_job(agent, "Summarize today's open incidents.", cron("0 9 * * *"))
    scheduler.add_job(agent, "Poll the status page.", every(minutes=5), max_runs=12)

    # Drive it (async, runs until stop() or `until`):
    await scheduler.astart()

    # Or tick it yourself (e.g. from your own loop / a test):
    results = scheduler.run_pending()

Schedules:
    - :func:`cron` parses a standard 5-field expression
      ``"minute hour day-of-month month day-of-week"`` with ``*``, ``*/step``,
      ``a-b``, ``a-b/step`` and ``a,b,c`` lists. Day-of-week is ``0-6`` with
      ``0`` = Sunday (``7`` also accepted for Sunday). When BOTH day-of-month
      and day-of-week are restricted, a day matches if EITHER matches (standard
      Vixie-cron semantics).
    - :func:`every` builds a fixed interval from seconds/minutes/hours.

Pair with Agent-as-API (:class:`selectools.serve.AgentAPI`) to expose a
scheduler's job list and last results over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol, Sequence, Tuple

from .stability import beta, register_stability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .agent.core import Agent

logger = logging.getLogger("selectools.scheduler")

# Search horizon for the next cron fire. 4 years covers Feb-29-only schedules
# without an unbounded loop on an impossible expression.
_MAX_CRON_LOOKAHEAD_MINUTES = 366 * 4 * 24 * 60


def _result_text(output: Any) -> str:
    """Extract the response text from an agent's run() result.

    ``Agent.run()`` returns an ``AgentResult`` whose ``.content`` is the final
    response text — NOT a string. Storing ``str(result)`` would record the
    dataclass repr instead of the answer, so prefer ``.content`` and fall back
    to ``str`` only for plain-string returns or unknown result types.
    """
    if isinstance(output, str):
        return output
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    return str(output)


# =============================================================================
# Schedules
# =============================================================================


@beta
class Schedule(Protocol):
    """A firing schedule: given a moment, return the next moment to fire."""

    def next_after(self, after: datetime) -> datetime:
        """Return the first fire time strictly after ``after``."""
        ...


def _parse_cron_field(raw: str, low: int, high: int) -> Tuple[frozenset, bool]:
    """Parse one cron field into (allowed values, is_star).

    Supports ``*``, ``*/step``, ``a``, ``a-b``, ``a-b/step`` and comma lists of
    those. ``is_star`` is True only when the whole field is the literal ``*``
    (needed for the day-of-month / day-of-week OR rule).
    """
    raw = raw.strip()
    if raw == "":
        raise ValueError("empty cron field")
    is_star = raw == "*"
    values: set = set()
    for part in raw.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            base, _, step_s = part.partition("/")
            if not step_s.isdigit() or int(step_s) < 1:
                raise ValueError(f"invalid step in cron field: '{part}'")
            step = int(step_s)
        else:
            base = part
        if base == "*":
            lo, hi = low, high
        elif "-" in base:
            lo_s, _, hi_s = base.partition("-")
            if not (lo_s.isdigit() and hi_s.isdigit()):
                raise ValueError(f"invalid range in cron field: '{part}'")
            lo, hi = int(lo_s), int(hi_s)
        else:
            if not base.isdigit():
                raise ValueError(f"invalid value in cron field: '{part}'")
            lo = hi = int(base)
        if lo > hi:
            raise ValueError(f"range start after end in cron field: '{part}'")
        if lo < low or hi > high:
            raise ValueError(f"cron field value out of range [{low}, {high}]: '{part}'")
        values.update(range(lo, hi + 1, step))
    if not values:
        raise ValueError(f"cron field matched no values: '{raw}'")
    return frozenset(values), is_star


@beta
class CronSchedule:
    """A schedule parsed from a standard 5-field cron expression.

    ``"minute hour day-of-month month day-of-week"``. Minute resolution. See the
    module docstring for the supported syntax and the day-of-month/day-of-week
    OR rule.
    """

    __slots__ = (
        "expression",
        "_minutes",
        "_hours",
        "_doms",
        "_months",
        "_dows",
        "_dom_star",
        "_dow_star",
    )

    def __init__(self, expression: str) -> None:
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"cron expression must have 5 fields "
                f"(minute hour day-of-month month day-of-week), got {len(parts)}: "
                f"'{expression}'"
            )
        self.expression = expression
        self._minutes, _ = _parse_cron_field(parts[0], 0, 59)
        self._hours, _ = _parse_cron_field(parts[1], 0, 23)
        self._doms, self._dom_star = _parse_cron_field(parts[2], 1, 31)
        self._months, _ = _parse_cron_field(parts[3], 1, 12)
        # Day-of-week 0-7, 7 == 0 == Sunday. Normalize 7 -> 0.
        dows, dow_star = _parse_cron_field(parts[4], 0, 7)
        self._dows = frozenset(0 if d == 7 else d for d in dows)
        self._dow_star = dow_star

    def _day_matches(self, dt: datetime) -> bool:
        # Python weekday(): Mon=0..Sun=6. Cron: Sun=0..Sat=6.
        cron_dow = (dt.weekday() + 1) % 7
        dom_ok = dt.day in self._doms
        dow_ok = cron_dow in self._dows
        if self._dom_star and self._dow_star:
            return True
        if self._dom_star:
            return dow_ok
        if self._dow_star:
            return dom_ok
        return dom_ok or dow_ok

    def matches(self, dt: datetime) -> bool:
        """Whether ``dt`` (at minute resolution) satisfies this expression."""
        return (
            dt.minute in self._minutes
            and dt.hour in self._hours
            and dt.month in self._months
            and self._day_matches(dt)
        )

    def next_after(self, after: datetime) -> datetime:
        """First fire time strictly after ``after`` (seconds/micros zeroed)."""
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(_MAX_CRON_LOOKAHEAD_MINUTES):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError(f"cron expression '{self.expression}' has no fire time within 4 years")

    def __repr__(self) -> str:
        return f"CronSchedule({self.expression!r})"


@beta
class IntervalSchedule:
    """Fire every fixed interval (>= 1 second)."""

    __slots__ = ("interval",)

    def __init__(self, seconds: float) -> None:
        if seconds < 1:
            raise ValueError("interval must be at least 1 second")
        self.interval = timedelta(seconds=seconds)

    def next_after(self, after: datetime) -> datetime:
        return after + self.interval

    def __repr__(self) -> str:
        return f"IntervalSchedule({self.interval.total_seconds()}s)"


@beta
def cron(expression: str) -> CronSchedule:
    """Build a :class:`CronSchedule` from a 5-field cron expression."""
    return CronSchedule(expression)


@beta
def every(
    seconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
) -> IntervalSchedule:
    """Build an :class:`IntervalSchedule` from seconds/minutes/hours."""
    total = seconds + minutes * 60 + hours * 3600
    if total < 1:
        raise ValueError("every(...) total interval must be at least 1 second")
    return IntervalSchedule(total)


# =============================================================================
# Jobs and results
# =============================================================================


@beta
@dataclass
class JobResult:
    """The outcome of one job firing."""

    job_name: str
    fired_at: datetime
    run_index: int
    output: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@beta
@dataclass
class ScheduledJob:
    """An agent + prompt bound to a schedule, plus its runtime state."""

    agent: "Agent"
    prompt: str
    schedule: Schedule
    name: str
    max_runs: Optional[int] = None
    enabled: bool = True
    on_result: Optional[Callable[[JobResult], None]] = None
    # runtime state
    run_count: int = 0
    next_fire: Optional[datetime] = None
    last_result: Optional[JobResult] = None

    @property
    def exhausted(self) -> bool:
        """True once ``max_runs`` firings have completed."""
        return self.max_runs is not None and self.run_count >= self.max_runs

    def is_due(self, now: datetime) -> bool:
        return (
            self.enabled
            and not self.exhausted
            and self.next_fire is not None
            and now >= self.next_fire
        )


# =============================================================================
# Scheduler
# =============================================================================


@beta
class AgentScheduler:
    """Owns scheduled jobs and fires each agent when its schedule comes due.

    Drive it three ways:
      - :meth:`astart` — async loop that sleeps until the next due job, runs it,
        and repeats until :meth:`stop` or the ``until`` deadline.
      - :meth:`arun_pending` / :meth:`run_pending` — fire everything due ``now``
        once and return the results (for your own loop, or tests).

    A job that raises is recorded on its :class:`JobResult` (``error``) and in
    ``job.last_result``; it never stops sibling jobs or the loop.
    """

    def __init__(self, *, now: Optional[Callable[[], datetime]] = None) -> None:
        # ``now`` is injectable so tests can drive time deterministically.
        self._now = now or datetime.now
        self._jobs: List[ScheduledJob] = []
        self._stopped = False

    @property
    def jobs(self) -> Sequence[ScheduledJob]:
        return tuple(self._jobs)

    def add_job(
        self,
        agent: "Agent",
        prompt: str,
        schedule: Schedule,
        *,
        name: Optional[str] = None,
        max_runs: Optional[int] = None,
        on_result: Optional[Callable[[JobResult], None]] = None,
        start_immediately: bool = False,
    ) -> ScheduledJob:
        """Register a job.

        By default the first fire is the next time the schedule comes due
        strictly after now (an ``every(minutes=5)`` job first fires in five
        minutes). Pass ``start_immediately=True`` for the common "run now, then
        on schedule" pattern — the job fires on the next tick and resumes its
        normal cadence afterward.
        """
        if max_runs is not None and max_runs < 1:
            raise ValueError("max_runs must be >= 1 when set")
        resolved_name = name or getattr(agent, "name", None) or f"job-{len(self._jobs) + 1}"
        job = ScheduledJob(
            agent=agent,
            prompt=prompt,
            schedule=schedule,
            name=resolved_name,
            max_runs=max_runs,
            on_result=on_result,
        )
        job.next_fire = self._now() if start_immediately else schedule.next_after(self._now())
        self._jobs.append(job)
        return job

    def remove_job(self, name: str) -> bool:
        """Remove the first job with ``name``. Returns whether one was removed."""
        for i, job in enumerate(self._jobs):
            if job.name == name:
                del self._jobs[i]
                return True
        return False

    def due_jobs(self, now: Optional[datetime] = None) -> List[ScheduledJob]:
        moment = now or self._now()
        return [j for j in self._jobs if j.is_due(moment)]

    def _advance(self, job: ScheduledJob, fired_at: datetime) -> None:
        """Roll a job's counters/next_fire forward after it fires."""
        job.run_count += 1
        # Schedule from the fire time, not wall-clock-after-run, so a slow agent
        # doesn't drift the cadence.
        job.next_fire = None if job.exhausted else job.schedule.next_after(fired_at)

    def _finish(self, job: ScheduledJob, result: JobResult) -> None:
        job.last_result = result
        if job.on_result is not None:
            try:
                job.on_result(result)
            except Exception:  # nosec B110 - a callback must never break the scheduler
                logger.exception("scheduler on_result callback raised for job %r", job.name)

    def run_pending(self, now: Optional[datetime] = None) -> List[JobResult]:
        """Synchronously fire every job due at ``now`` (uses ``agent.run``)."""
        moment = now or self._now()
        results: List[JobResult] = []
        for job in self._jobs:
            if not job.is_due(moment):
                continue
            result = JobResult(job_name=job.name, fired_at=moment, run_index=job.run_count)
            try:
                result.output = _result_text(job.agent.run(job.prompt))
            except Exception as exc:  # noqa: BLE001 - isolate job failures
                result.error = f"{type(exc).__name__}: {exc}"
                logger.exception("scheduled job %r raised", job.name)
            self._advance(job, moment)
            self._finish(job, result)
            results.append(result)
        return results

    async def arun_pending(self, now: Optional[datetime] = None) -> List[JobResult]:
        """Async variant of :meth:`run_pending` (uses ``agent.arun``)."""
        moment = now or self._now()
        results: List[JobResult] = []
        for job in self._jobs:
            if not job.is_due(moment):
                continue
            result = JobResult(job_name=job.name, fired_at=moment, run_index=job.run_count)
            try:
                result.output = _result_text(await job.agent.arun(job.prompt))
            except Exception as exc:  # noqa: BLE001 - isolate job failures
                result.error = f"{type(exc).__name__}: {exc}"
                logger.exception("scheduled job %r raised", job.name)
            self._advance(job, moment)
            self._finish(job, result)
            results.append(result)
        return results

    def _next_wakeup(self) -> Optional[datetime]:
        pending = [
            j.next_fire
            for j in self._jobs
            if j.enabled and not j.exhausted and j.next_fire is not None
        ]
        return min(pending) if pending else None

    def stop(self) -> None:
        """Signal :meth:`astart` to exit after its current sleep/iteration."""
        self._stopped = True

    async def astart(
        self,
        *,
        poll_interval: float = 1.0,
        until: Optional[datetime] = None,
    ) -> List[JobResult]:
        """Run the scheduler loop until :meth:`stop`, ``until``, or all jobs exhaust.

        Sleeps until the soonest due job (capped at ``poll_interval`` so
        :meth:`stop` and newly added jobs are noticed promptly), fires what is
        due, and repeats. Returns every :class:`JobResult` produced.
        """
        self._stopped = False
        all_results: List[JobResult] = []
        while not self._stopped:
            now = self._now()
            if until is not None and now >= until:
                break
            all_results.extend(await self.arun_pending(now))
            wakeup = self._next_wakeup()
            if wakeup is None:
                break  # nothing left to fire
            sleep_for = (wakeup - self._now()).total_seconds()
            if until is not None:
                sleep_for = min(sleep_for, (until - self._now()).total_seconds())
            sleep_for = max(0.0, min(sleep_for, poll_interval))
            await asyncio.sleep(sleep_for)
        return all_results


register_stability("scheduler", "beta")

__stability__ = "beta"

__all__ = [
    "AgentScheduler",
    "CronSchedule",
    "IntervalSchedule",
    "JobResult",
    "Schedule",
    "ScheduledJob",
    "cron",
    "every",
]
