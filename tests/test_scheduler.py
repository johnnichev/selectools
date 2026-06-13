"""Tests for selectools.scheduler — cron/interval scheduled agents."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from selectools.scheduler import (
    AgentScheduler,
    CronSchedule,
    IntervalSchedule,
    JobResult,
    cron,
    every,
)

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeAgent:
    """Minimal Agent stand-in: records prompts, returns a canned answer."""

    def __init__(self, name="fake", answer="done", raises=False):
        self.name = name
        self.answer = answer
        self.raises = raises
        self.calls = []

    def run(self, prompt):
        self.calls.append(prompt)
        if self.raises:
            raise RuntimeError("boom")
        return self.answer

    async def arun(self, prompt):
        self.calls.append(prompt)
        if self.raises:
            raise RuntimeError("boom")
        return self.answer


class Clock:
    """Deterministic, advanceable time source."""

    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


# --------------------------------------------------------------------------- #
# Cron parsing
# --------------------------------------------------------------------------- #


def test_cron_requires_five_fields():
    with pytest.raises(ValueError, match="5 fields"):
        CronSchedule("* * * *")
    with pytest.raises(ValueError, match="5 fields"):
        CronSchedule("* * * * * *")


def test_cron_matches_every_minute():
    sched = cron("* * * * *")
    assert sched.matches(datetime(2026, 6, 13, 14, 7))
    assert sched.matches(datetime(2026, 1, 1, 0, 0))


def test_cron_daily_at_nine():
    sched = cron("0 9 * * *")
    assert sched.matches(datetime(2026, 6, 13, 9, 0))
    assert not sched.matches(datetime(2026, 6, 13, 9, 1))
    assert not sched.matches(datetime(2026, 6, 13, 10, 0))


def test_cron_step_and_list_and_range():
    sched = cron("*/15 9-17 * * 1,3,5")  # every 15 min, 9-17h, Mon/Wed/Fri
    mon_9_15 = datetime(2026, 6, 15, 9, 15)  # 2026-06-15 is a Monday
    assert mon_9_15.weekday() == 0
    assert sched.matches(mon_9_15)
    assert not sched.matches(datetime(2026, 6, 15, 9, 16))  # not a 15-step minute
    assert not sched.matches(datetime(2026, 6, 16, 9, 15))  # Tuesday


def test_cron_next_after_rolls_to_next_day():
    sched = cron("0 9 * * *")
    nxt = sched.next_after(datetime(2026, 6, 13, 9, 30))
    assert nxt == datetime(2026, 6, 14, 9, 0)


def test_cron_next_after_is_strictly_after():
    sched = cron("0 9 * * *")
    # Exactly at fire time -> next is tomorrow, not the same instant.
    nxt = sched.next_after(datetime(2026, 6, 13, 9, 0))
    assert nxt == datetime(2026, 6, 14, 9, 0)


def test_cron_dom_dow_or_semantics():
    # day-of-month 1 OR day-of-week 0 (Sunday): both restricted -> OR.
    sched = cron("0 0 1 * 0")
    assert sched.matches(datetime(2026, 6, 1, 0, 0))  # the 1st (a Monday)
    assert sched.matches(datetime(2026, 6, 7, 0, 0))  # a Sunday, not the 1st
    assert not sched.matches(datetime(2026, 6, 2, 0, 0))  # neither


def test_cron_dow_seven_is_sunday():
    s7 = cron("0 0 * * 7")
    s0 = cron("0 0 * * 0")
    sunday = datetime(2026, 6, 7, 0, 0)
    assert sunday.weekday() == 6
    assert s7.matches(sunday)
    assert s0.matches(sunday)


@pytest.mark.parametrize(
    "expr", ["bad * * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "*/0 * * * *"]
)
def test_cron_invalid_fields_raise(expr):
    with pytest.raises(ValueError):
        CronSchedule(expr)


# --------------------------------------------------------------------------- #
# Interval
# --------------------------------------------------------------------------- #


def test_interval_next_after():
    sched = IntervalSchedule(300)
    base = datetime(2026, 6, 13, 12, 0)
    assert sched.next_after(base) == base + timedelta(seconds=300)


def test_every_composes_units():
    assert every(minutes=5).interval == timedelta(minutes=5)
    assert every(hours=1, minutes=30).interval == timedelta(minutes=90)


def test_interval_minimum():
    with pytest.raises(ValueError):
        IntervalSchedule(0.5)
    with pytest.raises(ValueError):
        every(seconds=0)


# --------------------------------------------------------------------------- #
# Scheduler — firing, state, isolation
# --------------------------------------------------------------------------- #


def test_add_job_computes_first_fire():
    clock = Clock(datetime(2026, 6, 13, 8, 0))
    s = AgentScheduler(now=clock)
    agent = FakeAgent()
    job = s.add_job(agent, "go", cron("0 9 * * *"))
    assert job.next_fire == datetime(2026, 6, 13, 9, 0)
    assert job.name == "fake"


def test_default_first_fire_is_deferred_not_immediate():
    # A job created exactly at a fire instant does NOT fire now; the next fire
    # is strictly in the future (interval jobs first fire one interval out).
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    job = s.add_job(FakeAgent(), "go", cron("0 9 * * *"))
    assert job.next_fire == datetime(2026, 6, 14, 9, 0)
    assert s.run_pending() == []


def test_run_pending_fires_only_due_jobs():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    due = FakeAgent(name="due")
    later = FakeAgent(name="later")
    s.add_job(due, "now", cron("0 9 * * *"), start_immediately=True)
    s.add_job(later, "later", cron("0 17 * * *"))  # next_fire 17:00 today
    results = s.run_pending()
    assert len(results) == 1
    assert results[0].job_name == "due"
    assert results[0].output == "done"
    assert due.calls == ["now"]
    assert later.calls == []


def test_run_pending_advances_next_fire_and_count():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    job = s.add_job(FakeAgent(), "go", cron("0 9 * * *"), start_immediately=True)
    s.run_pending()
    assert job.run_count == 1
    assert job.next_fire == datetime(2026, 6, 14, 9, 0)
    # Not due again at the same instant.
    assert s.run_pending() == []


def test_max_runs_exhausts_job():
    clock = Clock(datetime(2026, 6, 13, 12, 0))
    s = AgentScheduler(now=clock)
    job = s.add_job(FakeAgent(), "go", every(seconds=60), max_runs=2, start_immediately=True)
    s.run_pending()  # run 1 (immediate)
    clock.advance(seconds=60)
    s.run_pending()  # run 2
    clock.advance(seconds=60)
    assert s.run_pending() == []  # exhausted
    assert job.run_count == 2
    assert job.exhausted
    assert job.next_fire is None


def test_failing_job_is_isolated_and_recorded():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    bad = FakeAgent(name="bad", raises=True)
    good = FakeAgent(name="good")
    s.add_job(bad, "x", cron("0 9 * * *"), start_immediately=True)
    s.add_job(good, "y", cron("0 9 * * *"), start_immediately=True)
    results = s.run_pending()
    by_name = {r.job_name: r for r in results}
    assert by_name["bad"].error is not None
    assert "RuntimeError" in by_name["bad"].error
    assert not by_name["bad"].ok
    assert by_name["good"].ok
    assert by_name["good"].output == "done"
    # The failing job still advanced (won't wedge re-firing the same instant).
    assert s.due_jobs() == []


def test_on_result_callback_receives_results():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    seen = []
    s.add_job(FakeAgent(), "go", cron("0 9 * * *"), on_result=seen.append, start_immediately=True)
    s.run_pending()
    assert len(seen) == 1
    assert isinstance(seen[0], JobResult)
    assert seen[0].ok


def test_on_result_callback_error_does_not_break_scheduler():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)

    def boom(_):
        raise ValueError("callback fail")

    job = s.add_job(FakeAgent(), "go", cron("0 9 * * *"), on_result=boom, start_immediately=True)
    results = s.run_pending()  # must not raise
    assert results[0].ok
    assert job.run_count == 1


def test_disabled_job_does_not_fire():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    job = s.add_job(FakeAgent(), "go", cron("0 9 * * *"), start_immediately=True)
    job.enabled = False
    assert s.run_pending() == []


def test_remove_job():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    s.add_job(FakeAgent(name="a"), "go", cron("0 9 * * *"), name="a")
    assert s.remove_job("a") is True
    assert s.remove_job("a") is False
    assert s.jobs == ()


def test_max_runs_validation():
    s = AgentScheduler()
    with pytest.raises(ValueError):
        s.add_job(FakeAgent(), "go", every(seconds=60), max_runs=0)


# --------------------------------------------------------------------------- #
# Async loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_astart_runs_until_deadline():
    # Real (fast) interval; bounded by `until` so the test terminates.
    start = datetime.now()
    s = AgentScheduler()
    agent = FakeAgent()
    s.add_job(agent, "tick", every(seconds=1))
    results = await s.astart(poll_interval=0.05, until=start + timedelta(milliseconds=250))
    # Fires immediately on the first tick (next_after is ~1s out, but the
    # deadline bounds the loop); at minimum the loop exits cleanly.
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_arun_pending_uses_arun():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    agent = FakeAgent()
    s.add_job(agent, "go", cron("0 9 * * *"), start_immediately=True)
    results = await s.arun_pending()
    assert len(results) == 1
    assert agent.calls == ["go"]


@pytest.mark.asyncio
async def test_astart_stops_when_all_jobs_exhausted():
    clock = Clock(datetime(2026, 6, 13, 9, 0))
    s = AgentScheduler(now=clock)
    # Job due immediately, single run -> loop should exit once exhausted.
    s.add_job(FakeAgent(), "go", every(seconds=1), max_runs=1, start_immediately=True)
    results = await s.astart(poll_interval=0.01)
    assert len(results) == 1
    assert s.jobs[0].exhausted
