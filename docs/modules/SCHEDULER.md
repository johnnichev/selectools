# Scheduled Agents

Run an [`Agent`](AGENT.md) on a recurring **cron** or **interval** schedule for
periodic, unattended work — monitoring, reporting, digesting, cleanup.

`AgentScheduler` is stdlib-only (`asyncio` + `datetime`): no APScheduler or cron
daemon dependency. It is marked `@beta`.

## Quick start

```python
from selectools import Agent, AgentScheduler, cron, every

agent = Agent(tools=[...], provider=provider)

scheduler = AgentScheduler()
scheduler.add_job(agent, "Summarize today's open incidents.", cron("0 9 * * *"))
scheduler.add_job(agent, "Poll the status page for changes.", every(minutes=5))

# Run the loop (async). Exits on stop(), the `until` deadline, or when every
# job has exhausted its max_runs.
await scheduler.astart()
```

Prefer to drive the clock yourself (your own loop, a serverless tick, a test)?
Fire whatever is due in one call:

```python
results = scheduler.run_pending()      # sync, uses agent.run
results = await scheduler.arun_pending()  # async, uses agent.arun
```

## Schedules

### Cron

`cron("minute hour day-of-month month day-of-week")` — standard 5-field syntax,
minute resolution.

| Field | Range | Notes |
|---|---|---|
| minute | 0–59 | |
| hour | 0–23 | |
| day-of-month | 1–31 | |
| month | 1–12 | |
| day-of-week | 0–7 | 0 and 7 are both Sunday |

Each field supports `*`, `*/step`, `a`, `a-b`, `a-b/step`, and comma lists
(`a,b,c`). When **both** day-of-month and day-of-week are restricted, a day
matches if **either** matches — standard Vixie-cron semantics.

```python
cron("*/15 9-17 * * 1-5")   # every 15 min, 9am–5pm, Mon–Fri
cron("0 0 1 * *")           # midnight on the 1st of each month
cron("0 0 1 * 0")           # midnight on the 1st OR any Sunday
```

### Interval

`every(seconds=…, minutes=…, hours=…)` — a fixed interval, minimum one second.

```python
every(seconds=30)
every(minutes=5)
every(hours=1, minutes=30)   # 90 minutes
```

## Jobs

`add_job` returns a `ScheduledJob` you can inspect or mutate:

```python
job = scheduler.add_job(
    agent, "Daily digest", cron("0 8 * * *"),
    name="digest",          # defaults to the agent's name, then job-N
    max_runs=30,            # stop after N firings (None = unlimited)
    on_result=handle,       # callback(JobResult) after each run
    start_immediately=True, # fire on the next tick, then resume the cadence
)
job.enabled = False         # pause without removing
scheduler.remove_job("digest")
```

By default the first fire is the next time the schedule comes due **strictly
after now** (an `every(minutes=5)` job first fires in five minutes). Pass
`start_immediately=True` for the common "run now, then on schedule" pattern.

## Results and failure isolation

Each firing produces a `JobResult`:

```python
@dataclass
class JobResult:
    job_name: str
    fired_at: datetime
    run_index: int
    output: str | None     # the agent's answer, on success
    error: str | None      # "ExcType: message", on failure
    # .ok -> error is None
```

A job that raises is recorded on its `JobResult.error` (and `job.last_result`)
and **never stops sibling jobs or the scheduler loop**. The cadence is anchored
to the scheduled fire time, not wall-clock-after-run, so a slow agent does not
drift the schedule.

## Driving the loop

| Method | Use |
|---|---|
| `await scheduler.astart(poll_interval=1.0, until=None)` | Run until `stop()`, the `until` deadline, or all jobs exhaust. Returns every `JobResult`. |
| `scheduler.run_pending(now=None)` | Fire everything due now once (sync, `agent.run`). |
| `await scheduler.arun_pending(now=None)` | Same, async (`agent.arun`). |
| `scheduler.stop()` | Signal `astart` to exit after the current iteration. |
| `scheduler.due_jobs(now=None)` | The jobs that would fire now. |

For deterministic tests, inject a clock: `AgentScheduler(now=lambda: fixed_dt)`.

## Pairing with Agent-as-API

A scheduler and [Agent-as-API](SERVE.md) compose: serve your agent over HTTP for
on-demand calls while a scheduler drives the same agent on a cadence in the
background. Expose `scheduler.jobs` and each `job.last_result` from your own
route to surface schedule health.
