"""
Scheduled agents — run an Agent on a cron or interval schedule.

Demonstrates:
  - cron() and every() schedules
  - start_immediately for "run now, then on cadence"
  - run_pending() to drive the scheduler with your own clock
  - failure isolation + JobResult inspection

Run: python examples/113_scheduled_agents.py
"""

from datetime import datetime

from selectools import Agent, AgentConfig, AgentScheduler, cron, every, tool
from selectools.providers.stubs import LocalProvider


@tool(description="Look up the current incident count")
def incident_count() -> str:
    return "incidents: 0 open"


def main() -> None:
    # A trivial agent (LocalProvider needs no API key / network).
    agent = Agent(
        tools=[incident_count],
        provider=LocalProvider(),
        config=AgentConfig(name="reporter"),
    )

    scheduler = AgentScheduler()

    # Daily 9am digest.
    scheduler.add_job(agent, "Summarize today's open incidents.", cron("0 9 * * *"))

    # Every 5 minutes, starting on the next tick.
    scheduler.add_job(
        agent,
        "Poll the status page for changes.",
        every(minutes=5),
        name="status-poll",
        start_immediately=True,
    )

    print("Registered jobs:")
    for job in scheduler.jobs:
        print(f"  {job.name:12s} next_fire={job.next_fire}")

    # Drive it ourselves: fire everything due right now.
    results = scheduler.run_pending()
    print(f"\nFired {len(results)} job(s) on this tick:")
    for r in results:
        status = "ok" if r.ok else f"ERROR {r.error}"
        print(f"  {r.job_name:12s} run#{r.run_index}  {status}")

    # In production you would instead run the async loop:
    #     await scheduler.astart()
    # which sleeps until the next due job and fires it, until stop() or `until`.
    print(f"\n(now = {datetime.now():%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    main()
