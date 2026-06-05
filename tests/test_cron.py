from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime

from voiceui.cron import CronSchedule, CronScheduler, build_runtime_jobs
from voiceui.logs import reset_logging
from voiceui.models import CronConfig, CronJobConfig


class CronScheduleTests(unittest.TestCase):
    def test_schedule_matches_steps_ranges_and_names(self) -> None:
        schedule = CronSchedule.parse("*/15 8-10 * jan,feb mon-fri")

        self.assertTrue(schedule.matches(datetime(2026, 1, 5, 8, 30)))
        self.assertFalse(schedule.matches(datetime(2026, 1, 5, 8, 31)))
        self.assertFalse(schedule.matches(datetime(2026, 3, 5, 8, 30)))

    def test_day_of_month_and_day_of_week_use_cron_or_semantics(self) -> None:
        schedule = CronSchedule.parse("0 9 1 * mon")

        self.assertTrue(schedule.matches(datetime(2026, 6, 8, 9, 0)))
        self.assertTrue(schedule.matches(datetime(2026, 7, 1, 9, 0)))
        self.assertFalse(schedule.matches(datetime(2026, 7, 2, 9, 0)))

    def test_day_of_week_accepts_zero_or_seven_for_sunday(self) -> None:
        zero = CronSchedule.parse("0 9 * * 0")
        seven = CronSchedule.parse("0 9 * * 7")
        sunday = datetime(2026, 6, 7, 9, 0)

        self.assertTrue(zero.matches(sunday))
        self.assertTrue(seven.matches(sunday))

    def test_next_after_returns_next_matching_minute(self) -> None:
        schedule = CronSchedule.parse("*/20 * * * *")

        self.assertEqual(
            schedule.next_after(datetime(2026, 6, 5, 7, 21, 30)),
            datetime(2026, 6, 5, 7, 40),
        )

    def test_invalid_schedule_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CronSchedule.parse("60 * * * *")


class CronSchedulerTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_logging()

    def test_build_runtime_jobs_validates_enabled_jobs(self) -> None:
        with self.assertRaises(ValueError):
            build_runtime_jobs(
                CronConfig(
                    enabled=True,
                    jobs=[CronJobConfig(id="bad", schedule="* * * * *", text="")],
                )
            )

    def test_scheduler_runs_due_job_once_per_minute(self) -> None:
        calls: list[str] = []
        scheduler = CronScheduler(
            CronConfig(
                enabled=True,
                jobs=[
                    CronJobConfig(
                        id="morning_weather",
                        schedule="30 7 * * *",
                        text="weather",
                    )
                ],
            ),
            handler=lambda job: calls.append(job.text),
        )

        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                scheduler.run_pending(datetime(2026, 6, 5, 7, 30, 5)),
                ["morning_weather"],
            )
            self.assertEqual(scheduler.run_pending(datetime(2026, 6, 5, 7, 30, 55)), [])
            self.assertEqual(scheduler.run_pending(datetime(2026, 6, 5, 7, 31, 0)), [])

        self.assertEqual(calls, ["weather"])

    def test_scheduler_converts_aware_time_to_job_timezone(self) -> None:
        calls: list[str] = []
        scheduler = CronScheduler(
            CronConfig(
                enabled=True,
                jobs=[
                    CronJobConfig(
                        id="shanghai_evening",
                        schedule="0 20 * * *",
                        text="evening",
                        timezone="Asia/Shanghai",
                    )
                ],
            ),
            handler=lambda job: calls.append(job.text),
        )

        with redirect_stdout(io.StringIO()):
            fired = scheduler.run_pending(datetime.fromisoformat("2026-06-05T12:00:00+00:00"))

        self.assertEqual(fired, ["shanghai_evening"])
        self.assertEqual(calls, ["evening"])


if __name__ == "__main__":
    unittest.main()
