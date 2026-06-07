from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta

from voiceui.logs import reset_logging
from voiceui.reminders import ReminderScheduler, parse_reminder_request


class ReminderParsingTests(unittest.TestCase):
    def test_parse_relative_alarm_request(self) -> None:
        now = datetime(2026, 6, 8, 4, 1, tzinfo=UTC)

        parsed = parse_reminder_request("你可以定一个一分钟后的闹钟吗？", now=now)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.delay_seconds, 60)
        self.assertEqual(parsed.kind, "alarm")
        self.assertEqual(parsed.label, "1分钟后")
        self.assertEqual(parsed.text, "闹钟时间到了。")

    def test_parse_reminder_message(self) -> None:
        now = datetime(2026, 6, 8, 4, 1, tzinfo=UTC)

        parsed = parse_reminder_request("十分钟后提醒我喝水", now=now)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.delay_seconds, 600)
        self.assertEqual(parsed.kind, "reminder")
        self.assertEqual(parsed.text, "提醒你：喝水。")

    def test_do_not_treat_do_not_forget_as_cancel(self) -> None:
        now = datetime(2026, 6, 8, 4, 1, tzinfo=UTC)

        parsed = parse_reminder_request("不要忘了十分钟后提醒我喝水", now=now)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.text, "提醒你：喝水。")

    def test_parse_absolute_tomorrow_time(self) -> None:
        now = datetime(2026, 6, 8, 20, 1, tzinfo=UTC)

        parsed = parse_reminder_request("明天早上七点提醒我开会", now=now)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.due_at, datetime(2026, 6, 9, 7, 0, tzinfo=UTC))
        self.assertEqual(parsed.text, "提醒你：开会。")

    def test_parse_requires_explicit_reminder_intent(self) -> None:
        now = datetime(2026, 6, 8, 4, 1, tzinfo=UTC)

        self.assertIsNone(parse_reminder_request("一分钟以后呢", now=now))


class ReminderSchedulerTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_logging()

    def test_scheduler_runs_due_reminder_once(self) -> None:
        now = datetime(2026, 6, 8, 4, 1, tzinfo=UTC)
        calls: list[str] = []
        scheduler = ReminderScheduler(
            lambda reminder: calls.append(reminder.text),
            now_fn=lambda: now,
        )
        try:
            with redirect_stdout(io.StringIO()):
                reminder = scheduler.schedule_at(now + timedelta(seconds=30), "提醒时间到了。")
                self.assertEqual(scheduler.run_due(now + timedelta(seconds=29)), [])
                self.assertEqual(scheduler.run_due(now + timedelta(seconds=30)), [reminder.id])

            self.assertEqual(calls, ["提醒时间到了。"])
            self.assertEqual(scheduler.pending(), [])
        finally:
            scheduler.stop()

    def test_scheduler_cancel_all_clears_pending_reminders(self) -> None:
        now = datetime(2026, 6, 8, 4, 1, tzinfo=UTC)
        scheduler = ReminderScheduler(lambda _reminder: None, now_fn=lambda: now)
        try:
            with redirect_stdout(io.StringIO()):
                scheduler.schedule_at(now + timedelta(seconds=30), "提醒时间到了。")
                self.assertEqual(scheduler.cancel_all(), 1)

            self.assertEqual(scheduler.pending(), [])
        finally:
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()
