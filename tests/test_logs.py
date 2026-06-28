from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from voiceui.logs import (
    configure_log_files,
    configure_logging,
    format_log,
    format_voice_context,
    is_log_enabled,
    log_continuous,
    log_event,
    log_switch_rows,
    record_text_event,
    reset_logging,
)
from voiceui.models import AssistantConfig, LoggingConfig, WakeConfig


class LogTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_logging()

    def test_format_log_uses_fixed_fields_and_millisecond_time(self) -> None:
        line = format_log(
            "vad",
            "completed",
            {"duration_ms": 320, "ok": True, "text": "hello world"},
            timestamp=datetime(2026, 6, 3, 19, 30, 1, 234567),
        )

        self.assertEqual(
            line,
            '2026-06-03T19:30:01.234 | module=vad | event=completed | '
            'params=duration_ms=320 ok=true text="hello world"',
        )

    def test_stt_text_is_highlighted_on_separate_line(self) -> None:
        line = format_log(
            "stt",
            "completed",
            {"latency_ms": 120, "text": "hello world"},
            timestamp=datetime(2026, 6, 3, 19, 30, 1, 234567),
        )

        self.assertEqual(
            line,
            "2026-06-03T19:30:01.234 | module=stt | event=completed | "
            "params=latency_ms=120\n"
            "    >>> ASR TEXT: hello world",
        )

    def test_tts_text_is_highlighted_on_separate_line(self) -> None:
        line = format_log(
            "tts",
            "completed",
            {"latency_ms": 240, "ok": True, "text": "first line\nsecond line"},
            timestamp=datetime(2026, 6, 3, 19, 30, 1, 234567),
        )

        self.assertEqual(
            line,
            "2026-06-03T19:30:01.234 | module=tts | event=completed | "
            "params=latency_ms=240 ok=true\n"
            "    >>> TTS TEXT: first line\n"
            "        second line",
        )

    def test_voice_context_line_uses_timestamp_role_and_text(self) -> None:
        line = format_voice_context(
            "stt",
            "completed",
            {"latency_ms": 120, "text": "hello world"},
            timestamp=datetime(2026, 6, 3, 19, 30, 1, 234567),
        )

        self.assertEqual(
            line,
            '2026-06-03T19:30:01.234 | context=voice | role=user | text="hello world"',
        )

    def test_event_logs_default_on_and_continuous_logs_default_off(self) -> None:
        configure_logging(LoggingConfig())

        output = io.StringIO()
        with redirect_stdout(output):
            log_event("vad", "completed", duration_ms=100)
            log_continuous("wake", "score", log_id="wake.score", elapsed_s=1.0)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertRegex(
            lines[0],
            re.escape(" | module=vad | event=completed | params=duration_ms=100") + "$",
        )

    def test_explicit_switches_override_defaults(self) -> None:
        configure_logging(
            LoggingConfig(
                events={"vad.completed": False},
                continuous={"wake.score": True},
            )
        )

        self.assertFalse(is_log_enabled("vad.completed"))
        self.assertTrue(is_log_enabled("wake.score"))

        output = io.StringIO()
        with redirect_stdout(output):
            log_event("vad", "completed", duration_ms=100)
            log_continuous("wake", "score", log_id="wake.score", elapsed_s=1.0)

        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_wake_debug_does_not_enable_wake_score_by_default(self) -> None:
        config = AssistantConfig(wake=WakeConfig(debug=True))

        wake_score = next(row for row in log_switch_rows(config) if row["id"] == "wake.score")

        self.assertFalse(wake_score["default_enabled"])
        self.assertFalse(wake_score["enabled"])

    def test_runtime_logs_are_mirrored_to_debug_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_log_path = f"{temp_dir}/debug.log"
            configure_log_files(debug_log_path=debug_log_path)

            with redirect_stdout(io.StringIO()):
                log_event("vad", "completed", duration_ms=100)

            content = open(debug_log_path, encoding="utf-8").read()
            self.assertIn("module=vad | event=completed", content)
            self.assertIn("params=duration_ms=100", content)

    def test_service_stdout_mode_routes_logs_to_file_and_context_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_log_path = f"{temp_dir}/debug.log"
            configure_log_files(
                debug_log_path=debug_log_path,
                stdout_mode="errors_and_voice_context",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                log_event("vad", "completed", duration_ms=100)
                log_event("stt", "completed", latency_ms=10, text="hello")
                log_event("tts", "completed", latency_ms=20, text="hi")
                log_event("error", "runtime", error="boom")

            stdout_text = output.getvalue()
            self.assertNotIn("module=vad | event=completed", stdout_text)
            self.assertIn("context=voice | role=user | text=hello", stdout_text)
            self.assertIn("context=voice | role=assistant | text=hi", stdout_text)
            self.assertIn("module=error | event=runtime", stdout_text)

            file_text = Path(debug_log_path).read_text(encoding="utf-8")
            self.assertIn("module=vad | event=completed", file_text)
            self.assertIn("module=stt | event=completed", file_text)
            self.assertIn("module=tts | event=completed", file_text)
            self.assertIn("module=error | event=runtime", file_text)

    def test_text_records_are_written_as_daily_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_log_files(text_record_dir=temp_dir)

            with redirect_stdout(io.StringIO()):
                log_event("stt", "completed", latency_ms=10, text="hello")
                log_event("tts", "completed", latency_ms=20, text="hi")
            record_text_event("llm", "completed", "hi", mode="test")

            files = list(Path(temp_dir).glob("voice_text_*.jsonl"))
            self.assertEqual(len(files), 1)
            records = [
                json.loads(line)
                for line in files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(record["module"], record["role"], record["text"]) for record in records],
                [
                    ("stt", "user", "hello"),
                    ("tts", "assistant", "hi"),
                    ("llm", "assistant", "hi"),
                ],
            )

    def test_barge_in_stream_replay_is_not_written_as_duplicate_text_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_log_files(text_record_dir=temp_dir)

            with redirect_stdout(io.StringIO()):
                log_event(
                    "stt",
                    "completed",
                    latency_ms=10,
                    source="barge_in",
                    text="停一下",
                )
                log_event(
                    "stt",
                    "completed",
                    latency_ms=0,
                    source="barge_in_stream",
                    text="停一下",
                )

            files = list(Path(temp_dir).glob("voice_text_*.jsonl"))
            self.assertEqual(len(files), 1)
            records = [
                json.loads(line)
                for line in files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["text"] for record in records], ["停一下"])


if __name__ == "__main__":
    unittest.main()
