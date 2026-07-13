from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from voiceui.__main__ import main
from voiceui.audio_dump import configure_audio_dump
from voiceui.logs import log_event, reset_logging
from voiceui.models import AssistantConfig, WakeEvent


class _FakeAudioInput:
    sample_rate = 16000
    block_ms = 80
    selected_channel = 0


class MainTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_audio_dump(None)
        reset_logging()

    def test_wake_test_keeps_listening_and_only_prints_wake_logs(self) -> None:
        detector = MagicMock()
        detector.wait.side_effect = [
            WakeEvent(engine="wekws_mha", confidence=0.998, label="hey_leela", pcm=b"1"),
            WakeEvent(engine="wekws_mha", confidence=0.999, label="hello_leela", pcm=b"2"),
            KeyboardInterrupt(),
        ]
        audio_dump = MagicMock()
        audio_dump.debug_log_path.return_value = None
        audio_dump.text_record_dir.return_value = None

        def create_audio(*_args, **_kwargs):
            log_event("audio", "stream_opened", log_id="audio.stream_opened", device="test")
            return _FakeAudioInput()

        output = io.StringIO()
        with (
            patch("voiceui.__main__.load_dotenv"),
            patch("voiceui.__main__.load_config", return_value=AssistantConfig()),
            patch("voiceui.__main__.AudioDumpManager", return_value=audio_dump),
            patch("voiceui.__main__.create_audio_input", side_effect=create_audio),
            patch("voiceui.__main__.create_wake_detector", return_value=detector),
            patch("voiceui.__main__._save_wake_debug") as save_wake_debug,
            redirect_stdout(output),
        ):
            result = main(["--config", "test.yaml", "--wake-test"])

        self.assertEqual(result, 130)
        self.assertEqual(detector.wait.call_count, 3)
        self.assertEqual(save_wake_debug.call_count, 2)
        stdout_text = output.getvalue()
        self.assertEqual(stdout_text.count("module=wake | event=detected"), 2)
        self.assertEqual(stdout_text.count("module=wake | event=test_waiting"), 3)
        self.assertNotIn("module=audio", stdout_text)


if __name__ == "__main__":
    unittest.main()
